use crate::instancing::InstanceMaterialData;
use crate::{AppState, PhysicsContext, RunState, Timestamps};
use async_channel::{Receiver, Sender};
use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::tasks::ComputeTaskPool;
use wgcore::kernel::KernelInvocationQueue;
use wgcore::timestamps::GpuTimestamps;

#[derive(Resource)]
pub struct TimestampChannel {
    pub snd: Sender<Timestamps>,
    pub rcv: Receiver<Timestamps>,
}

pub fn step_simulation(
    mut timings: ResMut<Timestamps>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut physics: ResMut<PhysicsContext>,
    mut app_state: ResMut<AppState>,
    particles: Query<&InstanceMaterialData>,
    timings_channel: Res<TimestampChannel>,
) {
    if app_state.run_state == RunState::Paused {
        return;
    }

    let timings = &mut *timings;

    while let Ok(new_timings) = timings_channel.rcv.try_recv() {
        *timings = new_timings;
    }

    timings.timestamps.as_mut().map(|t| t.clear());

    // Run the simulation.
    let device = render_device.wgpu_device();
    let physics = &mut *physics;
    let compute_queue = &*render_queue.0;
    let mut queue = KernelInvocationQueue::new(device);
    let mut encoder = device.create_command_encoder(&Default::default());

    // TODO: we could do the queue_step just once?
    app_state
        .pipeline
        .queue_step(&mut physics.data, &mut queue, timings.timestamps.is_some());

    for _ in 0..app_state.num_substeps {
        queue.encode(&mut encoder, timings.timestamps.as_mut());
    }

    timings.timestamps.as_mut().map(|t| t.resolve(&mut encoder));

    // Prepare the vertex buffer.
    if let Ok(instances_buffer) = particles.get_single() {
        queue.clear();
        app_state.prep_vertex_buffer.queue(
            &mut queue,
            &app_state.gpu_render_config,
            &physics.data.particles,
            &physics.data.grid,
            &physics.data.sim_params,
            &instances_buffer.buffer.buffer,
        );
        queue.encode(&mut encoder, timings.timestamps.as_mut());
    }

    // Submit.
    compute_queue.submit(Some(encoder.finish()));

    if let Some(timestamps) = std::mem::take(&mut timings.timestamps) {
        let timings_snd = timings_channel.snd.clone();
        let timestamp_period = compute_queue.get_timestamp_period();
        let num_substeps = app_state.num_substeps;
        let timestamps_future = async move {
            let values = timestamps.wait_for_results_async().await.unwrap();
            let timestamps_ms = GpuTimestamps::timestamps_to_ms(&values, timestamp_period);
            let mut new_timings = Timestamps {
                timestamps: Some(timestamps),
                ..Default::default()
            };

            for i in 0..num_substeps {
                let times = &timestamps_ms[i * 12..];
                new_timings.grid_sort += times[1] - times[0];
                new_timings.p2g += times[3] - times[2];
                new_timings.grid_update += times[5] - times[4];
                new_timings.g2p += times[7] - times[6];
                new_timings.particles_update += times[9] - times[8];
                new_timings.integrate_bodies += times[11] - times[10];
            }
            timings_snd.send(new_timings).await.unwrap();
        };

        ComputeTaskPool::get().spawn(timestamps_future).detach();
    }

    if app_state.run_state == RunState::Step {
        app_state.run_state = RunState::Paused;
    }
}

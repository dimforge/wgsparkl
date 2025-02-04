use crate::instancing::InstanceMaterialData;
use crate::{AppState, PhysicsContext, RunState, Timestamps};
use async_channel::{Receiver, Sender};
use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue, WgpuWrapper};
use bevy::tasks::ComputeTaskPool;
use std::time::Instant;
use wgcore::kernel::KernelInvocationQueue;
use wgcore::re_exports::encase::StorageBuffer;
use wgcore::timestamps::GpuTimestamps;
use wgparry2d::math::GpuSim;
use wgpu::{Device, Queue};
use wgsparkl::rapier::math::Vector;
use wgsparkl::wgrapier::dynamics::{GpuBodySet, GpuVelocity};

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
    step_simulation_legacy(
        timings,
        render_device,
        render_queue,
        physics,
        app_state,
        particles,
        timings_channel,
    )
}

pub fn step_simulation_new(
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

    // Step the simulation.
    app_state
        .pipeline
        .queue_step(&mut physics.data, &mut queue, false); // timings.timestamps.is_some());

    let t0 = Instant::now();
    for _ in 0..app_state.num_substeps {
        simulate_single_substep(
            device,
            app_state.num_substeps,
            &compute_queue,
            &queue,
            physics,
        );
    }
    println!("Substep times: {}", t0.elapsed().as_secs_f32());

    // timings.timestamps.as_mut().map(|t| t.resolve(&mut encoder));

    // Prepare the vertex buffer for rendering the particles.
    if let Ok(instances_buffer) = particles.get_single() {
        let mut encoder = device.create_command_encoder(&Default::default());
        queue.clear();
        app_state.prep_vertex_buffer.queue(
            &mut queue,
            &app_state.gpu_render_config,
            &physics.data.particles,
            &physics.data.grid,
            &physics.data.sim_params,
            &instances_buffer.buffer.buffer,
        );
        queue.encode(&mut encoder, None); // timings.timestamps.as_mut());

        // Submit.
        compute_queue.submit(Some(encoder.finish()));
    }

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

fn simulate_single_substep(
    device: &Device,
    num_substeps: usize,
    compute_queue: &WgpuWrapper<Queue>,
    queue: &KernelInvocationQueue,
    physics: &mut PhysicsContext,
) {
    // Run the simulation.
    let mut encoder = device.create_command_encoder(&Default::default());

    // 1. Send updated bodies information to the gpu.
    // PERF: don’t reallocate the buffers at each step.
    /*
        let poses_data: Vec<GpuSim> = physics
            .rapier_data
            .colliders
            .iter()
            .map(|(_, c)| (*c.position()).into())
            .collect();
        compute_queue.write_buffer(
            physics.data.bodies.poses().buffer(),
            0,
            bytemuck::cast_slice(&poses_data),
        );

        let vels_data: Vec<_> = physics
            .rapier_data
            .colliders
            .iter()
            .map(|(_, c)| {
                c.parent()
                    .and_then(|rb| physics.rapier_data.bodies.get(rb))
                    .map(|rb| GpuVelocity {
                        linear: *rb.linvel(),
                        angular: rb.angvel().clone(),
                    })
                    .unwrap_or_default()
            })
            .collect();
        let mut vels_bytes = vec![];
        let mut buffer = StorageBuffer::new(&mut vels_bytes);
        buffer.write(&vels_data).unwrap();
        compute_queue.write_buffer(physics.data.bodies.vels().buffer(), 0, &vels_bytes);
    */

    // Run a single simulation substep.
    queue.encode(&mut encoder, None); // timings.timestamps.as_mut());

    // Submit.
    compute_queue.submit(Some(encoder.finish()));

    device.poll(wgpu::Maintain::Wait);

    /*
    // Step the rigid-body simulation.
    let mut params = physics.rapier_data.params;
    params.dt = 0.016 / num_substeps as f32;
    physics.rapier_data.physics_pipeline.step(
        &(Vector::y() * -9.81),
        &physics.rapier_data.params,
        &mut physics.rapier_data.islands,
        &mut physics.rapier_data.broad_phase,
        &mut physics.rapier_data.narrow_phase,
        &mut physics.rapier_data.bodies,
        &mut physics.rapier_data.colliders,
        &mut physics.rapier_data.impulse_joints,
        &mut physics.rapier_data.multibody_joints,
        &mut physics.rapier_data.ccd_solver,
        None,
        &(),
        &(),
    );
     */
}

pub fn step_simulation_legacy(
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

    // Send updated bodies information to the gpu.
    // PERF: don’t reallocate the buffers at each step.
    let poses_data: Vec<GpuSim> = physics
        .rapier_data
        .colliders
        .iter()
        .map(|(_, c)| (*c.position()).into())
        .collect();
    compute_queue.write_buffer(
        physics.data.bodies.poses().buffer(),
        0,
        bytemuck::cast_slice(&poses_data),
    );

    let gravity = Vector::y() * -9.81;
    let vels_data: Vec<_> = physics
        .rapier_data
        .colliders
        .iter()
        .map(|(_, c)| {
            c.parent()
                .and_then(|rb| physics.rapier_data.bodies.get(rb))
                .map(|rb| GpuVelocity {
                    linear: *rb.linvel()
                        + gravity * physics.rapier_data.params.dt * (rb.is_dynamic() as u32 as f32)
                            / app_state.num_substeps as f32,
                    angular: rb.angvel().clone(),
                })
                .unwrap_or_default()
        })
        .collect();
    let mut vels_bytes = vec![];
    let mut buffer = StorageBuffer::new(&mut vels_bytes);
    buffer.write(&vels_data).unwrap();
    compute_queue.write_buffer(physics.data.bodies.vels().buffer(), 0, &vels_bytes);

    //// Step the simulation.
    // Queue the impulse reset first.
    app_state
        .pipeline
        .impulses
        .queue_reset(&mut queue, &physics.data.impulses);
    queue.encode(&mut encoder, None);

    queue.clear();

    app_state
        .pipeline
        .queue_step(&mut physics.data, &mut queue, timings.timestamps.is_some());

    for _ in 0..app_state.num_substeps {
        queue.encode(&mut encoder, timings.timestamps.as_mut());
    }
    physics
        .data
        .impulses
        .total_impulses_staging
        .copy_from_encased(&mut encoder, &physics.data.impulses.total_impulses);

    // physics
    //     .data
    //     .grid
    //     .nodes_cdf_staging
    //     .copy_from(&mut encoder, &physics.data.grid.nodes_cdf);
    // physics
    //     .data
    //     .particles
    //     .cdf_read
    //     .copy_from_encased(&mut encoder, &physics.data.particles.cdf);

    timings.timestamps.as_mut().map(|t| t.resolve(&mut encoder));

    // Prepare the vertex buffer for rendering the particles.
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

    let impulses = futures::executor::block_on(
        physics
            .data
            .impulses
            .total_impulses_staging
            .read_encased(device),
    )
    .unwrap();

    println!("Impulses: {:?}", impulses[8]);

    for (i, (_, rb)) in physics.rapier_data.bodies.iter_mut().enumerate() {
        let prev_vel = *rb.linvel();
        rb.apply_impulse(impulses[i].linear, true);
        rb.apply_torque_impulse(impulses[i].angular, true);
        let new_vel = *rb.linvel();
        if i == 8 {
            println!(
                "Mass: {}, Vel before: {:?}, after: {:?}, diff: {:?}",
                rb.mass(),
                prev_vel,
                new_vel,
                new_vel - prev_vel
            );
        }
    }

    // let buf =
    //     futures::executor::block_on(physics.data.grid.nodes_cdf_staging.read(device)).unwrap();
    // println!(
    //     "{:.x?}, any nonzero: {}",
    //     &buf.iter()
    //         .filter(|e| e.affinities != 0)
    //         .take(10)
    //         .collect::<Vec<_>>(),
    //     buf.iter().any(|e| e.affinities != 0)
    // );

    // let buf =
    //     futures::executor::block_on(physics.data.particles.cdf_read.read_encased(device)).unwrap();
    // println!(
    //     "{:x?}, any nonzero: {}",
    //     &buf.iter()
    //         .filter(|e| e.affinity != 0)
    //         .take(10)
    //         .collect::<Vec<_>>(),
    //     buf.iter().any(|e| e.affinity != 0)
    // );

    physics.rapier_data.physics_pipeline.step(
        &gravity,
        &physics.rapier_data.params,
        &mut physics.rapier_data.islands,
        &mut physics.rapier_data.broad_phase,
        &mut physics.rapier_data.narrow_phase,
        &mut physics.rapier_data.bodies,
        &mut physics.rapier_data.colliders,
        &mut physics.rapier_data.impulse_joints,
        &mut physics.rapier_data.multibody_joints,
        &mut physics.rapier_data.ccd_solver,
        None,
        &(),
        &(),
    );

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

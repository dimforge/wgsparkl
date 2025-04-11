use crate::instancing::InstanceMaterialData;
use crate::startup::RigidParticlesTag;
use crate::{AppState, Callbacks, PhysicsContext, RenderContext, RunState, Timestamps};
use async_channel::{Receiver, Sender};
use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::tasks::ComputeTaskPool;
use wgcore::kernel::KernelInvocationQueue;
use wgcore::re_exports::encase::StorageBuffer;
use wgcore::timestamps::GpuTimestamps;
use wgsparkl::rapier::math::Vector;
use wgsparkl::rapier::prelude::RigidBodyPosition;
use wgsparkl::wgparry::math::GpuSim;
use wgsparkl::wgrapier::dynamics::GpuVelocity;

#[derive(Resource)]
pub struct TimestampChannel {
    pub snd: Sender<Timestamps>,
    pub rcv: Receiver<Timestamps>,
}

pub fn callbacks(
    mut render: ResMut<RenderContext>,
    mut physics: ResMut<PhysicsContext>,
    app_state: ResMut<AppState>,
    timings: Res<Timestamps>,
    mut callbacks: ResMut<Callbacks>,
) {
    for to_call in callbacks.0.iter_mut() {
        to_call(
            Some(&mut render),
            &mut physics,
            timings.as_ref(),
            &app_state,
        );
    }
}

pub fn step_simulation(
    mut timings: ResMut<Timestamps>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut physics: ResMut<PhysicsContext>,
    mut app_state: ResMut<AppState>,
    particles: Query<&InstanceMaterialData, Without<RigidParticlesTag>>,
    rigid_particles: Query<&InstanceMaterialData, With<RigidParticlesTag>>,
    timings_channel: Res<TimestampChannel>,
) {
    // for _ in 0..10 {
    step_simulation_legacy(
        &mut timings,
        &render_device,
        &render_queue,
        &mut physics,
        &mut app_state,
        &particles,
        &rigid_particles,
        &timings_channel,
    )
    // }
}

pub fn step_simulation_legacy(
    timings: &mut Timestamps,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    physics: &mut PhysicsContext,
    app_state: &mut AppState,
    particles: &Query<&InstanceMaterialData, Without<RigidParticlesTag>>,
    rigid_particles: &Query<&InstanceMaterialData, With<RigidParticlesTag>>,
    timings_channel: &TimestampChannel,
) {
    let timings = &mut *timings;

    while let Ok(new_timings) = timings_channel.rcv.try_recv() {
        *timings = new_timings;
    }

    if let Some(t) = timings.timestamps.as_mut() {
        t.clear()
    }

    // Run the simulation.
    let device = render_device.wgpu_device();
    let physics = &mut *physics;
    let compute_queue = &*render_queue.0;
    let mut queue = KernelInvocationQueue::new(device);
    let mut encoder = device.create_command_encoder(&Default::default());

    // Send updated bodies information to the gpu.
    // PERF: don’t reallocate the buffers at each step.
    let poses_data: Vec<GpuSim> = physics
        .data
        .coupling()
        .iter()
        .map(|coupling| {
            let c = &physics.rapier_data.colliders[coupling.collider];
            #[cfg(feature = "dim2")]
            return (*c.position()).into();
            #[cfg(feature = "dim3")]
            return GpuSim::from_isometry(*c.position(), 1.0);
        })
        .collect();
    // println!("poses: {:?}", poses_data);
    compute_queue.write_buffer(
        physics.data.bodies.poses().buffer(),
        0,
        bytemuck::cast_slice(&poses_data),
    );

    let divisor = 1.0; // app_state.num_substeps as f32;
    let gravity = Vector::y() * -9.81;
    let vels_data: Vec<_> = physics
        .data
        .coupling()
        .iter()
        .map(|coupling| {
            let rb = &physics.rapier_data.bodies[coupling.body];
            GpuVelocity {
                linear: *rb.linvel()
                    + gravity
                        * physics.rapier_data.integration_parameters.dt
                        * (rb.is_dynamic() as u32 as f32)
                        / (app_state.num_substeps as f32),
                #[allow(clippy::clone_on_copy)] // Needed for the 2d/3d switch.
                angular: rb.angvel().clone(),
            }
        })
        .collect();

    let mut vels_bytes = vec![];
    let mut buffer = StorageBuffer::new(&mut vels_bytes);
    buffer.write(&vels_data).unwrap();
    compute_queue.write_buffer(physics.data.bodies.vels().buffer(), 0, &vels_bytes);

    //// Step the simulation.
    app_state
        .pipeline
        .queue_step(&mut physics.data, &mut queue, timings.timestamps.is_some());

    for _ in 0..app_state.num_substeps {
        queue.encode(&mut encoder, timings.timestamps.as_mut());
    }
    physics
        .data
        .poses_staging
        .copy_from(&mut encoder, physics.data.bodies.poses());

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

    if let Some(t) = timings.timestamps.as_mut() {
        t.resolve(&mut encoder)
    }

    // Prepare the vertex buffer for rendering the particles.
    if let Ok(instances_buffer) = particles.get_single() {
        queue.clear();
        app_state.prep_vertex_buffer.queue(
            &mut queue,
            &app_state.gpu_render_config,
            &physics.data.particles,
            &physics.data.rigid_particles,
            &physics.data.grid,
            &physics.data.sim_params,
            &instances_buffer.buffer.buffer,
            rigid_particles
                .get_single()
                .ok()
                .map(|b| &**b.buffer.buffer),
        );
        queue.encode(&mut encoder, timings.timestamps.as_mut());
    }

    // Submit.
    compute_queue.submit(Some(encoder.finish()));

    // FIXME: make the readback work on wasm too.
    //        Currently, this means there won’t be any two-ways coupling on wasm.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let new_poses =
            futures::executor::block_on(physics.data.poses_staging.read(device)).unwrap();

        // println!("Impulses: {:?}", new_poses[8]);

        for (i, coupling) in physics.data.coupling().iter().enumerate() {
            let rb = &mut physics.rapier_data.bodies[coupling.body];
            if rb.is_dynamic() {
                let interpolator = RigidBodyPosition {
                    position: *rb.position(),
                    #[cfg(feature = "dim2")]
                    next_position: new_poses[i].similarity.isometry,
                    #[cfg(feature = "dim3")]
                    next_position: new_poses[i].isometry,
                };
                let vel = interpolator.interpolate_velocity(
                    1.0 / (physics.rapier_data.integration_parameters.dt / divisor),
                    &rb.mass_properties().local_mprops.local_com,
                );
                rb.set_linvel(vel.linvel, true);
                rb.set_angvel(vel.angvel, true);
                // println!("dvel: {:?}", vel.linvel - vel_before);
            }
        }
    }

    let mut params = physics.rapier_data.integration_parameters;
    params.dt /= divisor;
    physics.rapier_data.physics_pipeline.step(
        &nalgebra::zero(),
        &params, // physics.rapier_data.params,
        &mut physics.rapier_data.island_manager,
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
                let mut timings = [
                    &mut new_timings.update_rigid_particles,
                    &mut new_timings.grid_sort,
                    &mut new_timings.grid_update_cdf,
                    &mut new_timings.p2g_cdf,
                    &mut new_timings.g2p_cdf,
                    &mut new_timings.p2g,
                    &mut new_timings.grid_update,
                    &mut new_timings.g2p,
                    &mut new_timings.particles_update,
                    &mut new_timings.integrate_bodies,
                ];
                let times = &timestamps_ms[i * timings.len() * 2..];

                for (k, timing) in timings.iter_mut().enumerate() {
                    **timing += times[k * 2 + 1] - times[k * 2];
                }
            }
            timings_snd.send(new_timings).await.unwrap();
        };

        ComputeTaskPool::get().spawn(timestamps_future).detach();
    }

    if app_state.run_state == RunState::Step {
        app_state.run_state = RunState::Paused;
    }
}

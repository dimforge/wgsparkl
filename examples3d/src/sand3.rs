use bevy::log::warn;
use wgsparkl_testbed3d::step::callbacks;
use wgsparkl_testbed3d::{wgsparkl, Callbacks, RapierData};

use bevy::render::renderer::RenderDevice;
use nalgebra::vector;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder};
use wgsparkl::models::DruckerPrager;
use wgsparkl::{
    models::ElasticCoefficients,
    pipeline::MpmData,
    solver::{Particle, ParticleDynamics, SimulationParams},
};
use wgsparkl_testbed3d::{AppState, PhysicsContext};

pub fn sand_demo(
    render_device: RenderDevice,
    app_state: &mut AppState,
    callbacks: &mut Callbacks,
) -> PhysicsContext {
    let mut rapier_data = RapierData::default();
    let captured_render_device = render_device.clone();
    let device = render_device.wgpu_device();

    let nxz = 45;
    let cell_width = 1.0;
    let mut particles = vec![];

    let colors = std::fs::read("particles_colors.bin");
    let colors = if let Ok(colors) = colors {
        Some(rkyv::from_bytes::<Vec<[u8; 3]>, rkyv::rancor::Error>(&colors).unwrap())
    } else {
        warn!("Could not find particles_colors.bin, generate it with `cargo run --example color_positions`.");
        None
    };

    let mut particle_index = 0;
    for i in 0..nxz {
        for j in 0..100 {
            for k in 0..nxz {
                let position = vector![
                    i as f32 + 0.5 - nxz as f32 / 2.0,
                    j as f32 + 0.5 + 10.0,
                    k as f32 + 0.5 - nxz as f32 / 2.0
                ] * cell_width
                    / 2.0;
                let density = 2700.0;
                let radius = cell_width / 4.0;
                particles.push(Particle {
                    position,
                    dynamics: ParticleDynamics::with_density(radius, density),
                    model: ElasticCoefficients::from_young_modulus(2_000_000_000.0, 0.2),
                    plasticity: Some(DruckerPrager::new(2_000_000_000.0, 0.2)),
                    phase: None,
                    color: colors.as_ref().map(|colors| {
                        [
                            colors[particle_index][0],
                            colors[particle_index][1],
                            colors[particle_index][2],
                            255,
                        ]
                    }),
                });
                particle_index += 1;
            }
        }
    }

    if !app_state.restarting {
        app_state.num_substeps = 20;
        app_state.gravity_factor = 1.0;
    };

    let params = SimulationParams {
        gravity: vector![0.0, -9.81, 0.0] * app_state.gravity_factor,
        dt: (1.0 / 60.0) / (app_state.num_substeps as f32),
    };

    rapier_data.insert_body_and_collider(
        RigidBodyBuilder::fixed().translation(vector![0.0, -4.0, 0.0]),
        ColliderBuilder::cuboid(100.0, 4.0, 100.0),
    );

    rapier_data.insert_body_and_collider(
        RigidBodyBuilder::fixed().translation(vector![0.0, 5.0, -35.0]),
        ColliderBuilder::cuboid(35.0, 5.0, 0.5),
    );
    rapier_data.insert_body_and_collider(
        RigidBodyBuilder::fixed().translation(vector![0.0, 5.0, 35.0]),
        ColliderBuilder::cuboid(35.0, 5.0, 0.5),
    );
    rapier_data.insert_body_and_collider(
        RigidBodyBuilder::fixed().translation(vector![-35.0, 5.0, 0.0]),
        ColliderBuilder::cuboid(0.5, 5.0, 35.0),
    );
    rapier_data.insert_body_and_collider(
        RigidBodyBuilder::fixed().translation(vector![35.0, 5.0, 0.0]),
        ColliderBuilder::cuboid(0.5, 5.0, 35.0),
    );

    let rb = RigidBodyBuilder::kinematic_velocity_based()
        .translation(vector![0.0, 1.5, 0.0])
        .rotation(vector![0.0, 0.0, -0.5])
        .angvel(vector![0.0, -1.0, 0.0]);
    let co = ColliderBuilder::cuboid(0.5, 1.5, 30.0);
    let (rb_moving_helix, _) = rapier_data.insert_body_and_collider(rb, co);

    let mut step_left_before_stop = 600;
    callbacks.0.push(Box::new(
        move |_render,
              physics: &mut PhysicsContext,
              _timestamps,
              _app_state: &AppState,
              render_queue| {
            step_left_before_stop -= 1;
            if step_left_before_stop < 0 {
                physics
                    .rapier_data
                    .bodies
                    .get_mut(rb_moving_helix)
                    .unwrap()
                    .set_angvel(vector![0.0, 0.0, 0.0], false);
            }
            if step_left_before_stop == -300 {
                let particles_positions = read_particles::read_particles_positions(
                    captured_render_device.clone(),
                    render_queue,
                    &physics.data.particles,
                );
                // FIXME: when wgsparkl updates to rkyv 0.8, we can get rid of the mapping.
                let particle_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(
                    &particles_positions
                        .iter()
                        .map(|p| [p.x, p.y, p.z])
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                std::fs::write("particles.bin", &particle_bytes).unwrap();
                println!("Exported particles.bin");
            }
        },
    ));

    let data = MpmData::new(
        device,
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        cell_width,
        60_000,
    );
    PhysicsContext {
        data,
        rapier_data,
        particles,
    }
}

pub mod read_particles {
    use super::*;

    use bevy::render::{render_resource::BufferUsages, renderer::RenderQueue};
    use nalgebra::Vector4;
    use wgcore::tensor::GpuVector;
    use wgsparkl3d::solver::GpuParticles;

    pub fn read_particles_positions(
        render_device: RenderDevice,
        render_queue: RenderQueue,
        particles: &GpuParticles,
    ) -> Vec<Vector4<f32>> {
        // Get the `render_device` is the `Res<RenderDevice>` bevy resource.
        let device = render_device.wgpu_device();
        // Create the staging buffer.
        // Here `particles` is of type `GpuParticles`, accessible from
        // `PhysicsContext::data::particles`.
        let positions_staging: GpuVector<Vector4<f32>> = GpuVector::uninit(
            device,
            particles.len() as u32,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        );

        // Copy the buffer.
        // `render_queue` the `Res<RenderQueue>` bevy resource.
        let compute_queue = &*render_queue.0;
        let mut encoder = device.create_command_encoder(&Default::default());
        positions_staging.copy_from(&mut encoder, &particles.positions);
        compute_queue.submit(Some(encoder.finish()));

        // Run the copy. The fourth component of each entry can be ignored.
        let positions: Vec<Vector4<f32>> =
            futures::executor::block_on(positions_staging.read(device)).unwrap();
        positions
    }
}

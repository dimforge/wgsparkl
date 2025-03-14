use wgsparkl_testbed3d::{wgsparkl, CallBeforeSimulation, RapierData};

use bevy::prelude::*;
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

use crate::stop_angular_velocity;
pub fn sand_demo(
    mut commands: Commands,
    device: Res<RenderDevice>,
    mut app_state: ResMut<AppState>,
) {
    let mut rapier_data = RapierData::default();
    let device = device.wgpu_device();

    let nxz = 45;
    let cell_width = 1.0;
    let mut particles = vec![];
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
                    color: None,
                });
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
    let (rb, _) = rapier_data.insert_body_and_collider(rb, co);

    let data = MpmData::new(
        device,
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        cell_width,
        60_000,
    );
    commands.insert_resource(PhysicsContext {
        data,
        rapier_data,
        particles,
    });
    let system_id = commands.register_system(stop_angular_velocity);
    commands.spawn(CallBeforeSimulation(system_id));

    // TODO: export particles positions after a bigger delay
    commands.spawn(StopAngVelAfter {
        steps_left: 100,
        rigidbody_handle: rb,
    });
}

#[derive(Debug, Component)]
pub struct StopAngVelAfter {
    /// Number of steps left before stopping the angular velocity.
    pub steps_left: u32,
    pub rigidbody_handle: rapier3d::dynamics::RigidBodyHandle,
}

pub fn stop_angular_velocity(
    mut stop_angvel: Query<&mut StopAngVelAfter>,
    mut physics: ResMut<PhysicsContext>,
) {
    for mut stop_angvel in stop_angvel.iter_mut() {
        if stop_angvel.steps_left == 0 {
            physics
                .rapier_data
                .bodies
                .get_mut(stop_angvel.rigidbody_handle)
                .unwrap()
                .set_angvel(vector![0.0, 0.0, 0.0], false);
        } else {
            stop_angvel.steps_left -= 1;
        }
    }
}

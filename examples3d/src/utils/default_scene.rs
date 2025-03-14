#![allow(unused)]

use bevy::prelude::*;
use nalgebra::vector;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder};
use wgsparkl3d::{
    models::ElasticCoefficients,
    solver::{Particle, ParticleDynamics, ParticlePhase},
};
use wgsparkl_testbed3d::{AppState, RapierData};

pub const SAMPLE_PER_UNIT: f32 = 10.0;

pub fn set_default_app_state(mut app_state: ResMut<'_, AppState>) {
    if !app_state.restarting {
        app_state.num_substeps = 20;
        app_state.gravity_factor = 1.0;
    };
}

/// Spawns a ground and 4 walls.
pub fn spawn_ground_and_walls(rapier_data: &mut RapierData) {
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
}

pub fn create_particle(pos: &Vec3, color: Option<Color>) -> Particle {
    let radius = 1f32 / SAMPLE_PER_UNIT / 2f32;
    let density = 3700.0;
    let particle = Particle {
        position: pos.to_array().into(),
        dynamics: ParticleDynamics::with_density(radius, density),
        model: ElasticCoefficients::from_young_modulus(10_000_000.0, 0.2),
        plasticity: None,
        phase: Some(ParticlePhase {
            phase: 1.0,
            max_stretch: f32::MAX,
        }),
        color: color.map(|c| c.to_linear().to_u8_array()),
    };
    particle
}

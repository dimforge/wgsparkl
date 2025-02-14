use wgsparkl_testbed2d::{wgsparkl, RapierData};

use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use nalgebra::{point, vector, Similarity2, Vector2};
use rapier2d::prelude::{ColliderBuilder, RigidBodyBuilder};
use wgebra::GpuSim2;
use wgparry2d::parry::shape::Cuboid;
use wgrapier2d::dynamics::{BodyDesc, GpuVelocity};
use wgsparkl::models::DruckerPrager;
use wgsparkl::solver::ParticlePhase;
use wgsparkl::{
    models::ElasticCoefficients,
    pipeline::MpmData,
    solver::{Particle, ParticleMassProps, SimulationParams},
};
use wgsparkl_testbed2d::{init_testbed, AppState, PhysicsContext, SceneInits};

fn main() {
    panic!("Run the `testbed3` example instead.");
}

pub fn elastic_cut_demo(
    mut commands: Commands,
    device: Res<RenderDevice>,
    mut app_state: ResMut<AppState>,
) {
    let mut rapier_data = RapierData::default();
    let device = device.wgpu_device();

    let offset_y = 46.0;
    // let cell_width = 0.1;
    let cell_width = 0.2;
    let praticles_per_cell_width = 1;
    let mut particles = vec![];
    for i in 0..700 {
        for j in 0..700 {
            let position = vector![i as f32 + 0.5, j as f32 + 0.5] * cell_width
                / (2.0 * praticles_per_cell_width as f32)
                + Vector2::y() * offset_y;
            let density = 1000.0;
            particles.push(Particle {
                position,
                velocity: Vector2::zeros(),
                volume: ParticleMassProps::new(
                    density * (cell_width / (2.0 * praticles_per_cell_width as f32)).powi(2),
                    cell_width / (4.0 * praticles_per_cell_width as f32),
                ),
                model: ElasticCoefficients::from_young_modulus(5_000_000.0, 0.2),
                plasticity: None,
                phase: Some(ParticlePhase {
                    phase: 1.0,
                    max_stretch: f32::MAX,
                }),
            });
        }
    }

    if !app_state.restarting {
        app_state.num_substeps = 15;
        app_state.gravity_factor = 1.0;
    };

    let params = SimulationParams {
        gravity: vector![0.0, -9.81] * app_state.gravity_factor,
        dt: (1.0 / 60.0) / (app_state.num_substeps as f32),
        padding: 0.0,
    };

    const ANGVEL: f32 = 1.0; // 2.0;

    /*
     * Static platforms.
     */
    let rb = RigidBodyBuilder::fixed().translation(vector![35.0, 20.0]);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(70.0, 1.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let mut polyline = vec![];
    let subdivs = 100;
    let length = 84.0;
    let start = point![35.0, 70.0] - vector![length / 2.0, 0.0];

    for i in 0..=subdivs {
        let step = length / (subdivs as f32);
        let dx = i as f32 * step;
        polyline.push(start + vector![dx, dx.sin()])
    }

    let rb = RigidBodyBuilder::fixed();
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::polyline(polyline, None).build();
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    for k in 0..6 {
        let rb = RigidBodyBuilder::fixed();
        let rb_handle = rapier_data.bodies.insert(rb);
        let co = ColliderBuilder::polyline(
            vec![
                point![0.0 + k as f32 * 15.0, 20.0],
                point![-10.0 + k as f32 * 15.0, 45.0],
            ],
            None,
        )
        .build();
        rapier_data
            .colliders
            .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);
    }

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
}

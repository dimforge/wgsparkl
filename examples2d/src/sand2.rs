use wgsparkl_testbed2d::{wgsparkl, Callbacks, RapierData};

use bevy::render::renderer::RenderDevice;
use nalgebra::{vector, Vector2};
use rapier2d::prelude::{ColliderBuilder, RigidBodyBuilder};
use wgsparkl::models::DruckerPrager;
use wgsparkl::{
    models::ElasticCoefficients,
    pipeline::MpmData,
    solver::{Particle, SimulationParams},
};
use wgsparkl2d::solver::ParticleDynamics;
use wgsparkl_testbed2d::{AppState, PhysicsContext};

fn main() {
    panic!("Run the `testbed3` example instead.");
}

pub fn sand_demo(
    device: RenderDevice,
    app_state: &mut AppState,
    _callbacks: &mut Callbacks,
) -> PhysicsContext {
    let mut rapier_data = RapierData::default();
    let device = device.wgpu_device();

    let offset_y = 46.0;
    // let cell_width = 0.1;
    let cell_width = 0.2;
    let mut particles = vec![];
    for i in 0..700 {
        for j in 0..700 {
            let position = vector![i as f32 + 0.5, j as f32 + 0.5] * cell_width / 2.0
                + Vector2::y() * offset_y;
            let density = 1000.0;
            let radius = cell_width / 4.0;
            particles.push(Particle {
                position,
                dynamics: ParticleDynamics::with_density(radius, density),
                model: ElasticCoefficients::from_young_modulus(10_000_000.0, 0.2),
                plasticity: Some(DruckerPrager::new(10_000_000.0, 0.2)),
                phase: None,
                color: None,
            });
        }
    }

    if !app_state.restarting {
        app_state.num_substeps = 10;
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
    let rb = RigidBodyBuilder::fixed().translation(vector![35.0, -1.0]);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(42.0, 1.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::fixed()
        .translation(vector![-25.0, 45.0])
        .rotation(0.5);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1.0, 52.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::fixed()
        .translation(vector![95.0, 45.0])
        .rotation(-0.5);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1.0, 52.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    /*
     * Rotating platforms.
     */
    let rb = RigidBodyBuilder::kinematic_velocity_based()
        .translation(vector![5.0, 35.0])
        .angvel(ANGVEL);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1.0, 10.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::kinematic_velocity_based()
        .translation(vector![35.0, 35.0])
        .angvel(-ANGVEL);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(10.0, 1.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::kinematic_velocity_based()
        .translation(vector![65.0, 35.0])
        .angvel(ANGVEL);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1.0, 10.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::kinematic_velocity_based()
        .translation(vector![20.0, 20.0])
        .angvel(-ANGVEL);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::ball(5.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::kinematic_velocity_based()
        .translation(vector![50.0, 20.0])
        .angvel(-ANGVEL);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::capsule_y(5.0, 3.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    // let rb = RigidBodyBuilder::kinematic_velocity_based()
    //     .translation(vector![30.0, 0.0])
    //     // .rotation(std::f32::consts::PI / 4.0)
    //     .angvel(-ANGVEL)
    //     .linvel(vector![0.0, 4.0]);
    // let rb_handle = rapier_data.bodies.insert(rb);
    // let co = ColliderBuilder::cuboid(30.0, 30.0);
    // rapier_data
    //     .colliders
    //     .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    for k in 0..8 {
        let rb = RigidBodyBuilder::dynamic().translation(vector![35.0 + 3.0 * k as f32, 120.0]);
        let rb_handle = rapier_data.bodies.insert(rb);
        let co = ColliderBuilder::cuboid(5.0, 1.0).density(10.0 + k as f32 * 100.0);
        rapier_data
            .colliders
            .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);
    }

    // let rb = RigidBodyBuilder::kinematic_velocity_based()
    //     .translation(vector![35.0, 120.0])
    //     .linvel(Vector2::new(0.0, -10.0));
    // let rb_handle = rapier_data.bodies.insert(rb);
    // let co = ColliderBuilder::cuboid(4.0, 1.0).density(100.0);
    // rapier_data
    //     .colliders
    //     .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

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

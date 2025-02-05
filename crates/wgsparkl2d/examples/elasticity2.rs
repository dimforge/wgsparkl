use wgsparkl_testbed2d::{wgsparkl, RapierData};

use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use nalgebra::{vector, Similarity2, Vector2};
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

pub fn main() {
    let mut app = App::new();
    init_testbed(&mut app);
    app.add_systems(
        Startup,
        (register_scenes, start_default_scene)
            .chain()
            .after(wgsparkl_testbed2d::startup::setup_app),
    );
    app.run();
}

fn register_scenes(world: &mut World) {
    let scenes = vec![
        ("sand".to_string(), world.register_system(sand_demo)),
        ("elastic".to_string(), world.register_system(elastic_demo)),
    ];
    let mut inits = world.resource_mut::<SceneInits>();
    inits.scenes = scenes;
}

fn start_default_scene(mut commands: Commands, scenes: Res<SceneInits>) {
    scenes.init_scene(&mut commands, 0);
}

fn sand_demo(mut commands: Commands, device: Res<RenderDevice>, mut app_state: ResMut<AppState>) {
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
                model: ElasticCoefficients::from_young_modulus(10_000_000.0, 0.2),
                plasticity: Some(DruckerPrager::new(10_000_000.0, 0.2)),
                phase: None,
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
    commands.insert_resource(PhysicsContext {
        data,
        rapier_data,
        particles,
    });
}

fn elastic_demo(
    mut commands: Commands,
    device: Res<RenderDevice>,
    mut app_state: ResMut<AppState>,
) {
    let device = device.wgpu_device();

    let offset_y = 10.0;
    // let cell_width = 0.1;
    let cell_width = 0.2;
    let mut particles = vec![];
    for i in 0..700 {
        for j in 0..700 {
            let position =
                vector![i as f32 + 0.5 + (i / 50) as f32 * 2.0, j as f32 + 0.5] * cell_width / 2.0
                    + Vector2::y() * offset_y;
            let density = 1000.0;
            particles.push(Particle {
                position,
                velocity: Vector2::zeros(),
                volume: ParticleMassProps::new(
                    density * (cell_width / 2.0).powi(2),
                    cell_width / 4.0,
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
        app_state.gravity_factor = 2.0;
    };

    let params = SimulationParams {
        gravity: vector![0.0, -9.81] * app_state.gravity_factor,
        dt: (1.0 / 60.0) / (app_state.num_substeps as f32),
        padding: 0.0,
    };

    // let colliders = vec![
    //     BodyDesc {
    //         shape: Cuboid::new(vector![1000.0, 1.0]),
    //         pose: GpuSim2::from(Similarity2::new(vector![0.0, -1.0], 0.0, 1.0)),
    //         ..Default::default()
    //     },
    //     BodyDesc {
    //         shape: Cuboid::new(vector![1.0, 60.0]),
    //         pose: GpuSim2::from(Similarity2::new(vector![-20.0, 0.0], 0.5, 1.0)),
    //         ..BodyDesc::default()
    //     },
    //     BodyDesc {
    //         shape: Cuboid::new(vector![1.0, 60.0]),
    //         pose: GpuSim2::from(Similarity2::new(vector![90.0, 0.0], -0.5, 1.0)),
    //         ..BodyDesc::default()
    //     },
    // ];

    let rapier_data = RapierData::default();
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

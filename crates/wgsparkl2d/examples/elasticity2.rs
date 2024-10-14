use wgsparkl_testbed2d::wgsparkl;

use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use nalgebra::{vector, Similarity2, Vector2};
use wgebra::GpuSim2;
use wgparry2d::cuboid::GpuCuboid;
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
    let device = device.wgpu_device();

    let offset_y = 40.0;
    // let cell_width = 0.1;
    let cell_width = 0.2;
    let mut particles = vec![];
    for i in 0..700 {
        for j in 0..700 {
            let position = vector![i as f32 + 0.5, j as f32 + 0.5] * cell_width / 2.0
                + Vector2::y() * offset_y;
            let density = 1000.0;
            particles.push(Particle {
                position,
                velocity: Vector2::zeros(),
                volume: ParticleMassProps::new(
                    density * (cell_width / 2.0).powi(2),
                    cell_width / 4.0,
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

    const ANGVEL: f32 = 0.0; // 2.0;
    let colliders = vec![
        BodyDesc {
            shape: GpuCuboid::new(vector![1000.0, 1.0]),
            pose: GpuSim2::from(Similarity2::new(vector![0.0, -1.0], 0.0, 1.0)),
            ..Default::default()
        },
        BodyDesc {
            shape: GpuCuboid::new(vector![1.0, 60.0]),
            pose: GpuSim2::from(Similarity2::new(vector![-20.0, 0.0], 0.5, 1.0)),
            ..BodyDesc::default()
        },
        BodyDesc {
            shape: GpuCuboid::new(vector![1.0, 60.0]),
            pose: GpuSim2::from(Similarity2::new(vector![90.0, 0.0], -0.5, 1.0)),
            ..BodyDesc::default()
        },
        BodyDesc {
            shape: GpuCuboid::new(vector![5.0, 5.0]),
            pose: GpuSim2::from(Similarity2::new(vector![5.0, 30.0], 0.0, 1.0)),
            vel: GpuVelocity {
                linear: Vector2::zeros(),
                angular: ANGVEL,
            },
            ..BodyDesc::default()
        },
        BodyDesc {
            shape: GpuCuboid::new(vector![5.0, 5.0]),
            pose: GpuSim2::from(Similarity2::new(vector![35.0, 30.0], 0.0, 1.0)),
            vel: GpuVelocity {
                linear: Vector2::zeros(),
                angular: ANGVEL,
            },
            ..BodyDesc::default()
        },
        BodyDesc {
            shape: GpuCuboid::new(vector![5.0, 5.0]),
            pose: GpuSim2::from(Similarity2::new(vector![65.0, 30.0], 0.0, 1.0)),
            vel: GpuVelocity {
                linear: Vector2::zeros(),
                angular: ANGVEL,
            },
            ..BodyDesc::default()
        },
        BodyDesc {
            shape: GpuCuboid::new(vector![5.0, 5.0]),
            pose: GpuSim2::from(Similarity2::new(vector![20.0, 20.0], 0.0, 1.0)),
            vel: GpuVelocity {
                linear: Vector2::zeros(),
                angular: -ANGVEL,
            },
            ..BodyDesc::default()
        },
        BodyDesc {
            shape: GpuCuboid::new(vector![5.0, 5.0]),
            pose: GpuSim2::from(Similarity2::new(vector![50.0, 20.0], 0.0, 1.0)),
            vel: GpuVelocity {
                linear: Vector2::zeros(),
                angular: -ANGVEL,
            },
            ..BodyDesc::default()
        },
        BodyDesc {
            shape: GpuCuboid::new(vector![5.0, 5.0]),
            pose: GpuSim2::from(Similarity2::new(
                vector![35.0, 10.0],
                std::f32::consts::PI / 4.0,
                1.0,
            )),
            vel: GpuVelocity {
                linear: Vector2::zeros(),
                angular: ANGVEL,
            },
            ..BodyDesc::default()
        },
    ];
    let data = MpmData::new(device, params, &particles, &colliders, cell_width, 60_000);
    commands.insert_resource(PhysicsContext { data, particles });
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

    let colliders = vec![
        BodyDesc {
            shape: GpuCuboid::new(vector![1000.0, 1.0]),
            pose: GpuSim2::from(Similarity2::new(vector![0.0, -1.0], 0.0, 1.0)),
            ..Default::default()
        },
        BodyDesc {
            shape: GpuCuboid::new(vector![1.0, 60.0]),
            pose: GpuSim2::from(Similarity2::new(vector![-20.0, 0.0], 0.5, 1.0)),
            ..BodyDesc::default()
        },
        BodyDesc {
            shape: GpuCuboid::new(vector![1.0, 60.0]),
            pose: GpuSim2::from(Similarity2::new(vector![90.0, 0.0], -0.5, 1.0)),
            ..BodyDesc::default()
        },
    ];
    let data = MpmData::new(device, params, &particles, &colliders, cell_width, 60_000);
    commands.insert_resource(PhysicsContext { data, particles });
}

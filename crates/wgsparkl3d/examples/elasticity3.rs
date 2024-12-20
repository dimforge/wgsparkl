use wgsparkl_testbed3d::wgsparkl;

use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use nalgebra::{vector, Similarity3, Vector3};
use wgebra::GpuSim3;
use wgparry3d::cuboid::GpuCuboid;
use wgrapier3d::dynamics::BodyDesc;
use wgsparkl::models::DruckerPrager;
use wgsparkl::{
    models::ElasticCoefficients,
    pipeline::MpmData,
    solver::{Particle, ParticleMassProps, SimulationParams},
};
use wgsparkl_testbed3d::{init_testbed, AppState, PhysicsContext, SceneInits};

pub fn main() {
    let mut app = App::new();
    init_testbed(&mut app);
    app.add_systems(
        Startup,
        (register_scenes, start_default_scene)
            .chain()
            .after(wgsparkl_testbed3d::startup::setup_app),
    );
    app.run();
}

fn register_scenes(world: &mut World) {
    let scenes = vec![("sand".to_string(), world.register_system(sand_demo))];
    let mut inits = world.resource_mut::<SceneInits>();
    inits.scenes = scenes;
}

fn start_default_scene(mut commands: Commands, scenes: Res<SceneInits>) {
    scenes.init_scene(&mut commands, 0);
}

fn sand_demo(mut commands: Commands, device: Res<RenderDevice>, mut app_state: ResMut<AppState>) {
    let device = device.wgpu_device();

    let cell_width = 1.0;
    let mut particles = vec![];
    for i in 0..45 {
        for j in 0..100 {
            for k in 0..45 {
                let position = vector![i as f32 + 0.5, j as f32 + 0.5 + 10.0, k as f32 + 0.5]
                    * cell_width
                    / 2.0;
                let density = 2700.0;
                particles.push(Particle {
                    position,
                    velocity: Vector3::zeros(),
                    volume: ParticleMassProps::new(
                        density * (cell_width / 2.0).powi(3),
                        cell_width / 4.0,
                    ),
                    model: ElasticCoefficients::from_young_modulus(2_000_000_000.0, 0.2),
                    plasticity: Some(DruckerPrager::new(2_000_000_000.0, 0.2)),
                    phase: None,
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

    let bodies = vec![BodyDesc {
        shape: GpuCuboid::new(vector![10000.0, 4.0, 10000.0]),
        pose: GpuSim3::from(Similarity3::new(
            vector![0.0, -4.0, 0.0],
            Vector3::zeros(),
            1.0,
        )),
        ..BodyDesc::default()
    }];
    let data = MpmData::new(device, params, &particles, &bodies, cell_width, 60_000);
    commands.insert_resource(PhysicsContext { data, particles });
}

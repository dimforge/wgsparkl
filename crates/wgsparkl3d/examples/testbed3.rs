use wgsparkl_testbed3d::{wgsparkl, RapierData};

use bevy::prelude::*;
use wgparry3d::parry::shape::Cuboid;
use wgrapier3d::dynamics::BodyDesc;
use wgsparkl::models::DruckerPrager;
use wgsparkl::{
    models::ElasticCoefficients,
    pipeline::MpmData,
    solver::{Particle, ParticlePhase, SimulationParams},
};
use wgsparkl_testbed3d::{init_testbed, AppState, PhysicsContext, SceneInits};

mod elastic_cut3;
mod heightfield3;
mod sand3;

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
    let scenes = vec![
        ("sand".to_string(), world.register_system(sand3::sand_demo)),
        (
            "heightfield".to_string(),
            world.register_system(heightfield3::heightfield_demo),
        ),
        (
            "elastic_cut".to_string(),
            world.register_system(elastic_cut3::elastic_cut_demo),
        ),
    ];
    let mut inits = world.resource_mut::<SceneInits>();
    inits.scenes = scenes;
}

fn start_default_scene(mut commands: Commands, scenes: Res<SceneInits>) {
    scenes.init_scene(&mut commands, 0);
}

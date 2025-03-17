use bevy::prelude::*;
use wgsparkl_testbed2d::{init_testbed, SceneInits};

mod elastic_cut2;
mod elasticity2;
mod sand2;

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
        ("sand".to_string(), world.register_system(sand2::sand_demo)),
        (
            "elastic".to_string(),
            world.register_system(elasticity2::elastic_demo),
        ),
        (
            "elastic cut".to_string(),
            world.register_system(elastic_cut2::elastic_cut_demo),
        ),
    ];
    let mut inits = world.resource_mut::<SceneInits>();
    inits.scenes = scenes;
}

fn start_default_scene(mut commands: Commands, scenes: Res<SceneInits>) {
    scenes.init_scene(&mut commands, 0);
}

//! This example demonstrates how to load a GLB file and display it as a point cloud.
//!
//! No MPM simulation is involved here.

use bevy::{
    app::{App, Startup, Update},
    core_pipeline::core_3d::Camera3d,
    ecs::system::Commands,
    math::Vec3,
    prelude::*,
    render::camera::Camera,
    DefaultPlugins,
};
use bevy_editor_cam::{prelude::EditorCam, DefaultEditorCamPlugins};
use nalgebra::{Isometry3, Transform3, UnitQuaternion, Vector3};
use std::{fs::File, io::Read};
use wgsparkl3d::load_mesh3d::load_gltf::load_model_with_colors;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, DefaultEditorCamPlugins))
        .add_systems(Startup, init_scene)
        .add_systems(Update, display_point_cloud)
        .run();
}

#[derive(Component)]
pub struct PointCloud {
    pub positions: Vec<(Vec3, Color)>,
}

pub const SAMPLE_PER_UNIT: f32 = 20.0;

fn init_scene(mut commands: Commands) {
    commands.spawn((Camera3d::default(), Camera::default(), EditorCam::default()));

    let mut file = File::open("assets/shiba.glb").expect("Failed to open GLB file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");
    let pc_grid = load_model_with_colors(
        &buffer,
        Transform3::from_matrix_unchecked(
            Isometry3::from_parts(
                Vector3::new(0.0, 0.0, 0.0).into(),
                UnitQuaternion::identity(),
            )
            .to_matrix()
            .scale(3.0),
        ),
        None,
        SAMPLE_PER_UNIT,
    );

    commands.spawn(PointCloud {
        positions: pc_grid
            .iter()
            .map(|p| {
                let pos = Vec3::new(p.0.x, p.0.y, p.0.z);
                (pos, Color::from(Srgba::from_u8_array(p.1)))
            })
            .collect::<Vec<_>>(),
    });
}

fn display_point_cloud(pcs: Query<&PointCloud>, mut gizmos: Gizmos) {
    for pc in pcs.iter() {
        for p in pc.positions.iter() {
            gizmos.sphere(p.0, 1f32 / SAMPLE_PER_UNIT / 2f32, p.1);
        }
    }
}

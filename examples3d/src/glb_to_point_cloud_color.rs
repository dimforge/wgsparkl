use crate::utils::default_scene::{self, SAMPLE_PER_UNIT};

use bevy::{
    app::{App, Startup, Update},
    core_pipeline::core_3d::Camera3d,
    ecs::system::{Commands, Query},
    gizmos::gizmos::Gizmos,
    math::Vec3,
    pbr::wireframe::WireframePlugin,
    picking::mesh_picking::MeshPickingPlugin,
    prelude::*,
    render::render_resource::WgpuFeatures,
    render::{
        camera::Camera,
        renderer::RenderDevice,
        settings::{RenderCreation, WgpuSettings},
        RenderPlugin,
    },
    DefaultPlugins,
};
use bevy_editor_cam::{prelude::EditorCam, DefaultEditorCamPlugins};
use nalgebra::{vector, Isometry3, Transform3, UnitQuaternion, Vector3};
use std::{f32::consts::PI, fs::File, io::Read};
use wgsparkl3d::load_mesh3d::load_gltf::load_model_with_colors;
use wgsparkl3d::{pipeline::MpmData, solver::SimulationParams};
use wgsparkl_testbed3d::{AppState, PhysicsContext, RapierData};

#[allow(unused)]
fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(RenderPlugin {
                render_creation: RenderCreation::Automatic(WgpuSettings {
                    // WARN this is a native only feature. It will not work with webgl or webgpu
                    features: WgpuFeatures::POLYGON_MODE_LINE,
                    ..default()
                }),
                ..default()
            }),
            // You need to add this plugin to enable wireframe rendering
            WireframePlugin,
            MeshPickingPlugin,
            DefaultEditorCamPlugins,
        ))
        .add_systems(Startup, init_scene)
        .add_systems(Update, display_point_cloud)
        .run();
}

fn init_scene(mut commands: Commands) {
    commands.spawn((Camera3d::default(), Camera::default(), EditorCam::default()));
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        Transform {
            translation: Vec3::new(0.0, 2.0, 0.0),
            rotation: Quat::from_rotation_x(-PI / 4.),
            ..default()
        },
    ));

    let mut file = File::open("assets/shiba.glb").expect("Failed to open GLB file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");
    let pc_grid = load_model_with_colors(
        &buffer,
        Transform3::from_matrix_unchecked(
            Isometry3::from_parts(
                Vector3::new(0.0, 6.0, 0.0).into(),
                UnitQuaternion::identity(),
            )
            .to_matrix()
            .scale(3.0),
        ),
        None,
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

#[derive(Component)]
pub struct PointCloud {
    pub positions: Vec<(Vec3, Color)>,
}

pub fn elastic_color_model_demo(
    mut commands: Commands,
    device: Res<RenderDevice>,
    app_state: ResMut<AppState>,
) {
    let mut file = File::open("assets/shiba.glb").expect("Failed to open GLB file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");
    let pc_grid = load_model_with_colors(
        &buffer,
        Transform3::from_matrix_unchecked(
            Isometry3::from_parts(
                Vector3::new(0.0, 6.0, 0.0).into(),
                UnitQuaternion::identity(),
            )
            .to_matrix()
                * nalgebra::Matrix4::new_scaling(3.0),
        ),
        None,
    );
    let params = SimulationParams {
        gravity: vector![0.0, -9.81, 0.0] * app_state.gravity_factor,
        dt: (1.0 / 60.0) / (app_state.num_substeps as f32),
    };
    let mut rapier_data = RapierData::default();
    default_scene::set_default_app_state(app_state);
    default_scene::spawn_ground_and_walls(&mut rapier_data);

    let mut particles = vec![];
    for (pos, color) in pc_grid {
        let particle = default_scene::create_particle(
            &Vec3::new(pos.x, pos.y, pos.z),
            Some(Color::from(Srgba::from_u8_array(color))),
            1f32 / SAMPLE_PER_UNIT / 2f32,
        );
        particles.push(particle);
    }
    default_scene::spawn_ground_and_walls(&mut rapier_data);
    let data = MpmData::new(
        device.wgpu_device(),
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        1f32 / default_scene::SAMPLE_PER_UNIT,
        60_000,
    );

    let physics = PhysicsContext {
        data,
        rapier_data,
        particles,
    };
    commands.insert_resource(physics);
}

fn display_point_cloud(pcs: Query<&PointCloud>, mut gizmos: Gizmos) {
    for pc in pcs.iter() {
        for p in pc.positions.iter() {
            gizmos.sphere(p.0, 0.01f32, p.1);
        }
    }
}

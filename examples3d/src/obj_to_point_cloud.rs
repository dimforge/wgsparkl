use crate::utils::default_scene;
use crate::utils::extract_mesh;

use std::{fs::File, io::BufReader};

use bevy::{prelude::*, render::renderer::RenderDevice};
use bevy_editor_cam::{prelude::EditorCam, DefaultEditorCamPlugins};
use default_scene::{spawn_ground_and_walls, SAMPLE_PER_UNIT};
use nalgebra::{point, vector};
use obj::raw::object::Polygon;
use wgsparkl3d::{pipeline::MpmData, solver::SimulationParams};
use wgsparkl_testbed3d::{AppState, PhysicsContext, RapierData};

#[allow(unused)]
fn main() {
    App::new()
        .add_plugins((DefaultPlugins, MeshPickingPlugin, DefaultEditorCamPlugins))
        .add_systems(Startup, init_rapier_scene)
        .add_systems(Update, display_point_cloud)
        .run();
}

#[derive(Component)]
pub struct PointCloud {
    pub positions: Vec<Vec3>,
}

pub fn get_point_cloud() -> Vec<Vec3> {
    let obj_path = "assets/banana.obj";
    println!("Parsing and decomposing: {}", obj_path);
    let input = BufReader::new(File::open(obj_path).unwrap());

    let Ok(model) = obj::raw::parse_obj(input) else {
        return vec![];
    };
    let mut vertices: Vec<_> = model
        .positions
        .iter()
        .map(|v| point![v.0, v.1, v.2])
        .collect();
    let indices: Vec<_> = model
        .polygons
        .into_iter()
        .flat_map(|p| match p {
            Polygon::P(idx) => idx,
            Polygon::PT(idx) | Polygon::PN(idx) => idx.iter().map(|i| i.0).collect(),
            Polygon::PTN(idx) => idx.iter().map(|i| i.0).collect(),
        })
        .collect();
    extract_mesh::recenter_and_scale(&mut vertices, SAMPLE_PER_UNIT);
    extract_mesh::get_point_cloud_from_trimesh(&vertices, &indices, SAMPLE_PER_UNIT)
}

pub fn init_rapier_scene(mut commands: Commands) {
    commands.spawn((Camera3d::default(), Camera::default(), EditorCam::default()));

    commands.spawn(PointCloud {
        positions: get_point_cloud(),
    });

    // let decomposed_shape = SharedShape::trimesh_with_flags(vertices, indices, TriMeshFlags::FIX_INTERNAL_EDGES).unwrap()
}

fn display_point_cloud(mut gizmos: Gizmos, query: Query<&PointCloud>) {
    for pc in query.iter() {
        for pos in pc.positions.iter() {
            gizmos.sphere(*pos, 1f32 / SAMPLE_PER_UNIT / 2f32, Color::WHITE);
        }
    }
}

pub fn elastic_model_demo(
    mut commands: Commands,
    device: Res<RenderDevice>,
    app_state: ResMut<AppState>,
) {
    let point_cloud_color = get_point_cloud();

    let params = SimulationParams {
        gravity: vector![0.0, -9.81, 0.0] * app_state.gravity_factor,
        dt: (1.0 / 60.0) / (app_state.num_substeps as f32),
    };
    let mut rapier_data = RapierData::default();
    default_scene::set_default_app_state(app_state);
    default_scene::spawn_ground_and_walls(&mut rapier_data);

    let mut particles = vec![];
    for pos in point_cloud_color {
        let particle = default_scene::create_particle(&pos, None);
        particles.push(particle);
    }
    spawn_ground_and_walls(&mut rapier_data);
    let data = MpmData::new(
        device.wgpu_device(),
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        1f32 / SAMPLE_PER_UNIT,
        60_000,
    );

    let physics = PhysicsContext {
        data,
        rapier_data,
        particles,
    };
    commands.insert_resource(physics);
}

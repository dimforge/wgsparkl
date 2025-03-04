use std::{fs::File, io::BufReader};

use bevy::{prelude::*, render::renderer::RenderDevice};
use bevy_editor_cam::{prelude::EditorCam, DefaultEditorCamPlugins};
use nalgebra::{point, vector, zero, Point3, Quaternion, SimdValue, Vector3};
use obj::raw::object::Polygon;
use rapier3d::{
    parry::bounding_volume,
    prelude::{ColliderBuilder, QueryFilter, Real, RigidBodyBuilder, SharedShape, TriMeshFlags},
};
use wgsparkl3d::{
    models::ElasticCoefficients,
    pipeline::MpmData,
    solver::{Particle, ParticleDynamics, ParticlePhase, SimulationParams},
};
use wgsparkl_testbed3d::{AppState, PhysicsContext, RapierData};

fn main() {
    eprintln!("Run the `testbed3` example instead.");
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

const SAMPLE_PER_UNIT: f32 = 10.0;

pub fn get_point_cloud() -> Vec<Vec3> {
    let obj_path = "assets/camel_decimated.obj";
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
            Polygon::P(idx) => idx.into_iter(),
            Polygon::PT(idx) => Vec::from_iter(idx.into_iter().map(|i| i.0)).into_iter(),
            Polygon::PN(idx) => Vec::from_iter(idx.into_iter().map(|i| i.0)).into_iter(),
            Polygon::PTN(idx) => Vec::from_iter(idx.into_iter().map(|i| i.0)).into_iter(),
        })
        .collect();

    let aabb =
        bounding_volume::details::point_cloud_aabb(&rapier3d::na::Isometry::default(), &vertices);
    let center = aabb.center();
    let diag = (aabb.maxs - aabb.mins).norm();

    // Compute the size of the model, to scale it and have similar size for everything.
    vertices
        .iter_mut()
        .for_each(|p| *p = (*p - center.coords) * 10.0 / diag);
    let indices: Vec<_> = indices
        .chunks(3)
        .map(|idx| [idx[0] as u32, idx[1] as u32, idx[2] as u32])
        .collect();
    let aabb =
        bounding_volume::details::point_cloud_aabb(&rapier3d::na::Isometry::default(), &vertices);

    let decomposed_shape = SharedShape::convex_decomposition(&vertices, &indices);

    let mut rapier = RapierData::default();
    rapier.insert_body_and_collider(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::new(decomposed_shape.clone()),
    );
    rapier.update();
    let mut positions = vec![];
    dbg!(aabb);
    let aabb_sample = aabb.scaled(&Vector3::new(
        SAMPLE_PER_UNIT,
        SAMPLE_PER_UNIT,
        SAMPLE_PER_UNIT,
    ));
    for x in aabb_sample.mins.x as i32..aabb_sample.maxs.x as i32 {
        for y in aabb_sample.mins.y as i32..aabb_sample.maxs.y as i32 {
            for z in aabb_sample.mins.z as i32..aabb_sample.maxs.z as i32 {
                let point = Point3::new(x as f32, y as f32, z as f32) / SAMPLE_PER_UNIT;
                let pos = Vec3::new(point.x, point.y, point.z);
                rapier.intersections_with_point(&point, QueryFilter::default(), |handle| {
                    positions.push(pos);
                    false
                });
            }
        }
    }
    positions
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
    mut app_state: ResMut<AppState>,
) {
    let mut rapier_data = RapierData::default();
    let device = device.wgpu_device();

    let nxz = 45;
    let cell_width = 1f32 / SAMPLE_PER_UNIT;
    let mut particles = vec![];
    for pos in get_point_cloud() {
        let radius = 1f32 / SAMPLE_PER_UNIT / 2f32;
        let density = 3700.0;
        let pos = Quat::from_axis_angle(Vec3::X, 80f32.to_radians()) * pos + Vec3::Y * 15.0;
        particles.push(Particle {
            position: pos.to_array().into(),
            dynamics: ParticleDynamics::with_density(radius, density),
            model: ElasticCoefficients::from_young_modulus(10_000_000.0, 0.2),
            plasticity: None,
            phase: Some(ParticlePhase {
                phase: 1.0,
                max_stretch: f32::MAX,
            }),
        });
    }

    if !app_state.restarting {
        app_state.num_substeps = 20;
        app_state.gravity_factor = 1.0;
    };

    let params = SimulationParams {
        gravity: vector![0.0, -9.81, 0.0] * app_state.gravity_factor,
        dt: (1.0 / 60.0) / (app_state.num_substeps as f32),
    };

    rapier_data.insert_body_and_collider(
        RigidBodyBuilder::fixed().translation(vector![0.0, -4.0, 0.0]),
        ColliderBuilder::cuboid(100.0, 4.0, 100.0),
    );

    rapier_data.insert_body_and_collider(
        RigidBodyBuilder::fixed().translation(vector![0.0, 5.0, -35.0]),
        ColliderBuilder::cuboid(35.0, 5.0, 0.5),
    );
    rapier_data.insert_body_and_collider(
        RigidBodyBuilder::fixed().translation(vector![0.0, 5.0, 35.0]),
        ColliderBuilder::cuboid(35.0, 5.0, 0.5),
    );
    rapier_data.insert_body_and_collider(
        RigidBodyBuilder::fixed().translation(vector![-35.0, 5.0, 0.0]),
        ColliderBuilder::cuboid(0.5, 5.0, 35.0),
    );
    rapier_data.insert_body_and_collider(
        RigidBodyBuilder::fixed().translation(vector![35.0, 5.0, 0.0]),
        ColliderBuilder::cuboid(0.5, 5.0, 35.0),
    );

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

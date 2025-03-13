#![allow(unused)]

#[path = "libs/default_scene.rs"]
pub mod default_scene;
#[path = "libs/extract_mesh.rs"]
pub mod extract_mesh;
#[path = "libs/glb_to_point_cloud.rs"]
pub mod glb_to_point_cloud;

use glb_to_point_cloud::load_model_trimeshes;
use glb_to_point_cloud::load_model_with_colors;

use bevy::color::palettes::css;
use bevy::{prelude::*, render::renderer::RenderDevice, scene::SceneInstanceReady};
use nalgebra::vector;
use rapier3d::prelude::RigidBodyHandle;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, SharedShape, TriMeshFlags};
use wgsparkl3d::pipeline::MpmData;
use wgsparkl3d::solver::SimulationParams;
use wgsparkl_testbed3d::CallBeforeSimulation;
use wgsparkl_testbed3d::RapierData;
use wgsparkl_testbed3d::{AppState, PhysicsContext};

pub fn demo(
    mut commands: Commands,
    device: Res<RenderDevice>,
    mut app_state: ResMut<AppState>,
    asset_server: Res<AssetServer>,
) {
    let pc_grid = load_model_with_colors(
        "assets/banana.glb",
        Transform::from_scale(Vec3::splat(0.35))
            .with_translation(Vec3::Y * 2.6f32)
            .with_rotation(Quat::from_rotation_y(-90f32.to_radians())),
        Some(css::BLANCHED_ALMOND.into()),
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
        let particle = default_scene::create_particle(&pos, Some(color));
        particles.push(particle);
    }
    // Slicer
    let mut slicer_trimeshes = load_model_trimeshes("assets/chefs_knife_open_blade.glb");
    slicer_trimeshes.iter_mut().for_each(|trimesh| {
        trimesh.0.iter_mut().for_each(|v| {
            *v *= 10f32;
        });
    });

    let rb = RigidBodyBuilder::kinematic_position_based().translation(vector![0.0, 5.0, 0.0]);
    let parent_handle = rapier_data.bodies.insert(rb);
    for (vertices, indices) in slicer_trimeshes.iter() {
        // Insert collider into rapier state.

        let collider = ColliderBuilder::new(
            SharedShape::trimesh_with_flags(
                vertices.clone(),
                indices
                    .chunks_exact(3)
                    .map(|i| [i[0] as u32, i[1] as u32, i[2] as u32])
                    .collect(),
                TriMeshFlags::FIX_INTERNAL_EDGES,
            )
            .unwrap(),
        );
        rapier_data
            .colliders
            .insert_with_parent(collider, parent_handle, &mut rapier_data.bodies);
    }
    commands.spawn((
        Knife(parent_handle),
        SceneRoot(
            // This is a bit redundant with load_model_trimeshes, but it's a simple way to get the scene loaded visually.
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("chefs_knife_modified.glb")),
        ),
        Transform::from_scale(Vec3::splat(10f32)).with_translation(Vec3::Y * 5f32),
    ));

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
    //
    let system_id = commands.register_system(move_knife);
    commands.spawn(CallBeforeSimulation(system_id));
}

#[derive(Debug, Component)]
pub struct Knife(pub RigidBodyHandle);

fn move_knife(app_state: Res<AppState>, knife: Query<&Knife>, mut physics: ResMut<PhysicsContext>) {
    for knife in knife.iter() {
        let t = app_state.physics_time_seconds as f32;

        let body = physics.rapier_data.bodies.get_mut(knife.0).unwrap();
        let length = 1.50;
        let width = 0.5;
        let x_pos = 3.0;
        let y_pos = 1.5;
        let z_pos = -1.5;
        let velocity = 0.8;
        let period = (2f32 * length + 3f32 * width) / velocity;

        let i = (t / period).floor();
        let dis = velocity * (t - period * i);

        let new_pos = if dis < length {
            vector![x_pos - width * i, y_pos - dis, z_pos]
        } else if length <= dis && dis < length + width {
            vector![x_pos - width * i + (dis - length), y_pos - length, z_pos]
        } else if length + width <= dis && dis < 2f32 * length + width {
            vector![
                x_pos - width * i + width,
                y_pos - (2.0 * length + width - dis),
                z_pos
            ]
        } else {
            vector![
                x_pos - width * i + width - (dis - 2.0 * length - width),
                y_pos,
                z_pos
            ]
        };

        body.set_translation(new_pos, true);
    }
}

mod follow_rapier {
    use rapier3d::prelude::{RigidBodyHandle, RigidBodySet};

    use super::*;
    // TODO: use this for the knife.
    pub fn follow_body_position(
        entity_follower: Entity,
        body_to_follow_handle: RigidBodyHandle,
        bodies: &RigidBodySet,
        components: &mut Query<&mut Transform>,
    ) {
        if let Some(body) = bodies.get(body_to_follow_handle) {
            if let Ok(mut pos) = components.get_mut(entity_follower) {
                let co_pos = body.position();
                pos.translation.x = (co_pos.translation.vector.x) as f32;
                pos.translation.y = (co_pos.translation.vector.y) as f32;
                #[cfg(feature = "dim3")]
                {
                    pos.translation.z = (co_pos.translation.vector.z) as f32;
                    pos.rotation = Quat::from_xyzw(
                        co_pos.rotation.i as f32,
                        co_pos.rotation.j as f32,
                        co_pos.rotation.k as f32,
                        co_pos.rotation.w as f32,
                    );
                }
                #[cfg(feature = "dim2")]
                {
                    pos.rotation = Quat::from_rotation_z(co_pos.rotation.angle() as f32);
                }
            }
        }
    }
}

use super::model_to_point_cloud;
use super::model_to_point_cloud_color::load_model_trimeshes;
use super::model_to_point_cloud_color::load_model_with_colors;

use bevy::{prelude::*, render::renderer::RenderDevice, scene::SceneInstanceReady};
use nalgebra::vector;
use nalgebra::Quaternion;
use nalgebra::Rotation;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, SharedShape, TriMeshFlags};
use wgrapier3d::dynamics::body::BodyCoupling;
use wgrapier3d::dynamics::body::BodyCouplingEntry;
use wgsparkl_testbed3d::{AppState, PhysicsContext};

pub fn demo(
    mut commands: Commands,
    device: Res<RenderDevice>,
    mut app_state: ResMut<AppState>,
    asset_server: Res<AssetServer>,
) {
    let pc_grid = load_model_with_colors(
        "assets/banana.glb",
        Transform::from_scale(Vec3::splat(0.35)).with_translation(Vec3::Y * 5f32),
    );
    let mut physics = model_to_point_cloud::spawn_elastic_model_demo(
        device, app_state,
        &pc_grid,
        //Transform::from_scale(Vec3::splat(3.0)).with_translation(Vec3::Y * 1.0),
    );
    // Slicer
    let mut slicer_trimeshes = load_model_trimeshes("assets/chefs_knife_modified.glb");
    slicer_trimeshes.iter_mut().for_each(|trimesh| {
        trimesh.0.iter_mut().for_each(|v| {
            *v *= 10f32;
            *v = Rotation::from_axis_angle(&nalgebra::Vector3::y_axis(), 90f32.to_radians()) * *v;
        });
    });
    let rapier_data = &mut physics.rapier_data;

    let rb = RigidBodyBuilder::dynamic().translation(vector![0.0, 5.0, 0.0]);
    let parent_handle = rapier_data.bodies.insert(rb);
    let mut coupling = vec![];
    for (vertices, indices) in slicer_trimeshes.iter().skip(2) {
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
        coupling.push(rapier_data.colliders.insert_with_parent(
            collider,
            parent_handle,
            &mut rapier_data.bodies,
        ));
        break;
    }
    /*
    commands.spawn((
        SceneRoot(
            // This is a bit redundant with load_model_trimeshes, but it's a simple way to get the scene loaded visually.
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("chefs_knife_modified.glb")),
        ),
        Transform::from_scale(Vec3::splat(10f32))
            .with_rotation(Quat::from_axis_angle(Vec3::Y, 90f32.to_radians()))
            .with_translation(Vec3::Y * 5f32),
    ));*/

    let mut coupling: Vec<_> = coupling
        .iter()
        .map(|co_handle| {
            let co = &rapier_data.colliders[*co_handle];
            let rb_handle = co.parent().unwrap();
            BodyCouplingEntry {
                body: rb_handle,
                collider: co_handle.clone(),
                mode: BodyCoupling::OneWay,
            }
        })
        .collect();
    dbg!(&coupling);
    // FIXME: this doesn't work yet because the coupling cannot be changed now, coupling should be private and spawn_elastic_model_demo updated.
    // physics.data.coupling.append(&mut coupling);

    commands.insert_resource(physics);
    //
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

mod extract_mesh {
    use super::*;
    use bevy::render::mesh::{Indices, Mesh, VertexAttributeValues};
    use nalgebra::Point3;

    use f32 as Real;

    pub fn extract_mesh_vertices_indices(
        mesh: &Mesh,
    ) -> Option<(Vec<Point3<Real>>, Vec<[u32; 3]>)> {
        use rapier3d::na::point;

        let vertices = mesh.attribute(Mesh::ATTRIBUTE_POSITION)?;
        let indices = mesh.indices()?;

        let vtx: Vec<_> = match vertices {
            VertexAttributeValues::Float32(vtx) => Some(
                vtx.chunks(3)
                    .map(|v| point![v[0] as Real, v[1] as Real, v[2] as Real])
                    .collect(),
            ),
            VertexAttributeValues::Float32x3(vtx) => Some(
                vtx.iter()
                    .map(|v| point![v[0] as Real, v[1] as Real, v[2] as Real])
                    .collect(),
            ),
            _ => None,
        }?;

        let idx = match indices {
            Indices::U16(idx) => idx
                .chunks_exact(3)
                .map(|i| [i[0] as u32, i[1] as u32, i[2] as u32])
                .collect(),
            Indices::U32(idx) => idx.chunks_exact(3).map(|i| [i[0], i[1], i[2]]).collect(),
        };

        Some((vtx, idx))
    }
}

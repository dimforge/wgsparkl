#![allow(unused)]

#[path = "default_scene.rs"]
mod default_scene;

use bevy::color::palettes::css;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, Mesh, VertexAttributeValues};
use nalgebra::{Point3, Vector3};

use f32 as Real;
use rapier3d::parry::bounding_volume;
use rapier3d::prelude::{PointQuery, TriMesh, TriMeshFlags};

pub fn extract_mesh_vertices_indices(mesh: &Mesh) -> Option<(Vec<Point3<Real>>, Vec<[u32; 3]>)> {
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

pub fn recenter_and_scale(
    vertices: &mut Vec<nalgebra::OPoint<f32, nalgebra::Const<3>>>,
    sample_per_unit: f32,
) {
    // Compute the size of the model, to scale it and have similar size for everything.
    let aabb =
        bounding_volume::details::point_cloud_aabb(&rapier3d::na::Isometry::default(), &*vertices);
    let center = aabb.center();
    let diag = (aabb.maxs - aabb.mins).norm();
    vertices
        .iter_mut()
        .for_each(|p| *p = (*p - center.coords) * sample_per_unit / diag);
}

/// TODO: consider using [`wgsparkl3d::solver::particle3d::sample_mesh`] (should be exposed though!)
pub fn get_point_cloud_from_trimesh(
    vertices: &Vec<nalgebra::OPoint<f32, nalgebra::Const<3>>>,
    indices: &Vec<usize>,
    sample_per_unit: f32,
) -> Vec<Vec3> {
    let mut vertices = vertices.clone();

    let indices: Vec<_> = indices
        .chunks(3)
        .map(|idx| [idx[0] as u32, idx[1] as u32, idx[2] as u32])
        .collect();
    let trimesh =
        TriMesh::with_flags(vertices, indices, TriMeshFlags::ORIENTED).expect("Invalid mesh");
    let aabb = bounding_volume::details::point_cloud_aabb(
        &rapier3d::na::Isometry::default(),
        trimesh.vertices(),
    );
    let mut positions = vec![];

    let aabb_sample = aabb.scaled(&Vector3::new(
        sample_per_unit,
        sample_per_unit,
        sample_per_unit,
    ));
    for x in aabb_sample.mins.x as i32..aabb_sample.maxs.x as i32 {
        for y in aabb_sample.mins.y as i32..aabb_sample.maxs.y as i32 {
            for z in aabb_sample.mins.z as i32..aabb_sample.maxs.z as i32 {
                let point = Point3::new(x as f32, y as f32, z as f32) / sample_per_unit;
                let pos = Vec3::new(point.x, point.y, point.z);
                if trimesh.contains_local_point(&point) {
                    positions.push(pos);
                }
            }
        }
    }
    positions
}

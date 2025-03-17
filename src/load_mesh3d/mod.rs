#[cfg(feature = "load_bevy")]
pub mod load_bevy;
#[cfg(feature = "load_gltf")]
pub mod load_gltf;
#[cfg(feature = "load_obj")]
pub mod load_obj;

use nalgebra::{Point3, Vector3};

use crate::rapier;
use f32 as Real;
use rapier::parry::bounding_volume;
use rapier::prelude::{PointQuery, TriMesh, TriMeshFlags};

pub fn recenter_and_scale(
    vertices: &mut Vec<nalgebra::OPoint<f32, nalgebra::Const<3>>>,
    sample_per_unit: f32,
) {
    // Compute the size of the model, to scale it and have similar size for everything.
    let aabb =
        bounding_volume::details::point_cloud_aabb(&rapier::na::Isometry::default(), &*vertices);
    let center = aabb.center();
    let diag = (aabb.maxs - aabb.mins).norm();
    vertices
        .iter_mut()
        .for_each(|p| *p = (*p - center.coords) * sample_per_unit / diag);
}

pub fn get_point_cloud_from_trimesh(
    vertices: &Vec<nalgebra::OPoint<f32, nalgebra::Const<3>>>,
    indices: &Vec<usize>,
    sample_per_unit: f32,
) -> Vec<Point3<Real>> {
    let indices: Vec<_> = indices
        .chunks(3)
        .map(|idx| [idx[0] as u32, idx[1] as u32, idx[2] as u32])
        .collect();
    let trimesh = TriMesh::with_flags(vertices.clone(), indices, TriMeshFlags::ORIENTED)
        .expect("Invalid mesh");
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
                if trimesh.contains_local_point(&point) {
                    positions.push(point);
                }
            }
        }
    }
    positions
}

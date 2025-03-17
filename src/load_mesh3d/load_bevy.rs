use bevy::render::mesh::{Indices, Mesh, VertexAttributeValues};
use nalgebra::Point3;

use f32 as Real;

pub fn load_mesh_vertices_indices(mesh: &Mesh) -> Option<(Vec<Point3<Real>>, Vec<[u32; 3]>)> {
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

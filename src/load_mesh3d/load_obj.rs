use std::io::BufRead;

use nalgebra::{point, Point3};
use obj::{raw::object::Polygon, ObjResult};

use super::recenter_and_scale;

pub fn get_point_cloud<T: BufRead>(
    input: T,
    sample_per_unit: f32,
) -> ObjResult<(Vec<Point3<f32>>, Vec<usize>)> {
    let model = obj::raw::parse_obj(input)?;
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
    recenter_and_scale(&mut vertices, sample_per_unit);
    Ok((vertices, indices))
}

use f32 as Real;
use image::RgbaImage;
use nalgebra::{distance_squared, Matrix4, Point3, Transform3};

use super::get_point_cloud_from_trimesh;

fn extract_embedded_texture(
    gltf: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Option<RgbaImage> {
    for image in gltf.images() {
        dbg!(image.name());
        if let gltf::image::Source::View { view, .. } = image.source() {
            let buffer = &buffers[view.buffer().index()];
            let start = view.offset();
            let end = start + view.length();
            let image_data = &buffer.0[start..end];

            // Decode based on MIME type
            if let Ok(img) = image::load_from_memory(image_data) {
                return Some(img.to_rgba8());
            } else {
                eprintln!("Failed to decode texture.");
            }
        }
    }
    None
}

fn sample_texture(texture: &RgbaImage, uv: [f32; 2]) -> [u8; 4] {
    let (width, height) = texture.dimensions();

    // Convert UV (0.0 - 1.0) to pixel coordinates
    let x = (uv[0] * width as f32).clamp(0.0, (width - 1) as f32) as u32;
    let y = ((uv[1]) * height as f32).clamp(0.0, (height - 1) as f32) as u32;

    texture.get_pixel(x, y).0
}

fn closest_point(
    target: Point3<f32>,
    points: &[(Point3<f32>, [u8; 4])],
) -> Option<&(Point3<f32>, [u8; 4])> {
    points.iter().min_by(|a, b| {
        distance_squared(&a.0, &target)
            .partial_cmp(&distance_squared(&b.0, &target))
            .unwrap()
    })
}

// TODO: transform should not be here, but when we spawn the model.
// this function should have a "resolution" like the amount of points to sample in x/y/z.
/// Load a glb or gltf file and return a point cloud and a list of trimeshes.
///
/// ```
/// let mut file = File::open("path_to_file.glb").expect("Failed to open GLB file");
/// let mut buffer = Vec::new();
/// file.read_to_end(&mut buffer).expect("Failed to read file");
///
/// let _ = load_model_with_colors(&buffer);
/// ```
pub fn load_model_with_colors<S>(
    slice: S,
    transform: Transform3<Real>,
    color_inside: Option<[u8; 4]>,
    sample_per_unit: f32,
) -> Vec<(Point3<Real>, [u8; 4])>
where
    S: AsRef<[u8]>,
{
    let mut res = load_model_with_vertex_colors(slice);
    let mut pc_grid = vec![];

    res.0
        .iter_mut()
        .for_each(|p| p.0 = transform.transform_point(&p.0));

    for (t_id, trimesh) in res.1.iter_mut().enumerate() {
        if t_id != 0 {
            //continue;
        }
        trimesh.0.iter_mut().for_each(|p| {
            let new_p = transform.transform_point(&Point3::new(p.x, p.y, p.z));
            *p = Point3::new(new_p.x, new_p.y, new_p.z);
        });
        let mut pc = get_point_cloud_from_trimesh(&trimesh.0, &trimesh.1, sample_per_unit)
            .into_iter()
            .map(|p| {
                let closest_color = closest_point(p, &res.0).unwrap();
                if let Some(color_inside) = color_inside {
                    let distance = distance_squared(&p, &closest_color.0);
                    (
                        p,
                        if distance <= 0.04 {
                            closest_color.1
                        } else {
                            color_inside
                        },
                    )
                } else {
                    (p, closest_color.1)
                }
            })
            .collect();
        pc_grid.append(&mut pc);
    }
    pc_grid
}

fn get_node_transform(node: &gltf::Node, parent_transform: Matrix4<f32>) -> Matrix4<f32> {
    let matrix = node.transform().matrix();
    parent_transform * Matrix4::from_column_slice(&matrix.concat())
}

pub struct SceneIterator<'a, T, F>
where
    F: Fn(&gltf::Primitive, &Matrix4<f32>) -> T + 'a,
{
    stack: Vec<(gltf::Node<'a>, Matrix4<f32>)>,
    process_primitive: F,
}

impl<'a, T, F> SceneIterator<'a, T, F>
where
    F: Fn(&gltf::Primitive, &Matrix4<f32>) -> T + 'a,
{
    fn new(root_nodes: Vec<gltf::Node<'a>>, process_primitive: F) -> Self {
        let stack = root_nodes
            .into_iter()
            .map(|node| (node, Matrix4::identity()))
            .collect();
        SceneIterator {
            stack,
            process_primitive,
        }
    }
}

impl<'a, T, F> Iterator for SceneIterator<'a, T, F>
where
    F: Fn(&gltf::Primitive, &Matrix4<f32>) -> T + 'a,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, parent_transform)) = self.stack.pop() {
            let world_transform = get_node_transform(&node, parent_transform);

            if let Some(mesh) = node.mesh() {
                if let Some(primitive) = mesh.primitives().next() {
                    return Some((self.process_primitive)(&primitive, &world_transform));
                }
            }

            for child in node.children() {
                self.stack.push((child, world_transform));
            }
        }
        None
    }
}

/// Load a glb or gltf file into vertices and indices and its vertex colors from sampled texture.
///
/// ```
/// let mut file = File::open("path_to_file.glb").expect("Failed to open GLB file");
/// let mut buffer = Vec::new();
/// file.read_to_end(&mut buffer).expect("Failed to read file");
///
/// let (gltf, buffers, _) = gltf::import_slice(slice).expect("Failed to parse GLB");
///
/// let _ = load_model_with_point_cloud(&buffer);
/// ```
pub fn load_model_with_vertex_colors<S>(
    slice: S,
) -> (
    Vec<(nalgebra::Point3<f32>, [u8; 4])>,
    Vec<(Vec<nalgebra::Point3<f32>>, Vec<usize>)>,
)
where
    S: AsRef<[u8]>,
{
    let (gltf, buffers, _) = gltf::import_slice(slice).expect("Failed to parse GLB");

    // Extract embedded texture
    let texture = extract_embedded_texture(&gltf, &buffers).expect("Failed to extract texture");

    let mut pcs = vec![];
    let mut trimeshes = vec![];

    for scene in gltf.scenes() {
        let iterator = SceneIterator::new(scene.nodes().collect(), |primitive, world_transform| {
            let reader = primitive.reader(|buffer| Some(&*buffers[buffer.index()].0));

            let mut point_cloud = vec![];
            let mut positions = vec![];

            if let (Some(positions_iter), Some(tex_coords)) =
                (reader.read_positions(), reader.read_tex_coords(0))
            {
                for (pos, uv) in positions_iter.zip(tex_coords.into_f32()) {
                    let color = sample_texture(&texture, uv);
                    let position = Point3::new(pos[0], pos[1], pos[2]);
                    let transformed = world_transform.transform_point(&position);
                    point_cloud.push((transformed, color));
                    positions.push(transformed);
                }

                let indices = reader
                    .read_indices()
                    .expect("No indices found")
                    .into_u32()
                    .map(|i| i as usize)
                    .collect::<Vec<_>>();

                (point_cloud, (positions, indices))
            } else {
                (vec![], (vec![], vec![]))
            }
        });

        for (pc, trimesh) in iterator {
            pcs.extend(pc);
            trimeshes.push(trimesh);
        }
    }

    (pcs, trimeshes)
}

/// Load a glb or gltf file and return a point cloud and a list of trimeshes.
///
/// ```
/// let mut file = File::open("path_to_file.glb").expect("Failed to open GLB file");
/// let mut buffer = Vec::new();
/// file.read_to_end(&mut buffer).expect("Failed to read file");
///
/// let (gltf, buffers, _) = gltf::import_slice(slice).expect("Failed to parse GLB");
///
/// let _ = load_model_trimeshes(&buffer);
/// ```
pub fn load_model_trimeshes<S>(slice: S) -> Vec<(Vec<nalgebra::Point3<f32>>, Vec<usize>)>
where
    S: AsRef<[u8]>,
{
    let (gltf, buffers, _) = gltf::import_slice(slice).expect("Failed to parse GLB");

    let mut trimeshes = vec![];

    for scene in gltf.scenes() {
        let iterator = SceneIterator::new(scene.nodes().collect(), |primitive, world_transform| {
            let reader = primitive.reader(|buffer| Some(&*buffers[buffer.index()].0));

            let mut positions = vec![];

            if let Some(positions_iter) = reader.read_positions() {
                for pos in positions_iter {
                    let position = Point3::new(pos[0], pos[1], pos[2]);
                    let transformed = world_transform.transform_point(&position);
                    positions.push(transformed);
                }

                let indices = reader
                    .read_indices()
                    .expect("No indices found")
                    .into_u32()
                    .map(|i| i as usize)
                    .collect::<Vec<_>>();

                (positions, indices)
            } else {
                (vec![], vec![])
            }
        });

        for trimesh in iterator {
            trimeshes.push(trimesh);
        }
    }

    trimeshes
}

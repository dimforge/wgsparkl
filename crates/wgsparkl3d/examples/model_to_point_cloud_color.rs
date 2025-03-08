use bevy::{
    app::{App, Startup, Update},
    color::Color,
    core_pipeline::core_3d::Camera3d,
    ecs::{
        component::Component,
        system::{Commands, Query},
    },
    gizmos::gizmos::Gizmos,
    math::Vec3,
    picking::mesh_picking::MeshPickingPlugin,
    render::camera::Camera,
    DefaultPlugins,
};
use bevy_editor_cam::{prelude::EditorCam, DefaultEditorCamPlugins};
use image::RgbaImage;
use std::{fs::File, io::Read};

fn extract_embedded_texture<'a>(
    gltf: &gltf::Document,
    buffers: &'a [gltf::buffer::Data],
) -> Option<RgbaImage> {
    for mat in gltf.materials() {
        dbg!(mat.name());
        if (mat.name() == Some("Body_SG1")) {
            // dbg!(&mat);
        }

        let Some(texture) = mat.pbr_metallic_roughness().base_color_texture() else {
            continue;
        };
        dbg!(texture.texture().name());
    }
    for image in gltf.images() {
        //dbg!(tex.name());
        //let image = tex.source();
        dbg!(image.name());
        if let gltf::image::Source::View { view, mime_type } = image.source() {
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
    let y = ((1.0 - uv[1]) * height as f32).clamp(0.0, (height - 1) as f32) as u32; // Flip Y

    texture.get_pixel(x, y).0
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, MeshPickingPlugin, DefaultEditorCamPlugins))
        .add_systems(Startup, init_scene)
        .add_systems(Update, display_point_cloud)
        .run();
}

#[derive(Component)]
pub struct PointCloud {
    pub positions: Vec<(Vec3, Color)>,
}

fn init_scene(mut commands: Commands) {
    let path = "assets/Car 3D Model.glb"; // Replace with your actual GLB file path
    let pc = load_model(path);

    commands.spawn((Camera3d::default(), Camera::default(), EditorCam::default()));

    commands.spawn(PointCloud { positions: pc });
}

fn display_point_cloud(mut pcs: Query<&PointCloud>, mut gizmos: Gizmos) {
    for pc in pcs.iter() {
        for p in pc.positions.iter() {
            gizmos.sphere(p.0, 0.5f32, p.1);
        }
    }
}

fn load_model(path: &str) -> Vec<(Vec3, Color)> {
    let mut file = File::open(path).expect("Failed to open GLB file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");

    let (gltf, buffers, _) = gltf::import_slice(&buffer).expect("Failed to parse GLB");

    // Extract embedded texture
    let texture = extract_embedded_texture(&gltf, &buffers).expect("Failed to extract texture");

    let mut result = vec![];
    for scene in gltf.scenes() {
        for node in scene.nodes() {
            result.append(&mut recurse_inspect_scene(&buffers, &texture, node));
        }
    }
    result
}

fn recurse_inspect_scene(
    buffers: &Vec<gltf::buffer::Data>,
    texture: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    node: gltf::Node<'_>,
) -> Vec<(Vec3, Color)> {
    dbg!(node.name());
    let mut result = vec![];
    if let Some(mesh) = node.mesh() {
        dbg!(mesh.name());
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&*buffers[buffer.index()].0));

            if let (Some(positions), Some(tex_coords)) =
                (reader.read_positions(), reader.read_tex_coords(0))
            {
                let mut vertex_data = Vec::new();

                for (pos, uv) in positions.zip(tex_coords.into_f32()) {
                    let color = sample_texture(texture, uv);
                    vertex_data.push((pos, color));
                }

                for (i, (pos, color)) in vertex_data.iter().enumerate() {
                    result.push((
                        Vec3::new(pos[0], pos[1], pos[2]),
                        Color::linear_rgba(
                            color[0] as f32 / 255f32,
                            color[1] as f32 / 255f32,
                            color[2] as f32 / 255f32,
                            color[3] as f32 / 255f32,
                        ),
                    ));
                }
            } else {
                println!("No UV coordinates or positions found.");
            }
        }
    }
    for child in node.children() {
        result.append(&mut recurse_inspect_scene(buffers, texture, child));
    }
    result
}

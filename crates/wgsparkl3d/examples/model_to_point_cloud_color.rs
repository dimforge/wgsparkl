//mod model_to_point_cloud;
use super::model_to_point_cloud;

use bevy::{
    app::{App, Startup, Update},
    asset::RenderAssetUsages,
    color::{palettes::css, Color},
    core_pipeline::core_3d::Camera3d,
    ecs::{
        component::Component,
        system::{Commands, Query},
    },
    gizmos::gizmos::Gizmos,
    math::Vec3,
    pbr::{
        wireframe::{Wireframe, WireframePlugin},
        CascadeShadowConfigBuilder,
    },
    picking::mesh_picking::MeshPickingPlugin,
    prelude::*,
    render::{
        camera::Camera,
        mesh::{Indices, Mesh},
        render_resource::WgpuFeatures,
        renderer::RenderDevice,
        settings::{RenderCreation, WgpuSettings},
        RenderPlugin,
    },
    DefaultPlugins,
};
use bevy_editor_cam::{prelude::EditorCam, DefaultEditorCamPlugins};
use image::RgbaImage;
use nalgebra::{Matrix4, Point3, Vector3};
use std::{f32::consts::PI, fs::File, io::Read};
use wgpu::{Features, PrimitiveTopology};
use wgsparkl_testbed3d::AppState;

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
    let y = ((uv[1]) * height as f32).clamp(0.0, (height - 1) as f32) as u32;

    texture.get_pixel(x, y).0
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(RenderPlugin {
                render_creation: RenderCreation::Automatic(WgpuSettings {
                    // WARN this is a native only feature. It will not work with webgl or webgpu
                    features: WgpuFeatures::POLYGON_MODE_LINE,
                    ..default()
                }),
                ..default()
            }),
            // You need to add this plugin to enable wireframe rendering
            WireframePlugin,
            MeshPickingPlugin,
            DefaultEditorCamPlugins,
        ))
        .add_systems(Startup, init_scene)
        .add_systems(Update, display_point_cloud)
        .run();
}

#[derive(Component)]
pub struct PointCloud {
    pub positions: Vec<(Vec3, Color)>,
}

fn closest_point(target: Vec3, points: &[(Vec3, Color)]) -> Option<&(Vec3, Color)> {
    points.iter().min_by(|a, b| {
        a.0.distance_squared(target)
            .partial_cmp(&b.0.distance_squared(target))
            .unwrap()
    })
}

fn init_scene(mut commands: Commands) {
    commands.spawn((Camera3d::default(), Camera::default(), EditorCam::default()));
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        Transform {
            translation: Vec3::new(0.0, 2.0, 0.0),
            rotation: Quat::from_rotation_x(-PI / 4.),
            ..default()
        },
    ));

    let pc_grid = load_model_with_colors();

    commands.spawn(PointCloud { positions: pc_grid });
}

fn load_model_with_colors() -> Vec<(Vec3, Color)> {
    let path = "assets/shiba.glb";
    // Replace with your actual GLB file path
    let mut res = load_model(path);
    let mut pc_grid = vec![];

    let colors = [
        css::BLUE,
        css::RED,
        css::GREEN,
        css::YELLOW,
        css::SEASHELL,
        css::MAGENTA,
        css::WHITE,
        css::BLACK,
        css::BROWN,
    ];
    // TODO: load real gltf model to add a comparison
    // commands.spawn((
    //     Transform::from_xyz(0.0, 0.0, 0.0),
    //     SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("car/scene.gltf"))),
    // ));

    let scale = 3.0;
    res.0.iter_mut().for_each(|p| p.0 *= scale);

    for (t_id, mut trimesh) in res.1.iter_mut().enumerate() {
        if t_id != 0 {
            //continue;
        }
        let color = Color::from(colors[t_id % colors.len()]);
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            trimesh
                .0
                .iter()
                .map(|p| Vec3::new(p.x, p.y, p.z))
                .collect::<Vec<_>>(),
        )
        .with_inserted_indices(Indices::U32(trimesh.1.iter().map(|i| *i as u32).collect()));
        mesh.duplicate_vertices();
        mesh.compute_flat_normals();

        trimesh.0.iter_mut().for_each(|p| *p *= scale);
        let mut pc =
            model_to_point_cloud::get_point_cloud_from_trimesh(&trimesh.0, &trimesh.1, 18.0)
                .into_iter()
                .enumerate()
                .map(|(i, (p, color))| {
                    let closest_color = closest_point(p, &res.0).unwrap();
                    (p, closest_color.1)
                })
                .collect();
        pc_grid.append(&mut pc);
    }
    pc_grid
}

pub fn elastic_color_model_demo(
    mut commands: Commands,
    device: Res<RenderDevice>,
    mut app_state: ResMut<AppState>,
) {
    let pc_grid = load_model_with_colors();
    model_to_point_cloud::spawn_elastic_model_demo(
        commands.reborrow(),
        device,
        app_state,
        &pc_grid,
    );
}

fn display_point_cloud(mut pcs: Query<&PointCloud>, mut gizmos: Gizmos) {
    for pc in pcs.iter() {
        for p in pc.positions.iter() {
            gizmos.sphere(p.0, 0.01f32, p.1);
        }
    }
}

fn load_model(
    path: &str,
) -> (
    Vec<(Vec3, Color)>,
    Vec<(Vec<nalgebra::Point3<f32>>, Vec<usize>)>,
) {
    let mut file = File::open(path).expect("Failed to open GLB file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");

    let (gltf, buffers, _) = gltf::import_slice(&buffer).expect("Failed to parse GLB");

    // Extract embedded texture
    let texture = extract_embedded_texture(&gltf, &buffers).expect("Failed to extract texture");

    let mut pcs = vec![];
    let mut trimeshes = vec![];
    for scene in gltf.scenes() {
        for node in scene.nodes() {
            let mut res = recurse_inspect_scene(&buffers, &texture, node, Matrix4::identity());
            pcs.append(&mut res.0);
            trimeshes.append(&mut res.1);
        }
    }

    (pcs, trimeshes)
}

fn get_node_transform(node: &gltf::Node, parent_transform: Matrix4<f32>) -> Matrix4<f32> {
    let matrix = node.transform().matrix();
    parent_transform * Matrix4::from_column_slice(&matrix.concat())
}

fn recurse_inspect_scene(
    buffers: &Vec<gltf::buffer::Data>,
    texture: &image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    node: gltf::Node<'_>,
    parent_transform: Matrix4<f32>,
) -> (
    Vec<(Vec3, Color)>,
    Vec<(Vec<nalgebra::Point3<f32>>, Vec<usize>)>,
) {
    dbg!(node.name());
    let world_transform = get_node_transform(&node, parent_transform);
    let mut point_cloud = vec![];
    let mut trimeshes = vec![];
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

                let mut positions = vec![];
                for (i, (pos, color)) in vertex_data.iter().enumerate() {
                    let position = Point3::new(pos[0], pos[1], pos[2]);
                    let transformed = world_transform.transform_point(&position);
                    point_cloud.push((
                        Vec3::new(transformed.x, transformed.y, transformed.z),
                        Color::linear_rgba(
                            color[0] as f32 / 255f32,
                            color[1] as f32 / 255f32,
                            color[2] as f32 / 255f32,
                            color[3] as f32 / 255f32,
                        ),
                    ));

                    positions.push(transformed);
                }
                // read indices
                let indices = reader
                    .read_indices()
                    .expect("No indices found")
                    .into_u32()
                    .map(|i| i as usize)
                    .collect::<Vec<_>>();
                trimeshes.push((positions, indices));
            } else {
                println!("No UV coordinates or positions found.");
            }
        }
    }
    for child in node.children() {
        let mut res = recurse_inspect_scene(buffers, texture, child, world_transform);
        point_cloud.append(&mut res.0);
        trimeshes.append(&mut res.1);
    }
    (point_cloud, trimeshes)
}

use crate::instancing::{InstanceBuffer, InstanceData, InstanceMaterialData};
use crate::prep_vertex_buffer::{GpuRenderConfig, RenderConfig, RenderMode, WgPrepVertexBuffer};
use crate::step::TimestampChannel;
use crate::{AppState, PhysicsContext, RunState, Timestamps};
use bevy::asset::Assets;
use bevy::color::Color;
use bevy::hierarchy::DespawnRecursiveExt;
use bevy::math::{Vec3, Vec4};
use bevy::pbr::AmbientLight;
use bevy::prelude::*;
use bevy::render::render_resource::BufferUsages;
use bevy::render::renderer::RenderDevice;
use bevy::render::view::NoFrustumCulling;
use bevy_editor_cam::prelude::{EditorCam, EnabledMotion};
use std::sync::Arc;
use wgcore::hot_reloading::HotReloadState;
use wgcore::tensor::GpuVector;
use wgcore::timestamps::GpuTimestamps;
use wgcore::Shader;
use wgpu::Features;
use wgsparkl::pipeline::MpmPipeline;

/// set up a simple 3D scene
pub fn setup_app(mut commands: Commands, device: Res<RenderDevice>) {
    // app state
    let render_config = RenderConfig::new(RenderMode::Velocity);
    let gpu_render_config = GpuRenderConfig::new(device.wgpu_device(), render_config);
    let prep_vertex_buffer = WgPrepVertexBuffer::from_device(device.wgpu_device()).unwrap();

    let mut hot_reload = HotReloadState::new().unwrap();
    let pipeline = MpmPipeline::new(device.wgpu_device()).unwrap();
    pipeline.init_hot_reloading(&mut hot_reload);

    commands.insert_resource(AppState {
        render_config,
        gpu_render_config,
        prep_vertex_buffer,
        pipeline,
        run_state: RunState::Paused,
        num_substeps: 1,
        gravity_factor: 1.0,
        restarting: false,
        selected_scene: 0,
        hot_reload,
    });

    let (snd, rcv) = async_channel::unbounded();
    commands.insert_resource(TimestampChannel { snd, rcv });

    let features = device.features();
    let timestamps = features
        .contains(Features::TIMESTAMP_QUERY)
        .then(|| GpuTimestamps::new(device.wgpu_device(), 256));
    commands.insert_resource(Timestamps {
        timestamps,
        ..Default::default()
    });

    // light
    commands.insert_resource(AmbientLight {
        brightness: 1000.0,
        ..Default::default()
    });

    // camera
    #[cfg(feature = "dim2")]
    {
        commands.spawn((
            Camera3dBundle {
                transform: Transform::from_translation(Vec3::new(25.0, 25.0, 100.0)),
                // projection: Projection::Orthographic(OrthographicProjection {
                //     // scaling_mode: ScalingMode::FixedVertical(6.0),
                //     ..OrthographicProjection::default_3d()
                // }),
                ..default()
            },
            EditorCam {
                enabled_motion: EnabledMotion {
                    orbit: false,
                    ..Default::default()
                },
                last_anchor_depth: -99.0,
                ..Default::default()
            },
        ));
    }

    #[cfg(feature = "dim3")]
    {
        commands.spawn((
            Camera3dBundle {
                transform: Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
                ..default()
            },
            EditorCam::default(),
        ));
    }
}

pub fn setup_graphics(
    mut commands: Commands,
    device: Res<RenderDevice>,
    physics: Res<PhysicsContext>,
    mut meshes: ResMut<Assets<Mesh>>,
    to_clear: Query<Entity, With<InstanceMaterialData>>,
) {
    if let Ok(to_clear) = to_clear.get_single() {
        commands.entity(to_clear).despawn_recursive();
    }

    let device = device.wgpu_device();
    let colors = [
        Color::srgb_u8(124, 144, 255),
        Color::srgb_u8(8, 144, 255),
        Color::srgb_u8(124, 7, 255),
        Color::srgb_u8(124, 144, 7),
        Color::srgb_u8(200, 37, 255),
        Color::srgb_u8(124, 230, 25),
    ];
    let radius = physics.particles[0].volume.init_radius();
    let cube = meshes.add(Cuboid {
        half_size: Vec3::splat(radius),
    });

    let mut instances = vec![];
    for (rb_id, particle) in physics.particles.iter().enumerate() {
        let base_color = colors[rb_id % colors.len()].to_linear().to_f32_array();
        instances.push(InstanceData {
            deformation: [Vec4::X, Vec4::Y, Vec4::Z],
            #[cfg(feature = "dim2")]
            position: Vec4::new(particle.position.x, particle.position.y, 0.0, 0.0),
            #[cfg(feature = "dim3")]
            position: Vec4::new(
                particle.position.x,
                particle.position.y,
                particle.position.z,
                0.0,
            ),
            base_color,
            color: base_color,
        });
    }

    let instances_buffer = GpuVector::init(
        device,
        &instances,
        BufferUsages::STORAGE | BufferUsages::VERTEX,
    );

    let num_instances = instances.len();
    commands.spawn((
        Mesh3d(cube),
        SpatialBundle::INHERITED_IDENTITY,
        InstanceMaterialData {
            data: instances,
            buffer: InstanceBuffer {
                buffer: Arc::new(instances_buffer.into_inner().into()),
                length: num_instances,
            },
        },
        NoFrustumCulling,
    ));
}

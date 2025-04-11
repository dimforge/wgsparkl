use crate::instancing::{InstanceBuffer, InstanceData, InstanceMaterialData};
use crate::prep_vertex_buffer::{GpuRenderConfig, RenderConfig, RenderMode, WgPrepVertexBuffer};
use crate::rigid_graphics::{BevyMaterial, EntityWithGraphics};
use crate::step::TimestampChannel;
use crate::{AppState, Callbacks, PhysicsContext, RenderContext, RunState, Timestamps};
use bevy::asset::Assets;
use bevy::color::Color;
use bevy::hierarchy::DespawnRecursiveExt;
use bevy::math::{Vec3, Vec4};
use bevy::pbr::AmbientLight;
use bevy::prelude::*;
use bevy::render::render_resource::BufferUsages;
use bevy::render::renderer::RenderDevice;
use bevy::render::view::NoFrustumCulling;
use bevy_editor_cam::prelude::EditorCam;
use nalgebra::point;
use std::sync::Arc;
use wgcore::hot_reloading::HotReloadState;
use wgcore::tensor::GpuVector;
use wgcore::timestamps::GpuTimestamps;
use wgcore::Shader;
use wgpu::Features;
use wgsparkl::pipeline::MpmPipeline;
use wgsparkl::rapier::parry::math::Isometry;

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
        show_rigid_particles: false,
    });
    commands.init_resource::<RenderContext>();

    let (snd, rcv) = async_channel::unbounded();
    commands.insert_resource(TimestampChannel { snd, rcv });

    let features = device.features();
    let timestamps = features
        .contains(Features::TIMESTAMP_QUERY)
        .then(|| GpuTimestamps::new(device.wgpu_device(), 512));
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
            Camera3d::default(),
            Transform::from_translation(Vec3::new(25.0, 25.0, 100.0)),
            EditorCam {
                enabled_motion: bevy_editor_cam::prelude::EnabledMotion {
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
            Camera3d::default(),
            Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
            EditorCam::default(),
        ));
    }
}

fn setup_rapier_graphics(
    commands: &mut Commands,
    _device: &RenderDevice,
    physics: &PhysicsContext,
    rigid_render: &mut RenderContext,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<BevyMaterial>,
    _to_clear: &Query<Entity, With<InstanceMaterialData>>,
) {
    for rigid in rigid_render.rigid_entities.drain(..) {
        commands.entity(rigid.entity).despawn_recursive();
    }

    for (handle, collider) in physics.rapier_data.colliders.iter() {
        let parent = &physics.rapier_data.bodies[collider.parent().unwrap()];
        let color = if parent.is_fixed() {
            point![0.5, 0.5, 0.5]
        } else if parent.is_kinematic() {
            point![0.0, 0.0, 1.0]
        } else {
            point![0.0, 1.0, 0.0]
        };
        let e = EntityWithGraphics::spawn(
            commands,
            meshes,
            materials,
            &mut rigid_render.prefab_meshes,
            &mut rigid_render.instanced_materials,
            collider.shape(),
            Some(handle),
            *collider.position(),
            Isometry::identity(), // TODO
            color,
            false,
        );
        rigid_render.rigid_entities.push(e);
    }
}

#[derive(Component)]
pub struct RigidParticlesTag;

fn setup_particles_graphics(
    commands: &mut Commands,
    device: &RenderDevice,
    app_state: &AppState,
    physics: &PhysicsContext,
    meshes: &mut Assets<Mesh>,
    to_clear: &Query<Entity, With<InstanceMaterialData>>,
) {
    for to_clear in to_clear.iter() {
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
    let radius = physics.particles[0].dynamics.init_radius;
    let cube = meshes.add(Cuboid {
        half_size: Vec3::splat(radius),
    });

    /*
     * Particles rendering.
     */
    let mut instances = vec![];
    for (rb_id, particle) in physics.particles.iter().enumerate() {
        let base_color = particle.color.map_or_else(
            || colors[rb_id % colors.len()].to_linear().to_f32_array(),
            |c| c.map(|c| c as f32 / 255f32),
        );
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
        Mesh3d(cube.clone()),
        Transform::IDENTITY,
        Visibility::Inherited,
        InstanceMaterialData {
            data: instances,
            buffer: InstanceBuffer {
                buffer: Arc::new(instances_buffer.into_inner().into()),
                length: num_instances,
            },
        },
        NoFrustumCulling,
    ));

    /*
     * Rigid particles rendering.
     */
    if !physics.data.rigid_particles.is_empty() {
        let mut instances = vec![];
        for id in 0..physics.data.rigid_particles.len() as usize {
            let base_color = colors[id % colors.len()].to_linear().to_f32_array();
            instances.push(InstanceData {
                base_color,
                color: base_color,
                ..Default::default()
            });
        }

        let instances_buffer = GpuVector::init(
            device,
            &instances,
            BufferUsages::STORAGE | BufferUsages::VERTEX,
        );

        let num_instances = instances.len();
        let visibility = if app_state.show_rigid_particles {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };

        commands.spawn((
            Mesh3d(cube),
            Transform::IDENTITY,
            visibility,
            InstanceMaterialData {
                data: instances,
                buffer: InstanceBuffer {
                    buffer: Arc::new(instances_buffer.into_inner().into()),
                    length: num_instances,
                },
            },
            NoFrustumCulling,
            RigidParticlesTag,
        ));
    }
}

pub fn setup_graphics(
    mut commands: Commands,
    app_state: Res<AppState>,
    device: Res<RenderDevice>,
    physics: Res<PhysicsContext>,
    mut rigid_render: ResMut<RenderContext>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<BevyMaterial>>,
    to_clear: Query<Entity, With<InstanceMaterialData>>,
) {
    setup_particles_graphics(
        &mut commands,
        &device,
        &app_state,
        &physics,
        &mut meshes,
        &to_clear,
    );
    setup_rapier_graphics(
        &mut commands,
        &device,
        &physics,
        &mut rigid_render,
        &mut meshes,
        &mut materials,
        &to_clear,
    );
}

pub fn setup_app_state(mut callbacks: ResMut<Callbacks>) {
    callbacks.0.clear();
}

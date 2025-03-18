use std::fs::File;
use std::io::Read;

use crate::utils::default_scene::{self, SAMPLE_PER_UNIT};

use bevy::color::palettes::css;
use bevy::{prelude::*, render::renderer::RenderDevice};
use nalgebra::{vector, Isometry3, Transform3, UnitQuaternion, Vector3};
use rapier3d::prelude::RigidBodyHandle;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, SharedShape, TriMeshFlags};
use wgsparkl3d::load_mesh3d::load_gltf::{load_model_trimeshes, load_model_with_colors};
use wgsparkl3d::pipeline::MpmData;
use wgsparkl3d::solver::SimulationParams;
use wgsparkl_testbed3d::Callbacks;
use wgsparkl_testbed3d::{AppState, PhysicsContext};
use wgsparkl_testbed3d::{Callback, RapierData};

pub fn demo(
    device: RenderDevice,
    app_state: &mut AppState,
    callbacks: &mut Callbacks,
) -> PhysicsContext {
    let mut file = File::open("assets/banana.glb").expect("Failed to open GLB file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");

    let pc_grid = load_model_with_colors(
        &buffer,
        Transform3::from_matrix_unchecked(
            Isometry3::from_parts(
                Vector3::new(0.0, 2.4, 0.0).into(),
                UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -90f32.to_radians()),
            )
            .to_matrix()
                * nalgebra::Matrix4::new_scaling(0.3),
        ),
        Some(css::BLANCHED_ALMOND.to_u8_array()),
        SAMPLE_PER_UNIT,
    )
    .iter()
    .map(|p| {
        (
            Vec3::new(p.0.x, p.0.y, p.0.z),
            Color::from(Srgba::from_u8_array(p.1)),
        )
    })
    .collect::<Vec<_>>();

    let params = SimulationParams {
        gravity: vector![0.0, -9.81, 0.0] * app_state.gravity_factor,
        dt: (1.0 / 60.0) / (app_state.num_substeps as f32),
    };
    let mut rapier_data = RapierData::default();
    if !app_state.restarting {
        app_state.num_substeps = 30;
        app_state.gravity_factor = 1.0;
    };
    default_scene::spawn_ground_and_walls(&mut rapier_data);

    let mut particles = vec![];
    for (pos, color) in pc_grid {
        let particle =
            default_scene::create_particle(&pos, Some(color), 1f32 / SAMPLE_PER_UNIT / 2f32);
        particles.push(particle);
    }
    // Slicer
    let mut file =
        File::open("assets/chefs_knife_open_blade.glb").expect("Failed to open GLB file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");
    let mut slicer_trimeshes = load_model_trimeshes(&buffer);
    slicer_trimeshes.iter_mut().for_each(|trimesh| {
        trimesh.0.iter_mut().for_each(|v| {
            *v *= 10f32;
        });
    });

    let rb = RigidBodyBuilder::kinematic_position_based().translation(vector![0.0, 5.0, 0.0]);
    let parent_handle = rapier_data.bodies.insert(rb);
    /*
    // NOTE: Heightfield may be more efficient or more predictable than trimesh for the knife.
    let heights = nalgebra::DMatrix::zeros(10, 5);
    let heightfield = rapier3d::prelude::HeightField::new(heights, vector![2.0, 1.0, 10.0]);
    let (mut vtx, idx) = heightfield.to_trimesh();
    vtx.iter_mut().for_each(|pt| {
        *pt = Isometry3::rotation(vector![0f32, 0f32, -90f32.to_radians()]) * *pt
            + vector![0.0, 1f32, 0f32]
    });
    let co = ColliderBuilder::trimesh(vtx, idx).unwrap();
    rapier_data
        .colliders
        .insert_with_parent(co, parent_handle, &mut rapier_data.bodies);
    */
    for (vertices, indices) in slicer_trimeshes.iter() {
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
        rapier_data
            .colliders
            .insert_with_parent(collider, parent_handle, &mut rapier_data.bodies);
    }

    let data = MpmData::new(
        device.wgpu_device(),
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        1f32 / default_scene::SAMPLE_PER_UNIT,
        60_000,
    );

    let new_physics = PhysicsContext {
        data,
        rapier_data,
        particles,
    };
    //
    callbacks.0.push(move_knife_function(parent_handle));

    new_physics
}

fn move_knife_function(body_handle: RigidBodyHandle) -> Callback {
    Box::new(
        move |_render, physics: &mut PhysicsContext, _timestamps, app_state: &AppState| {
            let t = app_state.physics_time_seconds as f32;

            let body = physics.rapier_data.bodies.get_mut(body_handle).unwrap();
            let length = 1.31;
            let width = 0.35;
            let x_pos = 2.8;
            let y_pos = 1.3;
            let z_pos = -1.5;
            let velocity = 0.9;
            let extended_width = 0.15; // extended width for the horizontal move to the right
            let period =
                (2f32 * length + width + extended_width + 2f32 * width + extended_width) / velocity;

            let i = (t / period).floor();
            let dis = velocity * (t - period * i);

            // Determine the new position of the knife based on the current displacement
            let new_pos = if dis < length {
                // Moving downwards
                vector![x_pos - width * i, y_pos - dis, z_pos]
            } else if length <= dis && dis < length + width + extended_width {
                // Moving horizontally to the right with extended width
                vector![x_pos - width * i + (dis - length), y_pos - length, z_pos]
            } else if length + width + extended_width <= dis
                && dis < 2f32 * length + width + extended_width
            {
                // Moving upwards
                vector![
                    x_pos - width * i + width + extended_width,
                    y_pos - length + dis - (length + width + extended_width),
                    z_pos
                ]
            } else {
                // Moving horizontally to the left, back to the starting x position
                vector![
                    x_pos - width * i + width + extended_width
                        - (dis - 2.0 * length - width - extended_width),
                    y_pos,
                    z_pos
                ]
            };

            body.set_translation(new_pos, true);
        },
    )
}

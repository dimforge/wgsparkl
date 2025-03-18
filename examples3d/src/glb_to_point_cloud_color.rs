use crate::utils::default_scene::{self, SAMPLE_PER_UNIT};

use bevy::{
    ecs::system::Query, gizmos::gizmos::Gizmos, math::Vec3, prelude::*,
    render::renderer::RenderDevice,
};
use nalgebra::{vector, Isometry3, Transform3, UnitQuaternion, Vector3};
use std::{fs::File, io::Read};
use wgsparkl3d::load_mesh3d::load_gltf::load_model_with_colors;
use wgsparkl3d::{pipeline::MpmData, solver::SimulationParams};
use wgsparkl_testbed3d::{AppState, Callbacks, PhysicsContext, RapierData};

pub fn elastic_color_model_demo(
    device: RenderDevice,
    app_state: &mut AppState,
    _callbacks: &mut Callbacks,
) -> PhysicsContext {
    let mut file = File::open("assets/shiba.glb").expect("Failed to open GLB file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");
    let pc_grid = load_model_with_colors(
        &buffer,
        Transform3::from_matrix_unchecked(
            Isometry3::from_parts(
                Vector3::new(0.0, 6.0, 0.0).into(),
                UnitQuaternion::identity(),
            )
            .to_matrix()
                * nalgebra::Matrix4::new_scaling(3.0),
        ),
        None,
        SAMPLE_PER_UNIT,
    );
    let params = SimulationParams {
        gravity: vector![0.0, -9.81, 0.0] * app_state.gravity_factor,
        dt: (1.0 / 60.0) / (app_state.num_substeps as f32),
    };
    let mut rapier_data = RapierData::default();
    if !app_state.restarting {
        app_state.num_substeps = 20;
        app_state.gravity_factor = 1.0;
    };
    default_scene::spawn_ground_and_walls(&mut rapier_data);

    let mut particles = vec![];
    for (pos, color) in pc_grid {
        let particle = default_scene::create_particle(
            &Vec3::new(pos.x, pos.y, pos.z),
            Some(Color::from(Srgba::from_u8_array(color))),
            1f32 / SAMPLE_PER_UNIT / 2f32,
        );
        particles.push(particle);
    }
    default_scene::spawn_ground_and_walls(&mut rapier_data);
    let data = MpmData::new(
        device.wgpu_device(),
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        1f32 / default_scene::SAMPLE_PER_UNIT,
        60_000,
    );

    PhysicsContext {
        data,
        rapier_data,
        particles,
    }
}

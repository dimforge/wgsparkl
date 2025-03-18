#![allow(clippy::too_many_arguments)]
#![allow(clippy::module_inception)]

#[cfg(feature = "dim2")]
pub extern crate wgsparkl2d as wgsparkl;
#[cfg(feature = "dim3")]
pub extern crate wgsparkl3d as wgsparkl;

use bevy::render::renderer::RenderDevice;
#[cfg(feature = "dim2")]
pub use instancing2d as instancing;
#[cfg(feature = "dim3")]
pub use instancing3d as instancing;
use std::collections::HashMap;

#[cfg(feature = "dim2")]
pub mod instancing2d;
#[cfg(feature = "dim3")]
pub mod instancing3d;

mod hot_reload;
pub mod prep_vertex_buffer;
mod rigid_graphics;
pub mod startup;
pub mod step;
pub mod ui;

use bevy::asset::load_internal_asset;
use bevy::ecs::system::{BoxedSystem, SystemId};
use bevy::pbr::wireframe::WireframePlugin;
use bevy::prelude::*;
use bevy_editor_cam::prelude::DefaultEditorCamPlugins;
// use bevy_wasm_window_resize::WindowResizePlugin;
use crate::rigid_graphics::{EntityWithGraphics, InstancedMaterials};
use instancing::INSTANCING_SHADER_HANDLE;
use prep_vertex_buffer::{GpuRenderConfig, RenderConfig, WgPrepVertexBuffer};
use wgcore::hot_reloading::HotReloadState;
use wgcore::timestamps::GpuTimestamps;
use wgsparkl::rapier::dynamics::{CCDSolver, IntegrationParameters, RigidBodySet};
use wgsparkl::rapier::geometry::{ColliderSet, NarrowPhase};
use wgsparkl::rapier::prelude::{
    DefaultBroadPhase, ImpulseJointSet, IslandManager, MultibodyJointSet, PhysicsPipeline,
    ShapeType,
};
use wgsparkl::{
    pipeline::{MpmData, MpmPipeline},
    solver::Particle,
};

pub fn init_testbed(app: &mut App) {
    app.add_plugins(DefaultPlugins)
        // .add_plugins(WindowResizePlugin)
        .add_plugins((
            // bevy_mod_picking::DefaultPickingPlugins,
            DefaultEditorCamPlugins,
        ))
        .add_plugins(instancing::ParticlesMaterialPlugin)
        .add_plugins(bevy_egui::EguiPlugin)
        .init_resource::<SceneInits>()
        .init_resource::<Callbacks>()
        .add_systems(Startup, startup::setup_app)
        .add_systems(
            Update,
            (
                ui::update_ui,
                (step::step_simulation, step::callbacks)
                    .chain()
                    .run_if(|state: Res<AppState>| state.run_state != RunState::Paused),
                rigid_graphics::update_rigid_graphics,
                hot_reload::handle_hot_reloading,
            )
                .chain(),
        );

    #[cfg(not(target_arch = "wasm32"))]
    app.add_plugins(WireframePlugin);

    #[cfg(feature = "dim2")]
    load_internal_asset!(
        app,
        INSTANCING_SHADER_HANDLE,
        "./instancing2d.wgsl",
        Shader::from_wgsl
    );
    #[cfg(feature = "dim3")]
    load_internal_asset!(
        app,
        INSTANCING_SHADER_HANDLE,
        "./instancing3d.wgsl",
        Shader::from_wgsl
    );
}

#[derive(Resource)]
pub struct AppState {
    pub run_state: RunState,
    pub render_config: RenderConfig,
    pub gpu_render_config: GpuRenderConfig,
    pub pipeline: MpmPipeline,
    pub prep_vertex_buffer: WgPrepVertexBuffer,
    pub num_substeps: usize,
    pub gravity_factor: f32,
    pub restarting: bool,
    pub selected_scene: usize,
    pub hot_reload: HotReloadState,
    pub show_rigid_particles: bool,
    pub physics_time_seconds: f64,
}

pub use wgsparkl::rapier::prelude::PhysicsContext as RapierData;

#[derive(Resource)]
pub struct PhysicsContext {
    pub data: MpmData,
    pub rapier_data: RapierData,
    pub particles: Vec<Particle>,
}

#[derive(Resource, Default)]
pub struct Callbacks(pub Vec<Callback>);

pub type Callback = Box<
    dyn FnMut(Option<&mut RenderContext>, &mut PhysicsContext, &Timestamps, &AppState)
        + Send
        + Sync,
>;

#[derive(Resource, Default)]
pub struct RenderContext {
    pub instanced_materials: InstancedMaterials,
    pub prefab_meshes: HashMap<ShapeType, Handle<Mesh>>,
    pub rigid_entities: Vec<EntityWithGraphics>,
}

#[derive(Resource, Default)]
pub struct Timestamps {
    pub timestamps: Option<GpuTimestamps>,
    pub update_rigid_particles: f64,
    pub grid_sort: f64,
    pub grid_update_cdf: f64,
    pub p2g_cdf: f64,
    pub g2p_cdf: f64,
    pub p2g: f64,
    pub grid_update: f64,
    pub g2p: f64,
    pub particles_update: f64,
    pub integrate_bodies: f64,
}

impl Timestamps {
    pub fn total_time(&self) -> f64 {
        self.update_rigid_particles
            + self.grid_sort
            + self.grid_update_cdf
            + self.p2g_cdf
            + self.g2p_cdf
            + self.p2g
            + self.grid_update
            + self.g2p
            + self.particles_update
            + self.integrate_bodies
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RunState {
    Running,
    Paused,
    Step,
}

#[derive(Resource)]
pub struct SceneInits {
    pub scenes: Vec<(String, SceneInitFn)>,
    reset_graphics: SystemId,
    reset_app_state: SystemId,
}
pub type SceneInitFn =
    Box<dyn FnMut(RenderDevice, &mut AppState, &mut Callbacks) -> PhysicsContext + Send + Sync>;

impl SceneInits {
    pub fn init_scene(&self, commands: &mut Commands, scene_id: usize) {
        commands.run_system(self.reset_app_state);
        commands.run_system_cached_with(run_scene_init, scene_id);
        commands.run_system(self.reset_graphics);
    }
}

pub fn run_scene_init(
    scene_id: In<usize>,
    mut commands: Commands,
    mut scenes: ResMut<SceneInits>,
    device: Res<RenderDevice>,
    mut app_state: ResMut<AppState>,
    mut callbacks: ResMut<Callbacks>,
) {
    commands.insert_resource(scenes.scenes[scene_id.0].1(
        device.clone(),
        &mut app_state,
        &mut callbacks,
    ));
}

impl FromWorld for SceneInits {
    fn from_world(world: &mut World) -> Self {
        Self {
            scenes: vec![],
            reset_graphics: world.register_system(startup::setup_graphics),
            reset_app_state: world.register_system(startup::setup_app_state),
        }
    }
}

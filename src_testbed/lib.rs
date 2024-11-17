#[cfg(feature = "dim2")]
pub extern crate wgsparkl2d as wgsparkl;
#[cfg(feature = "dim3")]
pub extern crate wgsparkl3d as wgsparkl;

#[cfg(feature = "dim2")]
pub use instancing2d as instancing;
#[cfg(feature = "dim3")]
pub use instancing3d as instancing;

#[cfg(feature = "dim2")]
pub mod instancing2d;
#[cfg(feature = "dim3")]
pub mod instancing3d;

mod hot_reload;
pub mod prep_vertex_buffer;
pub mod startup;
pub mod step;
pub mod ui;

use bevy::asset::load_internal_asset;
use bevy::ecs::system::SystemId;
use bevy::prelude::*;
use bevy_editor_cam::prelude::DefaultEditorCamPlugins;
use bevy_wasm_window_resize::WindowResizePlugin;
use instancing::INSTANCING_SHADER_HANDLE;
use prep_vertex_buffer::{GpuRenderConfig, RenderConfig, WgPrepVertexBuffer};
use wgcore::hot_reloading::HotReloadState;
use wgcore::timestamps::GpuTimestamps;
use wgsparkl::{
    pipeline::{MpmData, MpmPipeline},
    solver::Particle,
};

pub fn init_testbed(app: &mut App) {
    app.add_plugins(DefaultPlugins)
        .add_plugins(WindowResizePlugin)
        .add_plugins((
            bevy_mod_picking::DefaultPickingPlugins,
            DefaultEditorCamPlugins,
        ))
        .add_plugins(instancing::ParticlesMaterialPlugin)
        .add_plugins(bevy_egui::EguiPlugin)
        .init_resource::<SceneInits>()
        .add_systems(Startup, startup::setup_app)
        .add_systems(
            Update,
            (
                ui::update_ui,
                step::step_simulation,
                hot_reload::handle_hot_reloading,
            ),
        );

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
}

#[derive(Resource)]
pub struct PhysicsContext {
    pub data: MpmData,
    pub particles: Vec<Particle>,
}

#[derive(Resource, Default)]
pub struct Timestamps {
    pub timestamps: Option<GpuTimestamps>,
    pub grid_sort: f64,
    pub p2g: f64,
    pub grid_update: f64,
    pub g2p: f64,
    pub particles_update: f64,
    pub integrate_bodies: f64,
}

impl Timestamps {
    pub fn total_time(&self) -> f64 {
        self.grid_sort
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
    pub scenes: Vec<(String, SystemId)>,
    reset_graphics: SystemId,
}

impl SceneInits {
    pub fn init_scene(&self, commands: &mut Commands, scene_id: usize) {
        commands.run_system(self.scenes[scene_id].1);
        commands.run_system(self.reset_graphics);
    }
}

impl FromWorld for SceneInits {
    fn from_world(world: &mut World) -> Self {
        Self {
            scenes: vec![],
            reset_graphics: world.register_system(startup::setup_graphics),
        }
    }
}

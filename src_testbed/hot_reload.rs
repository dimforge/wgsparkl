use crate::AppState;
use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;

pub fn handle_hot_reloading(render_device: Res<RenderDevice>, mut state: ResMut<AppState>) {
    let device = render_device.wgpu_device();
    let state = &mut *state;
    state.hot_reload.update_changes();
    match state.pipeline.reload_if_changed(device, &state.hot_reload) {
        Err(e) => {
            println!("Failed to hot reload the MPM pipeline: {e}");
        }
        Ok(changed) => {
            if changed {
                println!("Hot reloaded MPM pipeline successfully.");
            }
        }
    }
}

// NOTE: this is copied from https://github.com/Leinnan/bevy_wasm_window_resize/blob/master/src/lib.rs
//       that seems unmaintained (MIT license specified in the project’s Cargo.toml).

use bevy::prelude::*;

pub struct WindowResizePlugin;

impl Plugin for WindowResizePlugin {
    #[cfg(target_arch = "wasm32")]
    fn build(&self, app: &mut App) {
        app.add_systems(Update, handle_browser_resize);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn build(&self, _app: &mut App) {}
}

#[cfg(target_arch = "wasm32")]
fn handle_browser_resize(
    mut primary_query: bevy::ecs::system::Query<
        &mut bevy::window::Window,
        bevy::ecs::query::With<bevy::window::PrimaryWindow>,
    >,
) {
    let Some(wasm_window) = wgpu::web_sys::window() else {
        return;
    };
    let Ok(inner_width) = wasm_window.inner_width() else {
        return;
    };
    let Ok(inner_height) = wasm_window.inner_height() else {
        return;
    };
    let Some(target_width) = inner_width.as_f64() else {
        return;
    };
    let Some(target_height) = inner_height.as_f64() else {
        return;
    };
    for mut window in &mut primary_query {
        if window.resolution.width() != (target_width as f32)
            || window.resolution.height() != (target_height as f32)
        {
            window
                .resolution
                .set(target_width as f32, target_height as f32);
        }
    }
}

#![allow(clippy::too_many_arguments)]
#![allow(clippy::module_inception)]

#[cfg(feature = "dim2")]
pub extern crate rapier2d as rapier;
#[cfg(feature = "dim3")]
pub extern crate rapier3d as rapier;
#[cfg(feature = "dim2")]
pub extern crate wgparry2d as wgparry;
#[cfg(feature = "dim3")]
pub extern crate wgparry3d as wgparry;
#[cfg(feature = "dim2")]
pub extern crate wgrapier2d as wgrapier;
#[cfg(feature = "dim3")]
pub extern crate wgrapier3d as wgrapier;

mod collision;
pub mod grid;
#[cfg(feature = "dim3")]
pub mod load_mesh3d;
pub mod models;
pub mod pipeline;
pub mod solver;

pub(crate) fn dim_shader_defs() -> HashMap<String, ShaderDefValue> {
    let mut result = wgparry::dim_shader_defs();
    result.insert(
        "MACOS".to_string(),
        ShaderDefValue::Int(if cfg!(target_os = "macos") { 1 } else { 0 }),
    );
    result
}

use naga_oil::compose::ShaderDefValue;
use std::collections::HashMap;
pub(crate) use wgparry::substitute_aliases;

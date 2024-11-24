#[cfg(feature = "dim2")]
extern crate wgparry2d as wgparry;
#[cfg(feature = "dim3")]
extern crate wgparry3d as wgparry;
#[cfg(feature = "dim2")]
extern crate wgrapier2d as wgrapier;
#[cfg(feature = "dim3")]
extern crate wgrapier3d as wgrapier;

mod collision;
pub mod grid;
pub mod models;
pub mod pipeline;
pub mod solver;

pub(crate) fn dim_shader_defs() -> HashMap<String, ShaderDefValue> {
    let mut result = wgparry::dim_shader_defs();
    result.insert("MACOS".to_string(), ShaderDefValue::Int(if cfg!(target_os = "macos") { 1 } else { 0 }));
    result
}

use std::collections::HashMap;
use naga_oil::compose::ShaderDefValue;
pub(crate) use wgparry::{substitute_aliases};

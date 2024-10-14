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

pub(crate) use wgparry::{dim_shader_defs, substitute_aliases};

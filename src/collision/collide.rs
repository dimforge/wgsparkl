use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;
use wgparry::cuboid::WgCuboid;
use wgrapier::dynamics::WgBody;

#[derive(Shader)]
#[shader(
    derive(WgCuboid, WgBody),
    src = "collide.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
pub struct WgCollide;

wgcore::test_shader_compilation!(WgCollide);

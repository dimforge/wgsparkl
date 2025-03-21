use crate::grid::grid::WgGrid;
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;
use wgparry::shape::WgShape;
use wgrapier::dynamics::WgBody;

#[derive(Shader)]
#[shader(
    derive(WgShape, WgBody, WgGrid),
    src = "collide.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
pub struct WgCollide;

wgcore::test_shader_compilation!(WgCollide, wgcore, crate::dim_shader_defs());

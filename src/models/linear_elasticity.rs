use crate::dim_shader_defs;
use wgcore::Shader;
use wgebra::{WgSvd2, WgSvd3};

#[derive(Shader)]
#[shader(
    derive(WgSvd2, WgSvd3),
    src = "linear_elasticity.wgsl",
    shader_defs = "dim_shader_defs"
)]
pub struct WgLinearElasticity;

wgcore::test_shader_compilation!(WgLinearElasticity, wgcore, crate::dim_shader_defs());

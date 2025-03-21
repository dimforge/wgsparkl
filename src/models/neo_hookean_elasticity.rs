use crate::dim_shader_defs;
use wgcore::Shader;
use wgebra::{WgSvd2, WgSvd3};

#[derive(Shader)]
#[shader(
    derive(WgSvd2, WgSvd3),
    src = "neo_hookean_elasticity.wgsl",
    shader_defs = "dim_shader_defs"
)]
pub struct WgNeoHookeanElasticity;

wgcore::test_shader_compilation!(WgNeoHookeanElasticity, wgcore, crate::dim_shader_defs());

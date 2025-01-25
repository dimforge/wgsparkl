use crate::dim_shader_defs;
use wgcore::Shader;

#[derive(Shader)]
#[shader(src = "kernel.wgsl", shader_defs = "dim_shader_defs")]
pub struct WgKernel;

wgcore::test_shader_compilation!(WgKernel, wgcore, crate::dim_shader_defs());

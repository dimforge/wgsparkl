use crate::dim_shader_defs;
use wgcore::tensor::GpuScalar;
use wgcore::Shader;
use wgpu::{BufferUsages, Device};

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct SimulationParams {
    #[cfg(feature = "dim2")]
    pub gravity: nalgebra::Vector2<f32>,
    #[cfg(feature = "dim2")]
    pub padding: f32,
    #[cfg(feature = "dim3")]
    pub gravity: nalgebra::Vector3<f32>,
    pub dt: f32,
}

pub struct GpuSimulationParams {
    pub params: GpuScalar<SimulationParams>,
}

impl GpuSimulationParams {
    pub fn new(device: &Device, params: SimulationParams) -> Self {
        Self {
            params: GpuScalar::init(
                device,
                params,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
        }
    }
}

#[derive(Shader)]
#[shader(src = "params.wgsl", shader_defs = "dim_shader_defs")]
pub struct WgParams;

wgcore::test_shader_compilation!(WgParams, wgcore, crate::dim_shader_defs());

use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::GpuScalar;
use wgcore::Shader;
use wgebra::WgSvd2;
use wgebra::WgSvd3;
use wgpu::{Buffer, BufferUsages, ComputePipeline, Device};
use wgsparkl::grid::grid::{GpuGrid, WgGrid};
use wgsparkl::solver::{GpuParticles, GpuSimulationParams};
use wgsparkl::solver::{GpuRigidParticles, WgParticle};

pub enum RenderMode {
    Default = 0,
    Volume = 1,
    Velocity = 2,
    CdfNormals = 3,
    CdfDistances = 4,
    CdfSigns = 5,
}

impl RenderMode {
    pub fn text(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Volume => "volume",
            Self::Velocity => "velocity",
            Self::CdfNormals => "cdf (normals)",
            Self::CdfDistances => "cdf (distances)",
            Self::CdfSigns => "cdf (signs)",
        }
    }

    pub fn from_u32(val: u32) -> Self {
        match val {
            0 => Self::Default,
            1 => Self::Volume,
            2 => Self::Velocity,
            3 => Self::CdfNormals,
            4 => Self::CdfDistances,
            5 => Self::CdfSigns,
            _ => unreachable!(),
        }
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq, Debug, Default)]
#[repr(C)]
pub struct RenderConfig {
    pub mode: u32,
}

impl RenderConfig {
    pub fn new(mode: RenderMode) -> Self {
        Self { mode: mode as u32 }
    }
}

pub struct GpuRenderConfig {
    pub buffer: GpuScalar<RenderConfig>,
}

impl GpuRenderConfig {
    pub fn new(device: &Device, config: RenderConfig) -> Self {
        Self {
            buffer: GpuScalar::init(
                device,
                config,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
        }
    }
}

#[derive(Shader)]
#[shader(derive(WgParticle, WgGrid, WgSvd2, WgSvd3), composable = false)]
#[cfg_attr(feature = "dim2", shader(src = "prep_vertex_buffer2d.wgsl"))]
#[cfg_attr(feature = "dim3", shader(src = "prep_vertex_buffer3d.wgsl"))]
pub struct WgPrepVertexBuffer {
    main: ComputePipeline,
    main_rigid_particles: ComputePipeline,
}

impl WgPrepVertexBuffer {
    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        config: &GpuRenderConfig,
        particles: &GpuParticles,
        rigid_particles: &GpuRigidParticles,
        grid: &GpuGrid,
        params: &GpuSimulationParams,
        vertex_buffer: &Buffer,
        vertex_buffer_rigid_particles: Option<&Buffer>,
    ) {
        KernelInvocationBuilder::new(queue, &self.main)
            .bind0([
                vertex_buffer,
                particles.positions.buffer(),
                particles.volumes.buffer(),
                particles.velocities.buffer(),
                particles.cdf.buffer(),
                grid.meta.buffer(),
                params.params.buffer(),
                config.buffer.buffer(),
            ])
            .queue(particles.positions.len().div_ceil(64) as u32);

        if let Some(vertex_buffer) = vertex_buffer_rigid_particles {
            KernelInvocationBuilder::new(queue, &self.main_rigid_particles)
                .bind0([vertex_buffer, rigid_particles.sample_points.buffer()])
                .queue(rigid_particles.sample_points.len().div_ceil(64) as u32);
        }
    }
}

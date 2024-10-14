use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor};
use wgcore::composer::ComposerExt;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::shader::Shader;
use wgcore::tensor::GpuScalar;
use wgcore::utils;
use wgebra::WgSvd2;
use wgebra::WgSvd3;
use wgpu::{Buffer, BufferUsages, ComputePipeline, Device};
use wgsparkl::grid::grid::{GpuGrid, WgGrid};
use wgsparkl::solver::WgParticle;
use wgsparkl::solver::{GpuParticles, GpuSimulationParams};

pub enum RenderMode {
    Default = 0,
    Volume = 1,
    Velocity = 2,
}

impl RenderMode {
    pub fn text(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Volume => "volume",
            Self::Velocity => "velocity",
        }
    }

    pub fn from_u32(val: u32) -> Self {
        match val {
            0 => Self::Default,
            1 => Self::Volume,
            2 => Self::Velocity,
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

pub struct WgPrepVertexBuffer(ComputePipeline);

impl WgPrepVertexBuffer {
    #[cfg(feature = "dim2")]
    pub const SRC: &'static str = include_str!("./prep_vertex_buffer2d.wgsl");
    #[cfg(feature = "dim2")]
    pub const FILE_PATH: &'static str = "prep_vertex_buffer2d.wgsl";

    #[cfg(feature = "dim3")]
    pub const SRC: &'static str = include_str!("./prep_vertex_buffer3d.wgsl");
    #[cfg(feature = "dim3")]
    pub const FILE_PATH: &'static str = "prep_vertex_buffer3d.wgsl";

    pub fn new(device: &Device) -> Self {
        let module = Self::composer()
            .make_naga_module(NagaModuleDescriptor {
                source: Self::SRC,
                file_path: Self::FILE_PATH,
                ..Default::default()
            })
            .unwrap();
        let g2p = utils::load_module(device, "main", module.clone());
        Self(g2p)
    }

    pub fn compose(composer: &mut Composer) -> &mut Composer {
        WgParticle::compose(composer);
        WgGrid::compose(composer);
        WgSvd2::compose(composer);
        WgSvd3::compose(composer);
        composer
            .add_composable_module_once(ComposableModuleDescriptor {
                source: Self::SRC,
                file_path: Self::FILE_PATH,
                ..Default::default()
            })
            .unwrap();
        composer
    }

    pub fn composer() -> Composer {
        let mut composer = Composer::default();
        Self::compose(&mut composer);
        composer
    }

    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        config: &GpuRenderConfig,
        particles: &GpuParticles,
        grid: &GpuGrid,
        params: &GpuSimulationParams,
        vertex_buffer: &Buffer,
    ) {
        KernelInvocationBuilder::new(queue, &self.0)
            .bind0([
                vertex_buffer,
                particles.positions.buffer(),
                particles.volumes.buffer(),
                particles.velocities.buffer(),
                grid.meta.buffer(),
                params.params.buffer(),
                config.buffer.buffer(),
            ])
            .queue(particles.positions.len().div_ceil(64) as u32);
    }
}

wgcore::test_shader_compilation!(WgPrepVertexBuffer);

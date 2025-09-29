use crate::grid::grid::{GpuGrid, WgGrid};
use crate::grid::kernel::WgKernel;
use crate::solver::params::{GpuSimulationParams, WgParams};
use crate::solver::WgParticle;
use crate::{dim_shader_defs, substitute_aliases};
use nalgebra::Vector3;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::GpuVector;
use wgcore::Shader;
use wgpu::{BufferUsages, ComputePipeline, Device};
use wgrapier::dynamics::{GpuBodySet, WgBody};

pub struct GpuGhostParticles {
    pub positions: GpuVector<Vector3<f32>>,
}
impl GpuGhostParticles {
    pub fn empty(device: &Device) -> Self {
        Self {
            positions: GpuVector::uninit(device, 0, BufferUsages::STORAGE),
        }
    }
    pub fn from_particles(device: &Device, particles: &[Vector3<f32>]) -> Self {
        Self {
            positions: GpuVector::encase(
                device,
                particles,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
        }
    }
}

#[derive(Shader)]
#[shader(
    derive(WgParams, WgParticle, WgGrid, WgKernel, WgBody),
    src = "g2p_ghost.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
pub struct WgG2PGhost {
    pub g2p_ghost: ComputePipeline,
}

impl WgG2PGhost {
    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        sim_params: &GpuSimulationParams,
        grid: &GpuGrid,
        ghost_particles: &GpuGhostParticles,
        _bodies: &GpuBodySet,
    ) {
        KernelInvocationBuilder::new(queue, &self.g2p_ghost)
            .bind_at(
                0,
                [
                    (grid.meta.buffer(), 0),
                    (grid.hmap_entries.buffer(), 1),
                    // (grid.active_blocks.buffer(), 2),
                    (grid.nodes.buffer(), 3),
                ],
            )
            .bind(
                1,
                [
                    ghost_particles.positions.buffer(),
                    sim_params.params.buffer(),
                    // particles.positions.buffer(),
                ],
            )
            // .bind(2, [bodies.vels().buffer(), bodies.mprops().buffer()])
            .queue([ghost_particles.positions.len().div_ceil(64) as u32, 1, 1]);
    }
}

wgcore::test_shader_compilation!(WgG2PGhost, wgcore, crate::dim_shader_defs());

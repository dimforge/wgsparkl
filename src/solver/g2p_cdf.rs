use crate::grid::grid::{GpuGrid, WgGrid};
use crate::grid::kernel::WgKernel;
use crate::solver::params::{GpuSimulationParams, WgParams};
use crate::solver::GpuParticles;
use crate::solver::WgParticle;
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::Shader;
use wgebra::WgInv;
use wgpu::ComputePipeline;

#[derive(Shader)]
#[shader(
    derive(WgParams, WgParticle, WgGrid, WgKernel, WgInv),
    src = "g2p_cdf.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
pub struct WgG2PCdf {
    pub g2p_cdf: ComputePipeline,
}

impl WgG2PCdf {
    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        sim_params: &GpuSimulationParams,
        grid: &GpuGrid,
        particles: &GpuParticles,
    ) {
        KernelInvocationBuilder::new(queue, &self.g2p_cdf)
            .bind_at(
                0,
                [
                    (grid.meta.buffer(), 0),
                    (grid.hmap_entries.buffer(), 1),
                    (grid.active_blocks.buffer(), 2),
                    (grid.nodes.buffer(), 3),
                ],
            )
            .bind(
                1,
                [
                    particles.positions.buffer(),
                    particles.dynamics.buffer(),
                    particles.sorted_ids.buffer(),
                    sim_params.params.buffer(),
                ],
            )
            .queue_indirect(grid.indirect_n_g2p_p2g_groups.clone());
    }
}

wgcore::test_shader_compilation!(WgG2PCdf, wgcore, crate::dim_shader_defs());

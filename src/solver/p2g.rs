use crate::dim_shader_defs;
use crate::grid::grid::{GpuGrid, WgGrid};
use crate::grid::kernel::WgKernel;
use crate::models::WgLinearElasticity;
use crate::solver::params::WgParams;
use crate::solver::GpuParticles;
use crate::solver::WgParticle;
use naga_oil::compose::NagaModuleDescriptor;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::{utils, Shader};
use wgpu::ComputePipeline;

#[derive(Shader)]
#[shader(
    derive(WgParams, WgParticle, WgGrid, WgLinearElasticity, WgKernel),
    src = "p2g.wgsl",
    shader_defs = "dim_shader_defs"
)]
pub struct WgP2G {
    pub p2g: ComputePipeline,
}

impl WgP2G {
    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        grid: &GpuGrid,
        particles: &GpuParticles,
    ) {
        KernelInvocationBuilder::new(queue, &self.p2g)
            .bind(
                0,
                [
                    grid.meta.buffer(),
                    grid.hmap_entries.buffer(),
                    grid.active_blocks.buffer(),
                    grid.nodes.buffer(),
                ],
            )
            .bind(
                1,
                [
                    particles.positions.buffer(),
                    particles.velocities.buffer(),
                    particles.volumes.buffer(),
                    particles.affines.buffer(),
                    grid.nodes_linked_lists.buffer(),
                    particles.node_linked_lists.buffer(),
                ],
            )
            .queue_indirect(grid.indirect_n_g2p_p2g_groups.clone());
    }
}

wgcore::test_shader_compilation!(WgP2G);

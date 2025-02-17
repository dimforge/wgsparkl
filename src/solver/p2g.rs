use crate::dim_shader_defs;
use crate::grid::grid::{GpuGrid, WgGrid};
use crate::grid::kernel::WgKernel;
use crate::models::WgLinearElasticity;
use crate::solver::params::WgParams;
use crate::solver::{GpuImpulses, GpuParticles};
use crate::solver::{WgParticle, WgRigidImpulses};
use crate::substitute_aliases;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::Shader;
use wgpu::ComputePipeline;
use wgrapier::dynamics::{GpuBodySet, WgBody};

#[derive(Shader)]
#[shader(
    derive(
        WgParams,
        WgParticle,
        WgGrid,
        WgLinearElasticity,
        WgKernel,
        WgBody,
        WgRigidImpulses,
    ),
    src = "p2g.wgsl",
    src_fn = "substitute_aliases",
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
        impulses: &GpuImpulses,
        bodies: &GpuBodySet,
    ) {
        KernelInvocationBuilder::new(queue, &self.p2g)
            .bind_at(
                0,
                [
                    (grid.meta.buffer(), 0),
                    (grid.hmap_entries.buffer(), 1),
                    (grid.active_blocks.buffer(), 2),
                    (grid.nodes.buffer(), 3),
                    (grid.nodes_cdf.buffer(), 9),
                ],
            )
            .bind(
                1,
                [
                    particles.positions.buffer(),
                    particles.dynamics.buffer(),
                    grid.nodes_linked_lists.buffer(),
                    particles.node_linked_lists.buffer(),
                    impulses.incremental_impulses.buffer(),
                ],
            )
            .bind(2, [bodies.vels().buffer(), bodies.mprops().buffer()])
            .queue_indirect(grid.indirect_n_g2p_p2g_groups.clone());
    }
}

wgcore::test_shader_compilation!(WgP2G, wgcore, crate::dim_shader_defs());

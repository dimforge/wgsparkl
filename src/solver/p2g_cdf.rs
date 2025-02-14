use crate::dim_shader_defs;
use crate::grid::grid::{GpuGrid, WgGrid};
use crate::grid::kernel::WgKernel;
use crate::models::WgLinearElasticity;
use crate::solver::params::WgParams;
use crate::solver::{GpuImpulses, GpuParticles, GpuRigidParticles};
use crate::solver::{WgParticle, WgRigidImpulses};
use crate::substitute_aliases;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::Shader;
use wgparry::segment::WgSegment;
use wgparry::triangle::WgTriangle;
use wgpu::ComputePipeline;
use wgrapier::dynamics::{GpuBodySet, WgBody};

#[derive(Shader)]
#[shader(
    derive(WgParams, WgParticle, WgKernel, WgGrid, WgBody, WgSegment, WgTriangle),
    src = "p2g_cdf.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
pub struct WgP2GCdf {
    pub p2g_cdf: ComputePipeline,
}

impl WgP2GCdf {
    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        grid: &GpuGrid,
        particles: &GpuRigidParticles,
        bodies: &GpuBodySet,
    ) {
        if particles.is_empty() {
            return;
        }
        KernelInvocationBuilder::new(queue, &self.p2g_cdf)
            .bind_at(
                0,
                [
                    (grid.meta.buffer(), 0),
                    (grid.hmap_entries.buffer(), 1),
                    (grid.active_blocks.buffer(), 2),
                    (grid.nodes_cdf.buffer(), 9),
                ],
            )
            .bind(
                1,
                [
                    grid.nodes_rigid_linked_lists.buffer(),
                    particles.node_linked_lists.buffer(),
                    particles.sample_ids.buffer(),
                    bodies.shapes_vertex_buffers().buffer(),
                ],
            )
            .queue_indirect(grid.indirect_n_g2p_p2g_groups.clone());
    }
}

wgcore::test_shader_compilation!(WgP2GCdf, wgcore, crate::dim_shader_defs());

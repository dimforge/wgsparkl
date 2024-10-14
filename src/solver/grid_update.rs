use crate::collision::WgCollide;
use crate::dim_shader_defs;
use crate::grid::grid::{GpuGrid, WgGrid};
use crate::solver::params::GpuSimulationParams;
use naga_oil::compose::NagaModuleDescriptor;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::{utils, Shader};
use wgpu::ComputePipeline;
use wgrapier::dynamics::GpuBodySet;

#[derive(Shader)]
#[shader(
    derive(WgGrid, WgCollide),
    src = "grid_update.wgsl",
    shader_defs = "dim_shader_defs"
)]
pub struct WgGridUpdate {
    pub grid_update: ComputePipeline,
}

impl WgGridUpdate {
    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        sim_params: &GpuSimulationParams,
        grid: &GpuGrid,
        bodies: &GpuBodySet,
    ) {
        KernelInvocationBuilder::new(queue, &self.grid_update)
            .bind_at(
                0,
                [
                    (grid.meta.buffer(), 0),
                    (grid.active_blocks.buffer(), 2),
                    (grid.nodes.buffer(), 3),
                    (sim_params.params.buffer(), 4),
                ],
            )
            .bind(1, [])
            .bind(
                2,
                [
                    bodies.shapes().buffer(),
                    bodies.poses().buffer(),
                    // bodies.vels().buffer(),
                    // bodies.mprops().buffer(),
                ],
            )
            .queue_indirect(grid.indirect_n_g2p_p2g_groups.clone());
    }
}

wgcore::test_shader_compilation!(WgGridUpdate);

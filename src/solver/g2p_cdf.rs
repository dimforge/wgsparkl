use crate::grid::grid::{GpuGrid, WgGrid};
use crate::grid::kernel::WgKernel;
use crate::solver::params::{GpuSimulationParams, WgParams};
use crate::solver::GpuParticles;
use crate::solver::WgParticle;
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::kernel::KernelDispatch;
use wgcore::Shader;
use wgebra::WgInv;
use wgpu::{ComputePass, ComputePipeline, Device};

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
    pub fn dispatch<'a>(
        &'a self,
        device: &Device,
        pass: &mut ComputePass,
        sim_params: &GpuSimulationParams,
        grid: &GpuGrid,
        particles: &GpuParticles,
    ) {
        KernelDispatch::new(device, pass, &self.g2p_cdf)
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
            .dispatch_indirect(&grid.indirect_n_g2p_p2g_groups);
    }
}

wgcore::test_shader_compilation!(WgG2PCdf, wgcore, crate::dim_shader_defs());

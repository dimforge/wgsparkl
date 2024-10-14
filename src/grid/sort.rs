use crate::dim_shader_defs;
use crate::grid::grid::WgGrid;
use crate::solver::WgParticle;
use naga_oil::compose::NagaModuleDescriptor;
use wgcore::{utils, Shader};
use wgpu::ComputePipeline;

#[derive(Shader)]
#[shader(
    derive(WgParticle, WgGrid),
    src = "sort.wgsl",
    shader_defs = "dim_shader_defs"
)]
pub struct WgSort {
    pub(crate) touch_particle_blocks: ComputePipeline,
    pub(crate) update_block_particle_count: ComputePipeline,
    pub(crate) copy_particles_len_to_scan_value: ComputePipeline,
    pub(crate) copy_scan_values_to_first_particles: ComputePipeline,
    pub(crate) finalize_particles_sort: ComputePipeline,
}

wgcore::test_shader_compilation!(WgSort);

use crate::dim_shader_defs;
use crate::grid::grid::WgGrid;
use crate::solver::WgParticle;
use wgcore::Shader;
use wgpu::ComputePipeline;

#[derive(Shader)]
#[shader(
    derive(WgParticle, WgGrid),
    src = "sort.wgsl",
    shader_defs = "dim_shader_defs"
)]
pub struct WgSort {
    #[cfg(not(target_os = "macos"))]
    pub(crate) touch_particle_blocks: ComputePipeline,
    pub(crate) update_block_particle_count: ComputePipeline,
    pub(crate) copy_particles_len_to_scan_value: ComputePipeline,
    pub(crate) copy_scan_values_to_first_particles: ComputePipeline,
    pub(crate) finalize_particles_sort: ComputePipeline,
}

#[cfg(target_os = "macos")]
pub struct TouchParticleBlocks {
    pub(crate) touch_particle_blocks: ComputePipeline,
}

#[cfg(target_os = "macos")]
impl TouchParticleBlocks {
    pub fn from_device(device: &wgpu::Device) -> Self {
        #[cfg(feature = "dim2")]
        let src = wgpu::include_wgsl!("touch_particle_blocks2d.wgsl");
        #[cfg(feature = "dim3")]
        let src = wgpu::include_wgsl!("touch_particle_blocks3d.wgsl");
        let cs_module = device.create_shader_module(src);
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module,
            entry_point: Some("touch_particle_blocks"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            touch_particle_blocks: compute_pipeline,
        }
    }
}

wgcore::test_shader_compilation!(WgSort, wgcore, crate::dim_shader_defs());

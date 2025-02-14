use crate::dim_shader_defs;
use crate::grid::grid::{GpuGrid, WgGrid};
use crate::solver::{GpuRigidParticles, WgParticle};
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
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
    #[cfg(not(target_os = "macos"))]
    pub(crate) touch_rigid_particle_blocks: ComputePipeline,
    pub(crate) mark_rigid_particles_needing_block: ComputePipeline,
    pub(crate) update_block_particle_count: ComputePipeline,
    pub(crate) copy_particles_len_to_scan_value: ComputePipeline,
    pub(crate) copy_scan_values_to_first_particles: ComputePipeline,
    pub(crate) finalize_particles_sort: ComputePipeline,
    pub(crate) sort_rigid_particles: ComputePipeline,
}

impl WgSort {
    pub fn queue_sort_rigid_particles<'a>(
        &'a self,
        particles: &GpuRigidParticles,
        grid: &GpuGrid,
        queue: &mut KernelInvocationQueue<'a>,
    ) {
        if particles.is_empty() {
            // No work needed.
            return;
        }

        const GRID_WORKGROUP_SIZE: u32 = 64;
        let n_groups = (particles.len() as u32).div_ceil(GRID_WORKGROUP_SIZE);
        KernelInvocationBuilder::new(queue, &self.sort_rigid_particles)
            .bind_at(
                0,
                [
                    (grid.meta.buffer(), 0),
                    (grid.hmap_entries.buffer(), 1),
                    (grid.nodes_rigid_linked_lists.buffer(), 10),
                ],
            )
            .bind_at(
                1,
                [
                    (particles.sample_points.buffer(), 4),
                    (particles.node_linked_lists.buffer(), 5),
                ],
            )
            .queue(n_groups);
    }
}

#[cfg(target_os = "macos")]
pub struct TouchParticleBlocks {
    pub(crate) touch_particle_blocks: ComputePipeline,
    pub(crate) touch_rigid_particle_blocks: ComputePipeline,
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
        let rigid_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &cs_module,
                entry_point: Some("touch_rigid_particle_blocks"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self {
            touch_particle_blocks: compute_pipeline,
            touch_rigid_particle_blocks: rigid_compute_pipeline,
        }
    }
}

wgcore::test_shader_compilation!(WgSort, wgcore, crate::dim_shader_defs());

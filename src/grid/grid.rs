use crate::grid::prefix_sum::{PrefixSumWorkspace, WgPrefixSum};
use crate::grid::sort::WgSort;
use crate::solver::{GpuParticles, GpuRigidParticles, WgParams};
use crate::{dim_shader_defs, substitute_aliases};
use std::sync::Arc;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::Shader;
use wgpu::util::DispatchIndirectArgs;
use wgpu::{Buffer, BufferAddress, BufferDescriptor, BufferUsages, ComputePipeline, Device};

#[cfg(target_os = "macos")]
use crate::grid::sort::TouchParticleBlocks;

#[derive(Shader)]
#[shader(
    derive(WgParams),
    src = "grid.wgsl",
    shader_defs = "dim_shader_defs",
    src_fn = "substitute_aliases"
)]
pub struct WgGrid {
    reset_hmap: ComputePipeline,
    reset: ComputePipeline,
    init_indirect_workgroups: ComputePipeline,
}

impl WgGrid {
    // Returns the pair (number of active blocks, number of GPU dispatch blocks needed to cover all the particles).
    pub fn queue_sort<'a>(
        &'a self,
        particles: &GpuParticles,
        rigid_particles: &GpuRigidParticles,
        grid: &GpuGrid,
        prefix_sum: &mut PrefixSumWorkspace,
        sort_module: &'a WgSort,
        #[cfg(target_os = "macos")] touch_particle_blocks: &'a TouchParticleBlocks,
        prefix_sum_module: &'a WgPrefixSum,
        queue: &mut KernelInvocationQueue<'a>,
    ) {
        const GRID_WORKGROUP_SIZE: u32 = 64;

        // Retry until we allocated enough room on the sparse grid for all the blocks.
        let mut sparse_grid_has_the_correct_size = false;
        while !sparse_grid_has_the_correct_size {
            // - Reset next grid’s hashmap.
            // - Reset grid.num_active_blocks to 0.
            // - Run touch_particle_blocks on the next grid.
            // - Readback num_active_blocks.
            // - Update the hashmap & grid buffer sizes if its occupancy is to high.

            // NOTE: num_active_blocks := 0 is set in reset_hmap.
            KernelInvocationBuilder::new(queue, &self.reset_hmap)
                .bind0([grid.meta.buffer(), grid.hmap_entries.buffer()])
                .queue(grid.cpu_meta.hmap_capacity.div_ceil(GRID_WORKGROUP_SIZE));

            #[cfg(not(target_os = "macos"))]
            let touch_particle_blocks_pipeline = &sort_module.touch_particle_blocks;
            #[cfg(target_os = "macos")]
            let touch_particle_blocks_pipeline = &touch_particle_blocks.touch_particle_blocks;

            KernelInvocationBuilder::new(queue, touch_particle_blocks_pipeline)
                .bind_at(
                    0,
                    [
                        (grid.meta.buffer(), 0),
                        (grid.hmap_entries.buffer(), 1),
                        (grid.active_blocks.buffer(), 2),
                        (grid.debug.buffer(), 8),
                    ],
                )
                .bind(1, [particles.positions.buffer()])
                .queue((particles.len() as u32).div_ceil(GRID_WORKGROUP_SIZE));

            // Ensure blocks exist wherever we have rigid particles that might affect
            // other blocks. This is done in two passes:
            // 1. Mark all rigid particles that need to ensure it’s associated block exists
            // 2. Touch the blocks with marked rigid particles.
            KernelInvocationBuilder::new(queue, &sort_module.mark_rigid_particles_needing_block)
                .bind_at(
                    0,
                    [(grid.meta.buffer(), 0), (grid.hmap_entries.buffer(), 1)],
                )
                .bind_at(
                    1,
                    [
                        (rigid_particles.sample_points.buffer(), 0),
                        (rigid_particles.rigid_particle_needs_block.buffer(), 6),
                    ],
                )
                .queue((rigid_particles.len() as u32).div_ceil(GRID_WORKGROUP_SIZE));

            #[cfg(not(target_os = "macos"))]
            let touch_rigid_particle_blocks = &sort_module.touch_rigid_particle_blocks;
            #[cfg(target_os = "macos")]
            let touch_rigid_particle_blocks = &touch_particle_blocks.touch_rigid_particle_blocks;
            KernelInvocationBuilder::new(queue, touch_rigid_particle_blocks)
                .bind_at(
                    0,
                    [
                        (grid.meta.buffer(), 0),
                        (grid.hmap_entries.buffer(), 1),
                        (grid.active_blocks.buffer(), 2),
                        (grid.debug.buffer(), 8),
                    ],
                )
                .bind_at(
                    1,
                    [
                        (rigid_particles.sample_points.buffer(), 0),
                        (rigid_particles.rigid_particle_needs_block.buffer(), 6),
                    ],
                )
                .queue((rigid_particles.len() as u32).div_ceil(GRID_WORKGROUP_SIZE));

            // TODO: handle grid buffer resizing
            sparse_grid_has_the_correct_size = true;
        }

        // - Launch update_block_particle_count
        // - Launch copy_particle_len_to_scan_value
        // - Launch cumulated sum.
        // - Launch copy_scan_values_to_first_particles
        // - Launch finalize_particles_sort
        // - Launch write_blocks_multiplicity_to_scan_value
        // - Launch cumulated sum

        // Prepare workgroups for indirect dispatches based on the number of active blocks.
        KernelInvocationBuilder::new(queue, &self.init_indirect_workgroups)
            .bind_at(
                0,
                [
                    (grid.meta.buffer(), 0),
                    (&*grid.indirect_n_blocks_groups, 5),
                    (&*grid.indirect_n_g2p_p2g_groups, 7),
                ],
            )
            .queue(1);

        let n_groups = (particles.len() as u32).div_ceil(GRID_WORKGROUP_SIZE);
        let n_block_groups = grid.indirect_n_blocks_groups.clone();

        KernelInvocationBuilder::new(queue, &sort_module.update_block_particle_count)
            .bind(
                0,
                [
                    grid.meta.buffer(),
                    grid.hmap_entries.buffer(),
                    grid.active_blocks.buffer(),
                ],
            )
            .bind(1, [particles.positions.buffer()])
            .queue(n_groups);

        KernelInvocationBuilder::new(queue, &sort_module.copy_particles_len_to_scan_value)
            .bind_at(
                0,
                [(grid.meta.buffer(), 0), (grid.active_blocks.buffer(), 2)],
            )
            .bind_at(1, [(grid.scan_values.buffer(), 1)])
            .queue_indirect(n_block_groups.clone());

        prefix_sum_module.queue(queue, prefix_sum, &grid.scan_values);

        KernelInvocationBuilder::new(queue, &sort_module.copy_scan_values_to_first_particles)
            .bind_at(
                0,
                [(grid.meta.buffer(), 0), (grid.active_blocks.buffer(), 2)],
            )
            .bind_at(1, [(grid.scan_values.buffer(), 1)])
            .queue_indirect(n_block_groups.clone());

        // Reset here so the linked list heads get reset before `finalize_particles_sort` which
        // also setups the per-node linked list.
        KernelInvocationBuilder::new(queue, &self.reset)
            .bind_at(
                0,
                [
                    (grid.meta.buffer(), 0),
                    (grid.nodes.buffer(), 3),
                    (grid.nodes_linked_lists.buffer(), 6),
                    (grid.nodes_rigid_linked_lists.buffer(), 10),
                ],
            )
            .queue_indirect(n_block_groups.clone());

        KernelInvocationBuilder::new(queue, &sort_module.finalize_particles_sort)
            .bind_at(
                0,
                [
                    (grid.meta.buffer(), 0),
                    (grid.hmap_entries.buffer(), 1),
                    (grid.nodes_linked_lists.buffer(), 6),
                ],
            )
            .bind(
                1,
                [
                    particles.positions.buffer(),
                    grid.scan_values.buffer(),
                    particles.sorted_ids.buffer(),
                    particles.node_linked_lists.buffer(),
                ],
            )
            .queue(n_groups);
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq)]
#[repr(C)]
pub struct GpuGridMetadata {
    num_active_blocks: u32,
    cell_width: f32,
    hmap_capacity: u32,
    capacity: u32,
}

#[derive(Copy, Clone, PartialEq, encase::ShaderType)]
#[repr(C)]
pub struct GpuGridNode {
    momentum_velocity_mass: nalgebra::Vector4<f32>,
    cdf: GpuGridNodeCdf,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq)]
#[repr(C)]
pub struct BlockVirtualId {
    #[cfg(feature = "dim2")]
    id: nalgebra::Vector2<i32>,
    #[cfg(feature = "dim3")]
    id: nalgebra::Vector4<i32>, // Vector3 with padding.
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq)]
#[repr(C)]
pub struct GpuGridHashMapEntry {
    state: u32,
    #[cfg(feature = "dim2")]
    pad0: u32,
    #[cfg(feature = "dim3")]
    pad0: nalgebra::Vector3<u32>,
    key: BlockVirtualId,
    value: u32,
    #[cfg(feature = "dim2")]
    pad1: u32,
    #[cfg(feature = "dim3")]
    pad1: nalgebra::Vector3<u32>,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq)]
#[repr(C)]
pub struct GpuActiveBlockHeader {
    virtual_id: BlockVirtualId,
    first_particle: u32,
    num_particles: u32,
}

#[derive(Copy, Clone, PartialEq, Default, Debug, encase::ShaderType)]
#[repr(C)]
pub struct GpuGridNodeCdf {
    pub distance: f32,
    pub affinities: u32,
    pub closest_id: u32,
}

pub struct GpuGrid {
    pub cpu_meta: GpuGridMetadata,
    pub meta: GpuScalar<GpuGridMetadata>,
    pub hmap_entries: GpuVector<GpuGridHashMapEntry>,
    pub nodes: GpuVector<GpuGridNode>,
    pub active_blocks: GpuVector<GpuActiveBlockHeader>,
    pub scan_values: GpuVector<u32>,
    pub nodes_linked_lists: GpuVector<[u32; 2]>,
    pub nodes_rigid_linked_lists: GpuVector<[u32; 2]>,
    pub indirect_n_blocks_groups: Arc<Buffer>,
    pub indirect_n_g2p_p2g_groups: Arc<Buffer>,
    pub debug: GpuVector<u32>,
}

impl GpuGrid {
    pub fn with_capacity(device: &Device, capacity: u32, cell_width: f32) -> Self {
        const NODES_PER_BLOCK: u32 = 64; // 8 * 8 in 2D and 4 * 4 * 4 in 3D.
        let capacity = capacity.next_power_of_two();
        let cpu_meta = GpuGridMetadata {
            num_active_blocks: 0,
            cell_width,
            hmap_capacity: capacity,
            capacity,
        };
        let meta = GpuScalar::init(
            device,
            cpu_meta,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let hmap_entries = GpuVector::uninit(device, capacity, BufferUsages::STORAGE);
        let nodes =
            GpuVector::uninit_encased(device, capacity * NODES_PER_BLOCK, BufferUsages::STORAGE);
        let nodes_linked_lists =
            GpuVector::uninit(device, capacity * NODES_PER_BLOCK, BufferUsages::STORAGE);
        let nodes_rigid_linked_lists =
            GpuVector::uninit(device, capacity * NODES_PER_BLOCK, BufferUsages::STORAGE);
        let active_blocks = GpuVector::uninit(device, capacity, BufferUsages::STORAGE);
        let scan_values = GpuVector::uninit(device, capacity, BufferUsages::STORAGE);
        let indirect_n_blocks_groups = Arc::new(device.create_buffer(&BufferDescriptor {
            label: None,
            size: std::mem::size_of::<DispatchIndirectArgs>() as BufferAddress,
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        }));
        let indirect_n_g2p_p2g_groups = Arc::new(device.create_buffer(&BufferDescriptor {
            label: None,
            size: std::mem::size_of::<DispatchIndirectArgs>() as BufferAddress,
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        }));
        let debug = GpuVector::init(device, [0, 0], BufferUsages::STORAGE);

        Self {
            cpu_meta,
            meta,
            hmap_entries,
            nodes,
            active_blocks,
            scan_values,
            indirect_n_blocks_groups,
            indirect_n_g2p_p2g_groups,
            nodes_linked_lists,
            nodes_rigid_linked_lists,
            debug,
        }
    }
}

#[cfg(test)]
#[cfg(feature = "dim3")]
mod test {
    use super::{GpuGrid, PrefixSumWorkspace, WgGrid, WgPrefixSum};
    use crate::grid::sort::WgSort;
    use crate::models::ElasticCoefficients;
    use crate::solver::{GpuParticles, GpuRigidParticles, Particle, ParticleDynamics};
    use nalgebra::vector;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::KernelInvocationQueue;
    use wgcore::Shader;
    use wgpu::Maintain;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_grid_sort() {
        let gpu = GpuInstance::new().await.unwrap();
        let prefix_sum_module = WgPrefixSum::from_device(gpu.device()).unwrap();
        let grid_module = WgGrid::from_device(gpu.device()).unwrap();
        let sort_module = WgSort::from_device(gpu.device()).unwrap();

        let cell_width = 1.0;
        let mut cpu_particles = vec![];
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let position = vector![i as f32, j as f32, k as f32] / cell_width / 2.0;
                    cpu_particles.push(Particle {
                        position,
                        dynamics: ParticleDynamics::with_density(cell_width / 4.0, 1.0),
                        model: ElasticCoefficients::from_young_modulus(100_000.0, 0.33),
                        plasticity: None,
                        phase: None,
                        color: None,
                    });
                }
            }
        }

        let particles = GpuParticles::from_particles(gpu.device(), &cpu_particles);
        let grid = GpuGrid::with_capacity(gpu.device(), 100_000, cell_width);
        let mut prefix_sum = PrefixSumWorkspace::with_capacity(gpu.device(), 100_000);
        let mut queue = KernelInvocationQueue::new(gpu.device());
        #[cfg(target_os = "macos")]
        let touch_particle_blocks =
            crate::grid::sort::TouchParticleBlocks::from_device(gpu.device());
        let rigid_particles = GpuRigidParticles::new(gpu.device());

        grid_module.queue_sort(
            &particles,
            &rigid_particles,
            &grid,
            &mut prefix_sum,
            &sort_module,
            #[cfg(target_os = "macos")]
            &touch_particle_blocks,
            &prefix_sum_module,
            &mut queue,
        );

        // NOTE: run multiple times, the first execution is much slower.
        for _ in 0..3 {
            let mut encoder = gpu.device().create_command_encoder(&Default::default());
            queue.encode(&mut encoder, None);
            let t0 = std::time::Instant::now();
            gpu.queue().submit(Some(encoder.finish()));
            gpu.device().poll(Maintain::Wait);
            println!("Grid sort gpu time: {}", t0.elapsed().as_secs_f32());
        }
    }
}

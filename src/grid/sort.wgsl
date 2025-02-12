#define_import_path wgsparkl::grid::sort

#import wgsparkl::grid::grid as Grid;
#import wgsparkl::solver::particle as Particle


@group(1) @binding(0)
var<storage, read_write> particles_pos: array<Particle::Position>;
@group(1) @binding(1)
var<storage, read_write> scan_values: array<atomic<u32>>; // This has to be atomic for finalize_particles_sort. Should it be a different buffer?
@group(1) @binding(2)
var<storage, read_write> sorted_particle_ids: array<u32>;
@group(1) @binding(3)
var<storage, read_write> particle_node_linked_lists: array<u32>;
@group(1) @binding(4)
var<storage, read> rigid_particles_pos: array<Particle::Position>;
@group(1) @binding(5)
var<storage, read_write> rigid_particle_node_linked_lists: array<u32>;

// Disable this kernel on macos because of the underlying compareExchangeMap which is
// not working well with naga-oil. This is why we currently have the flattened
// toouch_particle_block2/3d.wgsl shaders as a workaround currently.
#if MACOS == 0
@compute @workgroup_size(Grid::GRID_WORKGROUP_SIZE, 1, 1)
fn touch_particle_blocks(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let id = invocation_id.x;
    if id < arrayLength(&particles_pos) {
        let particle = particles_pos[id];
        var blocks = Grid::blocks_associated_to_point(particle.pt);
        for (var i = 0u; i < Grid::NUM_ASSOC_BLOCKS; i += 1u) {
            Grid::mark_block_as_active(blocks[i]);
        }
    }
}
#endif

// TODO: can this kernel be combined with touch_particle_blocks?
@compute @workgroup_size(Grid::GRID_WORKGROUP_SIZE, 1, 1)
fn update_block_particle_count(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let id = invocation_id.x;
    if id < arrayLength(&particles_pos) {
        let particle = particles_pos[id];
        let block_id = Grid::block_associated_to_point(particle.pt);
        let active_block_id = Grid::find_block_header_id(block_id);
        let active_block_num_particles = &Grid::active_blocks[active_block_id.id].num_particles;
        atomicAdd(active_block_num_particles, 1u);
    }
}

@compute @workgroup_size(Grid::GRID_WORKGROUP_SIZE, 1, 1)
fn copy_particles_len_to_scan_value(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let id = invocation_id.x;
    if id < Grid::grid.num_active_blocks {
        scan_values[id] = Grid::active_blocks[id].num_particles;
    }
}

@compute @workgroup_size(Grid::GRID_WORKGROUP_SIZE, 1, 1)
fn copy_scan_values_to_first_particles(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let id = invocation_id.x;
    if id < Grid::grid.num_active_blocks {
        Grid::active_blocks[id].first_particle = scan_values[id];
    }
}

@compute @workgroup_size(Grid::GRID_WORKGROUP_SIZE, 1, 1)
fn finalize_particles_sort(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let id = invocation_id.x;
    if id < arrayLength(&particles_pos) {
        let particle = particles_pos[id];
        let block_id = Grid::block_associated_to_point(particle.pt);

        // Place the particle to its sorted place.
        let active_block_id = Grid::find_block_header_id(block_id);
        let target_index = atomicAdd(&scan_values[active_block_id.id], 1u);
        sorted_particle_ids[target_index] = id;

        // Setup the per-node particle linked-list.
        let node_local_id = Particle::associated_cell_index_in_block_off_by_one(particle, Grid::grid.cell_width);
        let node_global_id = Grid::node_id(Grid::block_header_id_to_physical_id(active_block_id), node_local_id);
        let node_linked_list = &Grid::nodes_linked_lists[node_global_id.id];
        let prev_head = atomicExchange(&(*node_linked_list).head, id);
        atomicAdd(&(*node_linked_list).len, 1u);
        particle_node_linked_lists[id] = prev_head;
    }
}

@compute @workgroup_size(Grid::GRID_WORKGROUP_SIZE, 1, 1)
fn sort_rigid_particles(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let id = invocation_id.x;
    if id < arrayLength(&rigid_particles_pos) {
        let particle = rigid_particles_pos[id];
        let block_id = Grid::block_associated_to_point(particle.pt);

        // Place the particle to its sorted place.
        let active_block_id = Grid::find_block_header_id(block_id);

        // NOTE: if the rigid particle doesn’t map to any block, we can just ignore it
        //       is it won’t affect the simulation.
        if active_block_id.id != Grid::NONE {
            // Setup the per-node rigid particle linked-list.
            let node_local_id = Particle::associated_cell_index_in_block_off_by_one(particle, Grid::grid.cell_width);
            let node_global_id = Grid::node_id(Grid::block_header_id_to_physical_id(active_block_id), node_local_id);
            let node_linked_list = &Grid::nodes_rigid_linked_lists[node_global_id.id];
            let prev_head = atomicExchange(&(*node_linked_list).head, id);
            atomicAdd(&(*node_linked_list).len, 1u);
            rigid_particle_node_linked_lists[id] = prev_head;
        }
    }
}
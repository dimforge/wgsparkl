#define_import_path wgsparkl::solver::grid_update

#import wgsparkl::grid::grid as Grid;
#import wgsparkl::collision::collide as Collide;

#if DIM == 2
const WORKGROUP_SIZE_X: u32 = 8;
const WORKGROUP_SIZE_Y: u32 = 8;
const WORKGROUP_SIZE_Z: u32 = 1;
#else
const WORKGROUP_SIZE_X: u32 = 4;
const WORKGROUP_SIZE_Y: u32 = 4;
const WORKGROUP_SIZE_Z: u32 = 4;
#endif

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn grid_update(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(local_invocation_index) tid_flat: u32,
    @builtin(workgroup_id) block_id: vec3<u32>
) {
    let bid = block_id.x;
    let vid = Grid::active_blocks[bid].virtual_id;

    let global_chunk_id = Grid::block_header_id_to_physical_id(Grid::BlockHeaderId(bid));
#if DIM == 2
    let global_node_id = Grid::node_id(global_chunk_id, tid.xy);
    let cell_pos = vec2<f32>(vid.id * 8 + vec2<i32>(tid.xy)) * Grid::grid.cell_width;
#else
    let global_node_id = Grid::node_id(global_chunk_id, tid);
    let cell_pos = vec3<f32>(vid.id * 4 + vec3<i32>(tid)) * Grid::grid.cell_width;
#endif

    // PERF: store the list of blocks with a collision so we can skip the ones without
    //       collisions in teh other `_cdf` kernels.
    //       Or maybe just use some sort of flag and skip the block at the start of the kernel?
    let global_id = global_node_id.id;
    Grid::nodes[global_id].cdf = Collide::collide(Grid::grid.cell_width, cell_pos);
}

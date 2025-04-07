#define_import_path wgsparkl::solver::grid_update

#import wgsparkl::grid::grid as Grid;

#if DIM == 2
const WORKGROUP_SIZE_X: u32 = 8;
const WORKGROUP_SIZE_Y: u32 = 8;
const WORKGROUP_SIZE_Z: u32 = 1;
#else
const WORKGROUP_SIZE_X: u32 = 4;
const WORKGROUP_SIZE_Y: u32 = 4;
const WORKGROUP_SIZE_Z: u32 = 4;
#endif

// NOTE: the only reason why this is its own kernel is because this makes us
//       exceed the 10 storage bindings on web platforms (because of the
//       collision-detection buffers).
//       If we ever end up moving the collision-detection to particles only,
//       we should consider doing the cell update in the p2g kernel.
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

    let global_id = global_node_id.id;
    let momentum_velocity_mass = Grid::nodes[global_id].momentum_velocity_mass;
    let new_grid_velocity_mass = update_single_cell(cell_pos, momentum_velocity_mass);
    Grid::nodes[global_id].momentum_velocity_mass = new_grid_velocity_mass;
}

#if DIM == 2
fn update_single_cell(cell_pos: vec2<f32>, momentum_velocity_mass: vec3<f32>) -> vec3<f32> {
    let mass = momentum_velocity_mass.z;
    let inv_mass = select(0.0, 1.0 / mass, mass > 0.0);
    var velocity = (momentum_velocity_mass.xy + mass * Grid::sim_params.gravity * Grid::sim_params.dt) * inv_mass;
    // Clamp the velocity so it doesn’t exceed 1 grid cell in one step.
    let vel_limit = vec2(Grid::grid.cell_width / Grid::sim_params.dt);
    velocity = clamp(velocity, -vel_limit, vel_limit);
    return vec3(velocity, mass);
}
#else
fn update_single_cell(cell_pos: vec3<f32>, momentum_velocity_mass: vec4<f32>) -> vec4<f32> {
    let mass = momentum_velocity_mass.w;
    let inv_mass = select(0.0, 1.0 / mass, mass > 0.0);
    var velocity = (momentum_velocity_mass.xyz + mass * Grid::sim_params.gravity * Grid::sim_params.dt) * inv_mass;

    // Clamp the velocity so it doesn’t exceed 1 grid cell in one step.
    let vel_limit = vec3(Grid::grid.cell_width / Grid::sim_params.dt);
    velocity = clamp(velocity, -vel_limit, vel_limit);
    return vec4(velocity, mass);
}
#endif

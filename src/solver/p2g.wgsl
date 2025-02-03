#define_import_path wgsparkl::solver::p2g

#import wgsparkl::solver::params as Params;
#import wgsparkl::solver::particle as Particle;
#import wgsparkl::grid::kernel as Kernel;
#import wgsparkl::grid::grid as Grid;
#import wgsparkl::solver::impulse as Impulse;
#import wgrapier::body as Body;


@group(1) @binding(0)
var<storage, read> particles_pos: array<Particle::Position>;
@group(1) @binding(1)
var<storage, read> particles_vel: array<Particle::Velocity>;
@group(1) @binding(2)
var<storage, read> particles_vol: array<Particle::Volume>;
#if DIM == 2
@group(1) @binding(3)
var<storage, read> particles_affine: array<mat2x2<f32>>;
#else
@group(1) @binding(3)
var<storage, read> particles_affine: array<mat3x3<f32>>;
#endif
@group(1) @binding(4)
var<storage, read> particles_cdf: array<Particle::Cdf>;
@group(1) @binding(5)
var<storage, read> nodes_linked_lists: array<Grid::NodeLinkedList>;
@group(1) @binding(6)
var<storage, read> particle_node_linked_lists: array<u32>;
@group(1) @binding(7)
var<storage, read_write> body_impulses: array<Impulse::IntegerImpulseAtomic>;

@group(2) @binding(0)
var<storage, read> body_vels: array<Body::Velocity>;
@group(2) @binding(1)
var<storage, read> body_mprops: array<Body::MassProperties>;



#if DIM == 2
const WORKGROUP_SIZE_X: u32 = 8;
const WORKGROUP_SIZE_Y: u32 = 8;
const WORKGROUP_SIZE_Z: u32 = 1;
const NUM_SHARED_CELLS: u32 = 10 * 10; // block-size plus 2 from adjacent blocks: (8 + 2)^2
var<workgroup> shared_vel_mass: array<vec3<f32>, NUM_SHARED_CELLS>;
var<workgroup> shared_affine: array<mat2x2<f32>, NUM_SHARED_CELLS>;
#else
const WORKGROUP_SIZE_X: u32 = 4;
const WORKGROUP_SIZE_Y: u32 = 4;
const WORKGROUP_SIZE_Z: u32 = 4;
const NUM_SHARED_CELLS: u32 = 6 * 6 * 6; // block-size plus 2 from adjacent blocks: (4 + 2)^3
var<workgroup> shared_vel_mass: array<vec4<f32>, NUM_SHARED_CELLS>;
var<workgroup> shared_affine: array<mat3x3<f32>, NUM_SHARED_CELLS>;
#endif
var<workgroup> shared_nodes: array<SharedNode, NUM_SHARED_CELLS>;
var<workgroup> shared_pos: array<Particle::Position, NUM_SHARED_CELLS>;
var<workgroup> shared_affinities: array<u32, NUM_SHARED_CELLS>;
var<workgroup> shared_normals: array<Vector, NUM_SHARED_CELLS>;
// TODO: is computing themax with an atomic faster than doing a reduction?
var<workgroup> max_linked_list_length: atomic<u32>;
// NOTE: workgroupUniformLoad doesn’t work on atomics, so we need that additional variable
//       to write `max_linked_list_length` into and then read with workgroupUniformLoad.
var<workgroup> max_linked_list_length_uniform: u32;

struct SharedNode {
    particle_id: u32,
    global_id: u32,
}

struct P2GStepResult {
#if DIM == 2
    new_momentum_velocity_mass: vec3<f32>,
    impulse: vec2<f32>,
#else
    new_momentum_velocity_mass: vec4<f32>,
    impulse: vec3<f32>,
#endif
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn p2g(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(local_invocation_index) tid_flat: u32,
    @builtin(workgroup_id) block_id: vec3<u32>
) {
    let bid = block_id.x;
    let vid = Grid::active_blocks[bid].virtual_id;

    // Figure out how many time we’ll have to iterate through
    // the particle linked-list to traverse them all.
    if tid_flat == 0 {
        // Technically not needed because the WebGpu spec. requires zeroing shared memory, but wgpu has an option to disable that.
        atomicStore(&max_linked_list_length, 0u);
    }
    workgroupBarrier();
    fetch_max_linked_lists_length(tid, vid, bid);
    workgroupBarrier();
    max_linked_list_length_uniform = max_linked_list_length;

    // Block -> shared memory transfer.
    fetch_nodes(tid, vid, bid);

    /* Run p2g. Note that we have one thread per cell we want to gather data into. */
    // NOTE: we shift by (8, 8) or (4, 4, 4) so we are in the top-most octant. This is the octant we
    //       got enough information for a full gather.
#if DIM == 2
    var new_momentum_velocity_mass = vec3(0.0);
    let packed_cell_index_in_block = flatten_shared_index(tid.x + 8u, tid.y + 8u);
#else
    var new_momentum_velocity_mass = vec4(0.0);
    let packed_cell_index_in_block = flatten_shared_index(tid.x + 4u, tid.y + 4u, tid.z + 4u);
#endif

    // TODO: we store the global_id in shared memory for convenience. Should we just recompute it instead?
    let global_id = shared_nodes[packed_cell_index_in_block].global_id;
    let node_affinities = Grid::nodes_cdf[global_id].affinities;
    let closest_body = Grid::nodes_cdf[global_id].closest_id;
    var total_result = P2GStepResult();

    // NOTE: read the linked list with workgroupUniformLoad so that is is considered
    //       part of a uniform execution flow (for the barriers to be valid).
    let len = workgroupUniformLoad(&max_linked_list_length_uniform);
    for (var i = 0u; i < len; i += 1u) {
        workgroupBarrier();
        fetch_next_particle(tid);
        workgroupBarrier();
        let partial_result = p2g_step(packed_cell_index_in_block, Grid::grid.cell_width, node_affinities, closest_body);
        total_result.new_momentum_velocity_mass += partial_result.new_momentum_velocity_mass;
        total_result.impulse += partial_result.impulse;
    }

    // Grid update.
#if DIM == 2
    let cell_pos = vec2<f32>(vid.id * 8 + vec2<i32>(tid.xy)) * Grid::grid.cell_width;
#else
    let cell_pos = vec3<f32>(vid.id * 4 + vec3<i32>(tid)) * Grid::grid.cell_width;
#endif

// NOTE: the only reason why we don’t do the grid update is because this makes us
//       exceed the 10 storage bindings on web paltforms (because of the
//       collision-detection buffers).
//       If we ever end up moving the collision-detection to particles only,
//       we should consider doing the cell update in the p2g kernel.

    // Write the node state to global memory.
    Grid::nodes[global_id].momentum_velocity_mass = total_result.new_momentum_velocity_mass;
    // Apply the impulse to the closest body.
    // PERF: we should probably run a reduction here to get per-collider accumulated impulses
    //       before adding to global memory. Because it is very likely that every single thread
    //       here targets the same body.
    atomicAdd(&body_impulses[closest_body].linear_x, Impulse::flt2int(total_result.impulse.x));
    atomicAdd(&body_impulses[closest_body].linear_y, Impulse::flt2int(total_result.impulse.y));
//    atomicAdd(&body_impulses[closest_body].angular, total_result.impulse.angular));
}

fn p2g_step(packed_cell_index_in_block: u32, cell_width: f32, node_affinity: u32, closest_body: u32) -> P2GStepResult {
    // NOTE: having these into a var is needed so we can index [i] them.
    //       Does this have any impact on performances?
    var NBH_SHIFTS = Kernel::NBH_SHIFTS;
    var NBH_SHIFTS_SHARED = Kernel::NBH_SHIFTS_SHARED;

    // Shift to reach the first node with particles contibuting to the current cell’s data.
#if DIM == 2
    let bottommost_contributing_node = flatten_shared_shift(2u, 2u);
    var new_momentum_velocity_mass = vec3(0.0);
#else
    let bottommost_contributing_node = flatten_shared_shift(2u, 2u, 2u);
    var new_momentum_velocity_mass = vec4(0.0);
#endif
    var impulse = Vector(0.0);

    for (var i = 0u; i < Kernel::NBH_LEN; i += 1u) {
        let packed_shift = NBH_SHIFTS_SHARED[i];
        let nbh_shared_index = packed_cell_index_in_block - bottommost_contributing_node + packed_shift;
        let particle_pos = shared_pos[nbh_shared_index];
        let particle_vel_mass = shared_vel_mass[nbh_shared_index];
        let particle_affine = shared_affine[nbh_shared_index];
        let ref_elt_pos_minus_particle_pos = Particle::dir_to_associated_grid_node(particle_pos, cell_width);
        // TODO: only compute the one weight we need.
        let w = Kernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

#if DIM == 2
        let particle_vel = particle_vel_mass.xy;
        let particle_mass = particle_vel_mass.z;
        let shift = vec2(2u, 2) - NBH_SHIFTS[i];
        let momentum = particle_vel * particle_mass;
        let dpt = ref_elt_pos_minus_particle_pos + vec2<f32>(shift) * cell_width; // cell_pos - particle_pos
        let weight = w.x[shift.x] * w.y[shift.y];
#else
        let particle_vel = particle_vel_mass.xyz;
        let particle_mass = particle_vel_mass.w;
        let shift = vec3(2u, 2, 2) - NBH_SHIFTS[i];
        let momentum = particle_vel * particle_mass;
        let dpt = ref_elt_pos_minus_particle_pos + vec3<f32>(shift) * cell_width; // cell_pos - particle_pos
        let weight = w.x[shift.x] * w.y[shift.y] * w.z[shift.z];
#endif

        let particle_affinity = shared_affinities[nbh_shared_index];
        if !Grid::affinities_are_compatible(node_affinity, particle_affinity) {
            let particle_normal = shared_normals[nbh_shared_index];
            let body_vel = body_vels[closest_body];
            let body_com = body_mprops[closest_body].com;
            let cell_center = dpt + particle_pos.pt;
            let body_pt_vel =  Body::velocity_at_point(body_com, body_vel, cell_center);
            let particle_ghost_vel = body_pt_vel + Grid::project_velocity(particle_vel - body_pt_vel, particle_normal);
            impulse += (particle_vel - particle_ghost_vel) * (weight * particle_mass);
            continue;
        } else {
#if DIM == 2
            new_momentum_velocity_mass += vec3(particle_affine * dpt + momentum, particle_mass) * weight;
#else
            new_momentum_velocity_mass += vec4(particle_affine * dpt + momentum, particle_mass) * weight;
#endif
        }
    }

    return P2GStepResult(new_momentum_velocity_mass, impulse);
}

#if DIM == 2
    const K_RANGE: u32 = 0;
#else
    const K_RANGE: u32 = 1;
#endif


fn fetch_max_linked_lists_length(tid: vec3<u32>, active_block_vid: Grid::BlockVirtualId, bid: u32) {
#if DIM == 2
    let base_block_pos_int = active_block_vid.id - vec2<i32>(1i, 1i);
#else
    let base_block_pos_int = active_block_vid.id - vec3<i32>(1i, 1i, 1i);
#endif

    for (var i = 0u; i <= 1u; i++) {
        for (var j = 0u; j <= 1u; j++) {
            for (var k = 0u; k <= K_RANGE; k++) {
#if DIM == 2
                if (i == 0 && tid.x < 6) || (j == 0 && tid.y < 6) {
                    // This thread is targetting a non-existent cell in shared memory.
                    continue;
                }

                let octant = vec2(i, j);
                let octant_hid = Grid::find_block_header_id(Grid::BlockVirtualId(base_block_pos_int + vec2<i32>(octant)));
#else
                if (i == 0 && tid.x < 2) || (j == 0 && tid.y < 2) || (k == 0 && tid.z < 2) {
                    // This thread is targetting a non-existent cell in shared memory.
                    continue;
                }

                let octant = vec3(i, j, k);
                let octant_hid = Grid::find_block_header_id(Grid::BlockVirtualId(base_block_pos_int + vec3<i32>(octant)));
#endif
                if octant_hid.id != Grid::NONE {
                    let global_chunk_id = Grid::block_header_id_to_physical_id(octant_hid);
#if DIM == 2
                    let global_node_id = Grid::node_id(global_chunk_id, tid.xy);
#else
                    let global_node_id = Grid::node_id(global_chunk_id, tid);
#endif
                    let len = nodes_linked_lists[global_node_id.id].len;
                    atomicMax(&max_linked_list_length, len);
                }
            }
        }
    }
}

fn fetch_nodes(tid: vec3<u32>, active_block_vid: Grid::BlockVirtualId, bid: u32) {
#if DIM == 2
    let base_block_pos_int = active_block_vid.id - vec2<i32>(1i, 1i);
#else
    let base_block_pos_int = active_block_vid.id - vec3<i32>(1i, 1i, 1i);
#endif

    for (var i = 0u; i <= 1u; i++) {
        for (var j = 0u; j <= 1u; j++) {
            for (var k = 0u; k <= K_RANGE; k++) {
#if DIM == 2
                if (i == 0 && tid.x < 6) || (j == 0 && tid.y < 6) {
                    // This thread is targetting a non-existent cell in shared memory.
                    continue;
                }

                let octant = vec2(i, j);
                let octant_hid = Grid::find_block_header_id(Grid::BlockVirtualId(base_block_pos_int + vec2<i32>(octant)));
                let shared_index = octant * 8 + tid.xy;
                let shared_node_index = flatten_shared_index(shared_index.x, shared_index.y);
#else
                if (i == 0 && tid.x < 2) || (j == 0 && tid.y < 2) || (k == 0 && tid.z < 2) {
                    // This thread is targetting a non-existent cell in shared memory.
                    continue;
                }

                let octant = vec3(i, j, k);
                let octant_hid = Grid::find_block_header_id(Grid::BlockVirtualId(base_block_pos_int + vec3<i32>(octant)));
                let shared_index = octant * 4 + tid;
                let shared_node_index = flatten_shared_index(shared_index.x, shared_index.y, shared_index.z);
#endif
                let shared_node = &shared_nodes[shared_node_index];

                if octant_hid.id != Grid::NONE {
                    let global_chunk_id = Grid::block_header_id_to_physical_id(octant_hid);
#if DIM == 2
                    let global_node_id = Grid::node_id(global_chunk_id, tid.xy);
#else
                    let global_node_id = Grid::node_id(global_chunk_id, tid);
#endif
                    let particle_id = nodes_linked_lists[global_node_id.id].head;
                    (*shared_node).particle_id = particle_id;
                    (*shared_node).global_id = global_node_id.id;
                } else {
                    // This octant doesn’t exist. Fill shared memory with zeros/NONE.
                    // NOTE: we don’t need to init global_id since it’s only read for the
                    //       current chunk that is guaranteed to exist, not the 2x2x2 adjacent ones.
                    (*shared_node).particle_id = Grid::NONE;
                }
            }
        }
    }
}

fn fetch_next_particle(tid: vec3<u32>) {
    for (var i = 0u; i <= 1u; i++) {
        for (var j = 0u; j <= 1u; j++) {
            for (var k = 0u; k <= K_RANGE; k++) {
#if DIM == 2
                if (i == 0 && tid.x < 6) || (j == 0 && tid.y < 6) {
                    continue;
                }
                let octant = vec2(i, j);
                let shared_index = octant * 8 + tid.xy;
                let shared_flat_index = flatten_shared_index(shared_index.x, shared_index.y);
#else
                if (i == 0 && tid.x < 2) || (j == 0 && tid.y < 2) || (k == 0 && tid.z < 2) {
                    continue;
                }
                let octant = vec3(i, j, k);
                let shared_index = octant * 4 + tid;
                let shared_flat_index = flatten_shared_index(shared_index.x, shared_index.y, shared_index.z);
#endif
                let shared_node = &shared_nodes[shared_flat_index];
                let curr_particle_id = (*shared_node).particle_id;

                if curr_particle_id != Grid::NONE {
                    shared_affinities[shared_flat_index] = particles_cdf[curr_particle_id].affinity;
                    shared_normals[shared_flat_index] = particles_cdf[curr_particle_id].normal;
                    shared_pos[shared_flat_index] = particles_pos[curr_particle_id];
                    shared_affine[shared_flat_index] = particles_affine[curr_particle_id];

#if DIM == 2
                    shared_vel_mass[shared_flat_index] = vec3(particles_vel[curr_particle_id].v, Particle::mass(particles_vol[curr_particle_id]));
#else
                    shared_vel_mass[shared_flat_index] = vec4(particles_vel[curr_particle_id].v, Particle::mass(particles_vol[curr_particle_id]));
#endif

                    let next_particle_id = particle_node_linked_lists[curr_particle_id];
                    (*shared_node).particle_id = next_particle_id;
                } else {
                    // TODO: would it be worth skipping writing zeros if we already
                    //       did it at the previous step? (if we already reached the end
                    //       of the particle linked list)
                    shared_affinities[shared_flat_index] = 0u;
                    shared_normals[shared_flat_index] = Vector(0.0);
#if DIM == 2
                    shared_pos[shared_flat_index].pt = vec2(0.0);
                    shared_affine[shared_flat_index] = mat2x2(vec2(0.0), vec2(0.0));
                    shared_vel_mass[shared_flat_index] = vec3(0.0);
#else
                    shared_pos[shared_flat_index].pt = vec3(0.0);
                    shared_affine[shared_flat_index] = mat3x3(vec3(0.0), vec3(0.0), vec3(0.0));
                    shared_vel_mass[shared_flat_index] = vec4(0.0);
#endif
                }
            }
        }
    }
}

#if DIM == 2
fn flatten_shared_index(x: u32, y: u32) -> u32 {
    return (x - 6) + (y - 6) * 10;
}
fn flatten_shared_shift(x: u32, y: u32) -> u32 {
    return x + y * 10;
}
#else
fn flatten_shared_index(x: u32, y: u32, z: u32) -> u32 {
    return (x - 2) + (y - 2) * 6 + (z - 2) * 6 * 6;
}
fn flatten_shared_shift(x: u32, y: u32, z: u32) -> u32 {
    return x + y * 6 + z * 6 * 6;
}
#endif

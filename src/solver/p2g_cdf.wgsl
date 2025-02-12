#define_import_path wgsparkl::solver::p2g

//#if DIM == 2
    #import wgparry::segment as Shape;
//#else
//    #import wgparry::triangle as Shape;
//#endif

#import wgsparkl::solver::params as Params;
#import wgsparkl::solver::particle as Particle;
#import wgsparkl::grid::kernel as Kernel;
#import wgsparkl::grid::grid as Grid;
#import wgrapier::body as Body;

@group(1) @binding(0)
var<storage, read> nodes_linked_lists: array<Grid::NodeLinkedList>;
@group(1) @binding(1)
var<storage, read> particle_node_linked_lists: array<u32>;
@group(1) @binding(2)
var<storage, read> rigid_particle_indices: array<Particle::RigidParticleIndices>;
@group(1) @binding(3)
var<storage, read> collider_vertices: array<Vector>;


#if DIM == 2
const WORKGROUP_SIZE_X: u32 = 8;
const WORKGROUP_SIZE_Y: u32 = 8;
const WORKGROUP_SIZE_Z: u32 = 1;
const NUM_SHARED_CELLS: u32 = 10 * 10; // block-size plus 2 from adjacent blocks: (8 + 2)^2
var<workgroup> shared_primitives: array<Shape::Segment, NUM_SHARED_CELLS>;
#else
const WORKGROUP_SIZE_X: u32 = 4;
const WORKGROUP_SIZE_Y: u32 = 4;
const WORKGROUP_SIZE_Z: u32 = 4;
const NUM_SHARED_CELLS: u32 = 6 * 6 * 6; // block-size plus 2 from adjacent blocks: (4 + 2)^3
var<workgroup> shared_primitives: array<Shape::Triangle, NUM_SHARED_CELLS>;
#endif
var<workgroup> shared_nodes: array<SharedNode, NUM_SHARED_CELLS>;
var<workgroup> shared_collider_ids: array<u32, NUM_SHARED_CELLS>;
// TODO: is computing themax with an atomic faster than doing a reduction?
var<workgroup> max_linked_list_length: atomic<u32>;
// NOTE: workgroupUniformLoad doesn’t work on atomics, so we need that additional variable
//       to write `max_linked_list_length` into and then read with workgroupUniformLoad.
var<workgroup> max_linked_list_length_uniform: u32;

struct SharedNode {
    particle_id: u32,
    global_id: u32,
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn p2g_cdf(
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
    let packed_cell_index_in_block = flatten_shared_index(tid.x + 8u, tid.y + 8u);
    let cell_pos = vec2<f32>(vid.id * 8 + vec2<i32>(tid.xy)) * Grid::grid.cell_width;
#else
    let packed_cell_index_in_block = flatten_shared_index(tid.x + 4u, tid.y + 4u, tid.z + 4u);
    let cell_pos = vec3<f32>(vid.id * 4 + vec3<i32>(tid)) * Grid::grid.cell_width;
#endif

    // TODO: we store the global_id in shared memory for convenience. Should we just recompute it instead?
    let global_id = shared_nodes[packed_cell_index_in_block].global_id;
    let node_affinities = Grid::nodes_cdf[global_id].affinities;
    let closest_body = Grid::nodes_cdf[global_id].closest_id;
    var node_cdf = Grid::nodes_cdf[global_id];

    // NOTE: read the linked list with workgroupUniformLoad so that is is considered
    //       part of a uniform execution flow (for the barriers to be valid).
    let len = workgroupUniformLoad(&max_linked_list_length_uniform);
    for (var i = 0u; i < len; i += 1u) {
        workgroupBarrier();
        fetch_next_particle(tid);
        workgroupBarrier();
        let partial_result = p2g_step(packed_cell_index_in_block, Grid::grid.cell_width, cell_pos);

        if partial_result.closest_id != Grid::NONE {
            node_cdf.affinities |= partial_result.affinities;

            if partial_result.distance < node_cdf.distance {
                node_cdf.distance = partial_result.distance;
                node_cdf.closest_id = node_cdf.closest_id;
            }
        }
    }

    // Write the node cdf to global memory.
    Grid::nodes_cdf[global_id] = node_cdf;
}

fn p2g_step(packed_cell_index_in_block: u32, cell_width: f32, cell_pos: Vector) -> Grid::NodeCdf {
    // NOTE: having these into a var is needed so we can index [i] them.
    //       Does this have any impact on performances?
    var NBH_SHIFTS_SHARED = Kernel::NBH_SHIFTS_SHARED;

    // Shift to reach the first node with particles contibuting to the current cell’s data.
#if DIM == 2
    let bottommost_contributing_node = flatten_shared_shift(2u, 2u);
#else
    let bottommost_contributing_node = flatten_shared_shift(2u, 2u, 2u);
#endif
    var result = Grid::NodeCdf(1.0e10, 0u, Grid::NONE);

    for (var i = 0u; i < Kernel::NBH_LEN; i += 1u) {
        let packed_shift = NBH_SHIFTS_SHARED[i];
        let nbh_shared_index = packed_cell_index_in_block - bottommost_contributing_node + packed_shift;

        let collider_id = shared_collider_ids[nbh_shared_index];

        if collider_id == Grid::NONE {
            continue;
        }

        let primitive = shared_primitives[nbh_shared_index];

        // Project the cell on the primitives.
        let proj = Shape::projectLocalPoint(primitive, cell_pos);

#if DIM == 2
        if any(proj != primitive.a) && any(proj != primitive.b) {
            // This is a valid projection.
            let dpt = cell_pos - proj;
            let distance = length(dpt);
            let ab = primitive.b - primitive.a;
            let sign = dot(dpt, vec2(-ab.y, ab.x)) < 0.0;
            result.affinities |= (1u << collider_id)
                | (u32(sign) << (collider_id + Grid::SIGN_BITS_SHIFT));

            if distance < result.distance {
                result.distance = min(result.distance, distance);
                result.closest_id = collider_id;
            }
        }
#endif
    }

    return result;
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
                    let rigid_idx = rigid_particle_indices[curr_particle_id];
                    shared_collider_ids[shared_flat_index] = rigid_idx.collider;
                    shared_primitives[shared_flat_index] = Shape::Segment(
                        collider_vertices[rigid_idx.segment.x],
                        collider_vertices[rigid_idx.segment.y]
                    );

                    let next_particle_id = particle_node_linked_lists[curr_particle_id];
                    (*shared_node).particle_id = next_particle_id;
                } else {
                    // TODO: would it be worth skipping writing zeros if we already
                    //       did it at the previous step? (if we already reached the end
                    //       of the particle linked list)
                    shared_collider_ids[shared_flat_index] = Grid::NONE;
#if DIM == 2
                    shared_primitives[shared_flat_index] = Shape::Segment(Vector(0.0), Vector(0.0));
#else
                    shared_primitives[shared_flat_index] = Shape::Triangle(Vector(0.0), Vector(0.0), Vector(0.0));
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

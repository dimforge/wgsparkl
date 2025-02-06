#define_import_path wgsparkl::grid::grid
#import wgsparkl::solver::params as Params;


@group(0) @binding(0)
var<storage, read_write> grid: Grid; // TODO: should be uniform? Currently it can’t due to the mutable num_active_blocks atomic.
@group(0) @binding(1)
var<storage, read_write> hmap_entries: array<GridHashMapEntry>;
@group(0) @binding(2)
var<storage, read_write> active_blocks: array<ActiveBlockHeader>;
@group(0) @binding(3)
var<storage, read_write> nodes: array<Node>;
@group(0) @binding(4)
var<uniform> sim_params: Params::SimulationParams;
@group(0) @binding(5)
var<storage, read_write> n_block_groups: DispatchIndirectArgs;
@group(0) @binding(6)
var<storage, read_write> nodes_linked_lists: array<NodeLinkedListAtomic>;
@group(0) @binding(7)
var<storage, read_write> n_g2p_p2g_groups: DispatchIndirectArgs;
@group(0) @binding(8)
var<storage, read_write> num_collisions: array<atomic<u32>>;
@group(0) @binding(9)
var<storage, read_write> nodes_cdf: array<NodeCdf>;


struct NodeLinkedListAtomic {
    head: atomic<u32>,
    len: atomic<u32>,
}

// Non-atomic version of NodeLinkedListAtomic
struct NodeLinkedList {
    head: u32,
    len: u32,
}


const GRID_WORKGROUP_SIZE: u32 = 64;
const G2P_P2G_WORKGROUP_SIZE: u32 = 64;
const NUM_CELL_PER_BLOCK: u32 = 64; // 8 * 8 in 2D and 4 * 4 * 4 in 3D.

// TODO: upstream this to wgcore?
struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}

/*
 * Some index types.
 */
struct BlockVirtualId {
    #if DIM == 2
    id: vec2<i32>,
    #else
    id: vec3<i32>,
    #endif
}

struct BlockHeaderId {
    id: u32,
}

struct BlockPhysicalId {
    id: u32,
}

struct NodePhysicalId {
    id: u32,
}

/*
 *
 * HashMap for the grid.
 *
 */
const NONE: u32 = 0xffffffffu;

#if DIM == 2
fn pack_key(key: BlockVirtualId) -> u32 {
    return (bitcast<u32>(key.id.x + 0x00007fff) & 0x0000ffffu) |
    ((bitcast<u32>(key.id.y + 0x00007fff) & 0x0000ffffu) << 16);
}
#else
fn pack_key(key: BlockVirtualId) -> u32 {
    // NOTE: we give the X and Z axis one more bit than Y.
    //       This is assuming Y-up and the fact that we want
    //       more room on the X-Z plane rather than along the up axis.
    return (bitcast<u32>(key.id.x + 0x000003ff) & 0x000007ffu) |
    ((bitcast<u32>(key.id.y + 0x000001ff) & 0x000003ffu) << 11) |
    ((bitcast<u32>(key.id.z + 0x000003ff) & 0x000007ffu) << 21);
}
#endif

fn hash(packed_key: u32) -> u32 {
    // Murmur3 hash function.
    var key = packed_key;
    key *= 0xcc9e2d51u;
    key = (key << 15) | (key >> 17);
    key *= 0x1b873593u;
    return key;
}

// IMPORTANT: if this struct is changed (including its layout), be sure to
//            modify the GpuGridHashMapEntry struct on the Rust side to ensure
//            it has the right size. Otherwise the hashmap will break.
struct GridHashMapEntry {
    // Indicates if the entry is free or empty.
    state: atomic<u32>,
    // The key stored on this entry.
    key: BlockVirtualId,
    // The associated value.
    value: BlockHeaderId
}

#if MACOS == 0
// The hash map ipmelementation is inspired from https://nosferalatu.com/SimpleGPUHashTable.html
fn insertion_index(capacity: u32, key: BlockVirtualId) -> u32 {
    let packed_key = pack_key(key);
    var slot = hash(packed_key) & (capacity - 1u);
    var retries = 0u;

    // NOTE: if there is no more room in the hashmap to store the data, we just do nothing.
    // It is up to the user to detect the high occupancy, resize the hashmap, and re-run
    // the failed insertion.
    for (var k = 0u; k < capacity; k++) {
        // TODO: would it be more efficient to move the `state` into its own
        //       vector with only atomics?
        let entry = &hmap_entries[slot];

        var my_retries = 0u;
        loop {
            let exch = atomicCompareExchangeWeak(&(*entry).state, NONE, packed_key);
            if exch.exchanged {
                // We found a slot.
                (*entry).key = key;
                // TODO: remove these atomicMax, it’s just for debugging.
                atomicMax(&num_collisions[0], k);
                atomicMax(&num_collisions[1], arrayLength(&hmap_entries));
                return slot;
            } else if exch.old_value == packed_key {
                // The entry already exists.
                // TODO: remove these atomicMax, it’s just for debugging.
                atomicMax(&num_collisions[0], k);
                atomicMax(&num_collisions[1], arrayLength(&hmap_entries));
                return NONE;
            } else if exch.old_value != NONE {
                // The slot is already taken.
                break;
            }
            // Otherwise we need to loop since we hit a case where the exchange could
            // have happened but didn’t due to the weak nature of the operation.
            my_retries += 1u;
        }

        retries = max(retries, my_retries);
        slot = (slot + 1u) % capacity & (capacity - 1u);
    }

    return NONE;
}
#endif

fn find_block_header_id(key: BlockVirtualId) -> BlockHeaderId {
    let packed_key = pack_key(key);
    var slot = hash(packed_key) & (grid.hmap_capacity - 1);

    loop {
        let entry = &hmap_entries[slot];
        let state = (*entry).state;
        if state == packed_key {
            return (*entry).value;
        } else if state == NONE {
            return BlockHeaderId(NONE);
        }

         slot = (slot + 1) & (grid.hmap_capacity - 1);
    }

    return BlockHeaderId(NONE);
}

@compute @workgroup_size(GRID_WORKGROUP_SIZE, 1, 1)
fn reset_hmap(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let id = invocation_id.x;
    if id < grid.hmap_capacity {
        hmap_entries[id].state = NONE;
        // Resetting the following two isn’t necessary for correctness,
        // but it makes debugging easier.
        #if DIM == 2
        hmap_entries[id].key = BlockVirtualId(vec2(0));
        #else
        hmap_entries[id].key = BlockVirtualId(vec3(0));
        #endif
        hmap_entries[id].value = BlockHeaderId(0);
    }
    if id == 0 {
        grid.num_active_blocks = 0u;
    }
}

/*
 * Sparse grid definition.
 */
#if DIM == 2
const NUM_ASSOC_BLOCKS: u32 = 4;
#else
const NUM_ASSOC_BLOCKS: u32 = 8;
#endif
const OFF_BY_ONE: i32 = 1;

struct ActiveBlockHeader {
    virtual_id: BlockVirtualId, // Needed to compute the world-space position of a block.
    first_particle: u32,
    num_particles: atomic<u32>,
}


struct Grid {
    num_active_blocks: atomic<u32>,
    cell_width: f32,
    // NOTE: the hashmap capacity MUST be a power of 2.
    hmap_capacity: u32,
    capacity: u32,
}

const AFFINITY_BITS_MASK: u32 = 0x0000ffffu;
const SIGN_BITS_SHIFT: u32 = 16;

struct NodeCdf {
    distance: f32,
    // Two bits per collider.
    // The 16 first bits are for affinity, the 16 last are for signs.
    affinities: u32,
    // Index to the closest collider.
    closest_id: u32,
}

fn affinity_bit(i_collider: u32, affinity: u32) -> bool {
    return (affinity & (1u << i_collider)) != 0;
}

fn sign_bit(i_collider: u32, affinity: u32) -> bool {
    return ((affinity >> SIGN_BITS_SHIFT) & (1u << i_collider)) != 0;
}

fn affinities_are_compatible(affinity1: u32, affinity2: u32) -> bool {
    let affinities_in_common = affinity1 & affinity2 & AFFINITY_BITS_MASK;
    let signs1 = (affinity1 >> SIGN_BITS_SHIFT) & affinities_in_common;
    let signs2 = (affinity2 >> SIGN_BITS_SHIFT) & affinities_in_common;
    return signs1 == signs2;
}

struct Node {
    /// The first three components contains either the cell’s momentum or its velocity
    /// (depending on the context). The fourth component contains the cell’s mass.
    #if DIM == 2
    momentum_velocity_mass: vec3<f32>,
    #else
    momentum_velocity_mass: vec4<f32>,
    #endif
}

#if DIM == 2
fn block_associated_to_point(pt: vec2<f32>) -> BlockVirtualId {
    let assoc_cell = round(pt / grid.cell_width) - 1.0;
    let assoc_block = floor(assoc_cell / 8.0);
    return BlockVirtualId(vec2(
        i32(assoc_block.x),
        i32(assoc_block.y),
    ));
}

fn blocks_associated_to_point(pt: vec2<f32>) -> array<BlockVirtualId, NUM_ASSOC_BLOCKS> {
    let main_block = block_associated_to_point(pt);
    return blocks_associated_to_block(main_block);
}
#else
fn block_associated_to_point(pt: vec3<f32>) -> BlockVirtualId {
    let assoc_cell = round(pt / grid.cell_width) - 1.0;
    let assoc_block = floor(assoc_cell / 4.0);
    return BlockVirtualId(vec3(
        i32(assoc_block.x),
        i32(assoc_block.y),
        i32(assoc_block.z),
    ));
}

fn blocks_associated_to_point(pt: vec3<f32>) -> array<BlockVirtualId, NUM_ASSOC_BLOCKS> {
    let main_block = block_associated_to_point(pt);
    return blocks_associated_to_block(main_block);
}
#endif

fn blocks_associated_to_block(block: BlockVirtualId) -> array<BlockVirtualId, NUM_ASSOC_BLOCKS> {
    #if DIM == 2
    return array<BlockVirtualId, NUM_ASSOC_BLOCKS>(
        BlockVirtualId(block.id + vec2(0, 0)),
        BlockVirtualId(block.id + vec2(0, 1)),
        BlockVirtualId(block.id + vec2(1, 0)),
        BlockVirtualId(block.id + vec2(1, 1)),
    );
    #else
    return array<BlockVirtualId, NUM_ASSOC_BLOCKS>(
        BlockVirtualId(block.id + vec3(0, 0, 0)),
        BlockVirtualId(block.id + vec3(0, 0, 1)),
        BlockVirtualId(block.id + vec3(0, 1, 0)),
        BlockVirtualId(block.id + vec3(0, 1, 1)),
        BlockVirtualId(block.id + vec3(1, 0, 0)),
        BlockVirtualId(block.id + vec3(1, 0, 1)),
        BlockVirtualId(block.id + vec3(1, 1, 0)),
        BlockVirtualId(block.id + vec3(1, 1, 1)),
    );
    #endif
}

#if MACOS == 0
fn mark_block_as_active(block: BlockVirtualId) {
    let slot = insertion_index(grid.hmap_capacity, block);

    if slot != NONE {
        let block_header_id = atomicAdd(&grid.num_active_blocks, 1u);
        let active_block = &active_blocks[block_header_id];
        (*active_block).virtual_id = block;
        (*active_block).first_particle = 0u;
        (*active_block).num_particles = 0u;
        hmap_entries[slot].value = BlockHeaderId(block_header_id);
    }
}
#endif

fn block_header_id_to_physical_id(hid: BlockHeaderId) -> BlockPhysicalId {
    return BlockPhysicalId(hid.id * NUM_CELL_PER_BLOCK);
}

#if DIM == 2
fn node_id(pid: BlockPhysicalId, shift_in_block: vec2<u32>) -> NodePhysicalId {
    return NodePhysicalId(pid.id + shift_in_block.x + shift_in_block.y * 8);
}
#else
fn node_id(pid: BlockPhysicalId, shift_in_block: vec3<u32>) -> NodePhysicalId {
    return NodePhysicalId(pid.id + shift_in_block.x + shift_in_block.y * 4 + shift_in_block.z * 4 * 4);
}
#endif

fn div_ceil(x: u32, y: u32) -> u32 {
    return (x + y - 1) / y;
}

@compute @workgroup_size(1)
fn init_indirect_workgroups() {
    let num_active_blocks = grid.num_active_blocks;
    n_block_groups = DispatchIndirectArgs(div_ceil(num_active_blocks, GRID_WORKGROUP_SIZE), 1, 1);
    n_g2p_p2g_groups = DispatchIndirectArgs(num_active_blocks, 1, 1);
}

@compute @workgroup_size(GRID_WORKGROUP_SIZE, 1, 1)
fn reset(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
   let num_threads = num_workgroups.x * GRID_WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
   let i = invocation_id.x;
   let num_nodes = grid.num_active_blocks * NUM_CELL_PER_BLOCK;
   for (var i = invocation_id.x; i < num_nodes; i += num_threads) {
       #if DIM == 2
       nodes[i].momentum_velocity_mass = vec3(0.0);
       #else
       nodes[i].momentum_velocity_mass = vec4(0.0);
       #endif
       nodes_cdf[i] = NodeCdf(0.0, 0, 0);
       nodes_linked_lists[i].head = NONE;
       nodes_linked_lists[i].len = 0u;
   }
}

struct SimulationParameters {
    dt: f32,
    #if DIM == 2
    gravity: vec2<f32>,
    #else
    gravity: vec3<f32>,
    #endif
}

fn project_velocity(vel: Vector, n: Vector) -> Vector {
    // TODO: this should depend on the collider’s material
    //       properties.
    let normal_vel = dot(vel, n);

    if normal_vel < 0.0 {
        let friction = 0.9;
        let tangent_vel = vel - n * normal_vel;
        let tangent_vel_len = length(tangent_vel);
        let tangent_vel_dir = select(Vector(0.0), tangent_vel / tangent_vel_len, tangent_vel_len > 1.0e-8);
        return tangent_vel_dir * max(0.0, tangent_vel_len + friction * normal_vel);
    } else {
        return vel;
    }
}
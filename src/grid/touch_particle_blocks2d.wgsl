@group(0) @binding(0)
var<storage, read_write> grid: Grid; // TODO: should be uniform? Currently it can’t due to the mutable num_active_blocks atomic.
@group(0) @binding(1)
var<storage, read_write> hmap_entries: array<GridHashMapEntry>;
@group(0) @binding(2)
var<storage, read_write> active_blocks: array<ActiveBlockHeader>;
@group(0) @binding(8)
var<storage, read_write> num_collisions: array<atomic<u32>>;
@group(1) @binding(6)
var<storage, read_write> rigid_particle_needs_block: array<u32>;

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
    id: vec2<i32>,
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

fn pack_key(key: BlockVirtualId) -> u32 {
    return (bitcast<u32>(key.id.x + 0x00007fff) & 0x0000ffffu) |
    ((bitcast<u32>(key.id.y + 0x00007fff) & 0x0000ffffu) << 16);
}

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

/*
 * Sparse grid definition.
 */
const NUM_ASSOC_BLOCKS: u32 = 4;
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

struct Node {
    /// The first three components contains either the cell’s momentum or its velocity
    /// (depending on the context). The fourth component contains the cell’s mass.
    momentum_velocity_mass: vec3<f32>,
}

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

fn blocks_associated_to_block(block: BlockVirtualId) -> array<BlockVirtualId, NUM_ASSOC_BLOCKS> {
    return array<BlockVirtualId, NUM_ASSOC_BLOCKS>(
        BlockVirtualId(block.id + vec2(0, 0)),
        BlockVirtualId(block.id + vec2(0, 1)),
        BlockVirtualId(block.id + vec2(1, 0)),
        BlockVirtualId(block.id + vec2(1, 1)),
    );
}

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

struct SimulationParameters {
    dt: f32,
    gravity: vec2<f32>,
}

// ~~~~~~~~~~~ Copied from sort.wgsl and particle2/3d.wgsl ~~~~~~~~~~~~~
struct Position {
    pt: vec2<f32>,
}

@group(1) @binding(0)
var<storage, read_write> particles_pos: array<Position>;

@compute @workgroup_size(GRID_WORKGROUP_SIZE, 1, 1)
fn touch_particle_blocks(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let id = invocation_id.x;
    if id < arrayLength(&particles_pos) {
        let particle = particles_pos[id];
        var blocks = blocks_associated_to_point(particle.pt);
        for (var i = 0u; i < NUM_ASSOC_BLOCKS; i += 1u) {
            mark_block_as_active(blocks[i]);
        }
    }
}
/*
 * TODO: is there a way to re-use the G2P pattern accross shaders?
 *       In particular, this is very similar to the `g2p.wgsl` file
 *       but with different quantities being transfered.
 */

#define_import_path wgsparkl::solver::g2p

#import wgebra::inv as Inv;
#import wgsparkl::solver::params as Params;
#import wgsparkl::solver::particle as Particle;
#import wgsparkl::grid::kernel as Kernel;
#import wgsparkl::grid::grid as Grid;

@group(1) @binding(0)
var<storage, read> particles_pos: array<Particle::Position>;
@group(1) @binding(1)
var<storage, read_write> particles_dyn: array<Particle::Dynamics>;
@group(1) @binding(2)
var<storage, read> sorted_particle_ids: array<u32>;
@group(1) @binding(3)
var<uniform> params: Params::SimulationParams;

#if DIM == 2
const WORKGROUP_SIZE_X: u32 = 8;
const WORKGROUP_SIZE_Y: u32 = 8;
const WORKGROUP_SIZE_Z: u32 = 1;
const NUM_SHARED_CELLS: u32 = 10 * 10; // block-size plus 2 from adjacent blocks: (8 + 2)^2
#else
const WORKGROUP_SIZE_X: u32 = 4;
const WORKGROUP_SIZE_Y: u32 = 4;
const WORKGROUP_SIZE_Z: u32 = 4;
const NUM_SHARED_CELLS: u32 = 6 * 6 * 6; // block-size plus 2 from adjacent blocks: (4 + 2)^3
#endif

var<workgroup> shared_nodes: array<Grid::NodeCdf, NUM_SHARED_CELLS>;

const WORKGROUP_SIZE: u32 = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Z;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn g2p_cdf(
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(local_invocation_index) tid_flat: u32,
    @builtin(workgroup_id) block_id: vec3<u32>
) {
    let bid = block_id.x;

    let active_block = &Grid::active_blocks[bid];
    // Block -> shared memory transfer.
    global_shared_memory_transfers(tid, (*active_block).virtual_id);

    // Sync after shared memory initialization.
    workgroupBarrier();

    // Particle update. Runs g2p on shared memory only.
    let max_particle_id = (*active_block).first_particle + (*active_block).num_particles;

    for (var sorted_particle_id = (*active_block).first_particle + tid_flat;
         sorted_particle_id < max_particle_id;
         sorted_particle_id += WORKGROUP_SIZE) {
        let particle_id = sorted_particle_ids[sorted_particle_id];
        particle_g2p(particle_id, Grid::grid.cell_width, params.dt);
    }
}

fn global_shared_memory_transfers(tid: vec3<u32>, active_block_vid: Grid::BlockVirtualId) {
    let base_block_pos_int = active_block_vid.id;

#if DIM == 2
    for (var i = 0u; i <= 1u; i++) {
        for (var j = 0u; j <= 1u; j++) {
            if (i == 1 && tid.x > 1) || (j == 1 && tid.y > 1) {
                // This shared node doesn’t exist.
                continue;
            }

            let octant = vec2(i, j);
            let octant_hid = Grid::find_block_header_id(Grid::BlockVirtualId(base_block_pos_int + vec2<i32>(octant)));
            let shared_index = octant * 8 + tid.xy;
            let flat_id = flatten_shared_index(shared_index.x, shared_index.y);
            let shared_node = &shared_nodes[flat_id];

            if octant_hid.id != Grid::NONE {
                let global_chunk_id = Grid::block_header_id_to_physical_id(octant_hid);
                let global_node_id = Grid::node_id(global_chunk_id, tid.xy);
                *shared_node = Grid::nodes_cdf[global_node_id.id];
            } else {
                // This octant doesn’t exist. Fill shared memory with zeros/NONE.
                // NOTE: we don’t need to init global_id since it’s only read for the
                //       current chunk that is guaranteed to exist, not the 2x2 adjacent ones.
                *shared_node = Grid::NodeCdf(0.0, 0, Grid::NONE);
            }
        }
    }
#else
    for (var i = 0u; i <= 1u; i++) {
        for (var j = 0u; j <= 1u; j++) {
            for (var k = 0u; k <= 1u; k++) {
                if (i == 1 && tid.x > 1) || (j == 1 && tid.y > 1) || (k == 1 && tid.z > 1) {
                    // This shared node doesn’t exist.
                    continue;
                }

                let octant = vec3(i, j, k);
                let octant_hid = Grid::find_block_header_id(Grid::BlockVirtualId(base_block_pos_int + vec3<i32>(octant)));
                let shared_index = octant * 4 + tid;
                let shared_node = &shared_nodes[flatten_shared_index(shared_index.x, shared_index.y, shared_index.z)];

                if octant_hid.id != Grid::NONE {
                    let global_chunk_id = Grid::block_header_id_to_physical_id(octant_hid);
                    let global_node_id = Grid::node_id(global_chunk_id, tid);
                    *shared_node = Grid::nodes_cdf[global_node_id.id];
                } else {
                    // This octant doesn’t exist. Fill shared memory with zeros/NONE.
                    // NOTE: we don’t need to init global_id since it’s only read for the
                    //       current chunk that is guaranteed to exist, not the 2x2x2 adjacent ones.
                    *shared_node = Grid::NodeCdf(0.0, 0, Grid::NONE);
                }
            }
        }
    }
#endif
}

fn particle_g2p(particle_id: u32, cell_width: f32, dt: f32) {
    // NOTE: having these into a var is needed so we can index [i] them.
    //       Does this have any impact on performances?
    var NBH_SHIFTS = Kernel::NBH_SHIFTS;
    var NBH_SHIFTS_SHARED = Kernel::NBH_SHIFTS_SHARED;

    var contact_dist = 0.0;
    var contact_normal = vec3(0.0);
    var particle_affinity = 0u;
    // TODO: would using a mat4 be faster?
    var affinity_signs = array(
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    );

    let prev_affinity = particles_dyn[particle_id].cdf.affinity;
    let particle_pos = particles_pos[particle_id];
    let inv_d = Kernel::inv_d(cell_width);
    let ref_elt_pos_minus_particle_pos = Particle::dir_to_associated_grid_node(particle_pos, cell_width);
    let w = Kernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

    let assoc_cell_before_integration = round(particle_pos.pt / cell_width);
    let assoc_cell_index_in_block = Particle::associated_cell_index_in_block_off_by_one(particle_pos, cell_width);
    let packed_cell_index_in_block = flatten_shared_index(
        assoc_cell_index_in_block.x,
        assoc_cell_index_in_block.y,
#if DIM == 3
        assoc_cell_index_in_block.z
#endif
    );

    // Eqn. 21 to determine the sign bits.
    // Also combines the affinity masks.
    for (var i = 0u; i < Kernel::NBH_LEN; i += 1u) {
        let shift = NBH_SHIFTS[i];
        let packed_shift = NBH_SHIFTS_SHARED[i];
        let cell_data = shared_nodes[packed_cell_index_in_block + packed_shift];
        particle_affinity |= cell_data.affinities & Grid::AFFINITY_BITS_MASK;

#if DIM == 2
        let weight = w.x[shift.x] * w.y[shift.y];
#else
        let weight = w.x[shift.x] * w.y[shift.y] * w.z[shift.z];
#endif

        for (var i_collider = 0u; i_collider < 16u; i_collider += 1u) {
            let compatible = f32(Grid::affinity_bit(i_collider, cell_data.affinities));
            let sign = select(1.0, -1.0, Grid::sign_bit(i_collider, cell_data.affinities) && !shape_has_solid_interior(i_collider));
            affinity_signs[i_collider] += compatible * weight * sign * cell_data.distance;
        }
    }

    // Convert the affinity signs to bits.
    for (var i_collider = 0u; i_collider < 16u; i_collider += 1u) {
        // Only change the sign bit matching affinities that didn’t exist before.
        let mask = 1u << (i_collider + Grid::SIGN_BITS_SHIFT);
        if (prev_affinity & (1u << i_collider)) == 0 {
            let sgn_bit = select(0u, mask, affinity_signs[i_collider] < 0.0);
            particle_affinity |= sgn_bit;
        } else {
            particle_affinity |= prev_affinity & mask;
        }
    }

    // At this state the `affinity` (+ sign) bitmask is filled.
    // Now, compute the contact distance/normal using MLS reconstruction (Eq. 4)
#if DIM == 2
    var qtq = mat3x3f(); // Matrix M
    var qtu = vec3(0.0);
#else
    var qtq = mat4x4f(); // Matrix M
    var qtu = vec4(0.0);
#endif

    for (var i = 0u; i < Kernel::NBH_LEN; i += 1u) {
        let shift = NBH_SHIFTS[i];
        let packed_shift = NBH_SHIFTS_SHARED[i];
        var cell_data = shared_nodes[packed_cell_index_in_block + packed_shift];
#if DIM == 2
        let dpt = ref_elt_pos_minus_particle_pos + vec2<f32>(shift) * cell_width;
        let weight = w.x[shift.x] * w.y[shift.y];
#else
        let dpt = ref_elt_pos_minus_particle_pos + vec3<f32>(shift) * cell_width;
        let weight = w.x[shift.x] * w.y[shift.y] * w.z[shift.z];
#endif
        let combined_affinity = cell_data.affinities & particle_affinity & Grid::AFFINITY_BITS_MASK;
        let sign_differences = ((cell_data.affinities >> Grid::SIGN_BITS_SHIFT)
            ^ (particle_affinity >> Grid::SIGN_BITS_SHIFT)) & combined_affinity;

#if DIM == 2
        let p = vec3(dpt, 1.0);
#else
        let p = vec4(dpt, 1.0);
#endif

        if combined_affinity != 0u {
            if sign_differences == 0u {
                // All signs match, positive distance.
                qtq += outer_product(p, p) * weight;
                qtu += p * weight * cell_data.distance;
            } else { // if (sign_differences & (sign_differences - 1u)) != 0u {
                // Exactly one sign difference, negative distance.
                qtq += outer_product(p, p) * weight;
                qtu += p * weight * -cell_data.distance;
            }
        }
    }

    if determinant(qtq) > 1.0e-8 {
#if DIM == 2
        let result = Inv::inv3(qtq) * qtu;
        let len = length(result.xy);
        let normal = select(vec2(0.0), result.xy / len, len > 1.0e-6);
        // PERF: init the rigid-velocities here instead of in g2p?
        particles_dyn[particle_id].cdf = Particle::Cdf(normal, vec2(0.0), result.z, particle_affinity);
#else
        let result = Inv::inv4(qtq) * qtu;
        let normal = result.xyz / length(result.xyz);
        particles_dyn[particle_id].cdf = Particle::Cdf(normal, vec3(0.0), result.w, particle_affinity);
#endif
    } else {
        // TODO: store the affinity in this case too?
        particles_dyn[particle_id].cdf = Particle::default_cdf();
    }
}

fn shape_has_solid_interior(i_collider: u32) -> bool {
    // TODO: needs to be false for unoriented trimeshes and polylines,
    //       true for geometric primitives.
    return false;
}

// TODO: upstream to wgebra?
#if DIM == 2
fn outer_product(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return mat3x3(
        a * b.x,
        a * b.y,
        a * b.z,
    );
}

// Note that this is different from p2g. We don’t need to shift the index since the truncated
// blocks (the neighbor blocks) are in the quadrants with larger indices.
fn flatten_shared_index(x: u32, y: u32) -> u32 {
    return x + y * 10;
}
#else
fn outer_product(a: vec4<f32>, b: vec4<f32>) -> mat4x4<f32> {
    return mat4x4(
        a * b.x,
        a * b.y,
        a * b.z,
        a * b.w
    );
}


// Note that this is different from p2g. We don’t need to shift the index since the truncated
// blocks (the neighbor blocks) are in the octants with larger indices.
fn flatten_shared_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * 6 + z * 6 * 6;
}
#endif
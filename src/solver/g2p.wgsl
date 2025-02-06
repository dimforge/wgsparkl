#define_import_path wgsparkl::solver::g2p

#import wgsparkl::solver::params as Params;
#import wgsparkl::solver::particle as Particle;
#import wgsparkl::grid::kernel as Kernel;
#import wgsparkl::grid::grid as Grid;
#import wgsparkl::models::linear_elasticity as ConstitutiveModel;
#import wgsparkl::models::drucker_prager as DruckerPrager;
#import wgrapier::body as Body;

@group(1) @binding(0)
var<storage, read> particles_pos: array<Particle::Position>;
@group(1) @binding(1)
var<storage, read_write> particles_vel: array<Particle::Velocity>;
#if DIM == 2
@group(1) @binding(2)
var<storage, read_write> particles_affine: array<mat2x2<f32>>;
#else
@group(1) @binding(2)
var<storage, read_write> particles_affine: array<mat3x3<f32>>;
#endif
@group(1) @binding(3)
var<storage, read_write> particles_cdf: array<Particle::Cdf>;
@group(1) @binding(4)
var<storage, read> sorted_particle_ids: array<u32>;
@group(1) @binding(5)
var<uniform> params: Params::SimulationParams;

@group(2) @binding(0)
var<storage, read> body_vels: array<Body::Velocity>;
@group(2) @binding(1)
var<storage, read> body_mprops: array<Body::MassProperties>;

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

var<workgroup> shared_nodes: array<Grid::Node, NUM_SHARED_CELLS>;
var<workgroup> shared_nodes_cdf: array<Grid::NodeCdf, NUM_SHARED_CELLS>; // PERF: we don’t need the distance field from the cdf

const WORKGROUP_SIZE: u32 = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Z;
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn g2p(
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
            let flat_shared_index = flatten_shared_index(shared_index.x, shared_index.y);

            if octant_hid.id != Grid::NONE {
                let global_chunk_id = Grid::block_header_id_to_physical_id(octant_hid);
                let global_node_id = Grid::node_id(global_chunk_id, tid.xy);
                shared_nodes[flat_shared_index] = Grid::nodes[global_node_id.id];
                shared_nodes_cdf[flat_shared_index] = Grid::nodes_cdf[global_node_id.id];
            } else {
                // This octant doesn’t exist. Fill shared memory with zeros/NONE.
                // NOTE: we don’t need to init global_id since it’s only read for the
                //       current chunk that is guaranteed to exist, not the 2x2 adjacent ones.
                shared_nodes[flat_shared_index] = Grid::Node(vec3(0.0));
                shared_nodes_cdf[flat_shared_index] = Grid::NodeCdf(0.0, 0, 0);
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
                let flat_shared_index = flatten_shared_index(shared_index.x, shared_index.y, shared_index.z);

                if octant_hid.id != Grid::NONE {
                    let global_chunk_id = Grid::block_header_id_to_physical_id(octant_hid);
                    let global_node_id = Grid::node_id(global_chunk_id, tid);
                    shared_nodes[flat_shared_index] = Grid::nodes[global_node_id.id];
                    shared_nodes_cdf[flat_shared_index] = Grid::nodes_cdf[global_node_id.id];
                } else {
                    // This octant doesn’t exist. Fill shared memory with zeros/NONE.
                    // NOTE: we don’t need to init global_id since it’s only read for the
                    //       current chunk that is guaranteed to exist, not the 2x2x2 adjacent ones.
                    shared_nodes[flat_shared_index] = Grid::Node(vec4(0.0));
                    shared_nodes_cdf[flat_shared_index] = Grid::NodeCdf(0.0, 0, 0);
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

#if DIM == 2
    var rigid_vel = vec2<f32>(0.0);
    var momentum_velocity_mass = vec3<f32>(0.0);
    var velocity_gradient = mat2x2<f32>(vec2(0.0), vec2(0.0));
#else
    var rigid_vel = vec3<f32>(0.0);
    var momentum_velocity_mass = vec4<f32>(0.0);
    var velocity_gradient = mat3x3<f32>(vec3(0.0), vec3(0.0), vec3(0.0));
#endif

    // G2P
    {
        let particle_cdf = particles_cdf[particle_id];
        let particle_pos = particles_pos[particle_id];
        let particle_vel = particles_vel[particle_id].v;

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

        for (var i = 0u; i < Kernel::NBH_LEN; i += 1u) {
            let shift = NBH_SHIFTS[i];
            let packed_shift = NBH_SHIFTS_SHARED[i];
            let shared_id = packed_cell_index_in_block + packed_shift;
            let cell_data = shared_nodes[shared_id].momentum_velocity_mass;
            let cell_cdf = shared_nodes_cdf[shared_id];
            let is_compatible = Grid::affinities_are_compatible(particle_cdf.affinity, cell_cdf.affinities);

#if DIM == 2
            let dpt = ref_elt_pos_minus_particle_pos + vec2<f32>(shift) * cell_width;
#else
            let dpt = ref_elt_pos_minus_particle_pos + vec3<f32>(shift) * cell_width;
#endif

            let body_vel = body_vels[cell_cdf.closest_id]; // TODO: invalid if there is no body.
            let body_com = body_mprops[cell_cdf.closest_id].com;
            let cell_center = dpt + particle_pos.pt;
            let body_pt_vel =  Body::velocity_at_point(body_com, body_vel, cell_center);
            let particle_ghost_vel = body_pt_vel + Grid::project_velocity(particle_vel - body_pt_vel, particle_cdf.normal);

#if DIM == 2
            let cpic_cell_data = select(vec3(particle_ghost_vel, cell_data.z), cell_data, is_compatible);
            let weight = w.x[shift.x] * w.y[shift.y];
            momentum_velocity_mass += cpic_cell_data * weight;
            velocity_gradient += (weight * inv_d) * outer_product(cpic_cell_data.xy, dpt);
#else
            let cpic_cell_data = select(vec4(particle_ghost_vel, cell_data.w), cell_data, is_compatible);
            let weight = w.x[shift.x] * w.y[shift.y] * w.z[shift.z];
            momentum_velocity_mass += cpic_cell_data * weight;
            velocity_gradient += (weight * inv_d) * outer_product(cpic_cell_data.xyz, dpt);
#endif
        }

        for (var i = 0u; i < 16u; i++) {
            if Grid::affinity_bit(i, particle_cdf.affinity) {
                let body_vel = body_vels[i];
                let body_com = body_mprops[i].com;
                rigid_vel += Body::velocity_at_point(body_com, body_vel, particle_pos.pt);
            }
        }
    }

    particles_cdf[particle_id].rigid_vel = rigid_vel;
    // Set the particle velocity, and store the velocity gradient into the affine matrix.
    // The rest will be dealt with in the particle update kernel(s).
    particles_affine[particle_id] = velocity_gradient;
#if DIM == 2
    particles_vel[particle_id].v = momentum_velocity_mass.xy;
#else
    particles_vel[particle_id].v = momentum_velocity_mass.xyz;
#endif
}

// TODO: upstream to wgebra?
#if DIM == 2
fn outer_product(a: vec2<f32>, b: vec2<f32>) -> mat2x2<f32> {
    return mat2x2(
        a * b.x,
        a * b.y,
    );
}

// Note that this is different from p2g. We don’t need to shift the index since the truncated
// blocks (the neighbor blocks) are in the quadrants with larger indices.
fn flatten_shared_index(x: u32, y: u32) -> u32 {
    return x + y * 10;
}
#else
fn outer_product(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return mat3x3(
        a * b.x,
        a * b.y,
        a * b.z,
    );
}


// Note that this is different from p2g. We don’t need to shift the index since the truncated
// blocks (the neighbor blocks) are in the octants with larger indices.
fn flatten_shared_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * 6 + z * 6 * 6;
}
#endif

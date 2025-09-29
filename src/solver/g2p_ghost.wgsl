#define_import_path wgsparkl::solver::g2p_ghost

#import wgsparkl::solver::params as Params;
#import wgsparkl::solver::particle as Particle;
#import wgsparkl::grid::kernel as Kernel;
#import wgsparkl::grid::grid as Grid;
#import wgsparkl::models::linear_elasticity as ConstitutiveModel;
#import wgsparkl::models::drucker_prager as DruckerPrager;
#import wgrapier::body as Body;

// TODO: Add particle_cdf as well.
@group(1) @binding(0)
var<storage, read_write> particles_pos: array<Particle::Position>;
@group(1) @binding(1)
var<uniform> params: Params::SimulationParams;
@group(1) @binding(2)
var<storage, read> orig_particles_pos: array<Particle::Position>;

@group(2) @binding(0)
var<storage, read> body_vels: array<Body::Velocity>;
@group(2) @binding(1)
var<storage, read> body_mprops: array<Body::MassProperties>;

@compute @workgroup_size(64, 1, 1)
fn g2p_ghost(
    @builtin(global_invocation_id) dispatch_id: vec3<u32>,
) {
    let particle_id = dispatch_id.x;
    if particle_id >= arrayLength(&particles_pos) {
        return;
    }

    var NBH_SHIFTS = Kernel::NBH_SHIFTS;

    // var rigid_vel = vec3<f32>(0.0);
#if DIM == 2
    var velocity = vec2<f32>(0.0);
#else
    var velocity = vec3<f32>(0.0);
#endif

    let particle_pos = particles_pos[particle_id];

    let cell_width = Grid::grid.cell_width;
    let dt = params.dt;

    let inv_d = Kernel::inv_d(cell_width);
    let ref_elt_pos_minus_particle_pos = Particle::dir_to_associated_grid_node(particle_pos, cell_width);
    let w = Kernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

#if DIM == 2
    let assoc_cell = vec2<i32>(round(particle_pos.pt / cell_width) - 1.0);
#else
    let assoc_cell = vec3<i32>(round(particle_pos.pt / cell_width) - 1.0);
#endif

    for (var i = 0u; i < Kernel::NBH_LEN; i += 1u) {
        let shift = NBH_SHIFTS[i];
#if DIM == 2
        let cell = get_node(assoc_cell + vec2<i32>(shift));
#else
        let cell = get_node(assoc_cell + vec3<i32>(shift));
#endif
        let cell_data = cell.momentum_velocity_mass;
        // let cell_cdf = cell.cdf;
        // let is_compatible = Grid::affinities_are_compatible(particle_cdf.affinity, cell_cdf.affinities);

#if DIM == 2
        let dpt = ref_elt_pos_minus_particle_pos + vec2<f32>(shift) * cell_width;
#else
        let dpt = ref_elt_pos_minus_particle_pos + vec3<f32>(shift) * cell_width;
#endif

        var cpic_cell_data = cell_data;

        // if !is_compatible {
        //     if cell_cdf.closest_id != Grid::NONE {
        //         let body_vel = body_vels[cell_cdf.closest_id]; // TODO: invalid if there is no body.
        //         let body_com = body_mprops[cell_cdf.closest_id].com;
        //         let cell_center = dpt + particle_pos.pt;
        //         let body_pt_vel =  Body::velocity_at_point(body_com, body_vel, cell_center);
        //         let particle_ghost_vel = body_pt_vel + Grid::project_velocity(particle_vel - body_pt_vel, particle_cdf.normal);
        // 
        //         cpic_cell_data = vec4(particle_ghost_vel, cell_data.w);
        //     } else {
        //         // If there is no adjacent collider, the ghost vel is the particle vel.
        //         cpic_cell_data = vec4(particle_vel, cell_data.w);
        //     }
        // }

#if DIM == 2
        let weight = w.x[shift.x] * w.y[shift.y];
        velocity += cpic_cell_data.xy * weight;
#else
        let weight = w.x[shift.x] * w.y[shift.y] * w.z[shift.z];
        velocity += cpic_cell_data.xyz * weight;
#endif
        // velocity_gradient += (weight * inv_d) * outer_product(cpic_cell_data.xyz, dpt);
    }

    if length(velocity) > cell_width / dt {
        velocity = velocity / length(velocity) * cell_width / dt;
    }
    particles_pos[particle_id].pt += velocity * dt;

    // for (var i = 0u; i < 16u; i++) {
    //     if Grid::affinity_bit(i, particle_cdf.affinity) {
    //         let body_vel = body_vels[i];
    //         let body_com = body_mprops[i].com;
    //         rigid_vel += Body::velocity_at_point(body_com, body_vel, particle_pos.pt);
    //     }
    // }

}

#if DIM == 2
fn get_node(pos: vec2<i32>) -> Grid::Node {
    let offset = ((pos % 8) + 8) % 8;
    let block_pos = (pos - offset) / 8;
    let block_header = Grid::find_block_header_id(Grid::BlockVirtualId(block_pos));
    if block_header.id != Grid::NONE {
        let global_chunk_id = Grid::block_header_id_to_physical_id(block_header);
        let global_node_id = Grid::node_id(global_chunk_id, vec2<u32>(offset));
        return Grid::nodes[global_node_id.id];
    } else {
        return Grid::Node(vec3(0.0), Grid::NodeCdf(0.0, 0, Grid::NONE));
    }
}
#else
fn get_node(pos: vec3<i32>) -> Grid::Node {
    let offset = ((pos % 4) + 4) % 4;
    let block_pos = (pos - offset) / 4;
    let block_header = Grid::find_block_header_id(Grid::BlockVirtualId(block_pos));
    if block_header.id != Grid::NONE {
        let global_chunk_id = Grid::block_header_id_to_physical_id(block_header);
        let global_node_id = Grid::node_id(global_chunk_id, vec3<u32>(offset));
        return Grid::nodes[global_node_id.id];
    } else {
        return Grid::Node(vec4(0.0), Grid::NodeCdf(0.0, 0, Grid::NONE));
    }
}
#endif

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

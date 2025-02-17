#define_import_path wgsparkl::solver::particle

struct Position {
    pt: vec3<f32>,
}

struct Dynamics {
    velocity: vec3<f32>,
    def_grad: mat3x3<f32>,
    affine: mat3x3<f32>,
    cdf: Cdf,
    init_volume: f32,
    init_radius: f32,
    mass: f32,
}

struct Cdf {
    // Should we pack this?
    normal: vec3<f32>,
    rigid_vel: vec3<f32>,
    signed_distance: f32,
    affinity: u32,
//    // Index to the closest collider.
//    closest_id: u32,
}

struct RigidParticleIndices {
    triangle: vec3<u32>,
    collider: u32,
}


fn default_cdf() -> Cdf {
    return Cdf(vec3(0.0), vec3(0.0), 0.0, 0);
}

fn closest_grid_pos(part_pos: Position, cell_width: f32) -> vec3<f32> {
    return round(part_pos.pt / cell_width) * cell_width;
}

fn associated_cell_index_in_block_off_by_one(part_pos: Position, cell_width: f32) -> vec3<u32> {
    let assoc_cell = round(part_pos.pt / cell_width) - 1.0;
    let assoc_block = floor(assoc_cell / 4.0) * 4;
    return vec3<u32>(assoc_cell - assoc_block); // Will always be positive.
}

fn associated_grid_pos(part_pos: Position, cell_width: f32) -> vec3<f32> {
    return (round(part_pos.pt / cell_width) - vec3(1.0, 1.0, 1.0)) * cell_width;
}

fn dir_to_closest_grid_node(part_pos: Position, cell_width: f32) -> vec3<f32> {
    return closest_grid_pos(part_pos, cell_width) - part_pos.pt;
}

fn dir_to_associated_grid_node(part_pos: Position, cell_width: f32) -> vec3<f32> {
    return associated_grid_pos(part_pos, cell_width) - part_pos.pt;
}
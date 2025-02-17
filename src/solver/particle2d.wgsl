#define_import_path wgsparkl::solver::particle

struct Position {
    pt: vec2<f32>,
}

struct Dynamics {
    // NOTE: with this arrangement, we have
    //       only a 4-bytes padding at the end
    //       of the struct.
    velocity: vec2<f32>,
    def_grad: mat2x2<f32>,
    affine: mat2x2<f32>,
    cdf: Cdf,
    init_volume: f32,
    init_radius: f32,
    mass: f32,
}

struct RigidParticleIndices {
    segment: vec2<u32>,
    collider: u32,
}

struct Cdf {
    // Should we pack this?
    normal: vec2<f32>,
    rigid_vel: vec2<f32>,
    signed_distance: f32,
    affinity: u32,
//    // Index to the closest collider.
//    closest_id: u32,
}

fn default_cdf() -> Cdf {
    return Cdf(vec2(0.0), vec2(0.0), 0.0, 0);
}

fn closest_grid_pos(part_pos: Position, cell_width: f32) -> vec2<f32> {
    return round(part_pos.pt / cell_width) * cell_width;
}

fn associated_cell_index_in_block_off_by_one(part_pos: Position, cell_width: f32) -> vec2<u32> {
    let assoc_cell = round(part_pos.pt / cell_width) - 1.0;
    let assoc_block = floor(assoc_cell / 8.0) * 8;
    return vec2<u32>(assoc_cell - assoc_block); // Will always be positive.
}

fn associated_grid_pos(part_pos: Position, cell_width: f32) -> vec2<f32> {
    return (round(part_pos.pt / cell_width) - vec2(1.0, 1.0)) * cell_width;
}

fn dir_to_closest_grid_node(part_pos: Position, cell_width: f32) -> vec2<f32> {
    return closest_grid_pos(part_pos, cell_width) - part_pos.pt;
}

fn dir_to_associated_grid_node(part_pos: Position, cell_width: f32) -> vec2<f32> {
    return associated_grid_pos(part_pos, cell_width) - part_pos.pt;
}
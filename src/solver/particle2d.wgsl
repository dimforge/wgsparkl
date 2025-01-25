#define_import_path wgsparkl::solver::particle

struct Position {
    pt: vec2<f32>,
}

struct Velocity {
    v: vec2<f32>,
}

struct Volume {
    def_grad: mat2x2<f32>,
    init_volume_radius: vec2<f32>, // init volume and init radius
    mass: f32,
}

struct Cdf {
    // Should we pack this?
    normal: vec2<f32>,
    signed_distance: f32,
    affinity: u32,
}

fn default_cdf() -> Cdf {
    return Cdf(vec2(0.0), 0.0, 0);
}

fn deformation_gradient(volume: Volume) -> mat2x2<f32> {
    return volume.def_grad;
}

fn set_deformation_gradient(vol: Volume, new_def_grad: mat2x2<f32>) -> Volume {
    return Volume(new_def_grad, vol.init_volume_radius, vol.mass);
}

fn mass(volume: Volume) -> f32 {
    return volume.mass;
}

fn init_volume(volume: Volume) -> f32 {
    return volume.init_volume_radius.x;
}

fn init_radius(volume: Volume) -> f32 {
    return volume.init_volume_radius.y;
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
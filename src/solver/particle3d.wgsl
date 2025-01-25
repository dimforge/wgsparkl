#define_import_path wgsparkl::solver::particle

struct Position {
    pt: vec3<f32>,
}

struct Velocity {
    v: vec3<f32>,
}

struct Volume {
    // First three rows contains the deformation gradient.
    // Fourth row contains (mass, init_volume, init_radius).
    packed: mat3x4<f32>
}

struct Cdf {
    // Should we pack this?
    normal: vec3<f32>,
    signed_distance: f32,
    affinity: u32,
}

fn default_cdf() -> Cdf {
    return Cdf(vec3(0.0), 0.0, 0);
}

fn deformation_gradient(volume: Volume) -> mat3x3<f32> {
    return mat3x3(volume.packed.x.xyz, volume.packed.y.xyz, volume.packed.z.xyz);
}

fn set_deformation_gradient(vol: Volume, new_def_grad: mat3x3<f32>) -> Volume {
    return Volume(
        mat3x4(
            vec4(new_def_grad.x, vol.packed.x.w),
            vec4(new_def_grad.y, vol.packed.y.w),
            vec4(new_def_grad.z, vol.packed.z.w),
        )
    );
}

fn mass(volume: Volume) -> f32 {
    return volume.packed.x.w;
}

fn init_volume(volume: Volume) -> f32 {
    return volume.packed.y.w;
}

fn init_radius(volume: Volume) -> f32 {
    return volume.packed.z.w;
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
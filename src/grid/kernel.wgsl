//! Quadratic kernel.

#define_import_path wgsparkl::grid::kernel

#if DIM == 2
const NBH_LEN: u32 = 9;
const NBH_SHIFTS: array<vec2<u32>, NBH_LEN> = array<vec2<u32>, NBH_LEN>(
    vec2(2, 2),
    vec2(2, 0),
    vec2(2, 1),
    vec2(0, 2),
    vec2(0, 0),
    vec2(0, 1),
    vec2(1, 2),
    vec2(1, 0),
    vec2(1, 1),
);
const NBH_SHIFTS_SHARED: array<u32, NBH_LEN> = array<u32, NBH_LEN>(
    22, 2, 12, 20, 0, 10, 21, 1, 11,
);
#else
const NBH_LEN: u32 = 27;
const NBH_SHIFTS: array<vec3<u32>, NBH_LEN> = array<vec3<u32>, NBH_LEN>(
    vec3(2, 2, 2),
    vec3(2, 0, 2),
    vec3(2, 1, 2),
    vec3(0, 2, 2),
    vec3(0, 0, 2),
    vec3(0, 1, 2),
    vec3(1, 2, 2),
    vec3(1, 0, 2),
    vec3(1, 1, 2),
    vec3(2, 2, 0),
    vec3(2, 0, 0),
    vec3(2, 1, 0),
    vec3(0, 2, 0),
    vec3(0, 0, 0),
    vec3(0, 1, 0),
    vec3(1, 2, 0),
    vec3(1, 0, 0),
    vec3(1, 1, 0),
    vec3(2, 2, 1),
    vec3(2, 0, 1),
    vec3(2, 1, 1),
    vec3(0, 2, 1),
    vec3(0, 0, 1),
    vec3(0, 1, 1),
    vec3(1, 2, 1),
    vec3(1, 0, 1),
    vec3(1, 1, 1),
);
const NBH_SHIFTS_SHARED: array<u32, NBH_LEN> = array<u32, NBH_LEN>(
    86, 74, 80, 84, 72, 78, 85, 73, 79, 14, 2, 8, 12, 0, 6, 13, 1, 7, 50, 38, 44, 48, 36, 42, 49, 37, 43
);
#endif

fn inv_d(cell_width: f32) -> f32 {
    return 4.0 / (cell_width * cell_width);
}

fn eval_all(x: f32) -> vec3<f32> {
    return vec3(
        0.5 * (1.5 - x) * (1.5 - x),
        0.75 - (x - 1.0) * (x - 1.0),
        0.5 * (x - 0.5) * (x - 0.5)
    );
}

fn eval(x: f32) -> f32 {
    let x_abs = abs(x);
    let part1 = 0.75 - x_abs * x_abs;
    let part2 = 0.5 * (1.5 - x_abs) * (1.5 - x_abs);
    let part3 = 0.0;
    return select(select(part3, part2, x_abs < 1.5), part1, x_abs < 0.5);
}

fn eval_derivative(x: f32) -> f32 {
    let x_abs = abs(x);
    let part1 = -2.0 * sign(x) * x_abs;
    let part2 = -sign(x) * (1.5 - x_abs);
    let part3 = 0.0;
    return select(select(part3, part2, x_abs < 1.5), part1, x_abs < 0.5);
}

#if DIM == 2
fn precompute_weights(
    ref_elt_pos_minus_particle_pos: vec2<f32>,
    h: f32,
) -> mat2x3<f32> {
    return mat2x3(
        eval_all(-ref_elt_pos_minus_particle_pos.x / h),
        eval_all(-ref_elt_pos_minus_particle_pos.y / h),
    );
}
#else
fn precompute_weights(
    ref_elt_pos_minus_particle_pos: vec3<f32>,
    h: f32,
) -> mat3x3<f32> {
    return mat3x3(
        eval_all(-ref_elt_pos_minus_particle_pos.x / h),
        eval_all(-ref_elt_pos_minus_particle_pos.y / h),
        eval_all(-ref_elt_pos_minus_particle_pos.z / h),
    );
}
#endif
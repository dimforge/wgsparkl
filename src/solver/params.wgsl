#define_import_path wgsparkl::solver::params

struct SimulationParams {
#if DIM == 2
    gravity: vec2<f32>,
    padding: f32, // Due to uniform size limits
#else
    gravity: vec3<f32>,
#endif
    dt: f32,
}
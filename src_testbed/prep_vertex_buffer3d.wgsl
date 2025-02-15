#define_import_path wgsparkl::examples::prep_vertex_buffer

#import wgsparkl::solver::particle as Particle;
#import wgsparkl::grid::grid as Grid;
#import wgsparkl::solver::params as Params;
#import wgebra::svd3 as Svd3;

@group(0) @binding(0)
var<storage, read_write> instances: array<InstanceData>;
@group(0) @binding(1)
var<storage, read> particles_pos: array<Particle::Position>;
@group(0) @binding(2)
var<storage, read> particles_vol: array<Particle::Volume>;
@group(0) @binding(3)
var<storage, read> particles_vel: array<Particle::Velocity>;
@group(0) @binding(4)
var<storage, read> particles_cdf: array<Particle::Cdf>;
@group(0) @binding(5)
var<storage, read_write> grid: Grid::Grid;
@group(0) @binding(6)
var<uniform> params: Params::SimulationParams;
@group(0) @binding(7)
var<storage, read> config: RenderConfig;

struct RenderConfig {
    mode: u32,
}

const DEFAULT: u32 = 0;
const VOLUME: u32 = 1;
const VELOCITY: u32 = 2;
const CDF_NORMALS: u32 = 3;
const CDF_DISTANCES: u32 = 4;
const CDF_SIGNS: u32 = 5;


struct InstanceData {
    deformation: mat3x3<f32>,
    position: vec3<f32>,
    base_color: vec4<f32>,
    color: vec4<f32>,
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) tid: vec3<u32>,
) {
    let particle_id = tid.x;

    if particle_id < arrayLength(&instances) {
        let def_grad = Particle::deformation_gradient(particles_vol[particle_id]);
        instances[particle_id].deformation = def_grad;
        instances[particle_id].position = particles_pos[particle_id].pt;

        let color = instances[particle_id].base_color;
        let cell_width = grid.cell_width;
        let dt = params.dt;
        let max_vel = cell_width / dt;

        if config.mode == DEFAULT {
            instances[particle_id].color = color;
        } else if config.mode == VELOCITY {
            let vel = particles_vel[particle_id].v;
            instances[particle_id].color = vec4(abs(vel) * dt * 100.0 + vec3(0.2), color.w);
        } else if config.mode == VOLUME {
            let svd = Svd3::svd(def_grad);
            let color_xyz = (vec3(1.0) - svd.S) / 0.005 + vec3(0.2);
            instances[particle_id].color = vec4(color_xyz, color.w);
        } else if config.mode == CDF_NORMALS {
            let particle_normal = particles_cdf[particle_id].normal;
            if all(particle_normal == vec3(0.0)) {
                instances[particle_id].color = vec4(0.0, 0.0, 0.0, color.w);
            } else {
                let n = (particle_normal + vec3(1.0)) / 2.0;
                instances[particle_id].color = vec4(n.x, n.y, n.z, color.w);
            }
        } else if config.mode == CDF_DISTANCES {
            let d = particles_cdf[particle_id].signed_distance / (cell_width * 1.5);
            if d > 0.0 {
                instances[particle_id].color = vec4(0.0, abs(d), 0.0, color.w);
            } else {
                instances[particle_id].color = vec4(abs(d), 0.0, 0.0, color.w);
            }
        } else if config.mode == CDF_SIGNS {
             let d = particles_cdf[particle_id].affinity;
             let a = (d >> 16) & (d & 0x0000ffff);
             if d == 0 {
                 instances[particle_id].color = vec4(0.0, 0.0, 0.0, color.w);
             } else if a == 0 {
                 instances[particle_id].color = vec4(0.0, 1.0, 0.0, color.w);
             } else {
                 instances[particle_id].color = vec4(1.0, 0.0, 0.0, color.w);
             }
         }
    }
}


@compute @workgroup_size(64, 1, 1)
fn main_rigid_particles(
    @builtin(global_invocation_id) tid: vec3<u32>,
) {
    let particle_id = tid.x;

    if particle_id < arrayLength(&instances) {
        instances[particle_id].deformation = mat3x3f(
            0.4, 0.0, 0.0,
            0.0, 0.4, 0.0,
            0.0, 0.0, 0.4
        );
        instances[particle_id].position = particles_pos[particle_id].pt;
        instances[particle_id].color = instances[particle_id].base_color;
    }
}
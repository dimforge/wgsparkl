#define_import_path wgsparkl::solver::particle_update

#import wgsparkl::solver::params as Params;
#import wgsparkl::solver::particle as Particle;
#import wgsparkl::grid::kernel as Kernel;
#import wgsparkl::grid::grid as Grid;
#import wgsparkl::models::neo_hookean_elasticity as ConstitutiveModel;
//#import wgsparkl::models::linear_elasticity as ConstitutiveModel;
#import wgsparkl::models::drucker_prager as DruckerPrager;
#import wgebra::svd2 as Svd2
#import wgebra::svd3 as Svd3

#if DIM == 2
#import wgebra::sim2 as Pose;
#else
#import wgebra::sim3 as Pose;
#endif
#import wgparry::cuboid as Cuboid;

@group(1) @binding(0)
var<storage, read_write> particles_pos: array<Particle::Position>;
@group(1) @binding(1)
var<storage, read_write> particles_vel: array<Particle::Velocity>;
@group(1) @binding(2)
var<storage, read_write> particles_vol: array<Particle::Volume>;
#if DIM == 2
@group(1) @binding(3)
var<storage, read_write> particles_affine: array<mat2x2<f32>>;
#else
@group(1) @binding(3)
var<storage, read_write> particles_affine: array<mat3x3<f32>>;
#endif
@group(1) @binding(4)
var<storage, read> particles_cdf: array<Particle::Cdf>;
@group(1) @binding(5)
var<storage, read> constitutive_model: array<ConstitutiveModel::ElasticCoefficients>;
@group(1) @binding(6)
var<storage, read> plasticity: array<DruckerPrager::Plasticity>;
@group(1) @binding(7)
var<storage, read_write> plastic_state: array<DruckerPrager::PlasticState>;
@group(1) @binding(8)
var<storage, read_write> phases: array<Phase>;
@group(1) @binding(9)
var<uniform> params: Params::SimulationParams;

@group(2) @binding(0)
var<storage, read> collision_shapes: array<Cuboid::Cuboid>;
@group(2) @binding(1)
var<storage, read> collision_shape_poses: array<Transform>;

struct Phase {
    phase: f32,
    max_stretch: f32,
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let particle_id = gid.x;

    if particle_id >= arrayLength(&particles_pos) {
        return;
    }

    let dt = params.dt;
    let cell_width = Grid::grid.cell_width;
    let velocity_gradient = particles_affine[particle_id]; // The velocity gradient was stored in the affine buffer.
    var new_particle_vel = particles_vel[particle_id].v;
    let particle_pos = particles_pos[particle_id].pt;
    let cdf = particles_cdf[particle_id];
    let curr_particle_vol = particles_vol[particle_id];

    /*
     * Advection.
     */
    if cdf.signed_distance < -0.05 * cell_width {
        new_particle_vel = cdf.rigid_vel + Grid::project_velocity((new_particle_vel - cdf.rigid_vel), cdf.normal);
    }
    let new_particle_pos = particle_pos + new_particle_vel * dt;

    /*
     * Penalty impulse.
     */
     const PENALTY_COEFF: f32 = 1.0e4;
     if cdf.signed_distance < -0.05 * cell_width && cdf.signed_distance > -0.3 * cell_width {
         let impulse = (dt * -cdf.signed_distance * PENALTY_COEFF) * cdf.normal;
         // new_particle_vel += impulse / curr_particle_vol.mass;
     }

    /*
     * Deformation gradient update.
     */
    let curr_deformation_gradient = Particle::deformation_gradient(curr_particle_vol);
    var new_deformation_gradient = curr_deformation_gradient +
       (velocity_gradient * dt) * curr_deformation_gradient;

    /*
     * Constitutive model.
     */
    var phase = phases[particle_id].phase;

    // Update Phase.
    // TODO: should be stress based instead.
    let max_stretch = phases[particle_id].max_stretch;
    if phase > 0.0 && max_stretch > 0.0 {
    #if DIM == 2
        let svd = Svd2::svd(new_deformation_gradient);
        if svd.S.x > max_stretch || svd.S.y > max_stretch {
            phases[particle_id].phase = 0.0;
            phase = 0.0;
        }
    #else
        let svd = Svd3::svd(new_deformation_gradient);
        if svd.S.x > max_stretch || svd.S.y > max_stretch || svd.S.z > max_stretch {
            phases[particle_id].phase = 0.0;
            phase = 0.0;
        }
    #endif
    }

    // Plasticity.
    if phase == 0.0 {
        let projection = DruckerPrager::project(plasticity[particle_id], plastic_state[particle_id], new_deformation_gradient);
        plastic_state[particle_id] = projection.state;
        new_deformation_gradient = projection.deformation_gradient;
    }

    // Elasticity.
    let stress = ConstitutiveModel::kirchoff_stress(constitutive_model[particle_id], new_deformation_gradient);

    /*
     * Affine matrix for APIC transfer.
     */
    let inv_d = Kernel::inv_d(cell_width);
    let volume0 = Particle::init_volume(curr_particle_vol);
    let mass = Particle::mass(curr_particle_vol);
    let affine = velocity_gradient * mass - stress * (volume0 * inv_d * dt);

    /*
     * Write back the new particle properties.
     */
    particles_pos[particle_id].pt = new_particle_pos;
    particles_vel[particle_id].v = new_particle_vel;
    particles_vol[particle_id] = Particle::set_deformation_gradient(curr_particle_vol, new_deformation_gradient);
    particles_affine[particle_id] = affine;
}

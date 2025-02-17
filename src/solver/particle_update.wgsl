#define_import_path wgsparkl::solver::particle_update

#import wgsparkl::solver::params as Params;
#import wgsparkl::solver::particle as Particle;
#import wgsparkl::grid::kernel as Kernel;
#import wgsparkl::grid::grid as Grid;
//#import wgsparkl::models::neo_hookean_elasticity as ConstitutiveModel;
#import wgsparkl::models::linear_elasticity as ConstitutiveModel;
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
var<storage, read_write> particles_dyn: array<Particle::Dynamics>;
@group(1) @binding(2)
var<storage, read> constitutive_model: array<ConstitutiveModel::ElasticCoefficients>;
@group(1) @binding(3)
var<storage, read> plasticity: array<DruckerPrager::Plasticity>;
@group(1) @binding(4)
var<storage, read_write> plastic_state: array<DruckerPrager::PlasticState>;
@group(1) @binding(5)
var<storage, read_write> phases: array<Phase>;
@group(1) @binding(6)
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
    let dynamics = particles_dyn[particle_id];
    let particle_pos = particles_pos[particle_id].pt;
    var new_particle_vel = dynamics.velocity;

    /*
     * Advection.
     */
    if dynamics.cdf.signed_distance < -0.05 * cell_width {
        new_particle_vel = dynamics.cdf.rigid_vel + Grid::project_velocity((new_particle_vel - dynamics.cdf.rigid_vel), dynamics.cdf.normal);
    }

    // Clamp the max velocity a particle can get.
    // TODO: clamp the grid velocities instead?
    if length(new_particle_vel) > cell_width / dt {
        new_particle_vel = new_particle_vel / length(new_particle_vel) * cell_width / dt;
    }

    let new_particle_pos = particle_pos + new_particle_vel * dt;

    /*
     * Penalty impulse.
     */
     const PENALTY_COEFF: f32 = 1.0e3;
     if dynamics.cdf.signed_distance < -0.05 * cell_width { // && dynamics.cdf.signed_distance > -0.3 * cell_width {
         let corrected_dist = max(dynamics.cdf.signed_distance, -0.3 * cell_width);
         let impulse = (dt * -corrected_dist * PENALTY_COEFF) * dynamics.cdf.normal;
         new_particle_vel += impulse; // / curr_particle_vol.mass;
     }

    /*
     * Deformation gradient update.
     */
    // NOTE: the velocity gradient was stored in the affine buffer.
    var new_deformation_gradient = dynamics.def_grad +
       (dynamics.affine * dt) * dynamics.def_grad;

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
    // NOTE: the velocity gradient was stored in the affine buffer.
    let affine = dynamics.affine * dynamics.mass - stress * (dynamics.init_volume * inv_d * dt);

    /*
     * Write back the new particle properties.
     */
    particles_pos[particle_id].pt = new_particle_pos;
    particles_dyn[particle_id].velocity = new_particle_vel;
    particles_dyn[particle_id].def_grad = new_deformation_gradient;
    particles_dyn[particle_id].affine = affine;
}

#define_import_path wgsparkl::solver::impulse

#import wgsparkl::solver::params as Params;
#import wgrapier::body as Body;
#import wgsparkl::grid::grid as Grid;

#if DIM == 2
    #import wgebra::sim2 as Pose;
#else
    #import wgebra::sim3 as Pose;
#endif

struct IntegerImpulse {
#if DIM == 2
    linear: vec2<i32>,
    angular: i32,
#else
    linear: vec3<i32>,
    angular: vec3<i32>,
#endif
}

struct IntegerImpulseAtomic {
#if DIM == 2
    linear_x: atomic<i32>,
    linear_y: atomic<i32>,
    angular: atomic<i32>,
    padding: i32,
#else
    linear_x: atomic<i32>,
    linear_y: atomic<i32>,
    linear_z: atomic<i32>,
    padding_a: i32,
    angular_x: atomic<i32>,
    angular_y: atomic<i32>,
    angular_z: atomic<i32>,
    padding_b: i32,
#endif
}


const FLOAT_TO_INT_FACTOR: f32 = 1e5;

fn flt2int(flt: f32) -> i32 {
    return i32(flt * FLOAT_TO_INT_FACTOR);
}

fn int2flt(i: i32) -> f32 {
    return f32(i) / FLOAT_TO_INT_FACTOR;
}

fn int_impulse_to_float(imp: IntegerImpulse) -> Body::Impulse {
#if DIM == 2
    return Body::Impulse(
        vec2(int2flt(imp.linear.x), int2flt(imp.linear.y)),
        int2flt(imp.angular)
    );
#else
    return Body::Impulse(
        vec3(int2flt(imp.linear.x), int2flt(imp.linear.y), int2flt(imp.linear.z)),
        vec3(int2flt(imp.angular.x), int2flt(imp.angular.y), int2flt(imp.angular.z))
    );
#endif
}

fn float_impulse_to_int(imp: Body::Impulse) -> IntegerImpulse {
#if DIM == 2
    return IntegerImpulse(
        vec2(flt2int(imp.linear.x), flt2int(imp.linear.y)),
        flt2int(imp.angular)
    );
#else
    return IntegerImpulse(
        vec3(flt2int(imp.linear.x), flt2int(imp.linear.y), flt2int(imp.linear.z)),
        vec3(flt2int(imp.angular.x), flt2int(imp.angular.y), flt2int(imp.angular.z))
    );
#endif
}

@group(0) @binding(1)
var<storage, read_write> incremental_impulses: array<IntegerImpulse>;
@group(0) @binding(2)
var<storage, read_write> vels: array<Body::Velocity>;
@group(0) @binding(3)
var<storage, read_write> local_mprops: array<Body::MassProperties>;
@group(0) @binding(4)
var<storage, read_write> mprops: array<Body::MassProperties>;
#if DIM == 2
@group(0) @binding(5)
var<storage, read_write> poses: array<Pose::Sim2>;
#else
@group(0) @binding(5)
var<storage, read_write> poses: array<Pose::Sim3>;
#endif
@group(0) @binding(6)
var<uniform> params: Params::SimulationParams;

// NOTE: this is set to 16 exactly becaure we are currently limited to 16 bodies
//       due to the CPIC affinity bitmask size.
@compute @workgroup_size(16, 1, 1)
fn update(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let id = gid.x;

    if id < arrayLength(&vels) {
        var inc_impulse = int_impulse_to_float(incremental_impulses[id]);

        // Reset the incremental impulse to zero for the next substep.
#if DIM == 2
        incremental_impulses[id] = IntegerImpulse(vec2(0i), 0i);
#else
        incremental_impulses[id] = IntegerImpulse(vec3(0i), vec3(0i));
#endif

        // Apply impulse and integrate
        var new_vel = Body::applyImpulse(mprops[id], vels[id], inc_impulse);

        // Cap the velocities to not move more than a fraction of a cell-width in a given substep.
        let linvel_norm = length(new_vel.linear);
        let angvel_norm = length(new_vel.angular);
        let lin_limit = 0.1 * Grid::grid.cell_width / params.dt;
        let ang_limit = 1.0; // TODO: what’s a good angular limit?

        if (length(inc_impulse.linear) != 0.0 || length(inc_impulse.angular) != 0.0) {
            if linvel_norm > lin_limit {
                new_vel.linear = new_vel.linear * (lin_limit / linvel_norm);
            }
            if angvel_norm > ang_limit {
                new_vel.angular = new_vel.angular * (ang_limit / angvel_norm);
            }
        }

        var new_pose = Body::integrateVelocity(poses[id], new_vel, local_mprops[id].com, params.dt);
        let new_mprops = Body::updateMprops(new_pose, local_mprops[id]);

        // Apply gravity.
        let mass_mask = Vector(mprops[id].inv_mass != Vector(0.0));
        new_vel.linear += params.gravity * mass_mask * params.dt;

        vels[id] = new_vel;
        mprops[id] = new_mprops;
        poses[id] = new_pose;
    }
}
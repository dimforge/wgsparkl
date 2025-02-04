#define_import_path wgsparkl::solver::impulse

#import wgsparkl::solver::params as Params;
#import wgrapier::body as Body;

#if DIM == 2
    #import wgebra::sim2 as Pose;
#else
    #import wgebra::sim3 as Pose;
#endif

struct IntegerImpulse {
    linear: vec2<i32>,
    angular: i32,
}

struct IntegerImpulseAtomic {
    linear_x: atomic<i32>,
    linear_y: atomic<i32>,
    angular: atomic<i32>,
    padding: i32,
}


const FLOAT_TO_INT_FACTOR: f32 = 1e5;

fn flt2int(flt: f32) -> i32 {
    return i32(flt * FLOAT_TO_INT_FACTOR);
}

fn int2flt(i: i32) -> f32 {
    return f32(i) / FLOAT_TO_INT_FACTOR;
}

fn int_impulse_to_float(imp: IntegerImpulse) -> Body::Impulse {
    return Body::Impulse(
        vec2(int2flt(imp.linear.x), int2flt(imp.linear.y)),
        int2flt(imp.angular)
    );
}

fn float_impulse_to_int(imp: Body::Impulse) -> IntegerImpulse {
    return IntegerImpulse(
        vec2(flt2int(imp.linear.x), flt2int(imp.linear.y)),
        flt2int(imp.angular)
    );
}

@group(0) @binding(0)
var<storage, read_write> total_impulses: array<Body::Impulse>;
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
        let inc_impulse = int_impulse_to_float(incremental_impulses[id]);

        total_impulses[id].linear += inc_impulse.linear;
        total_impulses[id].angular += inc_impulse.angular;

        // Reset the incremental impulse to zero for the next substep.
        incremental_impulses[id] = IntegerImpulse(vec2(0i), 0i);

        // Apply impulse and integrate
        var new_vel = Body::applyImpulse(mprops[id], vels[id], inc_impulse);

        if mprops[id].inv_mass.x > 0.0 { // TODO: has a body flags bitfield?
            new_vel.linear += params.gravity * params.dt;
        }

        let new_pose = Body::integrateVelocity(poses[id], new_vel, local_mprops[id].com, params.dt);
        let new_mprops = Body::updateMprops(new_pose, local_mprops[id]);

        vels[id] = new_vel;
        mprops[id] = new_mprops;
        poses[id] = new_pose;
    }
}

@compute @workgroup_size(16, 1, 1)
fn reset(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let id = gid.x;
    total_impulses[id] = Body::Impulse(vec2(0.0), 0.0);
}

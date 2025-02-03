#define_import_path wgsparkl::solver::impulse

#import wgrapier::body as Body;

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
var<storage, read_write> body_vels: array<Body::Velocity>;
@group(0) @binding(3)
var<storage, read_write> body_mprops: array<Body::MassProperties>;

// NOTE: this is set to 16 exactly becaure we are currently limited to 16 bodies
//       due to the CPIC affinity bitmask size.
@compute @workgroup_size(16, 1, 1)
fn update(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let id = gid.x;

    if id < arrayLength(&body_vels) {
        let body_vel = body_vels[id];
        let body_mprops = body_mprops[id];
        let inc_impulse = int_impulse_to_float(incremental_impulses[id]);

        total_impulses[id].linear += inc_impulse.linear;
        total_impulses[id].angular += inc_impulse.angular;

        // Reset the incremental impulse to zero for the next substep.
        incremental_impulses[id] = IntegerImpulse(vec2(0i), 0i);
//        body_vels[id] = Body::applyImpulse(body_mprops, body_vel, inc_impulse);
    }
}

@compute @workgroup_size(16, 1, 1)
fn reset(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let id = gid.x;
    total_impulses[id] = Body::Impulse(vec2(0.0), 0.0);
}

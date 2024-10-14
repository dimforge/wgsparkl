#define_import_path wgsparkl::collision::collide
#import wgparry::cuboid as Cuboid;
#import wgrapier::body as Body;

#if DIM == 2
#import wgebra::sim2 as Pose;
#else
#import wgebra::sim3 as Pose;
#endif


@group(2) @binding(0)
var<storage, read> collision_shapes: array<Cuboid::Cuboid>;
@group(2) @binding(1)
var<storage, read> collision_shape_poses: array<Transform>;
//@group(2) @binding(2)
//var<storage, read> body_vels: array<Body::Velocity>;
//@group(2) @binding(3)
//var<storage, read> body_mprops: array<Body::MassProperties>;


fn collide(point: Vector, velocity: Vector) -> Vector {
    var result_vel = velocity;

    // TODO: don’t  rely on the array length, e.g., if the user wants to
    //       preallocate the array to add more dynamically.
    for (var i = 0u; i < arrayLength(&collision_shapes); i++) {
        let shape = collision_shapes[i];
        let shape_pose = collision_shape_poses[i];
        let proj = Cuboid::projectPointOnBoundary(shape, shape_pose, point);

        // TODO: make a branchless version of this?
        if proj.is_inside {
            // Apply the unilateral constraint (prevent further penetrations).
            let dpt = proj.point - point;
            let dist = length(dpt);

            if dist > 0.0 {
                let normal = dpt / dist;
                let signed_normal_vel = dot(normal, velocity);

                if signed_normal_vel < 0.0 {
                    let tangent_vel = velocity - signed_normal_vel * normal;
                    let tangent_vel_norm = length(tangent_vel);

                    if tangent_vel_norm > 1.0e-10 {
                        let friction = 0.7; // TODO: make this configurable
                        result_vel = tangent_vel / tangent_vel_norm * max(tangent_vel_norm + signed_normal_vel * friction, 0.0f);
                    } else {
                        result_vel = Vector(0.0);
                    }
                }
            }
        }
    }

    return result_vel;
}
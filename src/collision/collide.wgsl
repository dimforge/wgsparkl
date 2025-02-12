#define_import_path wgsparkl::collision::collide
#import wgparry::shape as Shape;
#import wgrapier::body as Body;
#import wgsparkl::grid::grid as Grid;

#if DIM == 2
#import wgebra::sim2 as Pose;
#else
#import wgebra::sim3 as Pose;
#endif


@group(2) @binding(0)
var<storage, read> collision_shapes: array<Shape::Shape>;
@group(2) @binding(1)
var<storage, read> collision_shape_poses: array<Transform>;
//@group(2) @binding(2)
//var<storage, read> body_vels: array<Body::Velocity>;
//@group(2) @binding(3)
//var<storage, read> body_mprops: array<Body::MassProperties>;


fn collide(cell_width: f32, point: Vector) -> Grid::NodeCdf {
    const MAX_FLT: f32 = 1.0e10; // Is the f32::MAX constant defined somewhere in WGSL?
    var cdf = Grid::NodeCdf(MAX_FLT, 0u, 0u);

#if DIM == 2
    let dist_cap = vec2(cell_width * 1.5);
#else
    let dist_cap = vec3(cell_width * 1.5);
#endif

    // TODO: don’t  rely on the array length, e.g., if the user wants to
    //       preallocate the array to add more dynamically.
    for (var i = 0u; i < arrayLength(&collision_shapes); i++) {
        // FIXME: figure out a way to support more than 16 colliders.
        let shape = collision_shapes[i];
        let shape_pose = collision_shape_poses[i];
        let shape_type = Shape::shape_type(shape);
        if shape_type != Shape::SHAPE_TYPE_POLYLINE {
            let proj = Shape::projectPointOnBoundary(shape, shape_pose, point);
            let dpt = proj.point - point;

            if proj.is_inside || all(abs(dpt) <= dist_cap) {
                let dist = length(dpt);
                // TODO: take is_inside into account to select the deepest
                //       penetration as the closest collider?
                cdf.closest_id = select(cdf.closest_id, i, dist < cdf.distance);
                cdf.distance = min(cdf.distance, dist);
                cdf.affinities |= select(0x00000001u, 0x00010001u, proj.is_inside) << i;
            }
        }
    }

    return cdf;
}
#import wgrapier::body as Body;
#import wgsparkl::solver::particle as Particle;

#if DIM == 2
    #import wgebra::sim2 as Pose;
#else
    #import wgebra::sim3 as Pose;
#endif

@group(0) @binding(0)
var<storage, read> local_pts: array<Vector>;
@group(0) @binding(1)
var<storage, read_write> world_pts: array<Vector>;
@group(0) @binding(2)
var<storage, read> rigid_particle_indices: array<Particle::RigidParticleIndices>;
@group(0) @binding(3)
var<storage, read> vertex_collider_ids: array<u32>;
#if DIM == 2
@group(0) @binding(4)
var<storage, read_write> poses: array<Pose::Sim2>;
#else
@group(0) @binding(4)
var<storage, read_write> poses: array<Pose::Sim3>;
#endif

@compute @workgroup_size(64, 1, 1)
fn transform_sample_points(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let id = gid.x;

    if id < arrayLength(&local_pts) {
        let collider_id = rigid_particle_indices[id].collider;
        let pose = poses[collider_id];
        world_pts[id] = Pose::mulPt(pose, local_pts[id]);
    }
}

@compute @workgroup_size(64, 1, 1)
fn transform_shape_points(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let id = gid.x;

    if id < arrayLength(&local_pts) {
        let collider_id = vertex_collider_ids[id];
        let pose = poses[collider_id];
        world_pts[id] = Pose::mulPt(pose, local_pts[id]);
    }
}
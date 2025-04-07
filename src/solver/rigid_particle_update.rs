use crate::dim_shader_defs;
use crate::solver::GpuRigidParticles;
use crate::solver::WgParticle;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};
use wgparry::substitute_aliases;
use wgpu::ComputePipeline;
use wgrapier::dynamics::{GpuBodySet, WgBody};

#[derive(Shader)]
#[shader(
    derive(WgBody, WgSim2, WgSim3, WgParticle),
    src = "rigid_particle_update.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs",
    composable = false
)]
pub struct WgRigidParticleUpdate {
    pub transform_sample_points: ComputePipeline,
    pub transform_shape_points: ComputePipeline,
}

impl WgRigidParticleUpdate {
    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        bodies: &GpuBodySet,
        rigid_particles: &GpuRigidParticles,
    ) {
        KernelInvocationBuilder::new(queue, &self.transform_sample_points)
            .bind_at(
                0,
                [
                    (rigid_particles.local_sample_points.buffer(), 0),
                    (rigid_particles.sample_points.buffer(), 1),
                    (rigid_particles.sample_ids.buffer(), 2),
                    (bodies.poses().buffer(), 4),
                ],
            )
            .queue((rigid_particles.local_sample_points.len() as u32).div_ceil(64));

        KernelInvocationBuilder::new(queue, &self.transform_shape_points)
            .bind_at(
                0,
                [
                    (bodies.shapes_local_vertex_buffers().buffer(), 0),
                    (bodies.shapes_vertex_buffers().buffer(), 1),
                    (bodies.shapes_vertex_collider_id().buffer(), 3),
                    (bodies.poses().buffer(), 4),
                ],
            )
            .queue((bodies.shapes_vertex_buffers().len() as u32).div_ceil(64));
    }
}

wgcore::test_shader_compilation!(WgRigidParticleUpdate, wgcore, crate::dim_shader_defs());

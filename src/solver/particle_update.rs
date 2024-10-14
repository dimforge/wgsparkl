use crate::collision::WgCollide;
use crate::dim_shader_defs;
use crate::grid::grid::{GpuGrid, WgGrid};
use crate::grid::kernel::WgKernel;
use crate::models::{GpuModels, WgDruckerPrager, WgLinearElasticity, WgNeoHookeanElasticity};
use crate::solver::params::{GpuSimulationParams, WgParams};
use crate::solver::GpuParticles;
use crate::solver::WgParticle;
use naga_oil::compose::NagaModuleDescriptor;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::{utils, Shader};
use wgparry::substitute_aliases;
use wgpu::ComputePipeline;
use wgrapier::dynamics::GpuBodySet;

#[derive(Shader)]
#[shader(
    derive(
        WgParams,
        WgParticle,
        WgGrid,
        WgNeoHookeanElasticity,
        WgLinearElasticity,
        WgDruckerPrager,
        WgKernel,
        WgCollide
    ),
    src = "particle_update.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
pub struct WgParticleUpdate {
    pub main: ComputePipeline,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct ParticlePhase {
    pub phase: f32,
    pub max_stretch: f32,
}

impl WgParticleUpdate {
    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        sim_params: &GpuSimulationParams,
        grid: &GpuGrid,
        particles: &GpuParticles,
        models: &GpuModels,
        _bodies: &GpuBodySet,
    ) {
        KernelInvocationBuilder::new(queue, &self.main)
            .bind(0, [grid.meta.buffer()])
            .bind(
                1,
                [
                    particles.positions.buffer(),
                    particles.velocities.buffer(),
                    particles.volumes.buffer(),
                    particles.affines.buffer(),
                    models.linear_elasticity.buffer(),
                    models.drucker_prager_plasticity.buffer(),
                    models.drucker_prager_plastic_state.buffer(),
                    models.phases.buffer(),
                    sim_params.params.buffer(),
                ],
            )
            // .bind(2, [bodies.shapes().buffer(), bodies.poses().buffer()])
            .queue(particles.positions.len().div_ceil(64) as u32);
    }
}

wgcore::test_shader_compilation!(WgParticleUpdate);

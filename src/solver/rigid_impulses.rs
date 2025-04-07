use crate::dim_shader_defs;
use crate::grid::grid::{GpuGrid, WgGrid};
use crate::solver::params::{GpuSimulationParams, WgParams};
use encase::ShaderType;
use rapier::math::{AngVector, Point, Vector};
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::GpuVector;
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};
use wgparry::substitute_aliases;
use wgpu::{BufferUsages, ComputePipeline};
use wgrapier::dynamics::{GpuBodySet, WgBody};

#[derive(Shader)]
#[shader(
    derive(WgBody, WgSim2, WgSim3, WgParams, WgGrid),
    src = "rigid_impulses.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
pub struct WgRigidImpulses {
    pub update: ComputePipeline,
    pub update_world_mass_properties: ComputePipeline,
}

#[derive(ShaderType, Copy, Clone, PartialEq, Debug, Default)]
#[repr(C)]
pub struct RigidImpulse {
    pub com: Point<f32>, // For convenience, to reduce the number of bindings
    pub linear: Vector<f32>,
    pub angular: AngVector<f32>,
}

pub struct GpuImpulses {
    pub incremental_impulses: GpuVector<RigidImpulse>,
    pub total_impulses: GpuVector<RigidImpulse>,
    pub total_impulses_staging: GpuVector<RigidImpulse>,
}

impl GpuImpulses {
    pub fn new(device: &wgpu::Device) -> Self {
        const MAX_BODY_COUNT: usize = 16; // CPIC doesnt support more.
        let impulses = [RigidImpulse::default(); MAX_BODY_COUNT];
        Self {
            incremental_impulses: GpuVector::encase(device, impulses, BufferUsages::STORAGE),
            total_impulses: GpuVector::encase(
                device,
                impulses,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            total_impulses_staging: GpuVector::encase(
                device,
                impulses,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            ),
        }
    }
}

impl WgRigidImpulses {
    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        grid: &GpuGrid,
        sim_params: &GpuSimulationParams,
        impulses: &GpuImpulses,
        bodies: &GpuBodySet,
    ) {
        KernelInvocationBuilder::new(queue, &self.update)
            .bind0([
                grid.meta.buffer(),
                impulses.incremental_impulses.buffer(),
                bodies.vels().buffer(),
                bodies.local_mprops().buffer(),
                bodies.mprops().buffer(),
                bodies.poses().buffer(),
                sim_params.params.buffer(),
            ])
            .queue(1);
    }

    pub fn queue_update_world_mass_properties<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        impulses: &GpuImpulses,
        bodies: &GpuBodySet,
    ) {
        KernelInvocationBuilder::new(queue, &self.update_world_mass_properties)
            .bind_at(
                0,
                [
                    (impulses.incremental_impulses.buffer(), 1),
                    (bodies.local_mprops().buffer(), 3),
                    (bodies.mprops().buffer(), 4),
                    (bodies.poses().buffer(), 5),
                ],
            )
            .queue(1);
    }
}

wgcore::test_shader_compilation!(WgRigidImpulses, wgcore, crate::dim_shader_defs());

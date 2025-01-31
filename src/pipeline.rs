use crate::grid::grid::{GpuGrid, WgGrid};
use crate::grid::prefix_sum::{PrefixSumWorkspace, WgPrefixSum};
use crate::grid::sort::WgSort;
use crate::models::GpuModels;
use crate::solver::{
    GpuParticles, GpuSimulationParams, Particle, SimulationParams, WgG2P, WgG2PCdf, WgGridUpdate,
    WgGridUpdateCdf, WgP2G, WgParticleUpdate,
};
use naga_oil::compose::ComposerError;
use wgcore::hot_reloading::HotReloadState;
use wgcore::kernel::KernelInvocationQueue;
use wgcore::Shader;
use wgpu::Device;
use wgrapier::dynamics::{BodyDesc, GpuBodySet, WgIntegrate};

#[cfg(target_os = "macos")]
use crate::grid::sort::TouchParticleBlocks;

pub struct MpmPipeline {
    grid: WgGrid,
    prefix_sum: WgPrefixSum,
    sort: WgSort,
    #[cfg(target_os = "macos")]
    touch_particle_blocks: TouchParticleBlocks,
    p2g: WgP2G,
    grid_update_cdf: WgGridUpdateCdf,
    grid_update: WgGridUpdate,
    particles_update: WgParticleUpdate,
    g2p: WgG2P,
    g2p_cdf: WgG2PCdf,
    integrate_bodies: WgIntegrate,
}

impl MpmPipeline {
    pub fn init_hot_reloading(&self, state: &mut HotReloadState) {
        WgGrid::watch_sources(state).unwrap(); // TODO: don’t unwrap
        WgPrefixSum::watch_sources(state).unwrap(); // TODO: don’t unwrap
        WgSort::watch_sources(state).unwrap(); // TODO: don’t unwrap
        WgP2G::watch_sources(state).unwrap(); // TODO: don’t unwrap
        WgGridUpdate::watch_sources(state).unwrap(); // TODO: don’t unwrap
        WgGridUpdateCdf::watch_sources(state).unwrap(); // TODO: don’t unwrap
        WgParticleUpdate::watch_sources(state).unwrap(); // TODO: don’t unwrap
        WgG2P::watch_sources(state).unwrap(); // TODO: don’t unwrap
        WgG2PCdf::watch_sources(state).unwrap(); // TODO: don’t unwrap
        WgIntegrate::watch_sources(state).unwrap(); // TODO: don’t unwrap
    }

    pub fn reload_if_changed(
        &mut self,
        device: &Device,
        state: &HotReloadState,
    ) -> Result<bool, ComposerError> {
        let mut changed = false;
        changed = self.grid.reload_if_changed(device, state)? || changed;
        changed = self.prefix_sum.reload_if_changed(device, state)? || changed;
        changed = self.sort.reload_if_changed(device, state)? || changed;
        changed = self.p2g.reload_if_changed(device, state)? || changed;
        changed = self.grid_update.reload_if_changed(device, state)? || changed;
        changed = self.grid_update_cdf.reload_if_changed(device, state)? || changed;
        changed = self.particles_update.reload_if_changed(device, state)? || changed;
        changed = self.g2p.reload_if_changed(device, state)? || changed;
        changed = self.g2p_cdf.reload_if_changed(device, state)? || changed;
        changed = self.integrate_bodies.reload_if_changed(device, state)? || changed;

        Ok(changed)
    }
}

pub struct MpmData {
    pub sim_params: GpuSimulationParams,
    pub grid: GpuGrid,
    pub particles: GpuParticles, // TODO: keep private?
    pub bodies: GpuBodySet,
    prefix_sum: PrefixSumWorkspace,
    models: GpuModels,
}

impl MpmData {
    pub fn new(
        device: &Device,
        params: SimulationParams,
        particles: &[Particle],
        bodies: &[BodyDesc],
        cell_width: f32,
        grid_capacity: u32,
    ) -> Self {
        let sim_params = GpuSimulationParams::new(device, params);
        let models = GpuModels::from_particles(device, particles);
        let particles = GpuParticles::from_particles(device, particles);
        let grid = GpuGrid::with_capacity(device, grid_capacity, cell_width);
        let prefix_sum = PrefixSumWorkspace::with_capacity(device, grid_capacity);
        let bodies = GpuBodySet::new(device, bodies);

        Self {
            sim_params,
            particles,
            bodies,
            grid,
            prefix_sum,
            models,
        }
    }
}

impl MpmPipeline {
    pub fn new(device: &Device) -> Result<Self, ComposerError> {
        Ok(Self {
            grid: WgGrid::from_device(device)?,
            prefix_sum: WgPrefixSum::from_device(device)?,
            sort: WgSort::from_device(device)?,
            p2g: WgP2G::from_device(device)?,
            grid_update: WgGridUpdate::from_device(device)?,
            grid_update_cdf: WgGridUpdateCdf::from_device(device)?,
            particles_update: WgParticleUpdate::from_device(device)?,
            g2p: WgG2P::from_device(device)?,
            g2p_cdf: WgG2PCdf::from_device(device)?,
            integrate_bodies: WgIntegrate::from_device(device)?,
            #[cfg(target_os = "macos")]
            touch_particle_blocks: TouchParticleBlocks::from_device(device),
        })
    }

    pub fn queue_step<'a>(
        &'a self,
        data: &mut MpmData,
        queue: &mut KernelInvocationQueue<'a>,
        add_timestamps: bool,
    ) {
        queue.compute_pass("grid sort", add_timestamps);

        self.grid.queue_sort(
            &data.particles,
            &data.grid,
            &mut data.prefix_sum,
            &self.sort,
            #[cfg(target_os = "macos")]
            &self.touch_particle_blocks,
            &self.prefix_sum,
            queue,
        );

        queue.compute_pass("grid_update_cdf", add_timestamps);

        self.grid_update_cdf
            .queue(queue, &data.sim_params, &data.grid, &data.bodies);

        queue.compute_pass("g2p_cdf", add_timestamps);

        self.g2p_cdf
            .queue(queue, &data.sim_params, &data.grid, &data.particles);

        queue.compute_pass("p2g", add_timestamps);
        self.p2g.queue(queue, &data.grid, &data.particles);

        queue.compute_pass("grid_update", add_timestamps);

        self.grid_update.queue(queue, &data.sim_params, &data.grid);

        queue.compute_pass("g2p", add_timestamps);

        self.g2p.queue(
            queue,
            &data.sim_params,
            &data.grid,
            &data.particles,
            &data.bodies,
        );

        queue.compute_pass("particles_update", add_timestamps);

        self.particles_update.queue(
            queue,
            &data.sim_params,
            &data.grid,
            &data.particles,
            &data.models,
            &data.bodies,
        );

        queue.compute_pass("integrate_bodies", add_timestamps);

        // TODO: should this be in a separate pipeline? Within wgrapier probably?
        self.integrate_bodies.queue(queue, &data.bodies);
    }
}

#[cfg(test)]
#[cfg(feature = "dim3")]
mod test {
    use crate::grid::grid::GpuGrid;
    use crate::grid::prefix_sum::PrefixSumWorkspace;
    use crate::models::WgLinearElasticity;
    use crate::pipeline::{MpmData, MpmPipeline};
    use crate::solver::{GpuParticles, Particle, ParticleMassProps, SimulationParams};
    use nalgebra::{vector, Vector3};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::KernelInvocationQueue;
    use wgpu::Maintain;

    #[futures_test::test]
    #[serial_test::serial]
    async fn pipeline_queue_step() {
        let gpu = GpuInstance::new().await.unwrap();
        let pipeline = MpmPipeline::new(gpu.device());

        let cell_width = 1.0;
        let mut cpu_particles = vec![];
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let position = vector![i as f32, j as f32, k as f32] / cell_width / 2.0;
                    cpu_particles.push(Particle {
                        position,
                        velocity: Vector3::zeros(),
                        volume: ParticleMassProps::new(1.0, cell_width / 4.0),
                        model: LinearElasticity::from_young_modulus(100_000.0, 0.33),
                        plasticity: None,
                        phase: None,
                    });
                }
            }
        }

        let params = SimulationParams {
            gravity: vector![0.0, -9.81, 0.0],
            dt: (1.0 / 60.0) / 10.0,
        };
        let mut data = MpmData::new(gpu.device(), params, &cpu_particles, cell_width, 100_000);
        let mut queue = KernelInvocationQueue::new(gpu.device_arc());
        pipeline.queue_step(&mut data, &mut queue);

        for _ in 0..3 {
            let mut encoder = gpu.device().create_command_encoder(&Default::default());
            queue.encode(&mut encoder);
            let t0 = std::time::Instant::now();
            gpu.queue().submit(Some(encoder.finish()));
            gpu.device().poll(Maintain::Wait);
            println!("Sim step time: {}", t0.elapsed().as_secs_f32());
        }
    }
}

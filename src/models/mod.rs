use crate::solver::{Particle, ParticlePhase};
pub use drucker_prager::{DruckerPrager, DruckerPragerPlasticState, WgDruckerPrager};
pub use linear_elasticity::WgLinearElasticity;
pub use neo_hookean_elasticity::WgNeoHookeanElasticity;
use wgcore::tensor::GpuVector;
use wgpu::{BufferUsages, Device};

mod drucker_prager;
mod linear_elasticity;
mod neo_hookean_elasticity;

pub struct GpuModels {
    pub linear_elasticity: GpuVector<ElasticCoefficients>,
    pub drucker_prager_plasticity: GpuVector<DruckerPrager>,
    pub drucker_prager_plastic_state: GpuVector<DruckerPragerPlasticState>,
    pub phases: GpuVector<ParticlePhase>,
}

impl GpuModels {
    pub fn from_particles(device: &Device, particles: &[Particle]) -> Self {
        let models: Vec<_> = particles.iter().map(|p| p.model).collect();
        let plasticity: Vec<_> = particles
            .iter()
            .map(|p| p.plasticity.unwrap_or(DruckerPrager::new(-1.0, -1.0)))
            .collect();
        let plastic_states: Vec<_> = particles
            .iter()
            .map(|_| DruckerPragerPlasticState::default())
            .collect();
        let phases: Vec<_> = particles
            .iter()
            .map(|p| {
                p.phase.unwrap_or(ParticlePhase {
                    phase: 0.0,
                    max_stretch: -1.0,
                })
            })
            .collect();
        Self {
            linear_elasticity: GpuVector::init(device, &models, BufferUsages::STORAGE),
            drucker_prager_plasticity: GpuVector::init(device, &plasticity, BufferUsages::STORAGE),
            drucker_prager_plastic_state: GpuVector::init(
                device,
                &plastic_states,
                BufferUsages::STORAGE,
            ),
            phases: GpuVector::init(device, &phases, BufferUsages::STORAGE),
        }
    }
}

fn lame_lambda_mu(young_modulus: f32, poisson_ratio: f32) -> (f32, f32) {
    (
        young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)),
        shear_modulus(young_modulus, poisson_ratio),
    )
}

fn shear_modulus(young_modulus: f32, poisson_ratio: f32) -> f32 {
    young_modulus / (2.0 * (1.0 + poisson_ratio))
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct ElasticCoefficients {
    pub lambda: f32,
    pub mu: f32,
}

impl ElasticCoefficients {
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_lambda_mu(young_modulus, poisson_ratio);
        Self { lambda, mu }
    }
}

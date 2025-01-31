use crate::dim_shader_defs;
use crate::models::{DruckerPrager, ElasticCoefficients};
use crate::solver::ParticlePhase;
use encase::ShaderType;
use nalgebra::{Matrix2, Vector2};
use wgcore::tensor::GpuVector;
use wgcore::Shader;
use wgpu::{BufferUsages, Device};

#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
#[repr(C)]
pub struct ParticleMassProps {
    def_grad: Matrix2<f32>,
    init_volume_radius: Vector2<f32>,
    mass: f32,
}

impl ParticleMassProps {
    pub fn new(mass: f32, init_radius: f32) -> Self {
        let exponent = if cfg!(feature = "dim2") { 2 } else { 3 };
        let init_volume = (init_radius * 2.0).powi(exponent); // NOTE: the particles are square-ish.
        Self {
            def_grad: Matrix2::identity(),
            init_volume_radius: Vector2::new(init_volume, init_radius),
            mass,
        }
    }

    pub fn init_radius(&self) -> f32 {
        self.init_volume_radius[1]
    }
}

#[derive(Copy, Clone, PartialEq, Debug, Default, ShaderType)]
#[repr(C)]
pub struct Cdf {
    pub normal: Vector2<f32>,
    pub rigid_vel: Vector2<f32>,
    pub signed_distance: f32,
    pub affinity: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct Particle {
    pub position: Vector2<f32>,
    pub velocity: Vector2<f32>,
    pub volume: ParticleMassProps,
    pub model: ElasticCoefficients,
    pub plasticity: Option<DruckerPrager>,
    pub phase: Option<ParticlePhase>,
}

pub struct GpuParticles {
    pub positions: GpuVector<Vector2<f32>>,
    pub velocities: GpuVector<Vector2<f32>>,
    pub cdf: GpuVector<Cdf>,
    pub cdf_read: GpuVector<Cdf>,
    pub volumes: GpuVector<ParticleMassProps>,
    pub affines: GpuVector<Matrix2<f32>>,
    pub sorted_ids: GpuVector<u32>,
    pub node_linked_lists: GpuVector<u32>,
}

impl GpuParticles {
    pub fn len(&self) -> usize {
        self.positions.len() as usize
    }

    pub fn from_particles(device: &Device, particles: &[Particle]) -> Self {
        let positions: Vec<_> = particles.iter().map(|p| p.position).collect();
        let velocities: Vec<_> = particles.iter().map(|p| p.velocity).collect();
        let volumes: Vec<_> = particles.iter().map(|p| p.volume).collect();
        let cdf = vec![Cdf::default(); particles.len()];

        Self {
            positions: GpuVector::init(device, &positions, BufferUsages::STORAGE),
            velocities: GpuVector::init(device, &velocities, BufferUsages::STORAGE),
            volumes: GpuVector::encase(device, &volumes, BufferUsages::STORAGE),
            sorted_ids: GpuVector::uninit(device, particles.len() as u32, BufferUsages::STORAGE),
            affines: GpuVector::uninit(device, particles.len() as u32, BufferUsages::STORAGE),
            cdf: GpuVector::encase(device, &cdf, BufferUsages::STORAGE | BufferUsages::COPY_SRC),
            cdf_read: GpuVector::encase(
                device,
                &cdf,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            ),
            node_linked_lists: GpuVector::uninit(
                device,
                particles.len() as u32,
                BufferUsages::STORAGE,
            ),
        }
    }
}

#[derive(Shader)]
#[shader(src = "particle2d.wgsl", shader_defs = "dim_shader_defs")]
pub struct WgParticle;

wgcore::test_shader_compilation!(WgParticle);

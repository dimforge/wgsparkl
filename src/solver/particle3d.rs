use crate::dim_shader_defs;
use crate::models::{DruckerPrager, ElasticCoefficients};
use crate::solver::ParticlePhase;
use encase::ShaderType;
use nalgebra::{Matrix4, Matrix4x3, Vector3, Vector4};
use wgcore::tensor::GpuVector;
use wgcore::Shader;
use wgpu::{BufferUsages, Device};

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct ParticleMassProps {
    // First free rows contains the deformation gradient.
    // Bottom row contains (mass, init_volume, init_radius)
    props: Matrix4x3<f32>,
}

impl ParticleMassProps {
    pub fn new(mass: f32, init_radius: f32) -> Self {
        let init_volume = (init_radius * 2.0).powi(3); // NOTE: the particles are square-ish.
        Self {
            #[rustfmt::skip]
            props: Matrix4x3::new(
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
                mass, init_volume, init_radius,
            ),
        }
    }

    pub fn init_radius(&self) -> f32 {
        self.props.m43
    }
}

#[derive(Copy, Clone, PartialEq, Debug, Default, ShaderType)]
#[repr(C)]
pub struct Cdf {
    pub normal: Vector3<f32>,
    pub rigid_vel: Vector3<f32>,
    pub signed_distance: f32,
    pub affinity: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct Particle {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub volume: ParticleMassProps,
    pub model: ElasticCoefficients,
    pub plasticity: Option<DruckerPrager>,
    pub phase: Option<ParticlePhase>,
}

pub struct GpuParticles {
    pub positions: GpuVector<Vector4<f32>>,
    pub velocities: GpuVector<Vector4<f32>>,
    pub cdf: GpuVector<Cdf>,
    pub cdf_read: GpuVector<Cdf>,
    pub volumes: GpuVector<ParticleMassProps>,
    pub affines: GpuVector<Matrix4<f32>>,
    pub sorted_ids: GpuVector<u32>,
    pub node_linked_lists: GpuVector<u32>,
}

impl GpuParticles {
    pub fn len(&self) -> usize {
        self.positions.len() as usize
    }

    pub fn from_particles(device: &Device, particles: &[Particle]) -> Self {
        let positions: Vec<_> = particles.iter().map(|p| p.position.push(0.0)).collect();
        let velocities: Vec<_> = particles.iter().map(|p| p.velocity.push(0.0)).collect();
        let volumes: Vec<_> = particles.iter().map(|p| p.volume).collect();
        let cdf = vec![Cdf::default(); particles.len()];

        Self {
            positions: GpuVector::init(device, &positions, BufferUsages::STORAGE),
            velocities: GpuVector::init(device, &velocities, BufferUsages::STORAGE),
            volumes: GpuVector::init(device, &volumes, BufferUsages::STORAGE),
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
#[shader(src = "particle3d.wgsl", shader_defs = "dim_shader_defs")]
pub struct WgParticle;

wgcore::test_shader_compilation!(WgParticle);

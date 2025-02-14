use crate::dim_shader_defs;
use crate::models::{DruckerPrager, ElasticCoefficients};
use crate::solver::ParticlePhase;
use encase::ShaderType;
use nalgebra::{vector, Matrix2, Point2, Vector2};
use rapier::geometry::{ColliderSet, Polyline, Segment};
use std::ops::Div;
use wgcore::tensor::GpuVector;
use wgcore::Shader;
use wgpu::{BufferUsages, Device};
use wgrapier::dynamics::GpuBodySet;

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

pub struct GpuRigidParticles {
    pub sample_points: GpuVector<Point2<f32>>,
    pub rigid_particle_needs_block: GpuVector<u32>,
    pub node_linked_lists: GpuVector<u32>,
    pub sample_ids: GpuVector<GpuSampleIds>,
}

impl GpuRigidParticles {
    pub fn from_rapier(
        device: &Device,
        colliders: &ColliderSet,
        gpu_bodies: &GpuBodySet,
        sampling_step: f32,
    ) -> Self {
        let mut sampling_buffers = SamplingBuffers::default();
        for (collider_id, ((_, collider), gpu_data)) in colliders
            .iter()
            .zip(gpu_bodies.shapes_data().iter())
            .enumerate()
        {
            if let Some(polyline) = collider.shape().as_polyline() {
                let rngs = gpu_data.polyline_rngs();
                let sampling_params = SamplingParams {
                    collider_id: collider_id as u32,
                    base_vid: rngs[0],
                    sampling_step,
                };
                sample_polyline(polyline, &sampling_params, &mut sampling_buffers)
            }
        }

        Self {
            sample_points: GpuVector::init(
                device,
                &sampling_buffers.samples,
                BufferUsages::STORAGE,
            ),
            node_linked_lists: GpuVector::uninit(
                device,
                sampling_buffers.samples.len() as u32,
                BufferUsages::STORAGE,
            ),
            sample_ids: GpuVector::encase(
                device,
                &sampling_buffers.samples_ids,
                BufferUsages::STORAGE,
            ),
            // NOTE: this is a packed bitmask so each u32 contains
            //       the flag for 32 particles.
            rigid_particle_needs_block: GpuVector::uninit(
                device,
                sampling_buffers.samples.len().div_ceil(32) as u32,
                BufferUsages::STORAGE,
            ),
        }
    }

    pub fn len(&self) -> u64 {
        self.sample_points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
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

#[derive(Copy, Clone, Debug, ShaderType)]
pub struct GpuSampleIds {
    pub segment: Vector2<u32>,
    pub collider: u32,
}

#[derive(Copy, Clone, Debug)]
struct SamplingParams {
    base_vid: u32,
    collider_id: u32,
    sampling_step: f32,
}

#[derive(Default, Clone)]
struct SamplingBuffers {
    samples: Vec<Point2<f32>>,
    samples_ids: Vec<GpuSampleIds>,
}

// TODO: move this elsewhere?
fn sample_polyline(polyline: &Polyline, params: &SamplingParams, buffers: &mut SamplingBuffers) {
    for seg_idx in polyline.indices() {
        let seg = Segment::new(
            polyline.vertices()[seg_idx[0] as usize],
            polyline.vertices()[seg_idx[1] as usize],
        );
        let sample_id = GpuSampleIds {
            segment: vector![params.base_vid + seg_idx[0], params.base_vid + seg_idx[1]],
            collider: params.collider_id,
        };
        buffers.samples.push(seg.a);
        buffers.samples_ids.push(sample_id);

        if let Some(dir) = seg.direction() {
            for i in 0.. {
                let shift = (i as f32) * params.sampling_step;
                if shift > seg.length() {
                    break;
                }

                buffers.samples.push(seg.a + *dir * shift);
                buffers.samples_ids.push(sample_id);
            }

            buffers.samples.push(seg.b);
            buffers.samples_ids.push(sample_id);
        }
    }
}

wgcore::test_shader_compilation!(WgParticle);

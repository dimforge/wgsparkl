use crate::dim_shader_defs;
use crate::models::{DruckerPrager, ElasticCoefficients};
use crate::solver::ParticlePhase;
use encase::ShaderType;
use nalgebra::{vector, Matrix4, Matrix4x3, Point3, Vector3, Vector4};
use rapier::geometry::{Segment, Triangle};
use rapier::prelude::{ColliderSet, TriMesh};
use std::collections::HashSet;
use wgcore::tensor::GpuVector;
use wgcore::Shader;
use wgpu::{BufferUsages, Device};
use wgrapier::dynamics::GpuBodySet;

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

#[derive(Copy, Clone, Debug, ShaderType)]
pub struct GpuSampleIds {
    pub triangle: Vector3<u32>,
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
    samples: Vec<Point3<f32>>,
    samples_ids: Vec<GpuSampleIds>,
}

pub struct GpuRigidParticles {
    pub sample_points: GpuVector<Point3<f32>>,
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
            if let Some(trimesh) = collider.shape().as_trimesh() {
                let rngs = gpu_data.trimesh_rngs();
                let sampling_params = SamplingParams {
                    collider_id: collider_id as u32,
                    base_vid: rngs[0],
                    sampling_step,
                };
                sample_trimesh(trimesh, &sampling_params, &mut sampling_buffers)
            }
        }

        Self {
            sample_points: GpuVector::encase(
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

// TODO: move this elsewhere?
fn sample_trimesh(trimesh: &TriMesh, params: &SamplingParams, buffers: &mut SamplingBuffers) {
    // let samples = sample_mesh(
    //     trimesh.vertices(),
    //     trimesh.indices(),
    //     params.sampling_step / 2.0,
    // );
    let mut samples = vec![];
    generate_samples(
        trimesh.vertices(),
        trimesh.indices(),
        params.sampling_step,
        &mut samples,
    );
    for sample in samples {
        let tri_idx = trimesh.indices()[sample.triangle_id as usize];
        let tri = Triangle::new(
            trimesh.vertices()[tri_idx[0] as usize],
            trimesh.vertices()[tri_idx[1] as usize],
            trimesh.vertices()[tri_idx[2] as usize],
        );
        let sample_id = GpuSampleIds {
            triangle: vector![
                params.base_vid + tri_idx[0],
                params.base_vid + tri_idx[1],
                params.base_vid + tri_idx[2]
            ],
            collider: params.collider_id,
        };
        buffers.samples.push(sample.point);
        buffers.samples_ids.push(sample_id);

        // TODO: actually sample the triangle, currently we only push
        //       the triangle’s center.
    }
}

// Epsilon used as a length threshold in various steps of the sampling. In particular, this avoids
// degenerate geometries from generating invalid samples.
const EPS: f32 = 1.0e-5;

pub struct TriangleSample {
    pub triangle_id: u32,
    pub point: Point3<f32>,
}

/// Samples a triangle mesh with a set of points such that at least one point is generated
/// inside each cell on a grid on the x-y plane with cells sized by `xy_spacing`.
pub fn sample_mesh(
    vertices: &[Point3<f32>],
    indices: &[[u32; 3]],
    xy_spacing: f32,
) -> Vec<TriangleSample> {
    let mut samples = vec![];
    // TODO: switch to a matrix of boolean to avoid hashing if
    //       this proves to be a perf bottleneck.
    let mut visited_segs = HashSet::new();

    let mut seg_needs_sampling = |mut ia: u32, mut ib: u32| {
        if ib > ia {
            std::mem::swap(&mut ia, &mut ib);
        }

        visited_segs.insert([ia, ib])
    };

    for (tri_id, idx) in indices.iter().enumerate() {
        let tri = Triangle::new(
            vertices[idx[0] as usize],
            vertices[idx[1] as usize],
            vertices[idx[2] as usize],
        );
        sample_triangle(tri, &mut samples, xy_spacing, tri_id as u32);

        if seg_needs_sampling(idx[0], idx[1]) {
            let seg = Segment::new(vertices[idx[0] as usize], vertices[idx[1] as usize]);
            sample_edge(seg, &mut samples, xy_spacing, tri_id as u32);
        }

        if seg_needs_sampling(idx[1], idx[2]) {
            let seg = Segment::new(vertices[idx[1] as usize], vertices[idx[2] as usize]);
            sample_edge(seg, &mut samples, xy_spacing, tri_id as u32);
        }

        if seg_needs_sampling(idx[2], idx[0]) {
            let seg = Segment::new(vertices[idx[2] as usize], vertices[idx[0] as usize]);
            sample_edge(seg, &mut samples, xy_spacing, tri_id as u32);
        }
    }

    samples
}

/// Samples a triangle edge with a set of points such that at least one point is generated
/// inside each cell on a grid on the x-y plane with cells sized by `xy_spacing`.
///
/// The returned samples will not contain `edge.a`. It might contain `edge.b` (but it is unlikely)
/// if it aligns exactly with the internal sampling spacing.
pub fn sample_edge(
    edge: Segment,
    samples: &mut Vec<TriangleSample>,
    xy_spacing: f32,
    triangle_id: u32,
) {
    let ab = edge.b - edge.a;
    let edge_length = ab.norm();

    if edge_length > EPS {
        let edge_dir = ab / edge_length;
        let spacing = xy_spacing / 2.0f32.sqrt();
        let nsteps = (edge_length / spacing).ceil() as usize;

        // Start at one so we don’t push edge.a.
        for i in 1..nsteps {
            let point = edge.a + edge_dir * (spacing * i as f32);
            samples.push(TriangleSample { point, triangle_id })
        }
    }
}

/// Samples a triangle with a set of points such that at least one point is generated
/// inside each cell on a grid on the x-y plane with cells sized by `xy_spacing`.
///
/// Tha sampling has the following characteristics:
/// 1. Guarantees at least one sample per cell in the "ambient" XY grid.
/// 2. The sampling grid is oriented along the base (longest edge) and height (orthogonal to the
///    base) of the triangle.
/// 3. Samples are selected strictly from the domain of the triangle (up to rounding error).
/// 4. No sample is placed on the base or any of the triangle vertices. Samples will generally not
///    be on any of the two other edges either (but may due so some fortuitous alignment
///    with the internal stepping length along the height of the triangle).
///
/// Because this does not attempt to sample the edges of the triangles, small or thin triangles
/// might not result in any samples. Edges should be sampled separately with [`sample_edge`].
pub fn sample_triangle(
    triangle: Triangle,
    samples: &mut Vec<TriangleSample>,
    xy_spacing: f32,
    triangle_id: u32,
) {
    // select the longest edge as the base
    let distance_ab = nalgebra::distance(&triangle.b, &triangle.a);
    let distance_bc = nalgebra::distance(&triangle.c, &triangle.b);
    let distance_ca = nalgebra::distance(&triangle.a, &triangle.c);
    let max = distance_ab.max(distance_bc).max(distance_ca);

    let triangle = if max == distance_bc {
        Triangle {
            a: triangle.b,
            b: triangle.c,
            c: triangle.a,
        }
    } else if max == distance_ca {
        Triangle {
            a: triangle.c,
            b: triangle.a,
            c: triangle.b,
        }
    } else {
        triangle
    };

    let ac = triangle.c - triangle.a;
    let base = triangle.b - triangle.a;
    let base_length = base.norm();
    let base_dir = base / base_length;

    // Adjust the spacing so it matches the required spacing on the x-y plane.
    // For simplicity, we just divide by sqrt(2) so that the spacing in any direction is guaranteed
    // to be smaller or equal to the inner-circle diameter of any cell from the implicit grid with
    // spacing `xy_spacing`.
    // We could use a more fine-grained adjustment that depends on the angle between the base-dir
    // and the world x-y axes. But this doesn’t make a significant difference in point count or
    // computation times. However, the sampling looks worse (less uniform in practice). So we stick
    // to the simple sqrt(2) approach.
    let spacing = xy_spacing / 2.0f32.sqrt();

    // Calculate the step increment on the base.
    let base_step_count = (base_length / spacing).ceil();
    let base_step = base_dir * spacing;

    // Project C on the base AB.
    let ac_offset_length = ac.dot(&base_dir);
    let bc_offset_length = base_length - ac_offset_length;

    if ac_offset_length < EPS || bc_offset_length < EPS || base_length < EPS {
        return;
    }

    // Compute the triangle’s height vector.
    let height = ac - base_dir * ac_offset_length;
    let height_length = height.norm();
    let height_dir = height / height_length;
    // Calculate the tangents.
    let tan_alpha = height_length / ac_offset_length;
    let tan_beta = height_length / bc_offset_length;

    // Start at 1 so we don’t sample the perpendicular edge if it’s at a right angle
    // with `triangle.a`.
    for i in 1..base_step_count as u32 {
        let base_position = triangle.a + (i as f32) * base_step;

        // Compute the height at the current base_position. The point at the
        // end of that height is either in the line (AC) or (BC), whichever is closer.
        let height_ac = tan_alpha * nalgebra::distance(&triangle.a, &base_position);
        let height_bc = tan_beta * nalgebra::distance(&triangle.b, &base_position);
        let height_length = height_ac.min(height_bc);

        // Calculate the step increment on the height.
        let height_step_count = (height_length / spacing).ceil();
        let height_step = height_dir * spacing;

        // Start at 1 so we don’t sample the basis edge.
        for j in 1..height_step_count as u32 {
            let particle_position = base_position + (j as f32) * height_step;

            if particle_position.iter().any(|e| !e.is_finite()) {
                continue;
            }

            samples.push(TriangleSample {
                point: particle_position,
                triangle_id,
            });
        }
    }
}

pub fn generate_samples(
    vertices: &[Point3<f32>],
    indices: &[[u32; 3]],
    cell_width: f32,
    samples: &mut Vec<TriangleSample>,
) {
    // let cell_width = cell_width / 2.0;
    for (triangle_index, triangle) in indices.iter().enumerate() {
        let triangle = Triangle {
            a: vertices[triangle[0] as usize],
            b: vertices[triangle[1] as usize],
            c: vertices[triangle[2] as usize],
        };

        cover_triangle(triangle, samples, cell_width, triangle_index as u32);
    }
}

// Cover the triangle with rigid particles. They should be spaced apart no more than the cell_width.
fn cover_triangle(
    triangle: Triangle,
    samples: &mut Vec<TriangleSample>,
    cell_width: f32,
    triangle_id: u32,
) {
    // select the longest edge as the base
    let distance_ab = nalgebra::distance(&triangle.b, &triangle.a);
    let distance_bc = nalgebra::distance(&triangle.c, &triangle.b);
    let distance_ca = nalgebra::distance(&triangle.a, &triangle.c);
    let max = distance_ab.max(distance_bc).max(distance_ca);

    let triangle = if max == distance_bc {
        Triangle {
            a: triangle.b,
            b: triangle.c,
            c: triangle.a,
        }
    } else if max == distance_ca {
        Triangle {
            a: triangle.c,
            b: triangle.a,
            c: triangle.b,
        }
    } else {
        triangle
    };

    let ac = triangle.c - triangle.a;
    let base = triangle.b - triangle.a;
    let base_length = base.norm();
    let base_dir = base / base_length;

    // Calculate the step increment on the base.
    let base_step_count = (base_length / cell_width).ceil();
    let base_step = base_dir * base_length / base_step_count;

    // Project C on the base AB.
    let ac_offset_length = ac.dot(&base_dir);
    let bc_offset_length = base_length - ac_offset_length;
    // Compute the triangle’s height vector.
    let height = ac - base_dir * ac_offset_length;
    let height_length = height.norm();
    let height_dir = height / height_length;
    // Calculate the tangents.
    let tan_alpha = height_length / ac_offset_length;
    let tan_beta = height_length / bc_offset_length;

    for i in 0..=base_step_count as u32 {
        let base_position = triangle.a + (i as f32) * base_step;

        let height_ac = tan_alpha * nalgebra::distance(&triangle.a, &base_position);
        let height_bc = tan_beta * nalgebra::distance(&triangle.b, &base_position);
        let height_length = height_ac.min(height_bc);

        // Calculate the step increment on the height.
        let height_step_count = (height_length / cell_width).ceil();
        let height_step = height_dir * height_length / height_step_count;

        for j in 0..=height_step_count as u32 {
            let particle_position = base_position + (j as f32) * height_step;

            samples.push(TriangleSample {
                triangle_id,
                point: particle_position,
            });
        }
    }
}

#[derive(Shader)]
#[shader(src = "particle3d.wgsl", shader_defs = "dim_shader_defs")]
pub struct WgParticle;

wgcore::test_shader_compilation!(WgParticle);

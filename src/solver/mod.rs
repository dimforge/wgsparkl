pub use g2p::WgG2P;
pub use p2g::WgP2G;
pub use params::{GpuSimulationParams, SimulationParams, WgParams};
#[cfg(feature = "dim2")]
pub use particle2d::{GpuParticles, Particle, ParticleMassProps, WgParticle};
#[cfg(feature = "dim3")]
pub use particle3d::{GpuParticles, Particle, ParticleMassProps, WgParticle};
// pub use particle_update::WgParticleUpdate;
pub use grid_update::WgGridUpdate;
pub use particle_update::{ParticlePhase, WgParticleUpdate};

mod g2p;
mod p2g;
mod params;
mod particle_update;

mod grid_update;
#[cfg(feature = "dim2")]
mod particle2d;
#[cfg(feature = "dim3")]
mod particle3d;

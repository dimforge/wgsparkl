use crate::dim_shader_defs;
use crate::models::lame_lambda_mu;
use wgcore::Shader;
use wgebra::{WgSvd2, WgSvd3};

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct DruckerPrager {
    pub h0: f32,
    pub h1: f32,
    pub h2: f32,
    pub h3: f32,
    pub lambda: f32,
    pub mu: f32,
}

impl DruckerPrager {
    pub fn new(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = if young_modulus > 0.0 {
            lame_lambda_mu(young_modulus, poisson_ratio)
        } else {
            (-1.0, -1.0)
        };

        Self {
            h0: 35.0f32.to_radians(),
            h1: 9.0f32.to_radians(),
            h2: 0.2,
            h3: 10.0f32.to_radians(),
            lambda,
            mu,
        }
    }
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct DruckerPragerPlasticState {
    plastic_deformation_gradient_det: f32,
    plastic_hardening: f32,
    log_vol_gain: f32,
}

impl Default for DruckerPragerPlasticState {
    fn default() -> Self {
        Self {
            plastic_deformation_gradient_det: 1.0,
            plastic_hardening: 1.0,
            log_vol_gain: 0.0,
        }
    }
}

#[derive(Shader)]
#[shader(
    derive(WgSvd2, WgSvd3),
    src = "drucker_prager.wgsl",
    shader_defs = "dim_shader_defs"
)]
pub struct WgDruckerPrager;

wgcore::test_shader_compilation!(WgDruckerPrager, wgcore, crate::dim_shader_defs());

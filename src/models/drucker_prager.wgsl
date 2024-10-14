//! Drucker-Prager plasticity model

#define_import_path wgsparkl::models::drucker_prager
#import wgebra::svd2 as Svd2
#import wgebra::svd3 as Svd3


struct Plasticity {
    // TODO: why does it not like field names ending with a digit? (h0, h1, h2, h3)
    ha: f32,
    hb: f32,
    hc: f32,
    hd: f32,
    lambda: f32,
    mu: f32,
}

// TODO: should be this in the same buffer as the plasticity parameters themselves?
struct PlasticState {
    plastic_deformation_gradient_det: f32,
    plastic_hardening: f32,
    log_vol_gain: f32,
}

fn alpha(plasticity: Plasticity, q: f32) -> f32 {
    let angle = plasticity.ha + (plasticity.hb * q - plasticity.hd) * exp(-plasticity.hc * q);
    let s_angle = sin(angle);
    return sqrt(2.0 / 3.0) * (2.0 * s_angle) / (3.0 - s_angle);
}

#if DIM == 2
struct DruckerPragerResult {
    state: PlasticState,
    deformation_gradient: mat2x2<f32>,
}

struct ProjectionResult {
    singular_values: vec2<f32>,
    plastic_hardening: f32,
    valid: bool,
}

fn project_deformation_gradient(plasticity: Plasticity, singular_values: vec2<f32>, log_vol_gain: f32, alpha: f32) -> ProjectionResult {
    let d = 2.0; // NOTE: this is 3 in 2D
    let strain = log(singular_values) + vec2(log_vol_gain / d);
    let strain_trace = strain.x + strain.y;
    let deviatoric_strain = strain - vec2(strain_trace / d);

    if strain_trace > 0.0 || all(deviatoric_strain == vec2(0.0)) {
        return ProjectionResult(vec2(1.0), length(strain), true);
    }

    let deviatoric_strain_norm = length(deviatoric_strain);
    let gamma = deviatoric_strain_norm
        + (d * plasticity.lambda + 2.0 * plasticity.mu) / (2.0 * plasticity.mu) * strain_trace * alpha;
    if gamma <= 0.0 {
        return ProjectionResult(vec2(0.0), 0.0, false);
    }

    let h = strain - deviatoric_strain * (gamma / deviatoric_strain_norm);
    return ProjectionResult(exp(h), gamma, true);
}

fn project(plasticity: Plasticity, state: PlasticState, deformation_gradient: mat2x2<f32>) -> DruckerPragerResult {
//    if true {
//        let snowvd = Svd2::svd(deformation_gradient);
//        let yield_surface = exp(1.0 - 0.95);
//        let j = snowvd.S.x * snowvd.S.y;
//        var new_sigvals = clamp(snowvd.S, vec2(1.0 / yield_surface), vec2(yield_surface));
//        let new_j = new_sigvals.x * new_sigvals.y;
//        new_sigvals *= sqrt(j / new_j);
//        let new_def_grad = Svd2::recompose(Svd2::Svd(snowvd.U, new_sigvals, snowvd.Vt));
//        return DruckerPragerResult(state, new_def_grad);
//    }
    if plasticity.lambda == 0 {
        // Plasticity is disable on this particle.
        return DruckerPragerResult(state, deformation_gradient);
    }

    let svd = Svd2::svd(deformation_gradient);
    let alpha = alpha(plasticity, state.plastic_hardening);
    let projection = project_deformation_gradient(plasticity, svd.S, state.log_vol_gain, alpha);

    if projection.valid {
        let prev_det = svd.S.x * svd.S.y;
        let new_det = projection.singular_values.x * projection.singular_values.y;

        let new_plastic_deformation_gradient_det = state.plastic_deformation_gradient_det * prev_det / new_det;
        let new_log_vol_gain = state.log_vol_gain + log(prev_det) - log(new_det);
        let new_plastic_hardening = state.plastic_hardening + projection.plastic_hardening;
        let new_deformation_gradient = Svd2::recompose(Svd2::Svd(svd.U, projection.singular_values, svd.Vt));
        return DruckerPragerResult(
            PlasticState(new_plastic_deformation_gradient_det, new_plastic_hardening, new_log_vol_gain),
            new_deformation_gradient,
        );
    } else {
        return DruckerPragerResult(state, deformation_gradient);
    }
}
#else
struct DruckerPragerResult {
    state: PlasticState,
    deformation_gradient: mat3x3<f32>,
}

struct ProjectionResult {
    singular_values: vec3<f32>,
    plastic_hardening: f32,
    valid: bool,
}

fn project_deformation_gradient(plasticity: Plasticity, singular_values: vec3<f32>, log_vol_gain: f32, alpha: f32) -> ProjectionResult {
    let d = 3.0; // NOTE: this is 2 in 2D
    let strain = log(singular_values) + vec3(log_vol_gain / d);
    let strain_trace = strain.x + strain.y + strain.z;
    let deviatoric_strain = strain - vec3(strain_trace / d);

    if strain_trace > 0.0 || all(deviatoric_strain == vec3(0.0)) {
        return ProjectionResult(vec3(1.0), length(strain), true);
    }

    let deviatoric_strain_norm = length(deviatoric_strain);
    let gamma = deviatoric_strain_norm
        + (d * plasticity.lambda + 2.0 * plasticity.mu) / (2.0 * plasticity.mu) * strain_trace * alpha;
    if gamma <= 0.0 {
        return ProjectionResult(vec3(0.0), 0.0, false);
    }

    let h = strain - deviatoric_strain * (gamma / deviatoric_strain_norm);
    return ProjectionResult(exp(h), gamma, true);
}

fn project(plasticity: Plasticity, state: PlasticState, deformation_gradient: mat3x3<f32>) -> DruckerPragerResult {
    if plasticity.lambda == 0 {
        // Plasticity is disable on this particle.
        return DruckerPragerResult(state, deformation_gradient);
    }

    let svd = Svd3::svd(deformation_gradient);
    let alpha = alpha(plasticity, state.plastic_hardening);
    let projection = project_deformation_gradient(plasticity, svd.S, state.log_vol_gain, alpha);

    if projection.valid {
        let prev_det = svd.S.x * svd.S.y * svd.S.z;
        let new_det = projection.singular_values.x * projection.singular_values.y * projection.singular_values.z;

        let new_plastic_deformation_gradient_det = state.plastic_deformation_gradient_det * prev_det / new_det;
        let new_log_vol_gain = state.log_vol_gain + log(prev_det) - log(new_det);
        let new_plastic_hardening = state.plastic_hardening + projection.plastic_hardening;
        let new_deformation_gradient = Svd3::recompose(Svd3::Svd(svd.U, projection.singular_values, svd.Vt));
        return DruckerPragerResult(
            PlasticState(new_plastic_deformation_gradient_det, new_plastic_hardening, new_log_vol_gain),
            new_deformation_gradient,
        );
    } else {
        return DruckerPragerResult(state, deformation_gradient);
    }
}
#endif
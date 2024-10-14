//! Neo-hookean elasticity model.

#define_import_path wgsparkl::models::neo_hookean_elasticity


struct ElasticCoefficients {
    lambda: f32,
    mu: f32,
}

#if DIM == 2
fn kirchoff_stress(model: ElasticCoefficients, deformation_gradient: mat2x2<f32>) -> mat2x2<f32> {
#else
fn kirchoff_stress(model: ElasticCoefficients, deformation_gradient: mat3x3<f32>) -> mat3x3<f32> {
#endif
    let j = determinant(deformation_gradient);
    let diag = model.lambda * log(j) - model.mu / j;
    var stress = model.mu / j * (deformation_gradient * transpose(deformation_gradient));
    stress.x.x += diag;
    stress.y.y += diag;
    #if DIM == 3
    stress.z.z += diag;
    #endif

    return stress;
}
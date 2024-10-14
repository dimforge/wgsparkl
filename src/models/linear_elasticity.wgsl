//! Linear (corotated) elasticity model.

#define_import_path wgsparkl::models::linear_elasticity
#import wgebra::svd2 as Svd2
#import wgebra::svd3 as Svd3


struct ElasticCoefficients {
    lambda: f32,
    mu: f32,
}

#if DIM == 2
fn kirchoff_stress(model: ElasticCoefficients, deformation_gradient: mat2x2<f32>) -> mat2x2<f32> {
    var svd = Svd2::svd(deformation_gradient);
    let j = svd.S.x * svd.S.y;

    svd.S -= vec2(1.0);

    let diag = model.lambda * (j - 1.0) * j;
    var result = (Svd2::recompose(svd) * transpose(deformation_gradient)) * (2.0 * model.mu);
    result.x.x += diag;
    result.y.y += diag;

    return result;
}
#else
fn kirchoff_stress(model: ElasticCoefficients, deformation_gradient: mat3x3<f32>) -> mat3x3<f32> {
    var svd = Svd3::svd(deformation_gradient);
    let j = svd.S.x * svd.S.y * svd.S.z;

    svd.S -= vec3(1.0);

    let diag = model.lambda * (j - 1.0) * j;
    var result = (Svd3::recompose(svd) * transpose(deformation_gradient)) * (2.0 * model.mu);
    result.x.x += diag;
    result.y.y += diag;
    result.z.z += diag;

    return result;
}
#endif
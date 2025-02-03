#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,

    @location(3) def_x: vec3<f32>,
    @location(4) def_y: vec3<f32>,
    @location(5) def_z: vec3<f32>,
    @location(6) pos: vec3<f32>,
    @location(7) unused: vec4<f32>,
    @location(8) i_color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let deformation = mat3x3(vertex.def_x, vertex.def_y, vertex.def_z);
    let position = deformation * vertex.position + vertex.pos;
    var out: VertexOutput;
    let identity = mat4x4f(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    out.clip_position = mesh_position_local_to_clip(
        identity,
        vec4<f32>(position, 1.0)
    );
    out.color = vertex.i_color;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
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
    // NOTE: Passing 0 as the instance_index to get_world_from_local() is a hack
    // for this example as the instance_index builtin would map to the wrong
    // index in the Mesh array. This index could be passed in via another
    // uniform instead but it's unnecessary for the example.
    out.clip_position = mesh_position_local_to_clip(
        get_world_from_local(0u),
        vec4<f32>(position, 1.0)
    );
    out.color = vertex.i_color;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
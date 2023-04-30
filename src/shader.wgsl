struct VertIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertOut {
	@builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>
};
@group(1) @binding(0)
var<uniform> cam: CameraUniform;

@vertex
fn vs_main(in: VertIn) -> VertOut {
    return VertOut(cam.proj * cam.view * vec4<f32>(in.pos, 1.0), in.uv);
}

@group(0) @binding(0)
var tex: texture_2d<f32>;
@group(0)@binding(1)
var tex_sampler: sampler;

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    return textureSample(tex, tex_sampler, in.uv);
}

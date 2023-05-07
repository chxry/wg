struct VertIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>
};

struct VertOut {
	@builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>
}

struct Model {
    model: mat4x4<f32>,
    normal: mat3x3<f32>
}
@group(1) @binding(0)
var<uniform> model: Model;
struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>
};
@group(2) @binding(0)
var<uniform> cam: Camera;

@vertex
fn vs_main(in: VertIn) -> VertOut {
    return VertOut(cam.proj * cam.view * model.model * vec4(in.pos, 1.0), in.pos, in.uv,model.normal * in.normal);
}

@group(0) @binding(0)
var tex: texture_2d<f32>;
@group(0)@binding(1)
var tex_sampler: sampler;
struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}
@group(3) @binding(0)
var<uniform> light: Light;


@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    var albedo = textureSample(tex, tex_sampler, in.uv).rgb;
    let ambient = 0.1 * light.color;
    let light_dir = normalize(light.position - in.world_pos);
    let diffuse =  max(dot(in.normal, light_dir), 0.0) * light.color;
    return vec4(in.normal, 1.0);
    // return vec4(albedo * (diffuse), 1.0);
}

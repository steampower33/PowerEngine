#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_normal;

layout(location = 0) out vec3 out_dir;

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4x4 view;
    mat4x4 proj;

    uint vulkan_thumbnail_index;
    uint p0;
    uint p1;
    uint p2;
} global;

layout(set = 1, binding = 0) uniform SkyboxUBO {
    mat4x4 model;

    uint env_idx;
    uint radiance_idx;
    uint irradiance_idx;
    uint specular_mip_levels;

    uint brdf_lut_index;
    uint p0;
    uint p1;
    uint p2;
} skybox;

void main() {
    mat4 viewRot = mat4(mat3(global.view));

    vec4 pos_model =  vec4(in_pos, 1.0);
    vec4 pos_world = skybox.model * pos_model;
    vec4 pos_view = global.view * pos_world;
    gl_Position = global.proj * pos_view;

    out_dir = pos_world.xyz;
}
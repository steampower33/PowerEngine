#version 450
#extension GL_KHR_vulkan_glsl : enable

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 1, binding = 0) uniform Model {
    mat4x4 world;

    vec4 albedo;

    int albedo_idx;
    int metallic_idx;
    int normal_idx;
    int roughness_idx;

    int ao_idx;
    int height_idx;
    float metallic_factor;
    float roughness_factor;

    float ao_factor;
    float height_factor;
    float coat_factor;
    float coat_roughness_factor;

    float fuzz_factor;
    float fuzz_roughness_factor;
    vec2 tile_uv;

    uint albedo_enable;
    uint metallic_enable;
    uint normal_enable;
    uint roughness_enable;

    uint ao_enable;
    uint height_enable;
    uint checker_board_enable;
    uint p2;
} model;

layout(set = 1, binding = 1, std430) readonly buffer Positions { vec4 pos[]; };
layout(set = 1, binding = 2, std430) readonly buffer Normals { vec4 normals[]; };

layout(push_constant) uniform SoftPC {
    vec4 color;
} pc;

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_world_normal;

void main() {
    uint vid = gl_VertexIndex;
    vec3 p = pos[vid].xyz;

    out_world_normal = normals[vid].xyz;
    out_color = pc.color.xyz;
    gl_Position = ubo.proj * ubo.view * vec4(p, 1.0);
}

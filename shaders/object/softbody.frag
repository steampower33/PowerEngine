#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 1, binding = 0) uniform Model {
    mat4x4 model;

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

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_world_normal;

layout(location = 0) out vec4 out_albedo_metal;
layout(location = 1) out vec4 out_normal_rough;
layout(location = 2) out vec2 out_height_ao;
layout(location = 3) out vec4 out_cozz_fuzz;

void main() {
    vec3 normal  = (!gl_FrontFacing) ? -in_world_normal : in_world_normal;

    out_albedo_metal = vec4(in_color, 1.0);
    out_normal_rough = vec4(normal * 0.5 + 0.5, 1.0);
    out_height_ao = vec2(0.0, 1.0);
    out_cozz_fuzz = vec4(0.0, 0.0, 0.0, 0.0);
}

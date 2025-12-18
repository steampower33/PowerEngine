#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

#include "../common/render_common.glsl"

layout(set = 0, binding = 0) uniform GlobalBlock {
    Global global;
};

layout(set = 1, binding = 0, std140) uniform ModelBlock {
    Model model;
};

layout(set = 2, binding = 0) uniform sampler2D tex[];

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_normal_world;
layout(location = 2) in vec3 in_pos_world;
layout(location = 3) in vec3 in_tangent_world;

layout(location = 0) out vec4 out_albedo_metal;
layout(location = 1) out vec4 out_normal_rough;
layout(location = 2) out vec2 out_height_ao;
layout(location = 3) out vec4 out_cozz_fuzz;

void main() {
    vec4 albedo    = (model.albedo_enable == 0u || model.albedo_idx == -1) ? vec4(model.albedo.xyz, 0.0): texture(tex[nonuniformEXT(model.albedo_idx)], in_uv);
    vec3 normalTS  = in_normal_world;

    float ao = 0.0, roughness = 0.0, metallic = 0.0;
    if (model.arm_enable == 1u && model.arm_idx != -1)
    {
        vec4 arm = texture(tex[nonuniformEXT(model.arm_idx)], in_uv);
        ao = arm.r;
        roughness = arm.g;
        metallic = arm.b;
    }
    else
    {
        ao       = (model.ao_enable == 0u || model.ao_idx == -1) ? model.ao_factor : texture(tex[nonuniformEXT(model.ao_idx)], in_uv).r;
        roughness = (model.roughness_enable == 0u || model.roughness_idx == -1) ? model.roughness_factor : texture(tex[nonuniformEXT(model.roughness_idx)], in_uv).r;
        metallic = (model.metallic_enable == 0u || model.metallic_idx == -1) ? model.metallic_factor : texture(tex[nonuniformEXT(model.metallic_idx)], in_uv).r;
    }
    float height   = (model.height_enable == 0u || model.height_idx == -1) ? model.height_factor : texture(tex[nonuniformEXT(model.height_idx)], in_uv).r;

    if (model.normal_enable == 1u && model.normal_idx != -1)
    {
        vec3 normal = texture(tex[nonuniformEXT(model.normal_idx)], in_uv).xyz * 2.0 - 1.0;

        vec3 N = in_normal_world;
        vec3 T = normalize(in_tangent_world - dot(in_tangent_world, N) * N);
        vec3 B = cross(N, T);
        
        mat3x3 TBN = mat3x3(T, B, N);
        normalTS = normalize(normal * TBN);
    }

    if (model.checker_board_enable == 1u)
    {
        vec2 uv = in_pos_world.xz * 0.5;
        ivec2 cell = ivec2(floor(uv));
        int c = (cell.x + cell.y) & 1;
        albedo.xyz = (c == 0) ? vec3(0.85) : vec3(0.35);
    }
    out_albedo_metal = vec4(albedo.xyz, metallic);

    vec3 n = normalize(normalTS);
    out_normal_rough = vec4(n * 0.5 + 0.5, roughness);
    
    out_height_ao = vec2(height, ao);
    out_cozz_fuzz = vec4(model.coat_factor, model.coat_roughness_factor, model.fuzz_factor, model.fuzz_roughness_factor);
}
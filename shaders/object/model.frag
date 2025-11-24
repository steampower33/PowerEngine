#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

#include "../common/model_common.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_normal_world;
layout(location = 2) in vec3 in_pos_world;
layout(location = 3) in vec3 in_tangent_world;

layout(location = 0) out vec4 out_albedo_metal;
layout(location = 1) out vec4 out_normal_rough;
layout(location = 2) out vec4 out_height_ao;

void main() {
    vec4 albedo    = (object.albedo_enable == 0u) ? vec4(object.albedo_use.xyz, 0.0): texture(tex[nonuniformEXT(object.albedo_idx)], in_uv);
    vec3 normalTS  = in_normal_world;
    float metallic = (object.metallic_enable == 0u) ? object.metallic_factor : texture(tex[nonuniformEXT(object.metallic_idx)], in_uv).r;
    float rough    = (object.roughness_enable == 0u) ? object.roughness_factor : texture(tex[nonuniformEXT(object.roughness_idx)], in_uv).r;
    float ao       = (object.ao_enable == 0u) ? object.ao_factor : texture(tex[nonuniformEXT(object.ao_idx)], in_uv).r;
    float height   = (object.height_enable == 0u) ? object.height_factor : texture(tex[nonuniformEXT(object.height_idx)], in_uv).r;

    if (object.normal_enable != 0u)
    {
        vec3 normal = texture(tex[nonuniformEXT(object.normal_idx)], in_uv).xyz * 2.0 - 1.0;

        vec3 N = in_normal_world;
        vec3 T = normalize(in_tangent_world - dot(in_tangent_world, N) * N);
        vec3 B = cross(N, T);
        
        mat3x3 TBN = mat3x3(T, B, N);
        normalTS = normalize(normal * TBN);
    }

    if (object.checker_board_enable == 1u)
    {
        vec2 uv = in_pos_world.xz * 0.5;
        ivec2 cell = ivec2(floor(uv));
        int c = (cell.x + cell.y) & 1;
        albedo.xyz = (c == 0) ? vec3(0.9) : vec3(0.7);
    }
    // RT0: albedo.rgb + metallic
    out_albedo_metal = vec4(albedo.xyz, metallic);

    // RT1: world-normal + roughness (여기선 예시로 그냥 tangent-space normal)
    vec3 n = normalize(normalTS);
    // [-1,1] → [0,1]
    out_normal_rough = vec4(n * 0.5 + 0.5, rough);

    // RT2: height + AO
    out_height_ao = vec4(height, ao, 0.0, 0.0);
}
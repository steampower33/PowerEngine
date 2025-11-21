#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 2, binding = 0) uniform Render {
    vec4 albedo_use;

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
    uint p0;
    uint p1;

    uint albedo_enable;
    uint metallic_enable;
    uint normal_enable;
    uint roughness_enable;

    uint ao_enable;
    uint height_enable;
    uint p3;
    uint p4;
} render;

layout(set = 3, binding = 0) uniform sampler2D tex[];

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_world_normal;

layout(location = 0) out vec4 out_albedo_metal;
layout(location = 1) out vec4 out_normal_rough;
layout(location = 2) out vec4 out_height_ao;

void main() {
    vec4 albedo    = (render.albedo_enable == 0u) ? vec4(render.albedo_use.xyz, 0.0): texture(tex[nonuniformEXT(render.albedo_idx)], in_uv);
    vec3 normal  = in_world_normal;
    float metallic = (render.metallic_enable == 0u) ? render.metallic_factor : texture(tex[nonuniformEXT(render.metallic_idx)], in_uv).r;
    float roughness = (render.roughness_enable == 0u) ? render.roughness_factor : texture(tex[nonuniformEXT(render.roughness_idx)], in_uv).r;
    float ao       = (render.ao_enable == 0u) ? render.ao_factor : texture(tex[nonuniformEXT(render.ao_idx)], in_uv).r;
    float height   = (render.height_enable == 0u) ? render.height_factor : texture(tex[nonuniformEXT(render.height_idx)], in_uv).r;

    out_albedo_metal = vec4(albedo.xyz, metallic);
    out_normal_rough = vec4(normal * 0.5 + 0.5, roughness); // [-1,1] -> [0,1]
    out_height_ao = vec4(0.0, ao, 0.0, 0.0);
}

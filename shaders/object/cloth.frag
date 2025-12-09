#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 1, binding = 0) uniform Cloth {

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
    float p0;
    float p1;

    uint albedo_enable;
    uint metallic_enable;
    uint normal_enable;
    uint roughness_enable;

    uint ao_enable;
    uint height_enable;
    uint checker_board_enable;
    uint p2;
} cloth;

layout(set = 2, binding = 0) uniform sampler2D tex[];

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_world_normal;

layout(location = 0) out vec4 out_albedo_metal;
layout(location = 1) out vec4 out_normal_rough;
layout(location = 2) out vec2 out_height_ao;
layout(location = 3) out vec4 out_cozz_fuzz;

layout(push_constant) uniform ClothPC { vec4 color; uint nx1; uint ny1; uint p0; float p1; } pc;

void main() {
    vec4 albedo    = (cloth.albedo_enable == 0u || cloth.albedo_idx == -1) ? pc.color : texture(tex[nonuniformEXT(cloth.albedo_idx)], in_uv);
    vec3 normal  = (!gl_FrontFacing) ? -in_world_normal : in_world_normal;
    float metallic = (cloth.metallic_enable == 0u || cloth.metallic_idx == -1) ? cloth.metallic_factor : texture(tex[nonuniformEXT(cloth.metallic_idx)], in_uv).r;
    float roughness = (cloth.roughness_enable == 0u || cloth.roughness_idx == -1) ? cloth.roughness_factor : texture(tex[nonuniformEXT(cloth.roughness_idx)], in_uv).r;
    float ao       = (cloth.ao_enable == 0u || cloth.ao_idx == -1) ? cloth.ao_factor : texture(tex[nonuniformEXT(cloth.ao_idx)], in_uv).r;
    float height   = (cloth.height_enable == 0u || cloth.height_idx == -1) ? cloth.height_factor : texture(tex[nonuniformEXT(cloth.height_idx)], in_uv).r;

    out_albedo_metal = vec4(albedo.xyz, metallic);
    out_normal_rough = vec4(normal * 0.5 + 0.5, roughness); // [-1,1] -> [0,1]
    out_height_ao = vec2(height, ao);
    out_cozz_fuzz = vec4(cloth.coat_factor, cloth.coat_roughness_factor, cloth.fuzz_factor, cloth.fuzz_roughness_factor);
}

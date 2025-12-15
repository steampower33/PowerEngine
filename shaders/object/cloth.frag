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

layout(set = 2, binding = 0) uniform sampler2D tex[];

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_world_normal;

layout(location = 0) out vec4 out_albedo_metal;
layout(location = 1) out vec4 out_normal_rough;
layout(location = 2) out vec2 out_height_ao;
layout(location = 3) out vec4 out_cozz_fuzz;

layout(push_constant) uniform PushConstant { vec4 color; uint nx1; uint ny1; uint p0; float p1; } pc;

void main() {

    vec4 albedo    = (model.albedo_enable == 0u || model.albedo_idx == -1) ? pc.color : texture(tex[nonuniformEXT(model.albedo_idx)], in_uv);
    vec3 normal  = (!gl_FrontFacing) ? -in_world_normal : in_world_normal;
    float metallic = (model.metallic_enable == 0u || model.metallic_idx == -1) ? model.metallic_factor : texture(tex[nonuniformEXT(model.metallic_idx)], in_uv).r;
    float roughness = (model.roughness_enable == 0u || model.roughness_idx == -1) ? model.roughness_factor : texture(tex[nonuniformEXT(model.roughness_idx)], in_uv).r;
    float ao       = (model.ao_enable == 0u || model.ao_idx == -1) ? model.ao_factor : texture(tex[nonuniformEXT(model.ao_idx)], in_uv).r;
    float height   = (model.height_enable == 0u || model.height_idx == -1) ? model.height_factor : texture(tex[nonuniformEXT(model.height_idx)], in_uv).r;

    out_albedo_metal = vec4(albedo.xyz, metallic);
    out_normal_rough = vec4(normal * 0.5 + 0.5, roughness); // [-1,1] -> [0,1]
    out_height_ao = vec2(height, ao);
    out_cozz_fuzz = vec4(model.coat_factor, model.coat_roughness_factor, model.fuzz_factor, model.fuzz_roughness_factor);
}

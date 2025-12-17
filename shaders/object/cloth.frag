#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

#include "../common/render_common.glsl"

layout(set = 1, binding = 0, std140) uniform ModelBlock {
    Model model;
};

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

    out_albedo_metal = vec4(albedo.xyz, metallic);
    out_normal_rough = vec4(normal * 0.5 + 0.5, roughness); // [-1,1] -> [0,1]
    out_height_ao = vec2(height, ao);
    out_cozz_fuzz = vec4(model.coat_factor, model.coat_roughness_factor, model.fuzz_factor, model.fuzz_roughness_factor);
}

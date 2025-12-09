#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

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

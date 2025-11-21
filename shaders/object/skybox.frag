#version 450
#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

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

layout(set = 2, binding = 0) uniform samplerCube env_tex[];

layout(location = 0) in vec3 in_dir;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 color = texture(env_tex[nonuniformEXT(skybox.env_idx)], in_dir).rgb;

    out_color = vec4(color, 0.0);
}

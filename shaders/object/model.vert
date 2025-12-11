#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

#include "../common/model_common.glsl"

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec3 in_tangent;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_normal_world;
layout(location = 2) out vec3 out_pos_world;
layout(location = 3) out vec3 out_tangent_world;

void main() 
{
    vec4 world_pos = object.model * vec4(in_pos, 1.0);

    mat3 normal_mat = transpose(inverse(mat3(object.model)));
    out_normal_world = normalize(normal_mat * in_normal);

    if (object.height_enable == 1u)
    {
        float height = texture(tex[nonuniformEXT(object.height_idx)], in_uv).r;
        height = height * 2.0 - 1.0;
        world_pos += vec4(out_normal_world * height * object.height_factor, 0.0);
    }
    gl_Position = global.proj * global.view * world_pos;

    out_pos_world = world_pos.xyz;
    out_uv = in_uv;
    out_tangent_world = (object.model * vec4(in_tangent, 0.0)).xyz;
}
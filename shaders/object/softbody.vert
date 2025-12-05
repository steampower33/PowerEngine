#version 450
#extension GL_KHR_vulkan_glsl : enable

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 1, binding = 0, std430) readonly buffer Positions { vec4 pos[]; };
layout(set = 1, binding = 1, std430) readonly buffer Normals { vec4 normals[]; };

layout(push_constant) uniform SoftPC {
    vec4 color;
} pc;

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_world_normal;

void main() {
    uint vid = gl_VertexIndex;
    vec3 p = pos[vid].xyz;

    out_world_normal = normals[vid].xyz;
    out_color = pc.color.xyz;
    gl_Position = ubo.proj * ubo.view * vec4(p, 1.0);
}

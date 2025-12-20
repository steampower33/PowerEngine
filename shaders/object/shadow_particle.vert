#version 460

#extension GL_KHR_vulkan_glsl : enable

layout(set=0, binding=0, std430) readonly buffer X { vec4 x[]; };

layout(push_constant) uniform PushConstant { mat4 light_view_proj; uint is_vertex_ssbo; } pc;

layout(location=0) in vec3 in_pos;

void main()
{
    uint vid = gl_VertexIndex;
    gl_Position = pc.light_view_proj * x[vid];
}

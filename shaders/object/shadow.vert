#version 460

#extension GL_KHR_vulkan_glsl : enable

layout(set=0, binding=0) uniform ModelUBO {
    mat4 model;
} model;

layout(set=1, binding=0, std430) readonly buffer X { vec4 x[]; };

layout(push_constant) uniform PushConstant { mat4 light_view_proj; uint is_vertex_ssbo; } pc;

layout(location=0) in vec3 in_pos;

void main()
{

    vec4 pos = vec4(0.0);
    if (pc.is_vertex_ssbo == 0u)
        pos = model.model * vec4(in_pos, 1.0);
    else if (pc.is_vertex_ssbo == 1u)
    {
        uint vid = gl_VertexIndex;
        pos = x[vid];
    }
    gl_Position = pc.light_view_proj * pos;
}

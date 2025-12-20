#version 460

layout(set=0, binding=0) uniform ModelUBO {
    mat4 model;
} model;

layout(push_constant) uniform PushConstant { mat4 light_view_proj; uint is_vertex_ssbo; } pc;

layout(location=0) in vec3 in_pos;

void main()
{
    gl_Position = pc.light_view_proj * model.model * vec4(in_pos, 1.0);
}

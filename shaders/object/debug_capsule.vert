#version 450

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4x4 view;
    mat4x4 proj;
} global;

layout(push_constant) uniform DebugPC {
    mat4 model;
    vec4 color;
} pc;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_normal;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_normal_world;
layout(location = 2) out vec3 out_color;

void main()
{
    vec4 world_pos = pc.model * vec4(in_pos, 1.0);

    mat3 normal_mat = transpose(inverse(mat3(pc.model)));
    out_normal_world = normalize(normal_mat * in_normal);
    
    gl_Position = global.proj * global.view * world_pos;

    out_uv = in_uv;
    out_color = pc.color.rgb;
}
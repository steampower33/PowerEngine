#version 450
#extension GL_KHR_vulkan_glsl : enable

const float grid_size = 1.0;

const vec4 positions[4] = vec4[](
    vec4(-0.5, 0.0,  0.5, 1.0),
    vec4( 0.5, 0.0,  0.5, 1.0),
    vec4(-0.5, 0.0, -0.5, 1.0),
    vec4( 0.5, 0.0, -0.5, 1.0)
);

layout(location = 0) out vec2 outCoords;

layout(set = 0, binding = 0) uniform Global
{
    mat4 view;
    mat4 proj;
} global;

void main()
{
    uint id = uint(gl_VertexIndex);

    vec4 world_pos = positions[id];
    world_pos.xyz *= grid_size;

    gl_Position = global.proj * global.view * world_pos;

    outCoords = world_pos.xz;
}
/*
    Infinite Grid Shader (Vulkan GLSL 450)

    Inspired by OGLDEV (Etay Meiri) examples/tutorials:
      Repo: https://github.com/emeiri/ogldev
      Reference: https://github.com/emeiri/ogldev/blob/master/DemoLITION/Framework/Shaders/GL/infinite_grid.vs

    Notes:
    - This implementation is written for Vulkan and adapted to this engine's coordinate system,
      descriptor bindings, and render pipeline.
*/

#version 450

const float cell_size = 100.0;

const vec3 pos[4] = vec3[4](
    vec3(-1.0, 0.0, -1.0),      // bottom left
    vec3( 1.0, 0.0, -1.0),      // bottom right
    vec3( 1.0, 0.0,  1.0),      // top right
    vec3(-1.0, 0.0,  1.0)       // top left
);

const int indices[6] = int[6](0, 2, 1, 2, 0, 3);

layout(set = 0, binding = 0) uniform Grid
{
    mat4 view;
    mat4 proj;

    vec3 camera_pos;
    float p0;
} grid;

layout(location = 0) out vec3 out_world_pos;

void main()
{
    uint id = uint(gl_VertexIndex);
    int idx = indices[id];

    vec3 world_pos = pos[idx] * cell_size;
    world_pos.x += grid.camera_pos.x;
    world_pos.z += grid.camera_pos.z;

    out_world_pos = world_pos;

    gl_Position = grid.proj * grid.view * vec4(world_pos, 1.0);
}
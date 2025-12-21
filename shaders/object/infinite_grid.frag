/*
    Infinite Grid Shader (Vulkan GLSL 450)

    Inspired by OGLDEV (Etay Meiri) examples/tutorials:
      Repo: https://github.com/emeiri/ogldev
      Reference: https://github.com/emeiri/ogldev/blob/master/DemoLITION/Framework/Shaders/GL/infinite_grid.fs

    Notes:
    - This implementation is written for Vulkan and adapted to this engine's coordinate system,
      descriptor bindings, and render pipeline.
*/

#version 450

#extension GL_KHR_vulkan_glsl : enable

const float grid_size = 100.0;
const float grid_cell_size = 1.0;
const float grid_min_pixel_between_cells = 1.0;
const vec4 grid_color_thin = vec4(0.1, 0.1, 0.1, 1.0);
const vec4 grid_color_thick = vec4(0.35, 0.35, 0.35, 1.0);

float satf(float x) { return clamp(x, 0.0, 1.0); }
vec2  satv(vec2 x)  { return clamp(x, vec2(0.0), vec2(1.0)); }
float max2(vec2 v)  { return max(v.x, v.y); }

float log10_(float x) { return log(x) / log(10.0); }

layout(set = 0, binding = 0) uniform Grid
{
    mat4 view;
    mat4 proj;

    vec3 camera_pos;
    float p0;
} grid;

layout(location = 0) in vec3 in_world_pos;

layout(location = 0) out vec4 out_color;

void main()
{
    vec2 dvx = vec2(dFdx(in_world_pos.x), dFdy(in_world_pos.x));
    vec2 dvy = vec2(dFdx(in_world_pos.z), dFdy(in_world_pos.z));

    float lx = length(dvx);
    float ly = length(dvy);

    vec2 dudv = vec2(lx, ly);

    float l = length(dudv);
	
    float LOD = max(0.0, log10_(l * grid_min_pixel_between_cells / grid_cell_size) + 1.0);

    float cell0 = grid_cell_size * pow(10.0, floor(LOD));
    float cell1 = cell0 * 10.0;
    float cell2 = cell1 * 10.0;

    dudv *= 4.0;

    vec2 mod_div = mod(in_world_pos.xz, cell0) / dudv;
    float a0 = max2(vec2(1.0) - abs(satv(mod_div) * 2.0 - vec2(1.0)));

    mod_div = mod(in_world_pos.xz, cell1) / dudv;
    float a1 = max2(vec2(1.0) - abs(satv(mod_div) * 2.0 - vec2(1.0)));

    mod_div = mod(in_world_pos.xz, cell2) / dudv;
    float a2 = max2(vec2(1.0) - abs(satv(mod_div) * 2.0 - vec2(1.0)));

    float fade = fract(LOD);

    vec4 col;
    if (a2 > 0.0) {
        col = grid_color_thick;
        col *= a2;
    } else if (a1 > 0.0) {
        col = mix(grid_color_thick, grid_color_thin, fade);
        col *= a1;
    } else {
        col = grid_color_thin;
        col *= (a0 * (1.0 - fade));
    }
    
    float falloff = 1.0 - satf(length(in_world_pos.xz - grid.camera_pos.xz) / grid_size);
    col.a *= falloff;

    col.rgb *= col.a;
    col.a = 1.0;

    out_color = col;
}
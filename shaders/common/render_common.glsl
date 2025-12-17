#ifndef RENDER_COMMON_GLSL
#define RENDER_COMMON_GLSL

struct Global {
    mat4 view;
    mat4 proj;
};

struct Model {
    mat4  world;
    vec4  albedo;

    int   albedo_idx;
    int   metallic_idx;
    int   normal_idx;
    int   roughness_idx;

    int   ao_idx;
    int   height_idx;
    int   arm_idx;
    int   coat_idx;

    int   fuzz_idx;
    float metallic_factor;
    float roughness_factor;
    float ao_factor;

    float height_factor;
    float coat_factor;
    float coat_roughness_factor;
    float fuzz_factor;

    float fuzz_roughness_factor;
    uint  albedo_enable;
    uint  metallic_enable;
    uint  normal_enable;

    uint  roughness_enable;
    uint  ao_enable;
    uint  height_enable;
    uint  arm_enable;

    uint  coat_enable;
    uint  fuzz_enable;
    uint  checker_board_enable;
    uint  p0;

    vec2  tile_uv;
    float p1;
    float p2;
};

#endif
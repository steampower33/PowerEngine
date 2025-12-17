#ifndef PARAMS_GLSL
#define PARAMS_GLSL

layout(set = 0, binding = 0) uniform Global {
    mat4x4 view;
    mat4x4 proj;
} global;

layout(set = 1, binding = 0) uniform Model {
    mat4x4 world;

    vec4 albedo;

    int albedo_idx;
    int metallic_idx;
    int normal_idx;
    int roughness_idx;

    int ao_idx;
    int height_idx;
    float metallic_factor;
    float roughness_factor;

    float ao_factor;
    float height_factor;
    float coat_factor;
    float coat_roughness_factor;

    float fuzz_factor;
    float fuzz_roughness_factor;
    vec2 tile_uv;

    uint albedo_enable;
    uint metallic_enable;
    uint normal_enable;
    uint roughness_enable;

    uint ao_enable;
    uint height_enable;
    uint checker_board_enable;
    uint p2;
} model;

layout(set = 2, binding = 0) uniform sampler2D tex[];

#endif
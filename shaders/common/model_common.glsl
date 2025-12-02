#ifndef PARAMS_GLSL
#define PARAMS_GLSL

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4x4 view;
    mat4x4 proj;

    uint vulkan_thumbnail_index;
    uint p0;
    uint p1;
    uint p2;
} global;

layout(set = 1, binding = 0) uniform ObjectUBO {
    mat4x4 model;

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
    uint p0;
    uint p1;

    uint albedo_enable;
    uint metallic_enable;
    uint normal_enable;
    uint roughness_enable;

    uint ao_enable;
    uint height_enable;
    uint checker_board_enable;
    uint p4;
} object;

layout(set = 2, binding = 0) uniform sampler2D tex[];

#endif
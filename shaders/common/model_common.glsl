#ifndef PARAMS_GLSL
#define PARAMS_GLSL

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4x4 view;
    mat4x4 proj;

    uint vulkanThumbnailIndex;
    uint p0;
    uint p1;
    uint p2;
} global;

layout(set = 1, binding = 0) uniform ObjectUBO {
    mat4x4 model;

    vec4 albedo_use;

    uint albedoIdx;
    uint metallicIdx;
    uint normalIdx;
    uint roughnessIdx;

    uint aoIdx;
    uint heightIdx;
    float metallicFactor;
    float roughnessFactor;

    float aoFactor;
    float heightFactor;
    uint p0;
    uint p1;

    uint albedoEnable;
    uint metallicEnable;
    uint normalEnable;
    uint roughtnessEnable;

    uint aoEnable;
    uint heightEnable;
    uint p3;
    uint p4;
} object;

layout(set = 2, binding = 0) uniform sampler2D tex[];

#endif
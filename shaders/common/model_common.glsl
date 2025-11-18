#ifndef PARAMS_GLSL
#define PARAMS_GLSL

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 view;
    mat4 proj;
} global;

layout(set = 1, binding = 0) uniform ObjectUBO {
    mat4 model;

    vec4 albedo_use;

    uint albedoIdx;
    uint metalnessIdx;
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
    uint metalnessEnable;
    uint normalEnable;
    uint roughtnessEnable;

    uint aoEnable;
    uint heightEnable;
    uint p3;
    uint p4;
} object;

layout(set = 2, binding = 0) uniform sampler2D tex[];

#endif
#version 450

#extension GL_KHR_vulkan_glsl : enable

layout(set = 1, binding = 1) uniform sampler2D tex;

layout(set = 2, binding = 0) uniform Render {
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
    uint roughnessEnable;

    uint aoEnable;
    uint heightEnable;
    uint p3;
    uint p4;
} render;

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vWorldNormal; // world normal

layout(location = 0) out vec4 outAlbedoMetal;
layout(location = 1) out vec4 outNormalRough;
layout(location = 2) out vec4 outHeightAO;

void main() {
//    vec3 albedo = (render.albedoEnable == 0u) ? pow(render.albedo_use.rgb,  vec3(2.2)) : texture(tex, vUV).rgb;
//
//    float metallic = (render.metallicEnable == 0u) ? 0.0 : render.metallicFactor;
//    float roughness = (render.roughnessEnable == 0u) ? 0.0 : render.roughnessFactor;
//    float ao = (render.aoEnable == 0u) ? 0.0 : render.aoFactor;

    vec3 albedo = texture(tex, vUV).rgb;
    float metallic = render.metallicFactor;
    float roughness = render.roughnessFactor;
    float ao = render.aoFactor;

    outAlbedoMetal = vec4(albedo, metallic);
    outNormalRough = vec4(vWorldNormal * 0.5 + 0.5, roughness); // [-1,1] -> [0,1]
    outHeightAO = vec4(0.0, ao, 0.0, 0.0);
}

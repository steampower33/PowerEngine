#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

#include "../common/model_common.glsl"

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vNormalWorld;
layout(location = 2) in vec3 vPosWorld;
layout(location = 3) in vec3 vTangentWorld;

layout(location = 0) out vec4 outAlbedoMetal;
layout(location = 1) out vec4 outNormalRough;
layout(location = 2) out vec4 outHeightAO;

void main() {
    vec4 albedo    = (object.albedoEnable == 0u) ? vec4(object.albedo_use.xyz, 0.0): texture(tex[nonuniformEXT(object.albedoIdx)], vUV);
    vec3 normalTS  = vNormalWorld;
    float metallic = (object.metallicEnable == 0u) ? object.metallicFactor : texture(tex[nonuniformEXT(object.metallicIdx)], vUV).r;
    float rough    = (object.roughnessEnable == 0u) ? object.roughnessFactor : texture(tex[nonuniformEXT(object.roughnessIdx)], vUV).r;
    float ao       = (object.aoEnable == 0u) ? object.aoFactor : texture(tex[nonuniformEXT(object.aoIdx)], vUV).r;
    float height   = (object.heightEnable == 0u) ? object.heightFactor : texture(tex[nonuniformEXT(object.heightIdx)], vUV).r;

    if (object.normalEnable != 0u)
    {
        vec3 normal = texture(tex[nonuniformEXT(object.normalIdx)], vUV).xyz * 2.0 - 1.0;

        vec3 N = vNormalWorld;
        vec3 T = normalize(vTangentWorld - dot(vTangentWorld, N) * N);
        vec3 B = cross(N, T);
        
        mat3x3 TBN = mat3x3(T, B, N);
        normalTS = normalize(normal * TBN);
    }

    // RT0: albedo.rgb + metallic 
    outAlbedoMetal = vec4(albedo.xyz, metallic);

    // RT1: world-normal + roughness (여기선 예시로 그냥 tangent-space normal)
    vec3 n = normalize(normalTS);
    // [-1,1] → [0,1]
    outNormalRough = vec4(n * 0.5 + 0.5, rough);

    // RT2: height + AO
    outHeightAO = vec4(height, ao, 0.0, 0.0);
}
#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 2, binding = 0) uniform Render {
    vec4 albedo_use;

    int albedoIdx;
    int metallicIdx;
    int normalIdx;
    int roughnessIdx;

    int aoIdx;
    int heightIdx;
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

layout(set = 3, binding = 0) uniform sampler2D tex[];

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vWorldNormal; // world normal

layout(location = 0) out vec4 outAlbedoMetal;
layout(location = 1) out vec4 outNormalRough;
layout(location = 2) out vec4 outHeightAO;

void main() {
    vec4 albedo    = (render.albedoEnable == 0u) ? vec4(render.albedo_use.xyz, 0.0): texture(tex[nonuniformEXT(render.albedoIdx)], vUV);
    vec3 normalTS  = vWorldNormal;
    float metallic = (render.metallicEnable == 0u) ? render.metallicFactor : texture(tex[nonuniformEXT(render.metallicIdx)], vUV).r;
    float roughness    = (render.roughnessEnable == 0u) ? render.roughnessFactor : texture(tex[nonuniformEXT(render.roughnessIdx)], vUV).r;
    float ao       = (render.aoEnable == 0u) ? render.aoFactor : texture(tex[nonuniformEXT(render.aoIdx)], vUV).r;
    float height   = (render.heightEnable == 0u) ? render.heightFactor : texture(tex[nonuniformEXT(render.heightIdx)], vUV).r;

    outAlbedoMetal = vec4(albedo.xyz, metallic);
    outNormalRough = vec4(normalTS * 0.5 + 0.5, roughness); // [-1,1] -> [0,1]
    outHeightAO = vec4(0.0, ao, 0.0, 0.0);
}

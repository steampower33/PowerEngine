#version 450

#extension GL_KHR_vulkan_glsl : enable

layout(set = 1, binding = 1) uniform sampler2D tex;

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vWorldNormal; // world normal

layout(location = 0) out vec4 outAlbedoMetal;
layout(location = 1) out vec4 outNormalRough;
layout(location = 2) out vec4 outHeightAO;

void main() {
    vec4 albedo = texture(tex, vUV);
    
    vec3 albedoLinear = pow(albedo.rgb, vec3(2.2));

    float metallic = 0.5;
    float ao = 1.0;
    outAlbedoMetal = vec4(albedoLinear, metallic);
    outNormalRough = vec4(vWorldNormal * 0.5 + 0.5, 1.0 - metallic); // [-1,1] -> [0,1]
    outHeightAO = vec4(0.0, ao, 0.0, 0.0);
}

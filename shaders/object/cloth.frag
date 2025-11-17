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

    outAlbedoMetal = vec4(albedoLinear, 0.0);
    outNormalRough = vec4(vWorldNormal * 0.5 + 0.5, 0.7); // [-1,1] -> [0,1]
    outHeightAO = vec4(0.0, 0.0, 0.0, 1.0);
}

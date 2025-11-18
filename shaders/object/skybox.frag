#version 450
#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 1, binding = 0) uniform SkyboxUBO {
    mat4 model;
    uint envIdx;
    uint radianceIdx;
    uint irradianceIdx;
    uint p0;
} skybox;

// IBL용 env cube map 배열이라고 가정
layout(set = 2, binding = 0) uniform samplerCube envTex[];

layout(location = 0) in vec3 vDir;

layout(location = 0) out vec4 outAlbedoMetal;
layout(location = 1) out vec4 outNormalRough;
layout(location = 2) out vec4 outHeightAO;

void main() {
    int envIndex = int(skybox.envIdx);

    vec3 color = texture(envTex[nonuniformEXT(envIndex)], vDir).rgb;

    outAlbedoMetal = vec4(color, 0.0);
//    outAlbedoMetal = vec4(0.5, 0.5, 0.5, 0.0);
}

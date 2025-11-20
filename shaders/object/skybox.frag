#version 450
#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 1, binding = 0) uniform SkyboxUBO {
    mat4x4 model;

    int envIdx;
    int radianceIdx;
    int irradianceIdx;
    uint specularMipLevels;

    int brdfLUTIndex;
    uint p0;
    uint p1;
    uint p2;
} skybox;

layout(set = 2, binding = 0) uniform samplerCube envTex[];

layout(location = 0) in vec3 vDir;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 color = texture(envTex[nonuniformEXT(skybox.envIdx)], vDir).rgb;

    outColor = vec4(color, 0.0);
}

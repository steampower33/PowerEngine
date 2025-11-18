#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 vDir;

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 view;
    mat4 proj;
} global;

layout(set = 1, binding = 0) uniform SkyboxUBO {
    mat4x4 model;
    uint envIdx;
    uint radianceIdx;
    uint irradianceIdx;
    uint p0;
} skybox;

void main() {
    mat4 viewRot = mat4(mat3(global.view));

    vec4 posModel =  vec4(inPos, 1.0);
    vec4 posWorld = skybox.model * posModel;
    vec4 posView = global.view * posWorld;
    gl_Position = global.proj * posView;

    vDir = posWorld.xyz;
}
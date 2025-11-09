#version 450

#extension GL_KHR_vulkan_glsl : enable

layout(set = 1, binding = 1) uniform sampler2D tex;

layout(location=0) in vec2 vUV;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(tex, vUV);
}

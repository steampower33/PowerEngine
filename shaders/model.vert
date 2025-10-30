#version 450

layout(set = 0, binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec4 inPos;   // VSInput.pos
layout(location = 1) in vec4 inUV;    // VSInput.uv (float4������ vec4�� �޵� .xy�� ���)

layout(location = 0) out vec2 vUV;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * inPos;
    vUV = vec2(inUV.x, 1.0 - inUV.y); // ������ ������ Y flip
}
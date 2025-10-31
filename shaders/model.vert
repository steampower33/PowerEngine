#version 450

layout(set=0, binding=0) uniform GlobalUBO { mat4 view; mat4 proj; } global;
layout(set=0, binding=1) uniform ObjectUBO { mat4 model; } object;

layout(location = 0) in vec4 inPos;   // VSInput.pos
layout(location = 1) in vec4 inUV;    // VSInput.uv (float4였으니 vec4로 받되 .xy만 사용)

layout(location = 0) out vec2 vUV;

void main() {
    gl_Position = global.proj * global.view * object.model * inPos;
    vUV = vec2(inUV.x, 1.0 - inUV.y); // 기존과 동일한 Y flip
}
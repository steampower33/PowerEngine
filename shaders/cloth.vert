#version 450

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 proj;
} ubo;

layout(set=1, binding=2, std430) readonly buffer Positions { vec4 X[]; };
layout(set=1, binding=3, std430) readonly buffer Indices   { uint I[]; }; // ← 새로 추가

void main() {

    uint vid = gl_VertexIndex;
    vec3 p = X[vid].xyz;

    gl_Position = ubo.proj * ubo.view * vec4(p, 1.0);
}
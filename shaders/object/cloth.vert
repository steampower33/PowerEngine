#version 450

#extension GL_KHR_vulkan_glsl : enable

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 proj;
} ubo;

layout(set=1, binding=0, std430) readonly buffer Positions { vec4 pos[]; };

layout(push_constant) uniform ClothPC { uint nx1; uint ny1; } pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vWorldNormal; // world normal

void main() {

    uint vid = gl_VertexIndex;
    uint nx1 = pc.nx1;
    uint ny1 = pc.ny1;
    
    uint x = vid % nx1;
    uint y = vid / nx1;
    
    vec3 p = pos[vid].xyz;
    
    uint xL = (x > 0)      ? (x - 1) : x;
    uint xR = (x + 1 < nx1)? (x + 1) : x;
    uint yD = (y > 0)      ? (y - 1) : y;
    uint yU = (y + 1 < ny1)? (y + 1) : y;
    
    uint idL = y * nx1 + xL;
    uint idR = y * nx1 + xR;
    uint idD = yD * nx1 + x;
    uint idU = yU * nx1 + x;
    
    vec3 pL = pos[idL].xyz;
    vec3 pR = pos[idR].xyz;
    vec3 pD = pos[idD].xyz;
    vec3 pU = pos[idU].xyz;
    
    // 격자 방향 벡터
    vec3 dx = pR - pL;
    vec3 dy = pU - pD;

    // 노말 계산 (방향은 필요에 따라 cross 순서 바꿔서 뒤집을 수 있음)
    vec3 N = normalize(cross(dy, dx));
    
    vWorldNormal = N;

    // 간단히 UV도 격자 좌표로 계산 (0~1)
    vUV = vec2(float(x) / float(nx1 - 1),
               float(y) / float(ny1 - 1));

    gl_Position = ubo.proj * ubo.view * vec4(p, 1.0);
}
#version 450

layout(set=0, binding=0) uniform GlobalUBO { 
    mat4 view; 
    mat4 proj;
} global;

layout(set=1, binding=0) uniform ObjectUBO { 
    mat4 model; 
    vec4 color_use; 
	uint albedo;
	uint ao;
	uint roughness;
	uint metallic;
	uint height;
	uint normal;
	uint pad0;
	uint pad1;
} object;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vNormalWorld;   // 월드 공간 노멀
layout(location = 2) out vec3 vPosWorld;      // 월드 위치
layout(location = 3) out vec4 vTangentWorld;

void main() {
    // 월드 위치
    vec4 worldPos = object.model * vec4(inPos, 1.0);
    gl_Position = global.proj * global.view * worldPos;

    vUV = inUV;

    // 일반적인 노말 매트릭스
    mat3 normalMat = transpose(inverse(mat3(object.model)));
    vNormalWorld = normalize(normalMat * inNormal);
    vPosWorld = worldPos.xyz;
    vTangentWorld = object.model * inTangent;
}
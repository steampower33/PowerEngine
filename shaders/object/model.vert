#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

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
	uint chooseTexIdx;
	float heightScale;
} object;

layout(set=1, binding=1) uniform sampler2D tex[];

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vNormalWorld;   // 월드 공간 노멀
layout(location = 2) out vec3 vPosWorld;      // 월드 위치
layout(location = 3) out vec3 vTangentWorld;

void main() {
    vUV = inUV;

    // 일반적인 노말 매트릭스
    mat3 normalMat = transpose(inverse(mat3(object.model)));
    vNormalWorld = normalize(normalMat * inNormal);
    
    vec4 worldPos = object.model * vec4(inPos, 1.0);
//    if (object.height != 0u)
//    {
//        float height = texture(tex[nonuniformEXT(object.height)], vUV).r;
//        height = height * 2.0 - 1.0;
//        worldPos += vec4(vNormalWorld * height * object.heightScale, 0.0);
//    }
    
    vPosWorld = worldPos.xyz;
    vTangentWorld = (object.model * vec4(inTangent, 0.0)).xyz;
    
    gl_Position = global.proj * global.view * worldPos;
}
#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

#include "../common/model_common.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vNormalWorld;   // 월드 공간 노멀
layout(location = 2) out vec3 vPosWorld;      // 월드 위치
layout(location = 3) out vec3 vTangentWorld;

void main() {
    // 월드 위치
    vec4 worldPos = object.model * vec4(inPos, 1.0);

    // 일반적인 노말 매트릭스
    mat3 normalMat = transpose(inverse(mat3(object.model)));
    vNormalWorld = normalize(normalMat * inNormal);    

    if (object.heightEnable == 1u)
    {
        float height = texture(tex[nonuniformEXT(object.heightIdx)], inUV).r;
        height = height * 2.0 - 1.0;
        worldPos += vec4(vNormalWorld * height * object.heightFactor, 0.0);
    }
    gl_Position = global.proj * global.view * worldPos;

    vPosWorld = worldPos.xyz;
    vUV = inUV;
    vTangentWorld = (object.model * vec4(inTangent, 0.0)).xyz;
}
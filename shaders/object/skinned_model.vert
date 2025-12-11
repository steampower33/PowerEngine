#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

#include "../common/model_common.glsl"

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec3 in_tangent;
layout(location = 4) in uvec4 in_joints;
layout(location = 5) in vec4 in_weights;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_normal_world;
layout(location = 2) out vec3 out_pos_world;
layout(location = 3) out vec3 out_tangent_world;

layout(set = 3, binding = 0) uniform BonesUBO {
    mat4 uBones[128];
} bones;

void main()
{
    mat4 skinMat = mat4(0.0);

    skinMat += bones.uBones[in_joints.x] * in_weights.x;
    skinMat += bones.uBones[in_joints.y] * in_weights.y;
    skinMat += bones.uBones[in_joints.z] * in_weights.z;
    skinMat += bones.uBones[in_joints.w] * in_weights.w;

    vec4 skinnedPos    = skinMat * vec4(in_pos, 1.0);
    vec3 skinnedNormal = mat3(skinMat) * in_normal;
    vec3 skinnedTangent = mat3(skinMat) * in_tangent;

    vec4 worldPos      = object.model * skinnedPos;
    vec3 worldNormal   = normalize(mat3(object.model) * skinnedNormal);
    vec3 worldTangent  = normalize(mat3(object.model) * skinnedTangent);

    gl_Position = global.proj * global.view * worldPos;

    out_uv          = in_uv;
    out_pos_world   = worldPos.xyz;
    out_normal_world = worldNormal;
    out_tangent_world = worldTangent;
}
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
	uint pad1;
} object;

layout(set=1, binding=1) uniform sampler2D tex[];

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vNormalWorld;   // 월드 공간 노멀
layout(location = 2) in vec3 vPosWorld;      // 월드 위치
layout(location = 3) in vec4 vTangentWorld;

layout(location = 0) out vec4 outAlbedoMetal;
layout(location = 1) out vec4 outNormalRough;
layout(location = 2) out vec4 outHeightAO;

void main() {
    int albedoIndex    = int(object.albedo);
    int normalIndex    = int(object.normal);
    int metallicIndex  = int(object.metallic);
    int roughnessIndex = int(object.roughness);
    int heightIndex    = int(object.height);
    int aoIndex        = int(object.ao);

    vec4 albedo    = (albedoIndex == 0u) ? vec4(object.color_use.xyz, 0.0): texture(tex[nonuniformEXT(albedoIndex)], vUV);
    vec3 normalTS  = vNormalWorld;
    float metallic = (metallicIndex == 0u) ? 0.0 : texture(tex[nonuniformEXT(metallicIndex)], vUV).r;
    float rough    = (roughnessIndex == 0u) ? 0.0 : texture(tex[nonuniformEXT(roughnessIndex)], vUV).r;
    float height   = (heightIndex == 0u) ? 0.0 : texture(tex[nonuniformEXT(heightIndex)], vUV).r;
    float ao       = (aoIndex == 0u) ? 0.0 : texture(tex[nonuniformEXT(aoIndex)], vUV).r;

    if (normalIndex != 0u) // NormalWorld를 교체
    {
        vec3 normal = texture(tex[nonuniformEXT(normalIndex)], vUV).xyz * 2.0 - 1.0;

        vec3 N = vNormalWorld;
        vec3 T = normalize(vTangentWorld.xyz - dot(vTangentWorld.xyz, N) * N);
        vec3 B = cross(N, T);
        
        mat3x3 TBN = mat3x3(T, B, N);
        normalTS = normalize(normal * TBN);
    }

    // sRGB → linear 필요하면 여기서 변환
    vec3 albedoLinear = pow(albedo.rgb, vec3(2.2));

    // RT0: albedo.rgb + metallic 
    outAlbedoMetal = vec4(albedoLinear, metallic);

    // RT1: world-normal + roughness (여기선 예시로 그냥 tangent-space normal)
    vec3 n = normalize(normalTS);
    // [-1,1] → [0,1] 로 패킹
    outNormalRough = vec4(n * 0.5 + 0.5, rough);

    // RT2: height + AO (나머지 채널은 필요에 따라 reserve)
    outHeightAO = vec4(height, ao, 0.0, 0.0);
}
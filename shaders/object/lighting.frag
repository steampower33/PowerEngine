#version 450
#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform LightUBO {
    vec4 cameraPos;
    vec4 spotPos_range;
    vec4 spotDir_inner;
    vec4 spotColor_outer;
    mat4 invViewProj;
    float exposure;
    float p0;
    float p1;
    float p2;
} light;

layout(set = 0, binding = 1) uniform sampler2D gAlbedoMetal;
layout(set = 0, binding = 2) uniform sampler2D gNormalRough;
layout(set = 0, binding = 3) uniform sampler2D gHeightAo;
layout(set = 0, binding = 4) uniform sampler2D gDepth;

layout(set = 1, binding = 0) uniform SkyboxUBO {
    mat4x4 model;

    uint envIdx;
    uint radianceIdx;
    uint irradianceIdx;
    uint specularMipLevels;

    uint brdfLUTIndex;
    uint p0;
    uint p1;
    uint p2;
} skybox;

layout(set = 2, binding = 0) uniform sampler2D tex[];
layout(set = 3, binding = 0) uniform samplerCube envTex[];

layout(location = 0) in vec2 vUV;

layout(location = 0) out vec4 outColor;

vec3 reconstructWorldPos(vec2 uv, float depth01)
{
    // NDC 좌표 (x,y in [-1,1], z in [0,1])
    vec2 ndcXY = uv * 2.0 - 1.0;
    float ndcZ = depth01;   // 여기서 depth01 그대로 쓰는 패턴

    vec4 clipPos = vec4(ndcXY, ndcZ, 1.0);

    // invViewProj = inverse(P * V)
    vec4 world = light.invViewProj * clipPos;
    return world.xyz / world.w;
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    // Epic / Filament 스타일
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) *
                pow(1.0 - cosTheta, 5.0);
}

// Narkowicz ACES Filmic
vec3 TonemapACES(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e),
                 0.0, 1.0);
}

void main()
{
    float depth = texture(gDepth, vUV).r;
    
    // worldPos 복원
    vec3 worldPos = reconstructWorldPos(vUV, depth);
    
    if (depth >= 1.0 - 1e-5) {
        vec3 sky = texture(envTex[nonuniformEXT(skybox.radianceIdx)], worldPos).rgb; // or 별도 skyboxCube
        outColor = vec4(sky, 1.0);
        return;
    }
    
    vec4 am = texture(gAlbedoMetal, vUV);
    vec4 nr = texture(gNormalRough, vUV);
    vec4 ha = texture(gHeightAo, vUV);

    vec3 albedo    = am.rgb;
    float metallic = am.a;

    vec3 normal = normalize(nr.rgb * 2.0 - 1.0);
    float roughness = nr.a;
    float height = ha.r;
    float ao = ha.g;

    vec3 cameraPos   = light.cameraPos.xyz;

    // 기본 벡터 계산
    vec3 N = normalize(normal);
    vec3 V = normalize(cameraPos - worldPos);
    float NdotV = max(dot(N, V), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);  // 금속이면 albedo가 F0

    vec3 diffuseIrr = texture(envTex[nonuniformEXT(skybox.irradianceIdx)], N).rgb;
    vec3 diffuseIBL = diffuseIrr * albedo;

    vec3 R = reflect(-V, N);

    float maxMip = float(skybox.specularMipLevels); // push constant or #define
    float lod = roughness * maxMip;

    vec3 prefilteredColor = textureLod(envTex[nonuniformEXT(skybox.radianceIdx)], R, lod).rgb;

    // (N·V, roughness) 로 LUT 샘플링
    vec2 brdf = texture(tex[nonuniformEXT(skybox.brdfLUTIndex)], vec2(NdotV, roughness)).rg;

    vec3 F = FresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    vec3 specularIBL = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambientDiffuse = kD * diffuseIBL * ao;
    vec3 ambientSpec    = specularIBL; // 필요하면 여기도 약하게 ao 곱해도 됨

    vec3 color = ambientDiffuse + ambientSpec;

    color = max(color, vec3(0.0));
    
    // HDR → LDR (Tone Mapping)
    float exposure = light.exposure; // UBO에서 조절
    color *= exposure;
    color  = TonemapACES(color);

    color = pow(color, vec3(1.0 / 2.2));
     
    outColor = vec4(color, 1.0);
}

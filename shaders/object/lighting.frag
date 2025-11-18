#version 450

layout(set = 0, binding = 0) uniform LightUBO {
    vec4 cameraPos;
    vec4 spotPos_range;
    vec4 spotDir_inner;
    vec4 spotColor_outer;
    mat4 invViewProj;
} light;

layout(set = 0, binding = 1) uniform sampler2D gAlbedoMetal;
layout(set = 0, binding = 2) uniform sampler2D gNormalRough;
layout(set = 0, binding = 3) uniform sampler2D gHeightAo;
layout(set = 0, binding = 4) uniform sampler2D gDepth;

layout(set = 1, binding = 0) uniform SkyboxUBO {
    uint envIdx;
    uint radianceIdx;
    uint irradianceIdx;
    uint p0;
} skybox;

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

void main()
{
    vec4 am = texture(gAlbedoMetal, vUV);
    vec4 nr = texture(gNormalRough, vUV);
    vec4 ha = texture(gHeightAo, vUV);
    float depth01 = texture(gDepth, vUV).r;

    // G-buffer
    vec3 albedo    = am.rgb;
    float metallic = am.a;

    vec3 normal    = normalize(nr.rgb * 2.0 - 1.0);
    float rough    = nr.a;
    float height   = ha.r;
    float ao       = ha.g;

    // worldPos 복원
    vec3 worldPos = reconstructWorldPos(vUV, depth01);

    // 스팟 라이트 파라미터 (world space)
    vec3 camPos  = light.cameraPos.xyz;
    vec3 spotPos = light.spotPos_range.xyz;
    float spotRange = light.spotPos_range.w;

    vec3 spotDir    = normalize(light.spotDir_inner.xyz);
    float cosInner  = light.spotDir_inner.w;
    vec3 spotColor  = light.spotColor_outer.rgb;
    float cosOuter  = light.spotColor_outer.a;

    // 기본 벡터 계산
    vec3 N = normalize(normal);
    vec3 L = normalize(spotPos - worldPos);
    vec3 V = normalize(camPos - worldPos);

    float NdotL = max(dot(N, L), 0.0);

    // 각도 기반 스팟 falloff
    float cosTheta = dot(-L, spotDir); // L은 표면→라이트, 스팟 방향은 라이트→아래
    float spotFactor = smoothstep(cosOuter, cosInner, cosTheta);

    // 거리 감쇠
    float dist = length(spotPos - worldPos);
    float rangeFactor = clamp(1.0 - dist / spotRange, 0.0, 1.0);

    float intensity = NdotL * spotFactor * rangeFactor;

    vec3 diffuse = albedo * spotColor * intensity;

    // 아주 간단한 ambient
    vec3 ambient = albedo * 0.05 * ao;

//    outColor = vec4(diffuse + ambient, 1.0);
    outColor = vec4(albedo, 1.0);
}

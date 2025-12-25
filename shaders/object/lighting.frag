#version 450
#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform LightUBO {
    mat4 inv_view_proj;

    vec4 camera_pos;
    vec3 position;
    float intensity;

    vec3 direction;
    float inner;

    float outer;
    uint light_enable;
    uint ibl_enable;
    float exposure;

    int ggx_brdf_idx;
    int charlie_brdf_idx;
    int sheen_e_brdf_idx;
    int p0;

    mat4 light_view_proj;
    vec2 shadow_map_inv_size;
    float shadow_strength;
    uint shadow_enable;
} light;

layout(set = 0, binding = 1) uniform sampler2D albedo_metal;
layout(set = 0, binding = 2) uniform sampler2D normal_rough;
layout(set = 0, binding = 3) uniform sampler2D height_ao;
layout(set = 0, binding = 4) uniform sampler2D cozz_fuzz;
layout(set = 0, binding = 5) uniform sampler2D depth_tex;
layout(set = 0, binding = 6) uniform sampler2DShadow shadow_map;

layout(set = 1, binding = 0) uniform SkyboxUBO {
    int env_idx;
    int radiance_idx;
    int irradiance_idx;
    int specular_mip_levels;
} skybox;

layout(set = 2, binding = 0) uniform sampler2D tex[];
layout(set = 3, binding = 0) uniform samplerCube env_tex[];

layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

vec3 reconstruct_world_pos(vec2 uv, float depth01)
{
    vec2 ndc_xy = uv * 2.0 - 1.0;
    float ndc_z = depth01;

    vec4 clip_pos = vec4(ndc_xy, ndc_z, 1.0);

    vec4 world = light.inv_view_proj * clip_pos;
    return world.xyz / world.w;
}

vec3 fresnel_schlick_roughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) *
                pow(1.0 - cosTheta, 5.0);
}

vec3 tonemap_aces(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e),
                 0.0, 1.0);
}

const float PI = 3.14159265359;

float D_GGX(float n_dot_h, float roughness)
{
    float a  = roughness * roughness;
    float a2 = a * a;
    float n_dot_h2 = n_dot_h * n_dot_h;

    float denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float D_Charlie(float NdotH, float roughness)
{
    float invAlpha = 1.0 / (roughness * roughness + 1e-6); 

    float cos2 = NdotH * NdotH;
    float sin2 = max(1.0 - cos2, 0.0);
    float sinTheta = sqrt(sin2);

    // (2 + 1/alpha) * sinTheta^(1/alpha) / (2��)
    float power = pow(sinTheta, invAlpha);
    return ( (2.0 + invAlpha) * power ) * (1.0 / (2.0 * PI));
}

float G_SchlickGGX(float NdotX, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;  // UE4 ��Ÿ��

    return NdotX / (NdotX * (1.0 - k) + k);
}

float G_Smith(float n_dot_l, float n_dot_v, float roughness)
{
    float g1 = G_SchlickGGX(n_dot_l, roughness);
    float g2 = G_SchlickGGX(n_dot_v, roughness);
    return g1 * g2;
}

vec3 fresnel_schlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float V_Neubelt(float NdotV, float NdotL)
{
    return 1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV + 1e-5));
}

float F_Sheen(float LdotH)
{
    float x = 1.0 - LdotH;
    float x2 = x * x;
    return x2 * x2 * x; // (1 - L��H)^5
}

float apply_shadow_strength(float s, float shadowStrength)
{
    return mix(1.0, s, shadowStrength);
}

float shadow_PCF(vec3 world_pos, vec3 world_normal)
{
    vec4 lp = light.light_view_proj * vec4(world_pos, 1.0);
    vec3 ndc = lp.xyz / lp.w;

    vec2 uv = ndc.xy * 0.5 + 0.5;
    float z = ndc.z; // ZO assumed

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;

    float bias = 0.0001;
    z -= bias;

    vec2 texel = light.shadow_map_inv_size;

    float sum = 0.0;
    for (int y=-2; y<=2; ++y)
    for (int x=-2; x<=2; ++x)
        sum += texture(shadow_map, vec3(uv + vec2(x,y)*texel, z));

    return sum / 25.0;
}

void main()
{
    float depth = texture(depth_tex, in_uv).r;
    
    vec3 world_pos = reconstruct_world_pos(in_uv, depth);
    
    vec4 am = texture(albedo_metal, in_uv);
    vec4 nr = texture(normal_rough, in_uv);
    vec4 ha = texture(height_ao, in_uv);
    vec4 cf = texture(cozz_fuzz, in_uv);

    vec3  albedo   = am.rgb;
    float metallic = am.a;

    vec3  normal    = normalize(nr.rgb * 2.0 - 1.0);
    float roughness = nr.a;

    float height = ha.r;
    float ao     = ha.g;
    
    float coat_factor           = cf.r;
    float coat_roughness_factor = cf.g;
    float fuzz_factor           = cf.b;
    float fuzz_roughness_factor = cf.a;

    vec3 camera_pos = light.camera_pos.xyz;
    vec3 N = normal;
    vec3 color = vec3(0.0);

    if (light.ibl_enable == 1u)
    {
        if (depth >= 1.0 - 1e-5) {
            vec3 dir = normalize(world_pos - camera_pos);

            vec3 sky = texture(env_tex[nonuniformEXT(skybox.radiance_idx)], dir).rgb;

            float exposure = light.exposure;
            sky *= exposure;
            sky = tonemap_aces(sky);

            out_color = vec4(sky, 1.0);
            return;
        }

        vec3 V = normalize(camera_pos - world_pos);
        float n_dot_v = max(dot(N, V), 0.0);

        // base : diffuse IBL
        vec3 diffuse_irr = texture(env_tex[nonuniformEXT(skybox.irradiance_idx)], N).rgb;
        vec3 diffuse_ibl = diffuse_irr * albedo;

        // base : specular IBL
        vec3 F0_base = mix(vec3(0.04), albedo, metallic);
        vec3 F_base  = fresnel_schlick_roughness(n_dot_v, F0_base, roughness);
        vec3 kS_base = F_base;
        vec3 kD_base = (vec3(1.0) - kS_base) * (1.0 - metallic);

        vec3 R = reflect(-V, N);

        float max_mip = float(skybox.specular_mip_levels - 1);

        float lod_base = roughness * max_mip;
        vec3 prefiltered_base = textureLod(env_tex[nonuniformEXT(skybox.radiance_idx)], R, lod_base).rgb;

        vec2 brdf_base = texture(tex[nonuniformEXT(light.ggx_brdf_idx)], vec2(n_dot_v, roughness)).rg;
        vec3 specular_ibl_base = prefiltered_base * (F_base * brdf_base.x + brdf_base.y);
        
        vec3 ambient_diffuse = kD_base * diffuse_ibl * ao;
        vec3 ambient_spec    = specular_ibl_base;
     
        vec3 base_ibl = ambient_diffuse + ambient_spec;

        // coat
        vec3 coat_ibl = vec3(0.0);
        if (coat_factor > 0.0) {
            float coat_roughness = clamp(coat_roughness_factor, 0.0, 1.0);

            vec3 F0_coat = vec3(0.04);

            float lod_coat = coat_roughness * max_mip;
            vec3 prefiltered_coat = textureLod(
                env_tex[nonuniformEXT(skybox.radiance_idx)], R, lod_coat
            ).rgb;

            vec2 brdf_coat = texture(
                tex[nonuniformEXT(light.ggx_brdf_idx)],
                vec2(n_dot_v, coat_roughness)
            ).rg;

            vec3 F_coat = fresnel_schlick_roughness(n_dot_v, F0_coat, coat_roughness_factor);

            coat_ibl = prefiltered_coat * (F_coat * brdf_coat.x + brdf_coat.y);
            coat_ibl *= coat_factor;

            vec3 coat_transmit = vec3(1.0) - F_coat * coat_factor;
            coat_transmit = clamp(coat_transmit, 0.0, 1.0);
            base_ibl *= coat_transmit;
        }

        // fuzz
        vec3 fuzz_ibl = vec3(0.0);
        if (fuzz_factor > 0.0) {
            float fuzz_roughness = clamp(fuzz_roughness_factor, 0.0, 1.0);

            float lod_fuzz = fuzz_roughness * max_mip;
            vec3 prefiltered_fuzz = textureLod(
                env_tex[nonuniformEXT(skybox.radiance_idx)], R, lod_fuzz
            ).rgb;
            
            vec2 sheen_e = texture(tex[nonuniformEXT(light.sheen_e_brdf_idx)],
                       vec2(n_dot_v, fuzz_roughness)).rg;

            vec3 fuzz_color = mix(vec3(1.0), albedo, 0.5);

            vec3 fuzz_spec = prefiltered_fuzz * (fuzz_color * sheen_e.x + sheen_e.y);
            fuzz_spec *= fuzz_factor;
            fuzz_ibl = fuzz_spec;

            float fuzzEnergy = clamp(fuzz_factor * 0.5, 0.0, 1.0);
            base_ibl *= (1.0 - fuzzEnergy);
        }

        vec3 ibl = base_ibl + coat_ibl + fuzz_ibl;

        ibl = max(ibl, vec3(0.0));

        float exposure = light.exposure;
        ibl *= exposure;

        color += ibl;
    }
    
    if (light.light_enable == 1u)
    {
        if (depth >= 1.0 - 1e-5) {
            out_color = vec4(0.0f);
            return;
        }

        vec3 light_pos = light.position;

        vec3 V = normalize(camera_pos - world_pos);
        vec3 L = normalize(light_pos - world_pos);
        vec3 H = normalize(V + L);

        float n_dot_l = max(dot(N, L), 0.0);
        float n_dot_v = max(dot(N, V), 0.0);
        float n_dot_h = max(dot(N, H), 0.0);
        float v_dot_h = max(dot(V, H), 0.0);

        vec3 spotAxis = normalize(light.direction);
        float cos_theta = dot(spotAxis, -L);
        float inner_cos = cos(radians(light.inner));
        float outer_cos = cos(radians(light.outer));
        float spot = smoothstep(outer_cos, inner_cos, cos_theta);
        float dist = length(light_pos - world_pos);
        float attenuation = 1.0 / (dist * dist);

        vec3 F0_base = mix(vec3(0.04), albedo, metallic);
        vec3 F_base  = fresnel_schlick(v_dot_h, F0_base);
        
        float D_base = D_GGX(n_dot_h, roughness);
        float G_base = G_Smith(n_dot_l, n_dot_v, roughness);

        float denom_base = max(4.0 * n_dot_l * n_dot_v, 1e-4);
        vec3 spec_brdf_base = (D_base * G_base * F_base) / denom_base;

        vec3 kS_base = F_base;
        vec3 kD_base = (vec3(1.0) - kS_base) * (1.0 - metallic);

        vec3 diffuse_brdf_base = kD_base * albedo / PI;
        vec3 brdf_base = diffuse_brdf_base + spec_brdf_base;

        vec3 direct_base = brdf_base * n_dot_l * light.intensity * attenuation * spot;

        vec3 direct_coat = vec3(0.0);
        if (coat_factor > 0.0)
        {
            float coat_rough = clamp(coat_roughness_factor, 0.0, 1.0);

            vec3 F0_coat = vec3(0.04);
            vec3 F_coat = fresnel_schlick(v_dot_h, F0_coat);

            float D_coat = D_GGX(n_dot_h, coat_rough);
            float G_coat = G_Smith(n_dot_l, n_dot_v, coat_rough);

            float denom_coat = max(4.0 * n_dot_l * n_dot_v, 1e-4);
            vec3 spec_brdf_coat = (D_coat * G_coat * F_coat) / denom_coat;

            direct_coat = spec_brdf_coat * n_dot_l
                              * light.intensity * attenuation * spot
                              * coat_factor;
                              
            vec3 coatTransmit = vec3(1.0) - F_coat * coat_factor;
            coatTransmit = clamp(coatTransmit, 0.0, 1.0);
            direct_base *= coatTransmit;
        }

        vec3 direct_fuzz = vec3(0.0);
        if (fuzz_factor > 0.0)
        {
            float fuzz_rough = clamp(fuzz_roughness_factor, 0.0, 1.0);
            
            float D_fuzz = D_Charlie(n_dot_h, fuzz_rough);
            
            float Vis = V_Neubelt(n_dot_v, n_dot_l);
            
            float l_dot_h = clamp(dot(L, H), 0.0, 1.0);
            float F_sheen = F_Sheen(l_dot_h);
            
            vec3 fuzz_color = mix(vec3(1.0), albedo, 0.5);

            vec3 spec_brdf_fuzz = D_fuzz * Vis * F_sheen * fuzz_color;

            direct_fuzz = spec_brdf_fuzz * n_dot_l
                              * light.intensity * attenuation * spot
                              * fuzz_factor;

            float fuzzEnergy = clamp(fuzz_factor * 0.5, 0.0, 1.0);
            direct_base *= (1.0 - fuzzEnergy);
            direct_coat *= (1.0 - fuzzEnergy);
        }
        
        vec3 direct = direct_base + direct_coat + direct_fuzz;

        float s = 1.0;
        if (light.shadow_enable == 1u)
        {
            s = shadow_PCF(world_pos, N);
            s = apply_shadow_strength(s, light.shadow_strength);
        }
        color += direct * s;
    }

    out_color = vec4(tonemap_aces(color), 1.0);
}

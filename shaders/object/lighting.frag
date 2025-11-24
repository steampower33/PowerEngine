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
    uint pbr_enable;
    float exposure;
} light;

layout(set = 0, binding = 1) uniform sampler2D out_albedo_metal;
layout(set = 0, binding = 2) uniform sampler2D out_normal_rough;
layout(set = 0, binding = 3) uniform sampler2D out_height_ao;
layout(set = 0, binding = 4) uniform sampler2D out_depth;

layout(set = 1, binding = 0) uniform SkyboxUBO {
    mat4x4 model;

    uint env_idx;
    uint radiance_idx;
    uint irradiance_idx;
    uint specular_mip_levels;

    uint brdf_lut_index;
    uint p0;
    uint p1;
    uint p2;
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

void main()
{
    float depth = texture(out_depth, in_uv).r;
    
    vec3 world_pos = reconstruct_world_pos(in_uv, depth);
    
    vec4 am = texture(out_albedo_metal, in_uv);
    vec4 nr = texture(out_normal_rough, in_uv);
    vec4 ha = texture(out_height_ao, in_uv);

    vec3 albedo    = am.rgb;
    float metallic = am.a;

    vec3 normal = normalize(nr.rgb * 2.0 - 1.0);
    float roughness = nr.a;
    float height = ha.r;
    float ao = ha.g;

    vec3 camera_pos = light.camera_pos.xyz;

    vec3 N = normalize(normal);

    vec3 color = albedo;
    if (light.pbr_enable == 1u)
    {
    
        if (depth >= 1.0 - 1e-5) {
            vec3 sky = texture(env_tex[nonuniformEXT(skybox.radiance_idx)], world_pos).rgb;
            out_color = vec4(sky, 1.0);
            return;
        }

        vec3 V = normalize(camera_pos - world_pos);
        float n_dot_v = max(dot(N, V), 0.0);

        vec3 F0 = mix(vec3(0.04), albedo, metallic);

        vec3 diffuse_irr = texture(env_tex[nonuniformEXT(skybox.irradiance_idx)], N).rgb;
        vec3 diffuse_ibl = diffuse_irr * albedo;

        vec3 R = reflect(-V, N);

        float max_mip = float(skybox.specular_mip_levels);
        float lod = roughness * max_mip;

        vec3 prefiltered_color = textureLod(env_tex[nonuniformEXT(skybox.radiance_idx)], R, lod).rgb;

        vec2 brdf = texture(tex[nonuniformEXT(skybox.brdf_lut_index)], vec2(n_dot_v, roughness)).rg;

        vec3 F = fresnel_schlick_roughness(n_dot_v, F0, roughness);
        vec3 kS = F;
        vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

        vec3 specular_ibl = prefiltered_color * (F * brdf.x + brdf.y);

        vec3 ambient_diffuse = kD * diffuse_ibl * ao;
        vec3 ambient_spec    = specular_ibl;

        color = ambient_diffuse + ambient_spec;

        color = max(color, vec3(0.0));
    
        float exposure = light.exposure;
        color *= exposure;
        color  = tonemap_aces(color);

        color = pow(color, vec3(1.0 / 2.2));
    }
    
    if (light.light_enable == 1u)
    {
        if (depth >= 1.0 - 1e-5) {
            out_color = vec4(0.0f);
            return;
        }
        vec3 light_pos = light.position;
        vec3 V = normalize(light.camera_pos.xyz - world_pos);
        vec3 L = normalize(light_pos - world_pos);
        vec3 H = normalize(L + V);
        vec3 L_dir = normalize(light.direction);
        float cos_theta = dot(-L_dir, L);
        float spot = smoothstep(cos(radians(light.outer)), cos(radians(light.inner)), cos_theta);
        float dist = length(light_pos - world_pos);
        float attenuation = 1.0 / (dist * dist);
        float diff = max(dot(N, L), 0.0);
        float spec = pow(max(dot(N, H), 0.0), 1000.0f);
        vec3 specular_color = vec3(1.0);
        color = (color * diff + specular_color * spec) 
              * light.intensity
              * attenuation 
              * spot;
    }
    out_color = vec4(color, 1.0);
}

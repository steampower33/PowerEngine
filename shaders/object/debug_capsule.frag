#version 450

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_normal_world;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec4 out_albedo_metal;
layout(location = 1) out vec4 out_normal_rough;
layout(location = 2) out vec2 out_height_ao;
layout(location = 3) out vec4 out_cozz_fuzz;

void main()
{
	vec3 normalTS  = in_normal_world;

	out_albedo_metal = vec4(in_color, 0.0);

    vec3 n = normalize(normalTS);
    out_normal_rough = vec4(n * 0.5 + 0.5, 0.0);
    out_height_ao = vec2(0.0, 1.0);
    out_cozz_fuzz = vec4(0.0);
}
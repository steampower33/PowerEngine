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
layout(location = 0) out vec4 outColor;

void main() {
    // sampler2D 배열 인덱스는 int여야 해서 캐스팅
    int albedoIndex = int(object.chooseTexIdx);

    // nonuniformEXT(...) 로 "이 인덱스는 프래그먼트마다 다를 수 있다" 표시
    vec4 sampled = texture(tex[nonuniformEXT(albedoIndex)], vUV);

    vec4 srgb = (object.color_use.w > 0.5)
        ? vec4(object.color_use.xyz, 1.0)
        : sampled;

    vec3 linear = pow(srgb.xyz, vec3(2.2));
    outColor = vec4(linear, 1.0);
}
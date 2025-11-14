#version 450

#extension GL_KHR_vulkan_glsl : enable

layout(set=0, binding=0) uniform GlobalUBO { 
    mat4 view; 
    mat4 proj;
} global;

layout(set=1, binding=0) uniform ObjectUBO { 
    mat4 model; 
    vec4 color_use; 
} object;

layout(set=1, binding=1) uniform sampler2D tex;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 srgb = object.color_use.w > 0.5 ? vec4(object.color_use.xyz, 1.0) : texture(tex, vUV);
    vec3 linear = pow(srgb.xyz, vec3(2.2));
    outColor = vec4(linear, 1.0);
}

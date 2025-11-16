#version 450

layout(set=0, binding=0) uniform GlobalUBO { 
    mat4 view; 
    mat4 proj; 
    vec4 background_color;
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
	uint pad0;
	uint pad1;
} object;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec2 vUV;

void main() {
    gl_Position = global.proj * global.view * object.model * vec4(inPos, 1.0);
    vUV = vec2(inUV.x, inUV.y);
}
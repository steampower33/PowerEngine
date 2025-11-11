#version 450

#extension GL_KHR_vulkan_glsl : enable

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 proj;
} ubo;

layout(set=1, binding=0, std430) readonly buffer Positions { vec4 X[]; };

layout(push_constant) uniform ClothPC { uint Nx; uint Ny; } pc;

layout(location=0) out vec2 vUV;

void main() {

    uint vid = gl_VertexIndex;
    vec3 p = X[vid].xyz;
    
    uint i = vid % pc.Nx;
    uint j = vid / pc.Nx;

    vUV = vec2(float(i) / float(pc.Nx - 1),
               1.0 - float(j) / float(pc.Ny - 1));

    gl_Position = ubo.proj * ubo.view * vec4(p, 1.0);
}
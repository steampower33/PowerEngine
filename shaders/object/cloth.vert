#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 1, binding = 0) uniform Cloth {
    vec4 albedo;

    int albedo_idx;
    int metallic_idx;
    int normal_idx;
    int roughness_idx;

    int ao_idx;
    int height_idx;
    float metallic_factor;
    float roughness_factor;

    float ao_factor;
    float height_factor;
    float sheen_weight_factor;
    float sheen_roughness_factor;

    uint albedo_enable;
    uint metallic_enable;
    uint normal_enable;
    uint roughness_enable;

    uint ao_enable;
    uint height_enable;
    uint p3;
    uint p4;
} cloth;

layout(set=1, binding=1, std430) readonly buffer Positions { vec4 pos[]; };
layout(set=1, binding=2, std430) readonly buffer Normals { vec4 normals[]; };

layout(set = 2, binding = 0) uniform sampler2D tex[];

layout(push_constant) uniform ClothPC { vec4 color; uint nx1; uint ny1; uint p0; float p1; } pc;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_world_normal;

void main() {

    uint vid = gl_VertexIndex;
    
    vec3 p = pos[vid].xyz;
    
    uint nx1 = pc.nx1;
    uint ny1 = pc.ny1;
    uint x = vid % nx1;
    uint y = vid / ny1;
    out_uv = vec2(float(x) / float(nx1 - 1), float(y) / float(ny1 - 1));

    out_world_normal = normals[vid].xyz;
    
//    float height = (cloth.height_enable == 0u) ? cloth.height_factor : texture(tex[nonuniformEXT(cloth.height_idx)], out_uv).r;
//    p += N * height * cloth.height_factor;

    gl_Position = ubo.proj * ubo.view * vec4(p, 1.0);
}
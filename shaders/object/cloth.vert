#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

#include "../common/render_common.glsl"

layout(set = 0, binding = 0) uniform GlobalBlock {
    Global global;
};

layout(set = 1, binding = 0, std140) uniform ModelBlock {
    Model model;
};

layout(set=1, binding=1, std430) readonly buffer Positions { vec4 pos[]; };
layout(set=1, binding=2, std430) readonly buffer Normals { vec4 normals[]; };

layout(set = 2, binding = 0) uniform sampler2D tex[];

layout(push_constant) uniform ClothPC { vec4 color; uint nx1; uint ny1; uint offset_particle; float p1; } pc;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_world_normal;

void main() {

    uint vid = gl_VertexIndex;
    
    vec3 p = pos[vid].xyz;
    
    uint nx1 = pc.nx1;
    uint ny1 = pc.ny1;

    uint vid_no_offset = (vid - pc.offset_particle);
    uint x = vid_no_offset % nx1;
    uint y = vid_no_offset / ny1;
    out_uv = vec2(float(x) / float(nx1 - 1.0), float(y) / float(ny1 - 1.0)) * model.tile_uv;

    out_world_normal = normals[vid].xyz;
    
//    float height = (model.height_enable == 0u) ? model.height_factor : texture(tex[nonuniformEXT(model.height_idx)], out_uv).r;
//    p += out_world_normal * height * model.height_factor;

    gl_Position = global.proj * global.view * vec4(p, 1.0);
}
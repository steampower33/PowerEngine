#version 450

#extension GL_KHR_vulkan_glsl : enable
#extension GL_EXT_nonuniform_qualifier : require

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 proj;
} ubo;

layout(set=1, binding=0, std430) readonly buffer Positions { vec4 pos[]; };

layout(set = 2, binding = 0) uniform Render {
    vec4 albedo_use;

    int albedoIdx;
    int metallicIdx;
    int normalIdx;
    int roughnessIdx;

    int aoIdx;
    int heightIdx;
    float metallicFactor;
    float roughnessFactor;

    float aoFactor;
    float heightFactor;
    uint p0;
    uint p1;

    uint albedoEnable;
    uint metallicEnable;
    uint normalEnable;
    uint roughnessEnable;

    uint aoEnable;
    uint heightEnable;
    uint p3;
    uint p4;
} render;

layout(set = 3, binding = 0) uniform sampler2D tex[];

layout(push_constant) uniform ClothPC { uint nx1; uint ny1; } pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vWorldNormal; // world normal

void main() {

    uint vid = gl_VertexIndex;
    uint nx1 = pc.nx1;
    uint ny1 = pc.ny1;
    
    uint x = vid % nx1;
    uint y = vid / nx1;
    
    vec3 p = pos[vid].xyz;
    
    uint xL = (x > 0)      ? (x - 1) : x;
    uint xR = (x + 1 < nx1)? (x + 1) : x;
    uint yD = (y > 0)      ? (y - 1) : y;
    uint yU = (y + 1 < ny1)? (y + 1) : y;
    
    uint idL = y * nx1 + xL;
    uint idR = y * nx1 + xR;
    uint idD = yD * nx1 + x;
    uint idU = yU * nx1 + x;
    
    vec3 pL = pos[idL].xyz;
    vec3 pR = pos[idR].xyz;
    vec3 pD = pos[idD].xyz;
    vec3 pU = pos[idU].xyz;
    
    vec3 dx = pR - pL;
    vec3 dy = pU - pD;

    vec3 N = normalize(cross(dy, dx));
    
    vWorldNormal = N;

    vUV = vec2(float(x) / float(nx1 - 1), float(y) / float(ny1 - 1));

    if (render.heightEnable == 1u)
    {
        float height   = (render.heightEnable == 0u) ? render.heightFactor : texture(tex[nonuniformEXT(render.heightIdx)], vUV).r;
        p += vWorldNormal * height * render.heightFactor;
    }


    gl_Position = ubo.proj * ubo.view * vec4(p, 1.0);
}
#ifndef PARAMS_GLSL
#define PARAMS_GLSL

layout(std140, set = 0, binding = 0) uniform SimParams {
    float dt;
    uint   num_particles;
    uint   num_edges;
    int   wind_test;
    float wind_strength;
    float sphere_radius;
    float max_speed;
    float damping;
    float relaxation_factor;
    uint   num_bends;
    uint   num_shears;
    float collision_margin;
    vec4  sphere_center;
    vec4  wind_dir;
    vec4  gravity;
    float thickness;
    float friction;
    float pad1;
    float pad2;
} sim;

layout(std430, set = 1, binding = 0) buffer X { vec4 x[]; };
layout(std430, set = 1, binding = 1) buffer XP { vec4  xp[]; };
layout(std430, set = 1, binding = 2) buffer V { vec4 v[]; };
layout(std430, set = 1, binding = 3) buffer W { float w[]; };
layout(std430, set = 1, binding = 4) buffer DeltaX { float  delta_x[]; };
layout(std430, set = 1, binding = 5) buffer DeltaY { float  delta_y[]; };
layout(std430, set = 1, binding = 6) buffer DeltaZ { float  delta_z[]; };
layout(std430, set = 1, binding = 7) buffer DCount { uint delta_count[]; };

struct Edge { 
    uint i; 
    uint j; 
    float rest; 
    float lambda;
};
layout(std430, set = 1, binding = 8) buffer Edges { Edge  edges[]; };

struct Shear {
    uint i0, i1, i2;
    float rest_dot;
    float lambda;
    float p0, p1, p2;
};
layout(std430, set = 1, binding = 9) buffer Shears { Shear  shears[]; };

struct Bend {
    uint p1, p2, p3, p4;
    float rest_angle;
    float lambda;
    vec2 pad;
};
layout(std430, set = 1, binding = 10) buffer Bends { Bend  bends[]; };



#endif
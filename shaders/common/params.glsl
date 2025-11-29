#ifndef PARAMS_GLSL
#define PARAMS_GLSL

struct Solve {
    uint base;
    uint count;
    float compliance;
    float beta;
};

struct MouseInteract {
    vec3 ray_origin;
    uint select_mode; // 0: none, 1: select, 2: drag
    vec3 ray_dir;
    float radius;
};

layout(push_constant) uniform PushConstant {
    Solve solve;
    MouseInteract mouse;
} pc;


layout(std140, set = 0, binding = 0) uniform SimParams {
    vec4 gravity;
    vec4 sphere_center;
    float sphere_radius;
    float thickness;
    float friction;
    float dt;
    float global_damping;
    float relaxation_factor;
    uint num_particles;
    uint num_edges;
    uint num_shears;
    uint num_bends;
    uint num_areas;
    float self_collision_stiffness;

    float cell_size;
    uint num_tables;
    uint max_neighbors;
    float collision_radius;
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
    float p0;
    float p1;
    float p2;
};
layout(std430, set = 1, binding = 9) buffer Shears { Shear  shears[]; };

struct Bend {
    uint i0, i1, i2, i3;
    float rest_angle;
    float lambda;
    float p0;
    float p1;
};
layout(std430, set = 1, binding = 10) buffer Bends { Bend  bends[]; };

struct GrabState {
    uint id;
    uint dist_bits;
    float t;
};
layout(std430, set = 1, binding = 11) buffer GrabStates { GrabState grab_state[]; };

struct Area {
    uint i0, i1, i2;
    float rest_area;
    vec3 rest_normal;
    float lambda;
};
layout(std430, set = 1, binding = 12) buffer Areas { Area areas[]; };

layout(std430, set = 1, binding = 13) buffer Hashes { uint hashes[]; };
layout(std430, set = 1, binding = 14) buffer Indices { uint indices[]; };
layout(std430, set = 1, binding = 15) buffer Starts { uint starts[]; };
layout(std430, set = 1, binding = 16) buffer Ends { uint ends[]; };
layout(std430, set = 1, binding = 17) buffer Neighbors { uint neighbors[]; };
layout(std430, set = 1, binding = 18) buffer NeighborLambdas { float neighbor_lambdas[]; };

#endif
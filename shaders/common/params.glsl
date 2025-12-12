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
    uint depth_mode; // 0: none, 1: depth in, 2: depth out
    float p0;
    float p1;
    float p2;
};

layout(push_constant) uniform PushConstant {
    Solve solve;
    MouseInteract mouse;
} pc;


layout(std140, set = 0, binding = 0) uniform SimParams {
    vec4 gravity;

    float dt;
    float thickness;
    float friction;
    float max_speed;

    float global_damping;
    float relaxation_factor;
    float neighbor_friction;
    float p0;

    uint num_particles;
    uint num_edges;
    uint num_shears;
    uint num_bends;

    uint num_areas;
    uint num_tries;
    uint num_volumes;
    uint num_colliders;

    float cell_size;
    uint num_tables;
    uint max_neighbors;
    float collision_radius;

    float stretch_stiffness;
    float shear_stiffness;
    float bend_stiffness;
    float area_stiffness;

    float self_collision_stiffness;
    float volume_stiffness;
    float softbody_stretch_stiffness;
    float p1;
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
layout(std430, set = 1, binding = 14) buffer SoretedIndices { uint sorted_indices[]; };
layout(std430, set = 1, binding = 15) buffer Starts { uint starts[]; };
layout(std430, set = 1, binding = 16) buffer Ends { uint ends[]; };
layout(std430, set = 1, binding = 17) buffer Neighbors { uint neighbors[]; };
layout(std430, set = 1, binding = 18) buffer NeighborLambdas { float neighbor_lambdas[]; };
layout(std430, set = 1, binding = 19) buffer Index { uint indices[]; };
layout(std430, set = 1, binding = 20) buffer Normals { vec4 normals[]; };
layout(std430, set = 1, binding = 21) buffer TriNormals { vec4 tri_normals[]; };
layout(std430, set = 1, binding = 22) buffer VertexTriOffsets { uint vertex_tri_offsets[]; };
layout(std430, set = 1, binding = 23) buffer VertexTriIndices { uint vertex_tri_indices[]; };

struct Volume {
    uint i0, i1, i2, i3;
    float rest_volume;
    float lambda;
    float p0;
    float p1;
};
layout(std430, set = 1, binding = 24) buffer Volumes { Volume volumes[]; };

struct Collider {
    vec3 p0;
    int kind; // 0 = sphere, 1 = plane, 2 = capsule
    vec3 p1;
    float radius;
};
layout(std430, set = 1, binding = 25) buffer Colliders { Collider colliders[]; };


#endif
#ifndef PARAMS_GLSL
#define PARAMS_GLSL

layout(std140, set = 0, binding = 0) uniform SimParams {
    float dt;
    int   numParticles;
    int   numEdges;
    int   windTest;
    float windStrength;
    float sphereRadius;
    float maxSpeed;
    float damping;
    float relaxationFactor;
    int   numBends;
    uint  numColliders;
    float collisionMargin;
    vec4  sphereCenter;
    vec4  windDir;
    vec4  gravity;
    float thickness;
    float friction;
    float pad1;
    float pad2;
} sim;

layout(std430, set = 1, binding = 0) buffer X { vec4 x[]; };
layout(std430, set = 1, binding = 1) buffer V { vec4 v[]; };
layout(std430, set = 1, binding = 2) buffer W { float w[]; };
layout(std430, set = 1, binding = 3) buffer DeltaX { float  deltaX[]; };
layout(std430, set = 1, binding = 4) buffer DeltaY { float  deltaY[]; };
layout(std430, set = 1, binding = 5) buffer DeltaZ { float  deltaZ[]; };
layout(std430, set = 1, binding = 6) buffer DCount { uint dcount[]; };

struct Edge { 
    uint i; 
    uint j; 
    float rest; 
    float lambda;
};
layout(std430, set = 1, binding = 7) buffer Edges { Edge  edges[]; };
layout(std430, set = 1, binding = 8) buffer XP { vec4  xp[]; };

struct Bend {
    uint p1, p2, p3, p4;
    float restAngle;
    float lambda;
    vec2 pad;
};
layout(std430, set = 1, binding = 9) buffer Bends { Bend  bends[]; };

struct SDFCollider {
    int   type;
    vec3  center;
    float radius;
    vec3  normal;
    vec3  velocity;
    float pad;
};

#endif
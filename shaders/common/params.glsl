#ifndef PARAMS_GLSL
#define PARAMS_GLSL

// 변경될때마다 리컴파일 안되고있음

layout(std140, set = 0, binding = 0) uniform SimParams {
    float dt;
    float damping;
    int   numParticles;
    int   numAllEdges;
    vec4  gravity;
    vec4  sphereCenter;
    float sphereRadius;
    float collisionBeta;
    int  windTest;
    float windStrength;
    vec4  windDir;
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
    float stiff;
};
layout(std430, set = 1, binding = 7) buffer Edges { Edge  edges[]; };
layout(std430, set = 1, binding = 8) buffer XPrev { vec4  xPrev[]; };

#endif
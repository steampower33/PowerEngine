#ifndef PARAMS_GLSL
#define PARAMS_GLSL

// 변경될때마다 리컴파일 안되고있음

layout(std140, set = 0, binding = 0) uniform Params {
	float dt;
	float invDt;
	float substeps;
	float iterations;
	vec4  gravity;
	uint  numParticles;
	uint  numEdges;
	uint  _pad0;
	uint  _pad1;
	float damping;
	float collisionFriction;
	float _pad2;
	float _pad3;
} params;

layout(std430, set = 1, binding = 0) buffer X { vec4 x[]; };
layout(std430, set = 1, binding = 1) buffer V { vec4 v[]; };
layout(std430, set = 1, binding = 2) buffer XP { vec4 xp[]; };
layout(std430, set = 1, binding = 3) buffer W { float w[]; };
layout(std430, set = 1, binding = 4) buffer DeltaU { uint deltaU32[]; };
layout(std430, set = 1, binding = 5) buffer DCount { uint dcount[]; };
layout(std430, set = 1, binding = 6) buffer Edges { uvec2 edges[]; };
layout(std430, set = 1, binding = 7) buffer RestL { float restL[]; };
layout(std430, set = 1, binding = 8) buffer CompS { float compS[]; };
layout(std430, set = 1, binding = 9) buffer LambdaS { float lambdaS[]; };

#endif
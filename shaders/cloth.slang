
struct VSInput
{
	float4 pos;
	float4 uv;
};

struct VSOutput
{
	float4 pos : SV_POSITION;
	float2 uv;
};

struct UBO
{
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};
[[vk::binding(0,0)]] ConstantBuffer<UBO> ubo;
[[vk::binding(1,0)]] Sampler2D texture;

struct Particle {
	float4 pos;
	float4 vel;
	float4 uv;
	float4 normal;
};

[[vk::binding(0,0)]] StructuredBuffer<Particle> particleIn;
[[vk::binding(1,0)]] RWStructuredBuffer<Particle> particleOut;

struct UBOCompute
{
	float deltaT;
	float particleMass;
	float springStiffness;
	float damping;
	float restDistH;
	float restDistV;
	float restDistD;
	float sphereRadius;
	float4 spherePos;
	float4 gravity;
	int2 particleCount;
};
[[vk::binding(2, 0)]] ConstantBuffer<UBOCompute> params;

float3 springForce(float3 p0, float3 p1, float restDist)
{
    float3 dist = p0 - p1;
    float distLen = length(dist);
    
    // --- 안전장치 추가 ---
    if (distLen < 0.00001f) {
        return float3(0.0f, 0.0f, 0.0f);
    }
    
    return (dist / distLen) * params.springStiffness * (distLen - restDist);
}

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput output;
	output.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, input.pos)));
	output.uv = float2(input.uv.x, 1.0 - input.uv.y);
    return output;
} 

[shader("fragment")]
float4 fragmentMain(VSOutput input)
{
	return texture.Sample(input.uv);
}

[shader("compute")]
[numthreads(10, 10, 1)]
void computeMain(uint3 id: SV_DispatchThreadID, uniform uint calculateNormals)
{
	uint index = id.y * params.particleCount.x + id.x;
	if (index >= params.particleCount.x * params.particleCount.y)
		return;

	float3 force = params.gravity.xyz * params.particleMass;

	float3 pos = particleIn[index].pos.xyz;
	float3 vel = particleIn[index].vel.xyz;

	// force += (-params.damping * vel);

	float invMass = (params.particleMass > 0.0) ? (1.0 / params.particleMass) : 0.0;
	float3 acceleration = force * invMass;

    // 위치 업데이트 (Semi-implicit Euler integration)
	float3 newVel = vel + acceleration * params.deltaT;
	float3 newPos = pos + newVel * params.deltaT;

	particleOut[index].vel = float4(newVel, 0.0);
	particleOut[index].pos = float4(newPos, 1.0);
    
    // (uv, normal 등 계산하지 않는 데이터는 이전 값을 유지해야 함)
    particleOut[index].uv = particleIn[index].uv;
    particleOut[index].normal = particleIn[index].normal;
}
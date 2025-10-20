struct VSInput {
    float2 inPosition;
    float4 inColor;
};


struct VSOutput
{
    float4 pos : SV_Position;
    float pointSize : SV_PointSize;
    float3 fragColor : COLOR0;
};

struct PSInput
{
    float4 pos : SV_POSITION;
    float3 fragColor : COLOR0;
    float2 pointCoord : SV_PointCoord;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pointSize = 1.0;
    output.pos = float4(input.inPosition, 1.0, 1.0);
    output.fragColor = input.inColor.rgb;
    //output.pointCoord = (input.inPosition * float2(1.0, -1.0) + float2(1.0)) / 2.0;
    return output;
}

[shader("fragment")]
float4 fragMain(PSInput input) : SV_TARGET {
    float dist = length(input.pointCoord - float2(0.5)); 
    
    if (dist > 0.5) {
        discard;
    }

    // 안티 앨리어싱을 위한 부드러운 가장자리 처리 (선택 사항)
    float alpha = 1.0 - smoothstep(0.45, 0.5, dist);

    float4 outColor = float4(input.fragColor.rgb, alpha);
    return outColor;

}

struct Particle {
	float2 position;
	float2 velocity;
    float4 color;
};

struct UniformBuffer {
    float deltaTime;
};
ConstantBuffer<UniformBuffer> ubo;

struct ParticleSSBO {
    Particle particles;
};
StructuredBuffer<ParticleSSBO> particlesIn;
RWStructuredBuffer<ParticleSSBO> particlesOut;

[shader("compute")]
[numthreads(256,1,1)]
void compMain(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;

    particlesOut[index].particles.position = particlesIn[index].particles.position + particlesIn[index].particles.velocity.xy * ubo.deltaTime;
    particlesOut[index].particles.velocity = particlesIn[index].particles.velocity;

    // Flip movement at window border
    if ((particlesOut[index].particles.position.x <= -1.0) || (particlesOut[index].particles.position.x >= 1.0)) {
        particlesOut[index].particles.velocity.x = -particlesOut[index].particles.velocity.x;
    }
    if ((particlesOut[index].particles.position.y <= -1.0) || (particlesOut[index].particles.position.y >= 1.0)) {
        particlesOut[index].particles.velocity.y = -particlesOut[index].particles.velocity.y;
    }

}
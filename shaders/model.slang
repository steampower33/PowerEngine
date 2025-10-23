
struct VSInput
{
	float3 pos;
	float2 uv;
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

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput output;
	output.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.pos, 1.0))));
	output.uv = float2(input.uv.x, 1.0 - input.uv.y);
    return output;
} 

[shader("fragment")]
float4 fragmentMain(VSOutput input)
{
	return texture.Sample(input.uv);
}

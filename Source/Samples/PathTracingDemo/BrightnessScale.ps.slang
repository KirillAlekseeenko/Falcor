Texture2D    gInputColor;
SamplerState gSampler;

float4 main(float2 texC : TEXCOORD) : SV_TARGET0
{
    float4 color = gInputColor.Sample(gSampler, texC);
    return float4(color.rgb * 0.1f, color.a);
}

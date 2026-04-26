struct SampleData
{
    float4 position;
    float4 normal;
    float4 irradiance;
    uint4 meta;
};

StructuredBuffer<SampleData> gSamples;

cbuffer DebugSphereCB
{
    float4x4 gViewProj;
    float    gSphereRadius;
    float    gIrradianceScale;
    float3   gSurfaceColor;
    float    _pad0;
    float3   gVolumeColor;
    float    _pad1;
};

struct VSIn
{
    float3 position : POSITION;
    float3 normal   : NORMAL;
    uint   instanceID : SV_InstanceID;
};

struct VSOut
{
    float4 posH : SV_POSITION;
    float3 normalW : NORMAL;
    nointerpolation float3 sampleNormalW : NORMAL1;
    nointerpolation float3 irradianceColor : COLOR0;
};

VSOut vsMain(VSIn vIn)
{
    const SampleData sample = gSamples[vIn.instanceID];

    VSOut vOut;
    vOut.normalW = normalize(vIn.normal);
    vOut.sampleNormalW = normalize(sample.normal.xyz);
    vOut.posH = mul(gViewProj, float4(sample.position.xyz + vIn.position * gSphereRadius, 1.f));

    float3 fallbackColor = (sample.meta.x & 1u) != 0u ? gSurfaceColor : gVolumeColor;
    float3 scaledIrradiance = sample.irradiance.xyz * gIrradianceScale;
    bool hasIrradiance = gIrradianceScale > 0.f;
    vOut.irradianceColor = hasIrradiance ? scaledIrradiance : fallbackColor;
    return vOut;
}

float4 psMain(VSOut vIn) : SV_TARGET
{
    float3 sphereNormal = normalize(vIn.normalW);
    float3 sampleNormal = normalize(vIn.sampleNormalW);
    float marker = smoothstep(0.95f, 0.985f, dot(sphereNormal, sampleNormal));
    float3 color = lerp(vIn.irradianceColor, float3(1.f), marker);
    return float4(color, 1.f);
}

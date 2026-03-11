import Scene.Raster;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;

static const uint kCascadeCount = 4;

cbuffer CommonParameters
{
    float4x4 gCascadeLightVPs[kCascadeCount]; // world → light clip space, one per cascade
    float    gCascadeSplits[kCascadeCount];   // view-space Z far boundary of each cascade
    float2   gTexelSize;                       // 1 / shadowMapSize, used for PCF offsets
    int      globalLightIndex;
    int      gShadowsEnabled;
};

Texture2DArray<float> gShadowDepth;    // D32Float array, one slice per cascade
SamplerComparisonState gShadowSampler; // linear PCF comparison sampler

// Returns the shadow visibility in [0, 1] for world-space position P.
// viewZ is the linear camera-space depth of P (dot of view-forward with P - cameraPos).
float shadowVisibility(float3 P, float viewZ)
{
    // Select the tightest cascade whose far boundary covers viewZ.
    uint cascade = kCascadeCount - 1;
    for (uint i = 0; i < kCascadeCount; i++)
    {
        if (viewZ < gCascadeSplits[i])
        {
            cascade = i;
            break;
        }
    }

    // Project P into the selected cascade's shadow map.
    float4 l  = mul(gCascadeLightVPs[cascade], float4(P, 1.f));
    float2 uv = l.xy / l.w * 0.5f + 0.5f;
    uv.y      = 1.0f - uv.y; // flip Y for D3D UV convention
    float  d  = l.z / l.w;   // receiver depth (rasterizer slope-bias handles the offset)

    // If P falls outside this cascade's footprint return fully lit.
    // This acts as a safety net for pixels near cascade boundaries or beyond the last cascade.
    if (any(uv < 0.f) || any(uv > 1.f))
        return 1.f;

    // 5x5 PCF using hardware bilinear comparison taps.
    float v = 0;
    const int kPCFRadius = 3;
    [unroll] for (int y = -kPCFRadius; y <= kPCFRadius; y++)
    [unroll] for (int x = -kPCFRadius; x <= kPCFRadius; x++)
        v += gShadowDepth.SampleCmpLevelZero(gShadowSampler, float3(uv + gTexelSize * float2(x, y), float(cascade)), d);

    return v / float((2 * kPCFRadius + 1) * (2 * kPCFRadius + 1));
}

VSOut vsMain(VSIn vIn)
{
    return defaultVS(vIn);
}

float4 psMain(VSOut vsOut, uint triangleIndex: SV_PrimitiveID) : SV_TARGET
{
    let lod = ImplicitLodTextureSampler();
    if (alphaTest(vsOut, triangleIndex, lod))
        discard;

    float3 viewDir = normalize(gScene.camera.getPosition() - vsOut.posW);
    ShadingData sd = prepareShadingData(vsOut, triangleIndex, viewDir);

    let mi = gScene.materials.getMaterialInstance(sd, lod);

    uint2 pixelIndex = uint2(vsOut.posH.xy);
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixelIndex, triangleIndex);

    const LightData lightData = gScene.getLight(globalLightIndex);

    float3 color = float3(0.f);

    if (lightData.type == uint(LightType::Directional))
    {
        // Compute linear view-space depth for cascade selection.
        // data.cameraW points from camera toward the image plane; its length equals focal distance,
        // so normalize it first to get the true forward unit vector.
        float3 camFwd = normalize(gScene.camera.data.cameraW);
        float  viewZ  = dot(sd.posW - gScene.camera.getPosition(), camFwd);

        float visibility = gShadowsEnabled != 0 ? shadowVisibility(sd.posW, viewZ) : 1.f;

        AnalyticLightSample ls;
        evalLightApproximate(sd.posW, lightData, ls);
        color += mi.eval(sd, ls.dir, sg) * ls.Li * visibility;
    }

    //let bsdfInfo = mi.getProperties(sd);
    //color = bsdfInfo.diffuseReflectionAlbedo;

    return float4(color, 1.f);
}

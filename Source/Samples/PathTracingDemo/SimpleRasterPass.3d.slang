import Scene.Raster;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;

cbuffer CommonParameters
{
    float4x4                gLightVP;
    float2                  gTexelSize;
    float                   gShadowBias;
    int                     globalLightIndex;
};

// global parameter block
Texture2D<float>        gShadowDepth;
SamplerComparisonState  gShadowSampler;

float shadowVisibility(float3 P)
{
    float4 l = mul(gLightVP, float4(P,1));
    float2 uv = l.xy / l.w * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y; // flip y for D3D
    float  d  = l.z  / l.w - gShadowBias;

    
    float v = 0;
    // 5×5 PCF

    int shadowMapWidth;
    int shadowMapHeight;

    gShadowDepth.GetDimensions(shadowMapWidth, shadowMapHeight);

    const uint pcfSizeTimesTwo = 4;
    [unroll]for(int y=-pcfSizeTimesTwo;y<=pcfSizeTimesTwo;++y)
    [unroll]for(int x=-pcfSizeTimesTwo;x<=pcfSizeTimesTwo;++x)
        v+= (gShadowDepth.Load(int3(uv * int2(shadowMapWidth, shadowMapHeight), 0) + int3(x, y, 0)) >= d ? 1 : 0);
        //v += gShadowDepth.SampleCmpLevelZero(gShadowSampler, uv + gTexelSize * float2(x,y), d);
    return v / float((2 * pcfSizeTimesTwo + 1) * (2 * pcfSizeTimesTwo + 1));
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

    // Create material instance.
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    //return float4(mi.getProperties(sd).diffuseReflectionAlbedo, 1.f);

    uint2 pixelIndex = uint2(vsOut.posH.xy / gTexelSize);
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixelIndex, triangleIndex);

    const LightData lightData = gScene.getLight(globalLightIndex);

    bool isDirectional = lightData.type == LightType::Directional;

    float3 color = float3(0.f, 0.f, 0.f);

    if (isDirectional)
    {
        // For directional lights, we can use the shadow visibility function directly.
        float visibility = shadowVisibility(sd.posW);

        AnalyticLightSample ls;
        evalLightApproximate(sd.posW, lightData, ls);
        color += mi.eval(sd, ls.dir, sg) * ls.Li * visibility;
    }

    color *= 0.2f; // Reduce contribution

    return float4(color, 1.f);
}

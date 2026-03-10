import Scene.Raster;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;

cbuffer PerFrame
{
    float4x4 gLightVP;
};

VSOut vsMain(VSIn vIn)
{
    VSOut o = defaultVS(vIn);

    float3 posW = o.posW;
    o.posH = mul(gLightVP, float4(posW, 1));
    return o;
}

void psMain() {}        // no color writes – depth only

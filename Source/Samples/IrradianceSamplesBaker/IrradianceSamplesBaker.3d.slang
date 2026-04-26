import Scene.Raster;
import Scene.Material.BasicMaterialData;

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
    BasicMaterialData md = gScene.materials.getBasicMaterialData(sd.materialID);
    SamplerState s = gScene.materials.getDefaultTextureSampler(sd.materialID);

    float3 color = gScene.materials.sampleTexture(md.texBaseColor, s, sd.uv, md.baseColor, lod).rgb;

    return float4(color, 1.f);
}

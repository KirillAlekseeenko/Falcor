import Scene.Raytracing;
import Utils.Math.MathHelpers;
import Utils.Sampling.LowDiscrepancy.HammersleySequence;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;

static const uint kSurfaceCandidateFlag = 1;
static const uint kMaxBounces = 5;
static const uint kRayTypeCount = 3; // 0 = probe, 1 = indirect path, 2 = shadow
static const float kPi = 3.14159265358979323846f;

struct BakeCandidateData
{
    float4 position;
    float4 normal;
    uint4 meta;
};

struct BakeResultData
{
    float4 position;
    float4 normal;
    float4 irradiance;
    uint4 meta;
};

struct ProbePayload
{
    uint hit;
    uint backface;
};

struct PathPayload
{
    float3 radiance;
    float3 posW;
    float3 normalW;
    float3 albedo;
    float3 nextDir;
    uint bounce;
    uint sampleSeed;
    bool terminated;
};

struct ShadowPayload
{
    bool visible;
};

StructuredBuffer<BakeCandidateData> gCandidates;
RWStructuredBuffer<BakeResultData> gResults;

cbuffer BakeParams
{
    uint gSampleCount;
    uint gProbeRayCount;
    uint gIrradianceRayCount;
    uint gDirLightIndex;
    float gSurfaceOffset;
    float gBackfaceThreshold;
};

bool alphaTestCurrentHit(BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    let lod = ExplicitLodTextureSampler(0.f);
    return gScene.materials.alphaTest(v, gScene.getMaterialID(instanceID), lod);
}

[shader("miss")]
void probeMiss(inout ProbePayload payload)
{
    payload.hit = 0;
    payload.backface = 0;
}

[shader("miss")]
void primaryMiss(inout PathPayload payload)
{
    payload.radiance = float3(0.f);
    payload.terminated = true;
}

[shader("miss")]
void shadowMiss(inout ShadowPayload payload)
{
    payload.visible = true;
}

[shader("anyhit")]
void probeAnyHit(inout ProbePayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    if (alphaTestCurrentHit(attribs))
        IgnoreHit();
}

[shader("anyhit")]
void primaryAnyHit(inout PathPayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    if (alphaTestCurrentHit(attribs))
        IgnoreHit();
}

[shader("anyhit")]
void shadowAnyHit(inout ShadowPayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    if (alphaTestCurrentHit(attribs))
        IgnoreHit();
}

bool traceShadowRay(float3 origin, float3 dir, float tMax)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = max(gSurfaceOffset * 0.5f, 1e-4f);
    ray.TMax = min(tMax, 1e30f);

    ShadowPayload payload;
    payload.visible = false;
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xFF,
        2,
        kRayTypeCount,
        2,
        ray,
        payload
    );

    return payload.visible;
}

float3 sampleCosineHemisphereWorld(float3 normalW, float2 u)
{
    float3 tangent;
    float3 bitangent;
    branchlessONB(normalW, tangent, bitangent);

    float3 localDir = sample_cosine_hemisphere_polar(u);
    return normalize(tangent * localDir.x + bitangent * localDir.y + normalW * localDir.z);
}

float3 traceIncidentRadiance(float3 origin, float3 dir, uint sampleSeed)
{
    float3 radiance = float3(0.f);
    float3 throughput = float3(1.f);

    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = max(gSurfaceOffset * 0.5f, 1e-4f);
    ray.TMax = 1e30f;

    [loop]
    for (uint bounce = 0; bounce <= kMaxBounces; ++bounce)
    {
        PathPayload payload;
        payload.bounce = bounce;
        payload.sampleSeed = sampleSeed;
        payload.terminated = false;

        TraceRay(
            gScene.rtAccel,
            RAY_FLAG_NONE,
            0xFF,
            1,
            kRayTypeCount,
            1,
            ray,
            payload
        );

        radiance += throughput * payload.radiance;
        if (payload.terminated || bounce == kMaxBounces)
            break;

        throughput *= payload.albedo;

        ray.Origin = payload.posW + payload.normalW * gSurfaceOffset;
        ray.Direction = payload.nextDir;
        ray.TMin = max(gSurfaceOffset * 0.5f, 1e-4f);
        ray.TMax = 1e30f;
    }

    return radiance;
}

[shader("closesthit")]
void probeClosestHit(inout ProbePayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);

    payload.hit = 1;
    payload.backface = dot(v.faceNormalW, WorldRayDirection()) > 0.f ? 1u : 0u;
}

[shader("closesthit")]
void primaryClosestHit(inout PathPayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    float3 rayDir = WorldRayDirection();

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);
    let lod = ExplicitLodTextureSampler(0.f);
    let materialInstance = gScene.materials.getMaterialInstance(sd, lod);
    let props = materialInstance.getProperties(sd);

    uint sampleIdx = payload.sampleSeed * (kMaxBounces + 1u) + payload.bounce;
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(uint2(DispatchRaysIndex().x, payload.sampleSeed), sampleIdx);

    float3 direct = props.emission;

    if (gDirLightIndex != 0xFFFFFFFFu)
    {
        LightData light = gScene.getLight(gDirLightIndex);
        AnalyticLightSample lightSample;
        evalLightApproximate(sd.posW, light, lightSample);

        float3 shadowOrigin = sd.posW + sd.frame.N * gSurfaceOffset;
        if (traceShadowRay(shadowOrigin, lightSample.dir, lightSample.distance))
            direct += materialInstance.eval(sd, lightSample.dir, sg) * lightSample.Li;
    }

    float2 bounceSample = sampleNext2D(sg);
    float3 localDir = sample_cosine_hemisphere_polar(bounceSample);
    float3 worldDir = sd.frame.fromLocal(localDir);

    payload.radiance = direct;
    payload.posW = sd.posW;
    payload.normalW = sd.frame.N;
    payload.albedo = props.diffuseReflectionAlbedo;
    payload.nextDir = worldDir;
    payload.terminated = false;
}

[shader("raygeneration")]
void rayGen()
{
    const uint sampleIndex = DispatchRaysIndex().x;
    if (sampleIndex >= gSampleCount)
        return;

    const BakeCandidateData candidate = gCandidates[sampleIndex];
    const bool isSurfaceCandidate = (candidate.meta.y & kSurfaceCandidateFlag) != 0;

    float3 samplePosition = candidate.position.xyz;
    float3 sampleNormal = normalize(candidate.normal.xyz);

    if (isSurfaceCandidate)
        samplePosition += sampleNormal * gSurfaceOffset;

    uint hitCount = 0u;
    uint backfaceCount = 0u;

    [loop]
    for (uint probeIndex = 0; probeIndex < gProbeRayCount; ++probeIndex)
    {
        ProbePayload payload = {0u, 0u};

        RayDesc ray;
        ray.Origin = samplePosition;
        ray.Direction = sample_sphere(getHammersley(probeIndex, gProbeRayCount));
        ray.TMin = max(gSurfaceOffset * 0.5f, 1e-4f);
        ray.TMax = 1e30f;

        TraceRay(
            gScene.rtAccel,
            RAY_FLAG_NONE,
            0xFF,
            0,
            kRayTypeCount,
            0,
            ray,
            payload
        );

        hitCount += payload.hit;
        backfaceCount += payload.backface;
    }

    const float backfaceRatio = hitCount > 0u ? float(backfaceCount) / float(hitCount) : 0.f;
    const bool isValidSample = backfaceRatio <= gBackfaceThreshold;

    float3 irradiance = float3(0.f);
    if (isValidSample && gIrradianceRayCount > 0u)
    {
        [loop]
        for (uint irradianceRayIndex = 0; irradianceRayIndex < gIrradianceRayCount; ++irradianceRayIndex)
        {
            float3 worldDir = sampleCosineHemisphereWorld(sampleNormal, getHammersley(irradianceRayIndex, gIrradianceRayCount));
            irradiance += traceIncidentRadiance(samplePosition, worldDir, irradianceRayIndex);
        }

        irradiance *= kPi / float(gIrradianceRayCount);
    }

    BakeResultData result;
    result.position = float4(samplePosition, 1.f);
    result.normal = float4(sampleNormal, 0.f);
    result.irradiance = float4(irradiance, 1.f);
    result.meta = uint4(hitCount, backfaceCount, candidate.meta.y, 0u);
    gResults[sampleIndex] = result;
}

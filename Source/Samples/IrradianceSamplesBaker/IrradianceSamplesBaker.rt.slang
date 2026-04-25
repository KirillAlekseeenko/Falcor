import Scene.Raytracing;
import Utils.Math.MathHelpers;
import Utils.Sampling.LowDiscrepancy.HammersleySequence;

static const uint kSurfaceCandidateFlag = 1;

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
    uint4 meta;
};

struct ProbePayload
{
    uint hit;
    uint backface;
};

StructuredBuffer<BakeCandidateData> gCandidates;
RWStructuredBuffer<BakeResultData> gResults;

cbuffer BakeParams
{
    uint gSampleCount;
    uint gProbeRayCount;
    float gSurfaceOffset;
};

[shader("miss")]
void probeMiss(inout ProbePayload payload)
{
    payload.hit = 0;
    payload.backface = 0;
}

[shader("anyhit")]
void probeAnyHit(inout ProbePayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    let lod = ExplicitLodTextureSampler(0.f);
    if (gScene.materials.alphaTest(v, gScene.getMaterialID(instanceID), lod))
        IgnoreHit();
}

[shader("closesthit")]
void probeClosestHit(inout ProbePayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);

    payload.hit = 1;
    payload.backface = dot(v.faceNormalW, WorldRayDirection()) > 0.f ? 1u : 0u;
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
    {
        const GeometryInstanceID instanceID = GeometryInstanceID(candidate.meta.x, 0);
        const float4x4 worldMatrix = gScene.getWorldMatrix(instanceID);
        const float3x3 worldInvTranspose = gScene.getInverseTransposeWorldMatrix(instanceID);

        samplePosition = mul(worldMatrix, float4(samplePosition, 1.f)).xyz;
        sampleNormal = normalize(mul(worldInvTranspose, sampleNormal));
        samplePosition += sampleNormal * gSurfaceOffset;
    }

    uint hitCount = 0;
    uint backfaceCount = 0;

    if (!isSurfaceCandidate)
    {
        [loop]
        for (uint probeIndex = 0; probeIndex < gProbeRayCount; ++probeIndex)
        {
            ProbePayload payload = {0u, 0u};

            RayDesc ray;
            ray.Origin = samplePosition;
            ray.Direction = sample_sphere(getHammersley(probeIndex, gProbeRayCount));
            ray.TMin = max(gSurfaceOffset * 0.5f, 1e-4f);
            ray.TMax = 1e30f;

            TraceRayScene(RAY_FLAG_NONE, 0xFF, 0, 0, ray, payload);

            hitCount += payload.hit;
            backfaceCount += payload.backface;
        }
    }

    BakeResultData result;
    result.position = float4(samplePosition, 1.f);
    result.normal = float4(sampleNormal, 0.f);
    result.meta = uint4(hitCount, backfaceCount, candidate.meta.y, 0u);
    gResults[sampleIndex] = result;
}

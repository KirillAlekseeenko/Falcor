import Scene.Raytracing;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.MathHelpers;
import Rendering.Lights.LightHelpers;

// ---- Constants --------------------------------------------------------------

static const uint kMaxBounces   = 5;
static const uint kRayTypeCount = 2; // 0 = primary/indirect, 1 = shadow

// ---- Resources --------------------------------------------------------------

RWTexture2D<float4> gAccum; ///< Running-average accumulation buffer (RGBA32F)

cbuffer PerFrame
{
    uint2  gFrameDim;
    uint   gFrameIndex;
    uint   gDirLightIndex; ///< Index of the directional light, or 0xFFFFFFFF
};

// ---- Payload types ----------------------------------------------------------

/** Payload for primary / indirect rays.
    Closesthit fills all fields; miss sets terminated=true and radiance=skyColor.
    Raygen reads the fields to advance the path. */
struct PathPayload
{
    float3 radiance;    ///< Emission + direct lighting evaluated at this hit
    float3 posW;        ///< World-space hit position
    float3 normalW;     ///< Shading normal (for ray offset on next bounce)
    float3 albedo;      ///< Diffuse reflectance (throughput multiplier)
    float3 nextDir;     ///< Cosine-weighted scatter direction for next bounce
    uint   bounce;      ///< Current bounce index (raygen writes before TraceRay)
    bool   terminated;  ///< True when ray missed geometry
};

/** Payload for shadow rays. Only shadowMiss writes to it. */
struct ShadowPayload
{
    bool visible; ///< True when ray reaches the light unoccluded
};

// ---- Miss shaders -----------------------------------------------------------

[shader("miss")]
void primaryMiss(inout PathPayload p)
{
    p.radiance   = float3(0.f, 0.f, 0.f); // for now we dont have physically correct representation of direct sky
    p.terminated = true;
}

[shader("miss")]
void shadowMiss(inout ShadowPayload p)
{
    p.visible = true;
}

// ---- Any-hit shaders (alpha test) -------------------------------------------

[shader("anyhit")]
void primaryAnyHit(inout PathPayload p, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID iid = getGeometryInstanceID();
    VertexData v = getVertexData(iid, PrimitiveIndex(), attribs);
    let lod = ExplicitLodTextureSampler(0.f);
    if (gScene.materials.alphaTest(v, gScene.getMaterialID(iid), lod))
        IgnoreHit();
}

[shader("anyhit")]
void shadowAnyHit(inout ShadowPayload p, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID iid = getGeometryInstanceID();
    VertexData v = getVertexData(iid, PrimitiveIndex(), attribs);
    let lod = ExplicitLodTextureSampler(0.f);
    if (gScene.materials.alphaTest(v, gScene.getMaterialID(iid), lod))
        IgnoreHit();
}

// ---- Shadow helper ----------------------------------------------------------

bool traceShadowRay(float3 origin, float3 dir, float tMax)
{
    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = dir;
    ray.TMin      = 1e-4f;
    ray.TMax      = min(tMax, 1e30f);

    ShadowPayload sp;
    sp.visible = false;
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xFF,
        1,              // shadow hit-group index
        kRayTypeCount,
        1,              // shadow miss index
        ray, sp
    );
    return sp.visible;
}

// ---- Primary closest-hit ----------------------------------------------------

[shader("closesthit")]
void primaryClosestHit(inout PathPayload p, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID iid  = getGeometryInstanceID();
    VertexData         v    = getVertexData(iid, PrimitiveIndex(), attribs);
    uint               matID = gScene.getMaterialID(iid);
    float3             rayDir = WorldRayDirection();

    ShadingData sd   = gScene.materials.prepareShadingData(v, matID, -rayDir);
    let lod          = ExplicitLodTextureSampler(0.f);
    let mi           = gScene.materials.getMaterialInstance(sd, lod);
    let props        = mi.getProperties(sd);

    // Sample generator: unique per pixel × bounce × frame
    uint sampleIdx = gFrameIndex * (kMaxBounces + 1u) + p.bounce;
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(DispatchRaysIndex().xy, sampleIdx);

    // --- Direct lighting (NEE) ---
    float3 direct = props.emission;

    if (gDirLightIndex != 0xFFFFFFFFu)
    {
        LightData light = gScene.getLight(gDirLightIndex);
        AnalyticLightSample ls;
        evalLightApproximate(sd.posW, light, ls);

        //if (dot(sd.frame.N, ls.dir) > 0.f)
        {
            float3 shadowOrigin = sd.posW + sd.frame.N * 1e-3f;
            if (traceShadowRay(shadowOrigin, ls.dir, ls.distance))
                direct += mi.eval(sd, ls.dir, sg) * ls.Li;
        }
    }

    // --- Cosine-weighted hemisphere sample for next bounce ---
    float2 u        = sampleNext2D(sg);
    float3 localDir = sample_cosine_hemisphere_polar(u);
    float3 worldDir = sd.frame.fromLocal(localDir);

    // Fill payload
    p.radiance   = direct;
    p.posW       = sd.posW;
    p.normalW    = sd.frame.N;
    p.albedo     = props.diffuseReflectionAlbedo;
    p.nextDir    = worldDir;
    p.terminated = false;
}

// ---- Ray generation ---------------------------------------------------------

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    if (any(pixel >= gFrameDim)) return;

    float3 radiance   = float3(0.f);
    float3 throughput = float3(1.f);

    RayDesc ray = gScene.camera.computeRayPinhole(pixel, gFrameDim).toRayDesc();

    [loop]
    for (uint bounce = 0; bounce <= kMaxBounces; bounce++)
    {
        PathPayload p;
        p.bounce     = bounce;
        p.terminated = false;

        TraceRay(gScene.rtAccel, RAY_FLAG_NONE, 0xFF,
                 0, kRayTypeCount, 0,   // primary hit / miss
                 ray, p);

        radiance += throughput * p.radiance;
        if (p.terminated || bounce == kMaxBounces) break;

        // Cosine-weighted hemisphere sampling: throughput *= albedo (cos / pdf cancels)
        throughput *= p.albedo;

        // Advance ray
        ray.Origin    = p.posW + p.normalW * 1e-3f;
        ray.Direction = p.nextDir;
        ray.TMin      = 0.f;
        ray.TMax      = 1e30f;
    }

    // Running-average accumulation
    float3 prev   = gAccum[pixel].rgb;
    float  weight = 1.f / float(gFrameIndex + 1u);
    gAccum[pixel] = float4(lerp(prev, radiance, weight), 1.f);
}

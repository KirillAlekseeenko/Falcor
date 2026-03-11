#include "RenderShadowMap.h"

#include <algorithm> // std::min / std::max
#include <cmath>     // std::pow, std::floor, std::abs

// =============================================================================
// Construction
// =============================================================================

RenderShadowMap::RenderShadowMap(ref<Device> pDevice, ref<Scene> pScene)
    : mpScene(pScene)
{
    // ---- Texture2DArray: one D32Float slice per cascade ------------------
    mpShadowArray = pDevice->createTexture2D(
        kShadowMapSize,
        kShadowMapSize,
        ResourceFormat::D32Float,
        kCascadeCount,   // arraySize > 1 => Texture2DArray
        1,               // mip levels
        nullptr,
        ResourceBindFlags::DepthStencil | ResourceBindFlags::ShaderResource
    );

    // ---- Per-cascade FBOs pointing at individual array slices ------------
    for (uint32_t i = 0; i < kCascadeCount; i++)
    {
        mpCascadeFbos[i] = Fbo::create(pDevice);
        mpCascadeFbos[i]->attachDepthStencilTarget(mpShadowArray, /*mip=*/0, /*firstSlice=*/i, /*sliceCount=*/1);
    }

    // ---- Depth-only RasterPass -------------------------------------------
    auto shaderModules    = pScene->getShaderModules();
    auto typeConformances = pScene->getTypeConformances();
    auto defines          = pScene->getSceneDefines();

    ProgramDesc desc;
    desc.addShaderModules(shaderModules);
    desc.addShaderLibrary("Samples/PathTracingDemo/ShadowPass.3d.slang")
        .vsEntry("vsMain")
        .psEntry("psMain");
    desc.addTypeConformances(typeConformances);

    mpShadowPass = RasterPass::create(pDevice, desc, defines);

    // Back-face culling is standard: front faces are written to the shadow map,
    // which handles single-sided geometry (vegetation, thin walls) correctly.
    // For D32Float, the integer depthBias scales with 2^(exponent(z)-23), so
    // a value of 50000 gives ~0.003 of constant offset at mid-range depths.
    // The slope-scaled bias handles steep surfaces.
    RasterizerState::Desc rsDesc;
    rsDesc.setCullMode(RasterizerState::CullMode::Back)
          .setDepthBias(50000, 3.0f);
    mpShadowRsState = RasterizerState::create(rsDesc);
}

// =============================================================================
// Split distance computation
// =============================================================================

std::array<float, RenderShadowMap::kCascadeCount>
RenderShadowMap::computeSplitDistances(float nearZ, float farZ) const
{
    // Practical split scheme: blend between logarithmic and uniform distributions.
    // Logarithmic gives more resolution near the camera; uniform distributes evenly.
    std::array<float, kCascadeCount> splits;
    for (uint32_t i = 0; i < kCascadeCount; i++)
    {
        float p        = float(i + 1) / float(kCascadeCount);
        float logSplit = nearZ * std::pow(farZ / nearZ, p);
        float linSplit = nearZ + (farZ - nearZ) * p;
        splits[i]      = kLambda * logSplit + (1.f - kLambda) * linSplit;
    }
    return splits;
}

// =============================================================================
// Frustum corners
// =============================================================================

std::array<float3, 8>
RenderShadowMap::getFrustumCornersWS(const Camera& cam, float sliceNear, float sliceFar) const
{
    // Unproject the 8 NDC unit-cube corners of the full camera frustum to world space.
    // D3D convention: NDC z in [0, 1] (near = 0, far = 1).
    static const float3 kNdcCube[8] = {
        {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0}, {1, 1, 0}, // near face
        {-1, -1, 1}, {1, -1, 1}, {-1, 1, 1}, {1, 1, 1}, // far face
    };

    float4x4 invVP = math::inverse(cam.getViewProjMatrix());
    float3 fullCorners[8];
    for (int i = 0; i < 8; i++)
    {
        const float3& n = kNdcCube[i];
        float4 w = math::mul(invVP, float4(n.x, n.y, n.z, 1.f));
        fullCorners[i] = float3(w.x, w.y, w.z) / w.w;
    }

    // Linearly interpolate along each frustum edge to extract the [sliceNear, sliceFar] sub-frustum.
    // This is valid because frustum edges are straight lines in world space.
    float cNear = cam.getNearPlane();
    float cFar  = cam.getFarPlane();
    float tNear = (sliceNear - cNear) / (cFar - cNear);
    float tFar  = (sliceFar  - cNear) / (cFar - cNear);

    std::array<float3, 8> corners;
    for (int i = 0; i < 4; i++)
    {
        float3 edge    = fullCorners[i + 4] - fullCorners[i];
        corners[i]     = fullCorners[i] + tNear * edge; // near face of slice
        corners[i + 4] = fullCorners[i] + tFar  * edge; // far face of slice
    }
    return corners;
}

// =============================================================================
// Cascade light matrix
// =============================================================================

RenderShadowMap::CascadeData
RenderShadowMap::computeCascadeMatrix(const std::array<float3, 8>& corners, float3 lightDir, float splitFar) const
{
    // Look-at center = centroid of the frustum slice.
    float3 center(0.f);
    for (const auto& c : corners)
    {
        center.x += c.x;
        center.y += c.y;
        center.z += c.z;
    }
    center /= float(corners.size());

    lightDir    = normalize(lightDir);
    float3 up   = (std::abs(lightDir.y) > 0.99f) ? float3(0, 0, 1) : float3(0, 1, 0);
    // Pull the eye back by 2× the scene radius so that no scene geometry can end up
    // behind the camera (which would give it positive view-space Z and exclude it from
    // the shadow map entirely).
    const float sceneRadius = mpScene->getSceneBounds().radius();
    float4x4 lightView = math::matrixFromLookAt(center - lightDir * sceneRadius * 2.f, center, up);

    // Tight AABB of the 8 corners in light-view space.
    float3 minLS( 1e38f,  1e38f,  1e38f);
    float3 maxLS(-1e38f, -1e38f, -1e38f);
    for (const auto& c : corners)
    {
        float4 lc = math::mul(lightView, float4(c.x, c.y, c.z, 1.f));
        // View matrix is affine — w is guaranteed to be 1.
        float3 ls(lc.x, lc.y, lc.z);
        minLS.x = std::min(minLS.x, ls.x);  maxLS.x = std::max(maxLS.x, ls.x);
        minLS.y = std::min(minLS.y, ls.y);  maxLS.y = std::max(maxLS.y, ls.y);
        minLS.z = std::min(minLS.z, ls.z);  maxLS.z = std::max(maxLS.z, ls.z);
    }

    // Snap XY bounds to texel-sized increments to prevent shadow edge shimmering
    // when the camera translates by sub-texel amounts.
    float texelW = (maxLS.x - minLS.x) / float(kShadowMapSize);
    float texelH = (maxLS.y - minLS.y) / float(kShadowMapSize);
    minLS.x = std::floor(minLS.x / texelW) * texelW;
    minLS.y = std::floor(minLS.y / texelH) * texelH;
    maxLS.x = minLS.x + float(kShadowMapSize) * texelW;
    maxLS.y = minLS.y + float(kShadowMapSize) * texelH;

    // math::ortho is right-handed and takes positive near/far DISTANCES, mapping
    // view-space z∈[-near, -far] → depth [0, 1].
    // In right-handed light-view space, objects in front have NEGATIVE Z, so:
    //   - farDist = -minLS.z  (minLS.z is the most-negative = farthest receiver corner)
    // Near is fixed at 0.1 rather than derived from the AABB. The frustum corners only
    // tell us where the receivers are; casters can be anywhere between the light and the
    // scene. With the eye pulled back by 2× scene radius, 0.1 opens the near plane toward
    // "the sky" and captures every potential shadow caster without wasting depth precision
    // (there is no perspective z-fighting in an orthographic projection).
    float farDist = -minLS.z;
    float zPad    = (farDist - (-maxLS.z)) * 0.1f;
    float4x4 lightProj = math::ortho(minLS.x, maxLS.x, minLS.y, maxLS.y,
                                      0.1f,
                                      farDist + zPad);

    return { math::mul(lightProj, lightView), splitFar };
}

// =============================================================================
// Frame render
// =============================================================================

void RenderShadowMap::renderCascades(RenderContext* pRenderContext, float3 lightDir)
{
    const auto& pCam = mpScene->getCamera();
    float nearZ      = pCam->getNearPlane();
    float farZ       = pCam->getFarPlane();

    auto splits   = computeSplitDistances(nearZ, farZ);
    float prevSplit = nearZ;

    for (uint32_t i = 0; i < kCascadeCount; i++)
    {
        auto corners  = getFrustumCornersWS(*pCam, prevSplit, splits[i]);
        mCascades[i]  = computeCascadeMatrix(corners, lightDir, splits[i]);

        pRenderContext->clearFbo(mpCascadeFbos[i].get(), float4(0.f), 1.f, 0, FboAttachmentType::Depth);
        mpShadowPass->getState()->setFbo(mpCascadeFbos[i]);
        mpShadowPass->getVars()->getRootVar()["PerFrame"]["gLightVP"] = mCascades[i].lightVP;
        mpScene->rasterize(pRenderContext, mpShadowPass->getState().get(), mpShadowPass->getVars().get(),
                           mpShadowRsState, mpShadowRsState);

        prevSplit = splits[i];
    }
}

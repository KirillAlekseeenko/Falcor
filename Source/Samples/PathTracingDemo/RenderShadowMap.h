#pragma once
#include "Falcor.h"
#include "Core/Pass/RasterPass.h"

using namespace Falcor;

/// Renders a cascaded shadow map (CSM) for a single directional light.
///
/// Owns all GPU resources: a D32Float Texture2DArray (one slice per cascade),
/// per-cascade FBOs, and the depth-only RasterPass.
/// Call renderCascades() once per frame, then bind getShadowMapArray() and
/// getCascades() into the lighting pass.
class RenderShadowMap
{
public:
    static constexpr uint32_t kCascadeCount  = 4;
    static constexpr uint32_t kShadowMapSize = 2048;

    /// Per-cascade data needed by the lighting pass shader.
    struct CascadeData
    {
        float4x4 lightVP;   ///< World → light clip space
        float    splitFar;  ///< Linear view-space Z of the cascade far boundary
    };

    /// Allocates GPU resources and builds the shadow RasterPass.
    RenderShadowMap(ref<Device> pDevice, ref<Scene> pScene);

    /// Recomputes cascade matrices and renders N depth passes.
    /// Must be called before the lighting pass each frame.
    void renderCascades(RenderContext* pRenderContext, float3 lightDir);

    ref<Texture>                                   getShadowMapArray() const { return mpShadowArray; }
    const std::array<CascadeData, kCascadeCount>&  getCascades()       const { return mCascades; }

private:
    /// Returns kCascadeCount view-space Z split distances using the practical split scheme.
    std::array<float, kCascadeCount> computeSplitDistances(float nearZ, float farZ) const;

    /// Returns 8 world-space corners of the camera frustum slice [sliceNear, sliceFar].
    std::array<float3, 8> getFrustumCornersWS(const Camera& cam, float sliceNear, float sliceFar) const;

    /// Builds the ortho light-VP matrix for one cascade from its world-space corners.
    CascadeData computeCascadeMatrix(const std::array<float3, 8>& corners, float3 lightDir, float splitFar) const;

    ref<Scene>                             mpScene;
    ref<RasterPass>                        mpShadowPass;
    ref<RasterizerState>                   mpShadowRsState;    ///< Back-face culling + depth bias
    ref<Texture>                           mpShadowArray;      ///< D32Float Texture2DArray[kCascadeCount]
    std::array<ref<Fbo>, kCascadeCount>    mpCascadeFbos;      ///< One FBO per array slice
    std::array<CascadeData, kCascadeCount> mCascades;

    /// Blend between logarithmic (1.0) and uniform (0.0) split distributions.
    static constexpr float kLambda = 0.5f;
};

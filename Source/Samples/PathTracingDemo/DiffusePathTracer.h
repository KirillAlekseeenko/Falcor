#pragma once
#include "Falcor.h"

using namespace Falcor;

/** Diffuse global-illumination path tracer using DXR.
 *
 *  Each frame fires one primary ray per pixel and traces up to kMaxBounces
 *  diffuse bounces, with one shadow ray per bounce toward the directional
 *  light (NEE).  Results are accumulated into a persistent float4 texture
 *  using a running average; the accumulation resets automatically when the
 *  camera moves.
 *
 *  Usage:
 *    1. Construct once after the scene is loaded.
 *    2. Call onResize() whenever the framebuffer changes size.
 *    3. Call render() each frame; it blits the result into pTargetFbo.
 */
class DiffusePathTracer
{
public:
    DiffusePathTracer(ref<Device> pDevice, ref<Scene> pScene);

    /** Render one frame and blit the accumulated result into pTargetFbo.
     *  @param dirLightIndex  Scene index of the directional light, or -1. */
    void render(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo, int dirLightIndex);

    /** Recreate the accumulation texture when the window is resized. */
    void onResize(uint32_t width, uint32_t height);

    /** Force a full accumulation reset (e.g. after a scene edit). */
    void reset() { mFrameIndex = 0; }

private:
    void createProgram();
    void createAccumTexture(uint32_t width, uint32_t height);

    ref<Device>        mpDevice;
    ref<Scene>         mpScene;
    ref<Program>       mpProgram;
    ref<RtProgramVars> mpVars;
    ref<Texture>       mpAccum;         ///< RGBA32F running-average buffer

    uint32_t  mFrameIndex = 0;
    uint32_t  mWidth      = 0;
    uint32_t  mHeight     = 0;

    // Camera-change detection
    float4x4  mPrevViewProj = float4x4();
};

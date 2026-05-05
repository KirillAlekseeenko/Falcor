#pragma once
#include "Falcor.h"
#include "Rendering/NeuralIrradianceModel.h"

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
    enum class Mode
    {
        Reference,
        NeuralIndirect,
    };

    DiffusePathTracer(ref<Device> pDevice, ref<Scene> pScene);

    /** Render one frame and blit the accumulated result into pTargetFbo.
     *  @param dirLightIndex  Scene index of the directional light, or -1. */
    void render(
        RenderContext* pRenderContext,
        const ref<Fbo>& pTargetFbo,
        int dirLightIndex,
        Mode mode = Mode::Reference,
        const NeuralIrradianceModel* pNeuralModel = nullptr,
        float neuralIndirectScale = 1.f,
        float surfaceOffset = 1e-3f
    );

    /** Recreate the accumulation texture when the window is resized. */
    void onResize(uint32_t width, uint32_t height);

    /** Force a full accumulation reset (e.g. after a scene edit). */
    void reset() { mFrameIndex = 0; }

    /** Recreate raytracing programs after shader hot reload. */
    void recreatePrograms();

private:
    void createProgram(bool useNeuralIndirect, ref<Program>& pProgram, ref<RtProgramVars>& pVars);
    void createAccumTexture(uint32_t width, uint32_t height);

    ref<Device>        mpDevice;
    ref<Scene>         mpScene;
    ref<Program>       mpReferenceProgram;
    ref<RtProgramVars> mpReferenceVars;
    ref<Program>       mpNeuralProgram;
    ref<RtProgramVars> mpNeuralVars;
    ref<Texture>       mpAccum;         ///< RGBA32F running-average buffer

    uint32_t  mFrameIndex = 0;
    uint32_t  mWidth      = 0;
    uint32_t  mHeight     = 0;
    Mode      mPreviousMode = Mode::Reference;
    float     mPreviousNeuralIndirectScale = 1.f;
    float     mPreviousSurfaceOffset = 1e-3f;

    // Camera-change detection
    float4x4  mPrevViewProj = float4x4();
};

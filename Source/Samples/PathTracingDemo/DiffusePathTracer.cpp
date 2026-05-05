#include "DiffusePathTracer.h"

#include <cstring> // memcmp

// =============================================================================
// Construction
// =============================================================================

DiffusePathTracer::DiffusePathTracer(ref<Device> pDevice, ref<Scene> pScene)
    : mpDevice(pDevice)
    , mpScene(pScene)
{
    recreatePrograms();
}

void DiffusePathTracer::createProgram(bool useNeuralIndirect, ref<Program>& pProgram, ref<RtProgramVars>& pVars)
{
    // Two miss shaders (primary, shadow), two ray types (primary, shadow).
    uint32_t geometryCount = mpScene->getGeometryCount();
    ref<RtBindingTable> sbt = RtBindingTable::create(2, 2, geometryCount);

    ProgramDesc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary("Samples/PathTracingDemo/DiffusePathTracer.rt.slang");
    desc.addTypeConformances(mpScene->getTypeConformances());
    desc.setMaxTraceRecursionDepth(2); // raygen → closesthit → shadowMiss
    desc.setMaxPayloadSize(80);        // PathPayload (~72 B); round up

    sbt->setRayGen(desc.addRayGen("rayGen"));
    sbt->setMiss(0, desc.addMiss("primaryMiss"));
    sbt->setMiss(1, desc.addMiss("shadowMiss"));

    auto primaryHitGroup = desc.addHitGroup("primaryClosestHit", "primaryAnyHit");
    auto shadowHitGroup  = desc.addHitGroup("", "shadowAnyHit");

    auto meshIDs = mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh);
    sbt->setHitGroup(0, meshIDs, primaryHitGroup);
    sbt->setHitGroup(1, meshIDs, shadowHitGroup);

    DefineList defines = mpScene->getSceneDefines();
    defines.add("USE_NEURAL_INDIRECT", useNeuralIndirect ? "1" : "0");

    pProgram = Program::create(mpDevice, desc, defines);
    pVars    = RtProgramVars::create(mpDevice, pProgram, sbt);
}

void DiffusePathTracer::recreatePrograms()
{
    createProgram(false, mpReferenceProgram, mpReferenceVars);
    createProgram(true, mpNeuralProgram, mpNeuralVars);
    reset();
}

void DiffusePathTracer::createAccumTexture(uint32_t width, uint32_t height)
{
    mpAccum = mpDevice->createTexture2D(
        width, height,
        ResourceFormat::RGBA32Float,
        1, 1,
        nullptr,
        ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
    );
}

// =============================================================================
// Resize
// =============================================================================

void DiffusePathTracer::onResize(uint32_t width, uint32_t height)
{
    if (mWidth == width && mHeight == height) return;
    mWidth  = width;
    mHeight = height;
    createAccumTexture(width, height);
    reset();
}

// =============================================================================
// Per-frame render
// =============================================================================

void DiffusePathTracer::render(
    RenderContext* pRenderContext,
    const ref<Fbo>& pTargetFbo,
    int dirLightIndex,
    Mode mode,
    const NeuralIrradianceModel* pNeuralModel,
    float neuralIndirectScale,
    float surfaceOffset
)
{
    if (mode == Mode::NeuralIndirect && (pNeuralModel == nullptr || !pNeuralModel->isLoaded()))
        mode = Mode::Reference;

    // Lazy texture creation (first frame or after resize).
    uint32_t w = pTargetFbo->getWidth();
    uint32_t h = pTargetFbo->getHeight();
    if (!mpAccum || mWidth != w || mHeight != h)
        onResize(w, h);

    // Reset accumulation when the camera moves.
    float4x4 vp = mpScene->getCamera()->getViewProjMatrix();
    if (memcmp(&vp, &mPrevViewProj, sizeof(float4x4)) != 0)
    {
        mPrevViewProj = vp;
        mFrameIndex   = 0;
    }

    if (mode != mPreviousMode || neuralIndirectScale != mPreviousNeuralIndirectScale || surfaceOffset != mPreviousSurfaceOffset)
    {
        mPreviousMode = mode;
        mPreviousNeuralIndirectScale = neuralIndirectScale;
        mPreviousSurfaceOffset = surfaceOffset;
        reset();
    }

    ref<Program> pProgram = mode == Mode::NeuralIndirect ? mpNeuralProgram : mpReferenceProgram;
    ref<RtProgramVars> pVars = mode == Mode::NeuralIndirect ? mpNeuralVars : mpReferenceVars;

    // Bind per-frame constants.
    auto var = pVars->getRootVar();
    var["PerFrame"]["gFrameDim"]      = uint2(w, h);
    var["PerFrame"]["gFrameIndex"]    = mFrameIndex;
    var["PerFrame"]["gDirLightIndex"] = dirLightIndex >= 0
                                            ? static_cast<uint32_t>(dirLightIndex)
                                            : 0xFFFFFFFFu;
    var["PerFrame"]["gUseNeuralIndirect"] = mode == Mode::NeuralIndirect ? 1u : 0u;
    var["PerFrame"]["gNeuralIndirectScale"] = neuralIndirectScale;
    var["PerFrame"]["gSurfaceOffset"] = surfaceOffset;

    if (mode == Mode::NeuralIndirect)
        pNeuralModel->bindShaderData(var);

    // Bind accumulation UAV.
    var["gAccum"] = mpAccum;

    // Dispatch rays.
    mpScene->raytrace(pRenderContext, pProgram.get(), pVars, uint3(w, h, 1));

    // Blit accumulated result into the target FBO.
    pRenderContext->blit(mpAccum->getSRV(), pTargetFbo->getRenderTargetView(0));

    ++mFrameIndex;
}

/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "PathTracingDemo.h"

#include "Core/Platform/OS.h"
#include "Utils/UI/TextRenderer.h"

#include <algorithm>

FALCOR_EXPORT_D3D12_AGILITY_SDK

uint32_t mSampleGuiWidth = 250;
uint32_t mSampleGuiHeight = 200;
uint32_t mSampleGuiPositionX = 20;
uint32_t mSampleGuiPositionY = 40;

PathTracingDemo::PathTracingDemo(const SampleAppConfig& config) : SampleApp(config)
{
    //
}

PathTracingDemo::~PathTracingDemo()
{
    //
}

void PathTracingDemo::loadScene(const std::string& scenePath, Fbo* displayFbo)
{
    mpScene  = Scene::create(getDevice(), scenePath);
    mpScene->setIsAnimated(false);
    mpCamera = mpScene->getCamera();

    // Update the controllers
    float radius = mpScene->getSceneBounds().radius();
    mpScene->setCameraSpeed(radius * 0.25f);
    float nearZ = 0.1f; // std::max(0.1f, radius / 750.0f);
    float farZ  = radius * 2;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)displayFbo->getWidth() / (float)displayFbo->getHeight());

    // ---- Lighting raster pass -------------------------------------------
    createRasterPasses();

    // ---- Cascaded shadow map --------------------------------------------
    mpShadowMapRenderer = std::make_unique<RenderShadowMap>(getDevice(), mpScene);

    // ---- Diffuse path tracer --------------------------------------------
    mpDiffusePT = std::make_unique<DiffusePathTracer>(getDevice(), mpScene);
    mSurfaceOffset = computeSurfaceOffset();

    if (mNeuralModelPath.empty())
        mNeuralModelPath = getProjectDirectory() / "NeuralLightGridLearning" / "checkpoints" / "tiny_irradiance_mlp.falcor-mlp.bin";
    if (std::filesystem::exists(mNeuralModelPath))
        loadNeuralModel(mNeuralModelPath);
    else
        mNeuralModelStatus = fmt::format("No neural model loaded. Default path not found: {}", mNeuralModelPath.string());

    // ---- PCF comparison sampler -----------------------------------------
    Sampler::Desc sDesc;
    sDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point)
         .setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp)
         .setComparisonFunc(ComparisonFunc::LessEqual);
    mpShadowSampler = getDevice()->createSampler(sDesc);

    // ---- Brightness scale pass ------------------------------------------
    mpBrightnessScalePass = FullScreenPass::create(getDevice(), "Samples/PathTracingDemo/BrightnessScale.ps.slang");

    Sampler::Desc bsDesc;
    bsDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear)
          .setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    mpBrightnessSampler = getDevice()->createSampler(bsDesc);
}

void PathTracingDemo::createRasterPasses()
{
    if (!mpScene)
        return;

    ProgramDesc rasterProgDesc;
    rasterProgDesc.addShaderModules(mpScene->getShaderModules());
    rasterProgDesc.setCompilerFlags(SlangCompilerFlags::GenerateDebugInfo);
    rasterProgDesc.addShaderLibrary("Samples/PathTracingDemo/SimpleRasterPass.3d.slang")
        .vsEntry("vsMain")
        .psEntry("psMain");
    rasterProgDesc.addTypeConformances(mpScene->getTypeConformances());

    DefineList directDefines = mpScene->getSceneDefines();
    directDefines.add("USE_NEURAL_INDIRECT", "0");
    mpRasterPass = RasterPass::create(getDevice(), rasterProgDesc, directDefines);

    DefineList neuralDefines = mpScene->getSceneDefines();
    neuralDefines.add("USE_NEURAL_INDIRECT", "1");
    mpNeuralRasterPass = RasterPass::create(getDevice(), rasterProgDesc, neuralDefines);

    if (mpHdrFbo)
    {
        mpRasterPass->getState()->setFbo(mpHdrFbo);
        mpNeuralRasterPass->getState()->setFbo(mpHdrFbo);
    }
}

void PathTracingDemo::loadNeuralModel(const std::filesystem::path& path)
{
    try
    {
        auto pModel = std::make_unique<NeuralIrradianceModel>(getDevice());
        pModel->loadFromFile(path);
        mpNeuralModel = std::move(pModel);
        mNeuralModelPath = path;
        mNeuralModelStatus = fmt::format("Loaded neural model: {}", path.string());
        if (mpDiffusePT)
            mpDiffusePT->reset();
    }
    catch (const std::exception& e)
    {
        mpNeuralModel.reset();
        mNeuralModelStatus = fmt::format("Failed to load neural model '{}': {}", path.string(), e.what());
        logError("{}", mNeuralModelStatus);
    }
}

bool PathTracingDemo::isNeuralModelLoaded() const
{
    return mpNeuralModel && mpNeuralModel->isLoaded();
}

float PathTracingDemo::computeSurfaceOffset() const
{
    if (!mpScene)
        return 1e-3f;

    return std::max(1e-4f, mpScene->getSceneBounds().radius() * 1e-5f);
}

void PathTracingDemo::onLoad(RenderContext* pRenderContext)
{
    if (getDevice()->isFeatureSupported(Device::SupportedFeatures::Raytracing) == false)
    {
        FALCOR_THROW("Device does not support raytracing!");
    }

    const std::string kBistroScenePath = "Bistro_v5_2/BistroExterior.pyscene";
    loadScene(kBistroScenePath, getTargetFbo().get());
}

void PathTracingDemo::onShutdown()
{
    SampleApp::onShutdown();
}

void PathTracingDemo::onResize(uint32_t width, uint32_t height)
{
    SampleApp::onResize(width, height);

    if (mpCamera)
        mpCamera->setAspectRatio((float)width / (float)height);

    if (mpRasterPass)
    {
        // (Re)create intermediate HDR FBO at the new resolution
        auto pColorTex = getDevice()->createTexture2D(
            width, height,
            ResourceFormat::RGBA16Float,
            1, 1,
            nullptr,
            ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
        );
        auto pDepthTex = getDevice()->createTexture2D(
            width, height,
            ResourceFormat::D32Float,
            1, 1,
            nullptr,
            ResourceBindFlags::DepthStencil
        );
        mpHdrFbo = Fbo::create(getDevice());
        mpHdrFbo->attachColorTarget(pColorTex, 0);
        mpHdrFbo->attachDepthStencilTarget(pDepthTex);

        mpRasterPass->getState()->setFbo(mpHdrFbo);
        if (mpNeuralRasterPass)
            mpNeuralRasterPass->getState()->setFbo(mpHdrFbo);
    }

    if (mpDiffusePT)
        mpDiffusePT->onResize(width, height);
}

float3 PathTracingDemo::getFirstDirectionalLightDir(int& dirLightIndex) const
{
    dirLightIndex = -1;

    if (mpScene == nullptr)
    {
        logWarning("No scene loaded. Cannot get first directional light direction.");
        return float3(0, 0, 0);
    }

    for (uint32_t i = 0; i < mpScene->getLightCount(); ++i)
    {
        const auto& pLight = mpScene->getLight(i);
        if (pLight->getType() == LightType::Directional)
        {
            const DirectionalLight* pDir = static_cast<const DirectionalLight*>(pLight.get());
            dirLightIndex = i;
            return pDir->getWorldDirection();
        }
    }

    return float3(-0.6f, -1.0f, 0.4f);
}

void PathTracingDemo::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    if (mpScene == nullptr)
    {
        logWarning("No scene loaded. Cannot render.");
        return;
    }

    // 1. Locate the directional light
    int dirLightIndex = -1;
    float3 lightDir = getFirstDirectionalLightDir(dirLightIndex);

    // 2. Tick the scene (animations, updates)
    IScene::UpdateFlags updates = mpScene->update(pRenderContext, getGlobalClock().getTime());
    if (is_set(updates, IScene::UpdateFlags::GeometryChanged))
        FALCOR_THROW("This sample does not support scene geometry changes.");
    if (is_set(updates, IScene::UpdateFlags::RecompileNeeded))
        FALCOR_THROW("This sample does not support scene changes that require shader recompilation.");

    const bool neuralModelLoaded = isNeuralModelLoaded();
    const bool useReferencePathTracer = mRenderMode == RenderMode::ReferencePathTracer;
    const bool useNeuralRayTracing = mRenderMode == RenderMode::NeuralRayTracing && neuralModelLoaded;
    const bool usePathTracer = useReferencePathTracer || useNeuralRayTracing || mRenderMode == RenderMode::NeuralRayTracing;

    if (usePathTracer)
    {
        // ---- Path-traced GI -------------------------------------------------
        // Render into the intermediate HDR FBO; brightness scale pass reads from it.
        mpDiffusePT->render(
            pRenderContext,
            mpHdrFbo,
            dirLightIndex,
            useNeuralRayTracing ? DiffusePathTracer::Mode::NeuralIndirect : DiffusePathTracer::Mode::Reference,
            neuralModelLoaded ? mpNeuralModel.get() : nullptr,
            mNeuralIndirectScale,
            mSurfaceOffset
        );
    }
    else
    {
        // ---- Rasterised direct lighting + CSM shadows -----------------------

        // 3a. Render cascade shadow maps
        if (mShadowsEnabled)
            mpShadowMapRenderer->renderCascades(pRenderContext, lightDir);

        // 3b. Clear intermediate HDR FBO
        const float4 clearColor(0.38f, 0.52f, 0.10f, 1);
        pRenderContext->clearFbo(mpHdrFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

        // 3c. Bind shadow map array and cascade data
        const bool useNeuralRaster = mRenderMode == RenderMode::RasterNeuralIndirect && neuralModelLoaded;
        ref<RasterPass> pActiveRasterPass = useNeuralRaster ? mpNeuralRasterPass : mpRasterPass;

        const auto& pReflection = pActiveRasterPass->getProgram()->getReflector()->getDefaultParameterBlock();
        auto shadowMapLoc    = pReflection->getResourceBinding("gShadowDepth");
        auto shadowSamplerLoc = pReflection->getResourceBinding("gShadowSampler");

        pActiveRasterPass->getVars()->setSrv(shadowMapLoc, mpShadowMapRenderer->getShadowMapArray()->getSRV());
        pActiveRasterPass->getVars()->setSampler(shadowSamplerLoc, mpShadowSampler);

        auto var = pActiveRasterPass->getVars()->getRootVar();
        const auto& cascades = mpShadowMapRenderer->getCascades();
        for (uint32_t i = 0; i < RenderShadowMap::kCascadeCount; i++)
        {
            var["CommonParameters"]["gCascadeLightVPs"][i] = cascades[i].lightVP;
            var["CommonParameters"]["gCascadeSplits"][i]   = cascades[i].splitFar;
        }

        const float kTexelSize = 1.f / float(RenderShadowMap::kShadowMapSize);
        var["CommonParameters"]["gTexelSize"]        = float2(kTexelSize, kTexelSize);
        var["CommonParameters"]["globalLightIndex"]  = dirLightIndex;
        var["CommonParameters"]["gShadowsEnabled"]   = mShadowsEnabled ? 1 : 0;
        var["CommonParameters"]["gNeuralIndirectEnabled"] = useNeuralRaster ? 1 : 0;
        var["CommonParameters"]["gNeuralIndirectScale"] = mNeuralIndirectScale;
        var["CommonParameters"]["gSurfaceOffset"] = mSurfaceOffset;

        if (useNeuralRaster)
            mpNeuralModel->bindShaderData(var);

        // 3d. Lighting raster pass
        mpScene->rasterize(pRenderContext, pActiveRasterPass->getState().get(), pActiveRasterPass->getVars().get());
    }

    // ---- Brightness scale pass (HDR → display) --------------------------
    auto bsVars = mpBrightnessScalePass->getRootVar();
    bsVars["gInputColor"] = mpHdrFbo->getColorTexture(0);
    bsVars["gSampler"]    = mpBrightnessSampler;
    mpBrightnessScalePass->execute(pRenderContext, pTargetFbo);

    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});
}

void PathTracingDemo::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Falcor", {420, 280});
    renderGlobalUI(pGui);

    static const Gui::DropdownList kRenderModes = {
        {uint32_t(RenderMode::RasterDirect), "Raster direct"},
        {uint32_t(RenderMode::RasterNeuralIndirect), "Raster direct + neural GI"},
        {uint32_t(RenderMode::ReferencePathTracer), "Reference path tracer"},
        {uint32_t(RenderMode::NeuralRayTracing), "RT direct + neural GI"},
    };

    uint32_t renderMode = uint32_t(mRenderMode);
    if (w.dropdown("Render Mode", kRenderModes, renderMode))
    {
        mRenderMode = RenderMode(renderMode);
        if (mpDiffusePT)
            mpDiffusePT->reset();
    }

    if (mRenderMode == RenderMode::RasterDirect || mRenderMode == RenderMode::RasterNeuralIndirect)
        w.checkbox("Shadows", mShadowsEnabled);

    w.separator();
    w.text("Neural Irradiance");
    w.text(mNeuralModelPath.empty() ? "<none>" : mNeuralModelPath.string());
    if (w.button("Load Model"))
    {
        static const FileDialogFilterVec kModelFilters = {{ "bin", "Falcor MLP model" }};
        std::filesystem::path path = mNeuralModelPath;
        if (openFileDialog(kModelFilters, path))
            loadNeuralModel(path);
    }

    if (w.button("Reload Model", true) && !mNeuralModelPath.empty())
        loadNeuralModel(mNeuralModelPath);

    if (w.var("GI Scale", mNeuralIndirectScale, 0.f, 10.f) && mpDiffusePT)
        mpDiffusePT->reset();

    w.var("Surface Offset", mSurfaceOffset, 1e-6f, 1.f);
    w.text(mNeuralModelStatus);

    if ((mRenderMode == RenderMode::RasterNeuralIndirect || mRenderMode == RenderMode::NeuralRayTracing) && !isNeuralModelLoaded())
        w.text("Neural mode selected, but no model is loaded.");
}

bool PathTracingDemo::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (keyEvent.key == Input::Key::Escape && keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        shutdown();
        return true;
    }

    if (mpScene && mpScene->onKeyEvent(keyEvent))
        return true;

    return false;
}

bool PathTracingDemo::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpScene && mpScene->onMouseEvent(mouseEvent);
}

void PathTracingDemo::onHotReload(HotReloadFlags reloaded)
{
    if (is_set(reloaded, HotReloadFlags::Program))
    {
        createRasterPasses();
        if (mpDiffusePT)
            mpDiffusePT->recreatePrograms();
    }
}


int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.title = "Falcor Project Template";
    config.windowDesc.resizableWindow = true;

    PathTracingDemo project(config);
    return project.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}

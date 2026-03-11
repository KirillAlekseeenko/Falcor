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

#include "Utils/UI/TextRenderer.h"

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
    mpCamera = mpScene->getCamera();

    // Update the controllers
    float radius = mpScene->getSceneBounds().radius();
    mpScene->setCameraSpeed(radius * 0.25f);
    float nearZ = 0.1f; // std::max(0.1f, radius / 750.0f);
    float farZ  = radius * 2;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)displayFbo->getWidth() / (float)displayFbo->getHeight());

    // Get shader modules and type conformances for types used by the scene.
    // These need to be set on the program in order to use Falcor's material system.
    auto shaderModules    = mpScene->getShaderModules();
    auto typeConformances = mpScene->getTypeConformances();
    auto defines          = mpScene->getSceneDefines();

    // ---- Lighting raster pass -------------------------------------------
    ProgramDesc rasterProgDesc;
    rasterProgDesc.addShaderModules(shaderModules);
    rasterProgDesc.setCompilerFlags(SlangCompilerFlags::GenerateDebugInfo);
    rasterProgDesc.addShaderLibrary("Samples/PathTracingDemo/SimpleRasterPass.3d.slang")
        .vsEntry("vsMain")
        .psEntry("psMain");
    rasterProgDesc.addTypeConformances(typeConformances);

    mpRasterPass = RasterPass::create(getDevice(), rasterProgDesc, defines);

    // ---- Cascaded shadow map --------------------------------------------
    mpShadowMapRenderer = std::make_unique<RenderShadowMap>(getDevice(), mpScene);

    // ---- PCF comparison sampler -----------------------------------------
    Sampler::Desc sDesc;
    sDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point)
         .setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp)
         .setComparisonFunc(ComparisonFunc::LessEqual);
    mpShadowSampler = getDevice()->createSampler(sDesc);
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
        mpRasterPass->getState()->setFbo(getTargetFbo());
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

    // 2. Render all cascade shadow maps
    if (mShadowsEnabled)
        mpShadowMapRenderer->renderCascades(pRenderContext, lightDir);

    // 3. Clear target and tick the scene
    const float4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    IScene::UpdateFlags updates = mpScene->update(pRenderContext, getGlobalClock().getTime());
    if (is_set(updates, IScene::UpdateFlags::GeometryChanged))
        FALCOR_THROW("This sample does not support scene geometry changes.");
    if (is_set(updates, IScene::UpdateFlags::RecompileNeeded))
        FALCOR_THROW("This sample does not support scene changes that require shader recompilation.");

    // 4. Bind shadow map array and cascade data to the lighting pass
    const auto& pReflection = mpRasterPass->getProgram()->getReflector()->getDefaultParameterBlock();
    auto shadowMapLoc    = pReflection->getResourceBinding("gShadowDepth");
    auto shadowSamplerLoc = pReflection->getResourceBinding("gShadowSampler");

    mpRasterPass->getVars()->setSrv(shadowMapLoc, mpShadowMapRenderer->getShadowMapArray()->getSRV());
    mpRasterPass->getVars()->setSampler(shadowSamplerLoc, mpShadowSampler);

    auto var = mpRasterPass->getVars()->getRootVar();
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

    // 5. Render lighting pass
    mpRasterPass->getState()->setFbo(pTargetFbo);
    mpScene->rasterize(pRenderContext, mpRasterPass->getState().get(), mpRasterPass->getVars().get());

    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});
}

void PathTracingDemo::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Falcor", {250, 200});
    renderGlobalUI(pGui);
    w.checkbox("Shadows", mShadowsEnabled);
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
    //
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

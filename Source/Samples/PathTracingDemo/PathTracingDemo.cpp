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
    mpScene = Scene::create(getDevice(), scenePath);
    mpCamera = mpScene->getCamera();

    // Update the controllers
    float radius = mpScene->getSceneBounds().radius();
    mpScene->setCameraSpeed(radius * 0.25f);
    float nearZ = std::max(0.1f, radius / 750.0f);
    float farZ = radius * 10;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)displayFbo->getWidth() / (float)displayFbo->getHeight());

    // Get shader modules and type conformances for types used by the scene.
    // These need to be set on the program in order to use Falcor's material system.
    auto shaderModules = mpScene->getShaderModules();
    auto typeConformances = mpScene->getTypeConformances();

    // Get scene defines. These need to be set on any program using the scene.
    auto defines = mpScene->getSceneDefines();

    // Create raster pass.
    // This utility wraps the creation of the program and vars, and sets the necessary scene defines.
    ProgramDesc rasterProgDesc;
    rasterProgDesc.addShaderModules(shaderModules);
    rasterProgDesc.setCompilerFlags(SlangCompilerFlags::GenerateDebugInfo); // generate PDB/Spir-V debug info
    rasterProgDesc.addShaderLibrary("Samples/PathTracingDemo/SimpleRasterPass.3d.slang").vsEntry("vsMain").psEntry("psMain");
    rasterProgDesc.addTypeConformances(typeConformances);

    mpRasterPass = RasterPass::create(getDevice(), rasterProgDesc, defines);

    // ------------- depth texture + FBO ------------
    Fbo::Desc fboDesc;
    fboDesc.setDepthStencilTarget(ResourceFormat::D32Float);
    mpShadowFbo = Fbo::create2D(getDevice(), kShadowRes, kShadowRes, fboDesc);
    mpShadowMap = mpShadowFbo->getDepthStencilTexture();

    // ------------- depth-only RasterPass ----------
    ProgramDesc shadowPassDesc;
    shadowPassDesc.addShaderModules(shaderModules);
    shadowPassDesc.addShaderLibrary("Samples/PathTracingDemo/ShadowPass.3d.slang").vsEntry("vsMain").psEntry("psMain"); // empty pixel shader
    shadowPassDesc.addTypeConformances(typeConformances);

    mpShadowPass = RasterPass::create(getDevice(), shadowPassDesc, defines);

    // ------------- PCF comparison sampler --------
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

    const std::string kBistroScenePath = "Bistro_v5_2/BistroExterior.pyscene"; // "Arcade/Arcade.pyscene"
    // create implementation for the method loadScene
    loadScene(kBistroScenePath, getTargetFbo().get());
}

void PathTracingDemo::onShutdown()
{
    // in DXR sample there is no need to release resources here, as the SampleApp will probably take care of it.
    SampleApp::onShutdown();
}

void PathTracingDemo::onResize(uint32_t width, uint32_t height)
{
    SampleApp::onResize(width, height);

    // Resize the scene camera.
    if (mpCamera)
    {
        mpCamera->setAspectRatio((float)width / (float)height);
    }

    // Resize the raster pass FBO.
    if (mpRasterPass)
    {
        mpRasterPass->getState()->setFbo(getTargetFbo());
    }
}

static float4x4 computeDirLightVP(const AABB& box, float3 dir)
{
    dir = normalize(dir);
    float3 up = abs(dir.y) > 0.99f ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 pos = box.center() - dir * box.radius() * 2.f;

    float4x4 view = math::matrixFromLookAt(pos, box.center(), up); 
    float r = box.radius();
    float4x4 proj = math::ortho(-r, r, -r, r, 0.1f, r * 4.f);
    return math::mul(proj, view);
}


float3 PathTracingDemo::getFirstDirectionalLightDir(int& dirLightIndex) const
{
    dirLightIndex = -1;

    if (mpScene == nullptr)
    {
        logWarning("No scene loaded. Cannot get first directional light direction.");
        return float3(0, 0, 0);
    }

    // If the scene has a directional light, return its direction.
    // Otherwise, compute a default light direction.
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

    // Default direction if no directional light is found
    return float3(-0.6f, -1.0f, 0.4f);
}

void PathTracingDemo::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    if (mpScene == nullptr)
    {
        logWarning("No scene loaded. Cannot render.");
        return;
    }

    // 1.  compute/update the light matrix once per frame
    int dirLightIndex = -1;
    mLightVP = computeDirLightVP(mpScene->getSceneBounds(), getFirstDirectionalLightDir(dirLightIndex));

    // 2.  ---------- depth pass ----------
    pRenderContext->clearFbo(mpShadowFbo.get(), float4(0), 1.f, 0, FboAttachmentType::Depth);
    mpShadowPass->getState()->setFbo(mpShadowFbo);

    auto shadowPassVar = mpShadowPass->getVars()->getRootVar();
    shadowPassVar["PerFrame"]["gLightVP"] = mLightVP; // uniform used in vsMain

    mpScene->rasterize(pRenderContext, mpShadowPass->getState().get(), mpShadowPass->getVars().get());

    const float4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    IScene::UpdateFlags updates = mpScene->update(pRenderContext, getGlobalClock().getTime());
    if (is_set(updates, IScene::UpdateFlags::GeometryChanged))
        FALCOR_THROW("This sample does not support scene geometry changes.");
    if (is_set(updates, IScene::UpdateFlags::RecompileNeeded))
        FALCOR_THROW("This sample does not support scene changes that require shader recompilation.");

    mpRasterPass->getVars()->setSampler("gShadowSampler", mpShadowSampler);

    const auto& pRasterPassDefaultBlockReflection = mpRasterPass->getProgram()->getReflector()->getDefaultParameterBlock();
    auto shadowMapBindLocation = pRasterPassDefaultBlockReflection->getResourceBinding("gShadowDepth");
    auto shadowMapSamplerBindLocation = pRasterPassDefaultBlockReflection->getResourceBinding("gShadowSampler");

    mpRasterPass->getVars()->setSrv(shadowMapBindLocation, mpShadowMap->getSRV());
    mpRasterPass->getVars()->setSampler(shadowMapSamplerBindLocation, mpShadowSampler);

    auto rasterPassVar = mpRasterPass->getVars()->getRootVar();

    rasterPassVar["CommonParameters"]["gLightVP"] = mLightVP;
    rasterPassVar["CommonParameters"]["gTexelSize"] = 1.f / float2(kShadowRes, kShadowRes);
    rasterPassVar["CommonParameters"]["gShadowBias"] = 0.001f;
    rasterPassVar["CommonParameters"]["globalLightIndex"] = dirLightIndex;

    // Render the scene using the raster pass.
    mpRasterPass->getState()->setFbo(pTargetFbo);
    mpScene->rasterize(pRenderContext, mpRasterPass->getState().get(), mpRasterPass->getVars().get());

    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});
}

void PathTracingDemo::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Falcor", {250, 200});
    renderGlobalUI(pGui);
    w.text("Hello from PathTracingDemo");
    if (w.button("Click Here"))
    {
        msgBox("Info", "Now why would you do that?");
    }
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
    // If the scene didn't handle the mouse event, we can return false to let the SampleApp handle it.
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

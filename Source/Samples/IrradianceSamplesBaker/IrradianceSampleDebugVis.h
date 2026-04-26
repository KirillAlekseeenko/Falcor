#pragma once
#include "Falcor.h"
#include "Core/Pass/RasterPass.h"

#include <algorithm>
#include <filesystem>
#include <vector>

using namespace Falcor;

class IrradianceSampleDebugVis
{
public:
    struct FileHeader
    {
        uint32_t magic = 0x31534249; // "IBS1"
        uint32_t version = 2;
        uint32_t sampleCount = 0;
        uint32_t surfaceSampleCount = 0;
        float surfaceSampleRatio = 0.f;
        float backfaceThreshold = 0.f;
        float surfaceOffset = 0.f;
        uint32_t reserved = 0u;
    };

    struct Sample
    {
        float4 position = float4(0.f);
        float4 normal = float4(0.f, 1.f, 0.f, 0.f);
        float4 irradiance = float4(0.f);
        uint4 meta = uint4(0u);
    };

    static constexpr uint32_t kSurfaceSampleFlag = 1u;

    explicit IrradianceSampleDebugVis(ref<Device> pDevice);

    void setScene(const ref<Scene>& pScene, const ref<Camera>& pCamera);
    void onHotReload();
    void renderUI(Gui::Widgets& widgets);
    void render(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void loadSamplesFromFile(const std::filesystem::path& path);

    uint32_t getLoadedSampleCount() const { return static_cast<uint32_t>(mLoadedSamples.size()); }
    uint32_t getLoadedSurfaceSampleCount() const { return std::min(mLoadedSamplesHeader.surfaceSampleCount, getLoadedSampleCount()); }
    uint32_t getVisibleSampleCount() const { return mVisibleSampleCount; }
    const std::filesystem::path& getLoadedSamplesPath() const { return mLoadedSamplesPath; }

private:
    struct DebugSphereVertex
    {
        float3 position = float3(0.f);
        float3 normal = float3(0.f, 1.f, 0.f);
    };

    void createDebugSpherePass();
    void createDebugSphereGeometry();
    void updateVisibleSamples();
    void updateIrradianceStats();

    ref<Device> mpDevice;
    ref<Scene> mpScene;
    ref<Camera> mpCamera;
    ref<RasterPass> mpDebugSpherePass;
    ref<Vao> mpDebugSphereVao;
    ref<Buffer> mpDebugSampleBuffer;

    std::vector<Sample> mLoadedSamples;
    std::vector<Sample> mVisibleSamples;

    std::filesystem::path mLoadedSamplesPath;
    FileHeader mLoadedSamplesHeader;

    float mSceneRadius = 1.f;
    float mDebugSphereRadius = 0.1f;
    float mDebugCullDistance = 0.f;
    float mMaxIrradianceComponent = 0.f;
    uint32_t mDebugSphereIndexCount = 0u;
    uint32_t mVisibleSampleCount = 0u;
    bool mShowDebugSamples = true;
};

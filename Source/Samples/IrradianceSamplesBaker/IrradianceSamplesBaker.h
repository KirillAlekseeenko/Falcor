#pragma once
#include "IrradianceSampleDebugVis.h"
#include "Falcor.h"
#include "Core/SampleApp.h"
#include "Core/Pass/RasterPass.h"

#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace Falcor;

class IrradianceSamplesBaker : public SampleApp
{
public:
    IrradianceSamplesBaker(const SampleAppConfig& config);
    ~IrradianceSamplesBaker() override;

    void onLoad(RenderContext* pRenderContext) override;
    void onShutdown() override;
    void onResize(uint32_t width, uint32_t height) override;
    void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo) override;
    void onGuiRender(Gui* pGui) override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;
    void onDroppedFile(const std::filesystem::path& path) override;
    void onHotReload(HotReloadFlags reloaded) override;

private:
    struct BakeCandidateData
    {
        float4 position = float4(0.f);
        float4 normal = float4(0.f, 1.f, 0.f, 0.f);
        uint4 meta = uint4(0u);
    };

    struct BakeResultData
    {
        float4 position = float4(0.f);
        float4 normal = float4(0.f, 1.f, 0.f, 0.f);
        uint4 meta = uint4(0u);
    };

    struct MeshSamplingData
    {
        std::vector<float3> positions;
        std::vector<uint3> triangleIndices;
        std::vector<float3> triangleNormals;
        std::vector<double> triangleAreaCdf;
        double totalArea = 0.0;
    };

    struct SurfaceInstanceData
    {
        uint32_t instanceID = 0;
        uint32_t meshID = 0;
        float4x4 worldMatrix = float4x4::identity();
        std::vector<double> triangleAreaCdf;
        double totalArea = 0.0;
    };

    struct BakeSummary
    {
        uint32_t requestedSampleCount = 0;
        uint32_t surfaceTargetCount = 0;
        uint32_t volumeTargetCount = 0;
        uint32_t acceptedSurfaceCount = 0;
        uint32_t acceptedVolumeCount = 0;
        uint32_t testedVolumeCandidateCount = 0;
        uint32_t finalSampleCount = 0;
        std::filesystem::path outputPath;
    };

    using StoredSample = IrradianceSampleDebugVis::Sample;
    using BakeFileHeader = IrradianceSampleDebugVis::FileHeader;

    void loadScene(const std::string& scenePath, RenderContext* pRenderContext);
    void createBakeProgram();
    void createSceneRasterPass();
    void invalidateSamplingCache();
    void buildSamplingCache();
    void bake(RenderContext* pRenderContext);
    void renderScenePreview(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);

    std::vector<BakeCandidateData> generateSurfaceCandidates(uint32_t count);
    std::vector<BakeCandidateData> generateVolumeCandidates(uint32_t count);
    std::vector<BakeResultData> resolveCandidates(RenderContext* pRenderContext, const std::vector<BakeCandidateData>& candidates);

    std::filesystem::path getDefaultOutputPath() const;
    float getSurfaceOffset() const;
    bool saveSamples(const std::vector<StoredSample>& samples, uint32_t surfaceSampleCount, const std::filesystem::path& outputPath) const;

    static uint32_t sampleCdfIndex(const std::vector<double>& cdf, double u);
    static float3 sampleTriangleBarycentrics(float u0, float u1);
    static float3 sampleUnitVector(std::mt19937_64& rng);

private:
    static constexpr uint32_t kSurfaceCandidateFlag = IrradianceSampleDebugVis::kSurfaceSampleFlag;
    static constexpr uint32_t kProbeRayCount = 32u;
    static constexpr uint32_t kMaxCandidateAttempts = 12u;

    ref<Scene> mpScene;
    ref<Camera> mpCamera;
    ref<Program> mpBakeProgram;
    ref<RtProgramVars> mpBakeVars;
    ref<RasterPass> mpSceneRasterPass;
    std::unique_ptr<IrradianceSampleDebugVis> mpDebugVis;

    std::vector<MeshSamplingData> mMeshSamplingData;
    std::vector<SurfaceInstanceData> mSurfaceInstances;
    std::vector<double> mSurfaceInstanceCdf;

    std::mt19937_64 mRng;

    std::string mScenePath;
    std::string mStatusMessage = "Ready.";
    BakeSummary mLastBakeSummary;

    uint32_t mRequestedSampleCount = 16384u;
    float mSurfaceSampleRatio = 0.2f;
    float mBackfaceThreshold = 0.5f;

    bool mBakeRequested = false;
    bool mSamplingCacheValid = false;
};

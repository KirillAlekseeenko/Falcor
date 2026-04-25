#include "IrradianceSamplesBaker.h"

#include "Utils/UI/TextRenderer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>

FALCOR_EXPORT_D3D12_AGILITY_SDK

namespace
{
const char kDefaultScenePath[] = "Bistro_v5_2/BistroExterior.pyscene";
const uint32_t kGuiWidth = 440;
const float kSurfaceOffsetScale = 1e-5f;
const float kPi = 3.14159265358979323846f;
}

IrradianceSamplesBaker::IrradianceSamplesBaker(const SampleAppConfig& config)
    : SampleApp(config)
    , mRng(std::random_device{}())
{}

IrradianceSamplesBaker::~IrradianceSamplesBaker() = default;

void IrradianceSamplesBaker::onLoad(RenderContext* pRenderContext)
{
    if (!getDevice()->isFeatureSupported(Device::SupportedFeatures::Raytracing))
        FALCOR_THROW("Device does not support raytracing.");

    loadScene(kDefaultScenePath, pRenderContext);
}

void IrradianceSamplesBaker::onShutdown() {}

void IrradianceSamplesBaker::onResize(uint32_t width, uint32_t height)
{
    SampleApp::onResize(width, height);
    if (mpCamera)
        mpCamera->setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
}

void IrradianceSamplesBaker::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    pRenderContext->clearFbo(pTargetFbo.get(), float4(0.08f, 0.09f, 0.11f, 1.f), 1.f, 0, FboAttachmentType::All);

    if (mpScene)
    {
        const auto updates = mpScene->update(pRenderContext, getGlobalClock().getTime());
        if (is_set(updates, IScene::UpdateFlags::GeometryChanged))
            FALCOR_THROW("IrradianceSamplesBaker does not support animated geometry during baking.");
        if (is_set(updates, IScene::UpdateFlags::RecompileNeeded))
            FALCOR_THROW("Scene changes require shader recompilation. Reload the scene before baking.");
    }

    if (mBakeRequested && mpScene)
    {
        mBakeRequested = false;
        try
        {
            bake(pRenderContext);
        }
        catch (const std::exception& e)
        {
            mStatusMessage = std::string("Bake failed: ") + e.what();
            logError("{}", mStatusMessage);
        }
    }

    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});
}

void IrradianceSamplesBaker::onGuiRender(Gui* pGui)
{
    Gui::Window window(pGui, "Irradiance Samples Baker", {kGuiWidth, 320});
    renderGlobalUI(pGui);

    window.text("Drop a .pyscene file into the window to switch scenes.");

    int sampleCount = static_cast<int>(mRequestedSampleCount);
    if (window.var("Sample Count", sampleCount, 1, 1 << 24))
        mRequestedSampleCount = static_cast<uint32_t>(std::max(sampleCount, 1));

    window.slider("Surface Ratio", mSurfaceSampleRatio, 0.f, 1.f);
    window.slider("Backface Threshold", mBackfaceThreshold, 0.f, 1.f);

    window.text("Scene:");
    window.text(mScenePath.empty() ? "<none>" : mScenePath);

    const auto outputPath = getDefaultOutputPath();
    window.text("Output:");
    window.text(outputPath.string());

    if (window.button("Bake"))
        mBakeRequested = true;

    window.separator();
    window.text(mStatusMessage);

    if (mLastBakeSummary.finalSampleCount > 0)
    {
        window.text(
            fmt::format(
                "Last bake: {} samples (surface {}, volume {})",
                mLastBakeSummary.finalSampleCount,
                mLastBakeSummary.acceptedSurfaceCount,
                mLastBakeSummary.acceptedVolumeCount
            )
        );
        window.text(fmt::format("Volume candidates tested: {}", mLastBakeSummary.testedVolumeCandidateCount));
    }
}

bool IrradianceSamplesBaker::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (keyEvent.key == Input::Key::Escape && keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        shutdown();
        return true;
    }

    if (keyEvent.key == Input::Key::B && keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        mBakeRequested = true;
        return true;
    }

    if (mpScene && mpScene->onKeyEvent(keyEvent))
        return true;

    return false;
}

bool IrradianceSamplesBaker::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpScene && mpScene->onMouseEvent(mouseEvent);
}

void IrradianceSamplesBaker::onDroppedFile(const std::filesystem::path& path)
{
    try
    {
        loadScene(path.string(), getRenderContext());
    }
    catch (const std::exception& e)
    {
        mStatusMessage = std::string("Failed to load dropped scene: ") + e.what();
        logError("{}", mStatusMessage);
    }
}

void IrradianceSamplesBaker::onHotReload(HotReloadFlags reloaded)
{
    if (is_set(reloaded, HotReloadFlags::Program))
        createBakeProgram();
}

void IrradianceSamplesBaker::loadScene(const std::string& scenePath, RenderContext* pRenderContext)
{
    mScenePath = scenePath;
    mpScene = Scene::create(getDevice(), scenePath);
    mpScene->setIsAnimated(false);
    mpCamera = mpScene->getCamera();

    if (mpCamera)
    {
        const float radius = mpScene->getSceneBounds().radius();
        mpScene->setCameraSpeed(radius * 0.25f);
        mpCamera->setDepthRange(0.1f, radius * 2.f);
        mpCamera->setAspectRatio(static_cast<float>(getTargetFbo()->getWidth()) / static_cast<float>(getTargetFbo()->getHeight()));
    }

    // TODO: What is a usecase for this condition?
    const auto updates = mpScene->update(pRenderContext, 0.0);
    if (is_set(updates, IScene::UpdateFlags::RecompileNeeded))
        FALCOR_THROW("Scene requires shader recompilation after loading.");

    // TODO: maybe just ignore unsupported geometry???
    if (mpScene->getGeometryTypes() != Scene::GeometryTypeFlags::TriangleMesh)
        FALCOR_THROW("This prototype baker currently supports scenes containing triangle meshes only.");

    createBakeProgram();
    invalidateSamplingCache();

    mStatusMessage = fmt::format("Loaded scene '{}'.", scenePath);
    mLastBakeSummary = {};
}

void IrradianceSamplesBaker::createBakeProgram()
{
    if (!mpScene)
        return;

    const uint32_t geometryCount = mpScene->getGeometryCount();
    const auto meshIDs = mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh);

    auto sbt = RtBindingTable::create(1, 1, geometryCount);

    ProgramDesc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary("Samples/IrradianceSamplesBaker/IrradianceSamplesBaker.rt.slang");
    desc.addTypeConformances(mpScene->getTypeConformances());
    desc.setMaxTraceRecursionDepth(1);
    desc.setMaxPayloadSize(sizeof(uint32_t) * 2);

    sbt->setRayGen(desc.addRayGen("rayGen"));
    sbt->setMiss(0, desc.addMiss("probeMiss"));

    auto hitGroup = desc.addHitGroup("probeClosestHit", "probeAnyHit");
    sbt->setHitGroup(0, meshIDs, hitGroup);

    mpBakeProgram = Program::create(getDevice(), desc, mpScene->getSceneDefines());
    mpBakeVars = RtProgramVars::create(getDevice(), mpBakeProgram, sbt);
}

void IrradianceSamplesBaker::invalidateSamplingCache()
{
    mMeshSamplingData.clear();
    mSurfaceInstances.clear();
    mSurfaceInstanceCdf.clear();
    mSamplingCacheValid = false;
}

void IrradianceSamplesBaker::buildSamplingCache()
{
    if (!mpScene || mSamplingCacheValid)
        return;

    mMeshSamplingData.clear();
    mMeshSamplingData.resize(mpScene->getMeshCount());

    for (uint32_t meshIndex = 0; meshIndex < mpScene->getMeshCount(); ++meshIndex)
    {
        const MeshID meshID{meshIndex};
        const auto& meshDesc = mpScene->getMesh(meshID);

        MeshSamplingData& data = mMeshSamplingData[meshIndex];
        if (meshDesc.getTriangleCount() == 0 || meshDesc.vertexCount == 0)
            continue;

        std::map<std::string, ref<Buffer>> buffers;
        buffers["positions"] = getDevice()->createStructuredBuffer(sizeof(float3), meshDesc.vertexCount);
        buffers["texcrds"] = getDevice()->createStructuredBuffer(sizeof(float3), meshDesc.vertexCount);
        buffers["triangleIndices"] = getDevice()->createStructuredBuffer(sizeof(uint3), meshDesc.getTriangleCount());

        mpScene->getMeshVerticesAndIndices(meshID, buffers);

        data.positions = buffers["positions"]->getElements<float3>();
        data.triangleIndices = buffers["triangleIndices"]->getElements<uint3>();
        data.triangleNormals.reserve(data.triangleIndices.size());
        data.triangleAreaCdf.reserve(data.triangleIndices.size());

        double areaSum = 0.0;
        for (const auto& tri : data.triangleIndices)
        {
            const float3& p0 = data.positions[tri.x];
            const float3& p1 = data.positions[tri.y];
            const float3& p2 = data.positions[tri.z];

            float3 faceNormal = cross(p1 - p0, p2 - p0);
            const float normalLength = length(faceNormal);
            const float area = 0.5f * normalLength;

            if (area > 0.f)
            {
                faceNormal /= normalLength;
                if (meshDesc.isFrontFaceCW())
                    faceNormal = -faceNormal;
            }
            else
            {
                faceNormal = float3(0.f, 1.f, 0.f);
            }

            areaSum += static_cast<double>(area);
            data.triangleNormals.push_back(faceNormal);
            data.triangleAreaCdf.push_back(areaSum);
        }

        data.totalArea = areaSum;
    }

    const auto instanceIDs = mpScene->getGeometryInstanceIDsByType(Scene::GeometryType::TriangleMesh);
    double instanceAreaSum = 0.0;
    for (uint32_t instanceID : instanceIDs)
    {
        const auto& instance = mpScene->getGeometryInstance(instanceID);
        const uint32_t meshIndex = instance.geometryID;
        if (meshIndex >= mMeshSamplingData.size())
            continue;

        const double totalArea = mMeshSamplingData[meshIndex].totalArea;
        if (totalArea <= 0.0)
            continue;

        mSurfaceInstances.push_back({instanceID, meshIndex, totalArea});
        instanceAreaSum += totalArea;
        mSurfaceInstanceCdf.push_back(instanceAreaSum);
    }

    if (mSurfaceInstances.empty())
        FALCOR_THROW("No triangle mesh instances with positive area were found.");

    mSamplingCacheValid = true;
}

std::vector<IrradianceSamplesBaker::BakeCandidateData> IrradianceSamplesBaker::generateSurfaceCandidates(uint32_t count)
{
    std::vector<BakeCandidateData> candidates;
    candidates.reserve(count);

    std::uniform_real_distribution<double> unit01(0.0, 1.0);

    for (uint32_t i = 0; i < count; ++i)
    {
        const uint32_t instanceIndex = sampleCdfIndex(mSurfaceInstanceCdf, unit01(mRng));
        const auto& instance = mSurfaceInstances[instanceIndex];
        const auto& meshData = mMeshSamplingData[instance.meshID];

        const uint32_t triangleIndex = sampleCdfIndex(meshData.triangleAreaCdf, unit01(mRng));
        const uint3 tri = meshData.triangleIndices[triangleIndex];

        const float3 bary = sampleTriangleBarycentrics(static_cast<float>(unit01(mRng)), static_cast<float>(unit01(mRng)));
        const float3 localPosition =
            meshData.positions[tri.x] * bary.x +
            meshData.positions[tri.y] * bary.y +
            meshData.positions[tri.z] * bary.z;

        BakeCandidateData candidate;
        candidate.position = float4(localPosition, 1.f);
        candidate.normal = float4(meshData.triangleNormals[triangleIndex], 0.f);
        candidate.meta = uint4(instance.instanceID, kSurfaceCandidateFlag, 0u, 0u);
        candidates.push_back(candidate);
    }

    return candidates;
}

std::vector<IrradianceSamplesBaker::BakeCandidateData> IrradianceSamplesBaker::generateVolumeCandidates(uint32_t count)
{
    std::vector<BakeCandidateData> candidates;
    candidates.reserve(count);

    const AABB bounds = mpScene->getSceneBounds();
    std::uniform_real_distribution<float> distX(bounds.minPoint.x, bounds.maxPoint.x);
    std::uniform_real_distribution<float> distY(bounds.minPoint.y, bounds.maxPoint.y);
    std::uniform_real_distribution<float> distZ(bounds.minPoint.z, bounds.maxPoint.z);

    for (uint32_t i = 0; i < count; ++i)
    {
        BakeCandidateData candidate;
        candidate.position = float4(distX(mRng), distY(mRng), distZ(mRng), 1.f);
        candidate.normal = float4(sampleUnitVector(mRng), 0.f);
        candidate.meta = uint4(0u, 0u, 0u, 0u);
        candidates.push_back(candidate);
    }

    return candidates;
}

std::vector<IrradianceSamplesBaker::BakeResultData> IrradianceSamplesBaker::resolveCandidates(
    RenderContext* pRenderContext,
    const std::vector<BakeCandidateData>& candidates
)
{
    std::vector<BakeResultData> results;
    if (candidates.empty())
        return results;

    FALCOR_ASSERT(mpBakeVars);
    auto rootVar = mpBakeVars->getRootVar();

    auto pCandidateBuffer = getDevice()->createStructuredBuffer(
        rootVar["gCandidates"],
        static_cast<uint32_t>(candidates.size()),
        ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal,
        candidates.data()
    );

    auto pResultBuffer = getDevice()->createStructuredBuffer(
        rootVar["gResults"],
        static_cast<uint32_t>(candidates.size()),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr
    );

    rootVar["gCandidates"] = pCandidateBuffer;
    rootVar["gResults"] = pResultBuffer;
    rootVar["BakeParams"]["gSampleCount"] = static_cast<uint32_t>(candidates.size());
    rootVar["BakeParams"]["gProbeRayCount"] = kProbeRayCount;
    rootVar["BakeParams"]["gSurfaceOffset"] = getSurfaceOffset();

    mpScene->raytrace(pRenderContext, mpBakeProgram.get(), mpBakeVars, uint3(static_cast<uint32_t>(candidates.size()), 1u, 1u));

    results = pResultBuffer->getElements<BakeResultData>();
    return results;
}

void IrradianceSamplesBaker::bake(RenderContext* pRenderContext)
{
    buildSamplingCache();

    const uint32_t clampedRequestedCount = std::max(1u, mRequestedSampleCount);
    const uint32_t surfaceTarget = std::min(
        clampedRequestedCount,
        static_cast<uint32_t>(std::lround(static_cast<double>(clampedRequestedCount) * std::clamp(mSurfaceSampleRatio, 0.f, 1.f)))
    );
    const uint32_t volumeTarget = clampedRequestedCount - surfaceTarget;

    mStatusMessage = "Baking samples...";
    logInfo("Starting bake for {} samples (surface ratio {}, backface threshold {}).", clampedRequestedCount, mSurfaceSampleRatio, mBackfaceThreshold);

    std::vector<StoredSample> finalSamples;
    finalSamples.reserve(clampedRequestedCount);

    if (surfaceTarget > 0)
    {
        const auto surfaceCandidates = generateSurfaceCandidates(surfaceTarget);
        const auto surfaceResults = resolveCandidates(pRenderContext, surfaceCandidates);

        for (const auto& result : surfaceResults)
        {
            StoredSample sample;
            sample.position = result.position;
            sample.normal = result.normal;
            sample.meta = uint4(kSurfaceCandidateFlag, 0u, 0u, 0u);
            finalSamples.push_back(sample);
        }
    }

    uint32_t acceptedVolumeCount = 0;
    uint32_t testedVolumeCandidateCount = 0;

    for (uint32_t attempt = 0; attempt < kMaxVolumeAttempts && acceptedVolumeCount < volumeTarget; ++attempt)
    {
        const uint32_t remaining = volumeTarget - acceptedVolumeCount;
        const uint32_t batchSize = std::max(remaining * 2u, 1024u);

        auto volumeCandidates = generateVolumeCandidates(batchSize);
        auto volumeResults = resolveCandidates(pRenderContext, volumeCandidates);
        testedVolumeCandidateCount += batchSize;

        for (const auto& result : volumeResults)
        {
            const uint32_t hitCount = result.meta.x;
            const uint32_t backfaceCount = result.meta.y;
            const float backfaceRatio = hitCount > 0 ? static_cast<float>(backfaceCount) / static_cast<float>(hitCount) : 0.f;

            if (backfaceRatio > mBackfaceThreshold)
                continue;

            StoredSample sample;
            sample.position = result.position;
            sample.normal = result.normal;
            sample.meta = uint4(0u, hitCount, backfaceCount, 0u);
            finalSamples.push_back(sample);
            ++acceptedVolumeCount;

            if (acceptedVolumeCount >= volumeTarget)
                break;
        }
    }

    if (acceptedVolumeCount < volumeTarget)
    {
        logWarning(
            "Only accepted {} out of {} requested volume samples after {} candidate rays.",
            acceptedVolumeCount,
            volumeTarget,
            testedVolumeCandidateCount
        );
    }

    const auto outputPath = getDefaultOutputPath();
    if (!saveSamples(finalSamples, surfaceTarget, outputPath))
        FALCOR_THROW("Failed to save baked sample file '{}'.", outputPath);

    mLastBakeSummary.requestedSampleCount = clampedRequestedCount;
    mLastBakeSummary.surfaceTargetCount = surfaceTarget;
    mLastBakeSummary.volumeTargetCount = volumeTarget;
    mLastBakeSummary.acceptedSurfaceCount = surfaceTarget;
    mLastBakeSummary.acceptedVolumeCount = acceptedVolumeCount;
    mLastBakeSummary.testedVolumeCandidateCount = testedVolumeCandidateCount;
    mLastBakeSummary.finalSampleCount = static_cast<uint32_t>(finalSamples.size());
    mLastBakeSummary.outputPath = outputPath;

    mStatusMessage = fmt::format(
        "Bake complete: {} samples written to {}",
        mLastBakeSummary.finalSampleCount,
        outputPath.string()
    );
    logInfo("{}", mStatusMessage);
}

std::filesystem::path IrradianceSamplesBaker::getDefaultOutputPath() const
{
    std::filesystem::path outputDir = getRuntimeDirectory() / "baked_samples";
    std::filesystem::create_directories(outputDir);

    std::filesystem::path scenePath = mScenePath.empty() ? std::filesystem::path("scene") : std::filesystem::path(mScenePath);
    const std::string stem = scenePath.stem().empty() ? "scene" : scenePath.stem().string();

    return outputDir / (stem + ".irradiance-samples.bin");
}

float IrradianceSamplesBaker::getSurfaceOffset() const
{
    if (!mpScene)
        return 1e-4f;
    // TODO: seems a bit overkill
    return std::max(1e-4f, mpScene->getSceneBounds().radius() * kSurfaceOffsetScale);
}

bool IrradianceSamplesBaker::saveSamples(
    const std::vector<StoredSample>& samples,
    uint32_t surfaceSampleCount,
    const std::filesystem::path& outputPath
) const
{
    std::filesystem::create_directories(outputPath.parent_path());

    std::ofstream stream(outputPath, std::ios::binary);
    if (!stream)
        return false;

    BakeFileHeader header;
    header.sampleCount = static_cast<uint32_t>(samples.size());
    header.surfaceSampleCount = surfaceSampleCount;
    header.surfaceSampleRatio = mSurfaceSampleRatio;
    header.backfaceThreshold = mBackfaceThreshold;
    header.surfaceOffset = getSurfaceOffset();

    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!samples.empty())
        stream.write(reinterpret_cast<const char*>(samples.data()), static_cast<std::streamsize>(samples.size() * sizeof(StoredSample)));

    return stream.good();
}

uint32_t IrradianceSamplesBaker::sampleCdfIndex(const std::vector<double>& cdf, double u)
{
    FALCOR_CHECK(!cdf.empty(), "Cannot sample from an empty CDF.");

    const double total = cdf.back();
    const double target = std::min(u * total, std::nextafter(total, 0.0));
    const auto it = std::lower_bound(cdf.begin(), cdf.end(), target);
    return static_cast<uint32_t>(std::distance(cdf.begin(), it));
}

float3 IrradianceSamplesBaker::sampleTriangleBarycentrics(float u0, float u1)
{
    const float su0 = std::sqrt(std::clamp(u0, 0.f, 1.f));
    const float b0 = 1.f - su0;
    const float b1 = u1 * su0;
    const float b2 = 1.f - b0 - b1;
    return float3(b0, b1, b2);
}

float3 IrradianceSamplesBaker::sampleUnitVector(std::mt19937_64& rng)
{
    std::uniform_real_distribution<float> unit01(0.f, 1.f);
    const float z = 1.f - 2.f * unit01(rng);
    const float phi = 2.f * kPi * unit01(rng);
    const float r = std::sqrt(std::max(0.f, 1.f - z * z));
    return float3(r * std::cos(phi), r * std::sin(phi), z);
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.title = "Irradiance Samples Baker";
    config.windowDesc.resizableWindow = true;

    IrradianceSamplesBaker project(config);
    return project.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}

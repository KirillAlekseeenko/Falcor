#include "IrradianceSamplesBaker.h"

#include "Utils/UI/TextRenderer.h"

#include <algorithm>
#include <cctype>
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

float3 computeUnnormalizedWorldNormal(const float4x4& worldMatrix, const std::vector<float3>& vertexPositions, const uint3& triangleIndices)
{
    const float4 p0H = math::mul(worldMatrix, float4(vertexPositions[triangleIndices.x], 1.f));
    const float4 p1H = math::mul(worldMatrix, float4(vertexPositions[triangleIndices.y], 1.f));
    const float4 p2H = math::mul(worldMatrix, float4(vertexPositions[triangleIndices.z], 1.f));
    const float3 p0(p0H.x, p0H.y, p0H.z);
    const float3 p1(p1H.x, p1H.y, p1H.z);
    const float3 p2(p2H.x, p2H.y, p2H.z);
    return cross(p1 - p0, p2 - p0);
}
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

    mpDebugVis = std::make_unique<IrradianceSampleDebugVis>(getDevice());
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

    renderScenePreview(pRenderContext, pTargetFbo);
    if (mpDebugVis)
        mpDebugVis->render(pRenderContext, pTargetFbo);

    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});
}

void IrradianceSamplesBaker::onGuiRender(Gui* pGui)
{
    Gui::Window window(pGui, "Irradiance Samples Baker", {kGuiWidth, 460});
    renderGlobalUI(pGui);

    window.text("Drop a .pyscene file to switch scenes or a baked .bin file to inspect samples.");

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

    if (window.button("Load Current Bake"))
    {
        try
        {
            if (mpDebugVis)
            {
                mpDebugVis->loadSamplesFromFile(outputPath);
                mStatusMessage = fmt::format("Loaded {} debug samples from {}", mpDebugVis->getLoadedSampleCount(), outputPath.string());
            }
        }
        catch (const std::exception& e)
        {
            mStatusMessage = std::string("Failed to load baked samples: ") + e.what();
            logError("{}", mStatusMessage);
        }
    }

    if (mpDebugVis)
    {
        window.separator();
        mpDebugVis->renderUI(window);
    }

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
        std::string extension = path.extension().string();
        std::transform(
            extension.begin(),
            extension.end(),
            extension.begin(),
            [](unsigned char c)
            {
                return static_cast<char>(std::tolower(c));
            }
        );

        if (extension == ".pyscene")
        {
            loadScene(path.string(), getRenderContext());
        }
        else if (extension == ".bin")
        {
            if (mpDebugVis)
            {
                mpDebugVis->loadSamplesFromFile(path);
                mStatusMessage = fmt::format("Loaded {} debug samples from {}", mpDebugVis->getLoadedSampleCount(), path.string());
            }
        }
        else
        {
            mStatusMessage = fmt::format("Ignored dropped file '{}'.", path.string());
        }
    }
    catch (const std::exception& e)
    {
        mStatusMessage = std::string("Failed to load dropped file: ") + e.what();
        logError("{}", mStatusMessage);
    }
}

void IrradianceSamplesBaker::onHotReload(HotReloadFlags reloaded)
{
    if (is_set(reloaded, HotReloadFlags::Program))
    {
        createBakeProgram();
        createSceneRasterPass();
        if (mpDebugVis)
            mpDebugVis->onHotReload();
    }
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
    createSceneRasterPass();
    if (mpDebugVis)
        mpDebugVis->setScene(mpScene, mpCamera);
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

void IrradianceSamplesBaker::createSceneRasterPass()
{
    if (!mpScene)
        return;

    ProgramDesc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary("Samples/IrradianceSamplesBaker/IrradianceSamplesBaker.3d.slang").vsEntry("vsMain").psEntry("psMain");
    desc.addTypeConformances(mpScene->getTypeConformances());

    mpSceneRasterPass = RasterPass::create(getDevice(), desc, mpScene->getSceneDefines());
}

void IrradianceSamplesBaker::renderScenePreview(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    if (!mpScene || !mpSceneRasterPass)
        return;

    mpSceneRasterPass->getState()->setFbo(pTargetFbo);
    mpScene->rasterize(pRenderContext, mpSceneRasterPass->getState().get(), mpSceneRasterPass->getVars().get());
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

    const auto* pAnimationController = mpScene->getAnimationController();
    FALCOR_CHECK(pAnimationController != nullptr, "Scene is missing an animation controller.");
    const auto& globalMatrices = pAnimationController->getGlobalMatrices();

    const auto instanceIDs = mpScene->getGeometryInstanceIDsByType(Scene::GeometryType::TriangleMesh);
    double instanceAreaSum = 0.0;
    for (uint32_t instanceID : instanceIDs)
    {
        const auto& instance = mpScene->getGeometryInstance(instanceID);
        const uint32_t meshIndex = instance.geometryID;
        if (meshIndex >= mMeshSamplingData.size())
            continue;

        if (instance.globalMatrixID >= globalMatrices.size())
            continue;

        const auto& meshData = mMeshSamplingData[meshIndex];
        if (meshData.triangleIndices.empty())
            continue;

        SurfaceInstanceData instanceData;
        instanceData.instanceID = instanceID;
        instanceData.meshID = meshIndex;
        instanceData.worldMatrix = globalMatrices[instance.globalMatrixID];
        instanceData.triangleAreaCdf.reserve(meshData.triangleIndices.size());

        double totalArea = 0.0;
        for (const auto& tri : meshData.triangleIndices)
        {
            const double triangleArea = 0.5 * static_cast<double>(
                length(computeUnnormalizedWorldNormal(instanceData.worldMatrix, meshData.positions, tri))
            );
            totalArea += triangleArea;
            instanceData.triangleAreaCdf.push_back(totalArea);
        }

        if (totalArea <= 0.0)
            continue;

        instanceData.totalArea = totalArea;
        mSurfaceInstances.push_back(std::move(instanceData));
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

        const uint32_t triangleIndex = sampleCdfIndex(instance.triangleAreaCdf, unit01(mRng));
        const uint3 triangleIndices = meshData.triangleIndices[triangleIndex];

        const float3 bary = sampleTriangleBarycentrics(static_cast<float>(unit01(mRng)), static_cast<float>(unit01(mRng)));
        const float3 localPosition =
            meshData.positions[triangleIndices.x] * bary.x +
            meshData.positions[triangleIndices.y] * bary.y +
            meshData.positions[triangleIndices.z] * bary.z;

        float3 worldNormal = computeUnnormalizedWorldNormal(instance.worldMatrix, meshData.positions, triangleIndices);
        if (const float normalLength = length(worldNormal); normalLength > 0.f)
            worldNormal /= normalLength;
        else
            worldNormal = float3(0.f, 1.f, 0.f);

        const float4 samplePositionH = math::mul(instance.worldMatrix, float4(localPosition, 1.f));

        BakeCandidateData candidate;
        candidate.position = float4(samplePositionH.x, samplePositionH.y, samplePositionH.z, 1.f);
        candidate.normal = float4(worldNormal, 0.f);
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

    auto passesBackfaceThreshold = [this](uint32_t hitCount, uint32_t backfaceCount)
    {
        const float backfaceRatio = hitCount > 0 ? static_cast<float>(backfaceCount) / static_cast<float>(hitCount) : 0.f;
        return backfaceRatio <= mBackfaceThreshold;
    };

    struct AcceptanceStats
    {
        uint32_t acceptedCount = 0;
        uint32_t testedCandidateCount = 0;
    };

    std::vector<StoredSample> finalSamples;
    finalSamples.reserve(clampedRequestedCount);

    auto collectAcceptedSamples =
        [&](uint32_t targetCount, uint32_t sampleFlag, const char* sampleLabel, auto&& generateCandidates)
    {
        AcceptanceStats stats;

        for (uint32_t attempt = 0; attempt < kMaxCandidateAttempts && stats.acceptedCount < targetCount; ++attempt)
        {
            const uint32_t remaining = targetCount - stats.acceptedCount;
            const uint32_t batchSize = std::max(remaining * 2u, 1024u);

            auto candidates = generateCandidates(batchSize);
            auto results = resolveCandidates(pRenderContext, candidates);
            stats.testedCandidateCount += batchSize;

            for (const auto& result : results)
            {
                const uint32_t hitCount = result.meta.x;
                const uint32_t backfaceCount = result.meta.y;
                if (!passesBackfaceThreshold(hitCount, backfaceCount))
                    continue;

                StoredSample sample;
                sample.position = result.position;
                sample.normal = result.normal;
                sample.meta = uint4(sampleFlag, hitCount, backfaceCount, 0u);
                finalSamples.push_back(sample);
                ++stats.acceptedCount;

                if (stats.acceptedCount >= targetCount)
                    break;
            }
        }

        if (stats.acceptedCount < targetCount)
        {
            logWarning(
                "Only accepted {} out of {} requested {} samples after {} candidate rays.",
                stats.acceptedCount,
                targetCount,
                sampleLabel,
                stats.testedCandidateCount
            );
        }

        return stats;
    };

    const auto surfaceStats = collectAcceptedSamples(
        surfaceTarget,
        kSurfaceCandidateFlag,
        "surface",
        [&](uint32_t batchSize)
        {
            return generateSurfaceCandidates(batchSize);
        }
    );

    const auto volumeStats = collectAcceptedSamples(
        volumeTarget,
        0u,
        "volume",
        [&](uint32_t batchSize)
        {
            return generateVolumeCandidates(batchSize);
        }
    );

    const auto outputPath = getDefaultOutputPath();
    if (!saveSamples(finalSamples, surfaceStats.acceptedCount, outputPath))
        FALCOR_THROW("Failed to save baked sample file '{}'.", outputPath);
    if (mpDebugVis)
        mpDebugVis->loadSamplesFromFile(outputPath);

    mLastBakeSummary.requestedSampleCount = clampedRequestedCount;
    mLastBakeSummary.surfaceTargetCount = surfaceTarget;
    mLastBakeSummary.volumeTargetCount = volumeTarget;
    mLastBakeSummary.acceptedSurfaceCount = surfaceStats.acceptedCount;
    mLastBakeSummary.acceptedVolumeCount = volumeStats.acceptedCount;
    mLastBakeSummary.testedVolumeCandidateCount = volumeStats.testedCandidateCount;
    mLastBakeSummary.finalSampleCount = static_cast<uint32_t>(finalSamples.size());
    mLastBakeSummary.outputPath = outputPath;

    mStatusMessage = fmt::format(
        "Bake complete: {} samples written to {} and loaded for debug.",
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

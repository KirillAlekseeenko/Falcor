#include "IrradianceSampleDebugVis.h"

#include "Scene/TriangleMesh.h"

#include <algorithm>
#include <fstream>

namespace
{
const float3 kDebugSurfaceColor = float3(1.0f, 0.56f, 0.18f);
const float3 kDebugVolumeColor = float3(0.22f, 0.86f, 1.0f);

struct LegacySampleV1
{
    float4 position = float4(0.f);
    float4 normal = float4(0.f, 1.f, 0.f, 0.f);
    uint4 meta = uint4(0u);
};
}

IrradianceSampleDebugVis::IrradianceSampleDebugVis(ref<Device> pDevice)
    : mpDevice(std::move(pDevice))
{
    createDebugSpherePass();
    createDebugSphereGeometry();
}

void IrradianceSampleDebugVis::setScene(const ref<Scene>& pScene, const ref<Camera>& pCamera)
{
    mpScene = pScene;
    mpCamera = pCamera;
    mSceneRadius = mpScene ? std::max(mpScene->getSceneBounds().radius(), 1.f) : 1.f;
    mDebugSphereRadius = mSceneRadius * 0.0025f;
    mDebugCullDistance = mSceneRadius * 0.35f;
}

void IrradianceSampleDebugVis::onHotReload()
{
    createDebugSpherePass();
}

void IrradianceSampleDebugVis::renderUI(Gui::Widgets& widgets)
{
    widgets.checkbox("Show Debug Samples", mShowDebugSamples);
    widgets.var("Sphere Radius", mDebugSphereRadius, 1e-4f, mSceneRadius * 0.1f);
    widgets.var("Cull Distance (0 = off)", mDebugCullDistance, 0.f, mSceneRadius * 20.f);

    if (!mLoadedSamplesPath.empty())
    {
        widgets.text("Loaded Samples:");
        widgets.text(mLoadedSamplesPath.string());
    }

    if (!mLoadedSamples.empty())
    {
        const uint32_t surfaceSampleCount = getLoadedSurfaceSampleCount();
        widgets.text(fmt::format("Debug samples: {} total, {} visible", getLoadedSampleCount(), mVisibleSampleCount));
        widgets.text(fmt::format("Surface {}, volume {}", surfaceSampleCount, getLoadedSampleCount() - surfaceSampleCount));
        widgets.text(fmt::format("Max irradiance component {:.6f}", mMaxIrradianceComponent));
    }
}

void IrradianceSampleDebugVis::render(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    updateVisibleSamples();
    if (!mpDebugSpherePass || !mpDebugSphereVao || !mpCamera || !mpDebugSampleBuffer || mVisibleSampleCount == 0u)
        return;

    auto rootVar = mpDebugSpherePass->getRootVar();
    rootVar["gSamples"] = mpDebugSampleBuffer;
    rootVar["DebugSphereCB"]["gViewProj"] = mpCamera->getViewProjMatrix();
    rootVar["DebugSphereCB"]["gSphereRadius"] = std::max(1e-4f, mDebugSphereRadius);
    rootVar["DebugSphereCB"]["gIrradianceScale"] = mMaxIrradianceComponent > 0.f ? 1.f / mMaxIrradianceComponent : 0.f;
    rootVar["DebugSphereCB"]["gSurfaceColor"] = kDebugSurfaceColor;
    rootVar["DebugSphereCB"]["gVolumeColor"] = kDebugVolumeColor;

    mpDebugSpherePass->getState()->setFbo(pTargetFbo);
    mpDebugSpherePass->getState()->setVao(mpDebugSphereVao);

    pRenderContext->drawIndexedInstanced(
        mpDebugSpherePass->getState().get(),
        mpDebugSpherePass->getVars().get(),
        mDebugSphereIndexCount,
        mVisibleSampleCount,
        0,
        0,
        0
    );
}

void IrradianceSampleDebugVis::loadSamplesFromFile(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        FALCOR_THROW("Failed to open baked sample file '{}'.", path);

    const uint64_t fileSize = static_cast<uint64_t>(stream.tellg());
    stream.seekg(0, std::ios::beg);

    FileHeader header;
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!stream)
        FALCOR_THROW("Failed to read header from '{}'.", path);

    const FileHeader expectedHeader;
    if (header.magic != expectedHeader.magic)
        FALCOR_THROW("File '{}' has an invalid irradiance sample magic number.", path);

    std::vector<Sample> loadedSamples;
    loadedSamples.resize(header.sampleCount);

    if (header.version == 1u)
    {
        const uint64_t requiredSize = sizeof(FileHeader) + static_cast<uint64_t>(header.sampleCount) * sizeof(LegacySampleV1);
        if (fileSize < requiredSize)
            FALCOR_THROW("File '{}' is truncated. Expected at least {} bytes, found {}.", path, requiredSize, fileSize);

        std::vector<LegacySampleV1> legacySamples(header.sampleCount);
        if (header.sampleCount > 0u)
        {
            stream.read(reinterpret_cast<char*>(legacySamples.data()), static_cast<std::streamsize>(header.sampleCount * sizeof(LegacySampleV1)));
            if (!stream)
                FALCOR_THROW("Failed to read {} baked samples from '{}'.", header.sampleCount, path);
        }

        for (uint32_t i = 0; i < header.sampleCount; ++i)
        {
            loadedSamples[i].position = legacySamples[i].position;
            loadedSamples[i].normal = legacySamples[i].normal;
            loadedSamples[i].irradiance = float4(0.f);
            loadedSamples[i].meta = legacySamples[i].meta;
        }
    }
    else if (header.version == expectedHeader.version)
    {
        const uint64_t requiredSize = sizeof(FileHeader) + static_cast<uint64_t>(header.sampleCount) * sizeof(Sample);
        if (fileSize < requiredSize)
            FALCOR_THROW("File '{}' is truncated. Expected at least {} bytes, found {}.", path, requiredSize, fileSize);

        if (header.sampleCount > 0u)
        {
            stream.read(reinterpret_cast<char*>(loadedSamples.data()), static_cast<std::streamsize>(header.sampleCount * sizeof(Sample)));
            if (!stream)
                FALCOR_THROW("Failed to read {} baked samples from '{}'.", header.sampleCount, path);
        }
    }
    else
    {
        FALCOR_THROW("File '{}' has unsupported irradiance sample version {}.", path, header.version);
    }

    mLoadedSamples = std::move(loadedSamples);
    mLoadedSamplesHeader = header;
    mLoadedSamplesPath = path;
    updateIrradianceStats();
}

void IrradianceSampleDebugVis::createDebugSpherePass()
{
    mpDebugSpherePass = RasterPass::create(mpDevice, "Samples/IrradianceSamplesBaker/DebugSpheres.3d.slang", "vsMain", "psMain");

    RasterizerState::Desc rsDesc;
    rsDesc.setCullMode(RasterizerState::CullMode::Back);
    mpDebugSpherePass->getState()->setRasterizerState(RasterizerState::create(rsDesc));

    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthEnabled(true).setDepthWriteMask(true).setDepthFunc(ComparisonFunc::LessEqual);
    mpDebugSpherePass->getState()->setDepthStencilState(DepthStencilState::create(dsDesc));
}

void IrradianceSampleDebugVis::createDebugSphereGeometry()
{
    const auto pSphere = TriangleMesh::createSphere(1.f, 12u, 8u);
    const auto& vertices = pSphere->getVertices();
    const auto& indices = pSphere->getIndices();

    std::vector<DebugSphereVertex> sphereVertices;
    sphereVertices.reserve(vertices.size());
    for (const auto& vertex : vertices)
        sphereVertices.push_back({vertex.position, vertex.normal});

    auto pVertexBuffer = mpDevice->createBuffer(
        sphereVertices.size() * sizeof(DebugSphereVertex),
        ResourceBindFlags::Vertex,
        MemoryType::DeviceLocal,
        sphereVertices.data()
    );

    auto pIndexBuffer = mpDevice->createBuffer(
        indices.size() * sizeof(uint32_t),
        ResourceBindFlags::Index,
        MemoryType::DeviceLocal,
        indices.data()
    );

    auto pBufferLayout = VertexBufferLayout::create();
    pBufferLayout->addElement("POSITION", offsetof(DebugSphereVertex, position), ResourceFormat::RGB32Float, 1, 0);
    pBufferLayout->addElement("NORMAL", offsetof(DebugSphereVertex, normal), ResourceFormat::RGB32Float, 1, 1);

    auto pLayout = VertexLayout::create();
    pLayout->addBufferLayout(0, pBufferLayout);

    mpDebugSphereVao = Vao::create(Vao::Topology::TriangleList, pLayout, {pVertexBuffer}, pIndexBuffer, ResourceFormat::R32Uint);
    mDebugSphereIndexCount = static_cast<uint32_t>(indices.size());
}

void IrradianceSampleDebugVis::updateVisibleSamples()
{
    mVisibleSamples.clear();
    mVisibleSampleCount = 0u;

    if (!mShowDebugSamples || mLoadedSamples.empty() || !mpCamera)
        return;

    const bool useCullDistance = mDebugCullDistance > 0.f;
    const float cullDistanceSqr = mDebugCullDistance * mDebugCullDistance;
    const float3 cameraPosition = mpCamera->getPosition();

    mVisibleSamples.reserve(mLoadedSamples.size());
    for (const auto& sample : mLoadedSamples)
    {
        const float3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
        const float3 delta = samplePosition - cameraPosition;
        if (useCullDistance && dot(delta, delta) > cullDistanceSqr)
            continue;

        mVisibleSamples.push_back(sample);
    }

    mVisibleSampleCount = static_cast<uint32_t>(mVisibleSamples.size());

    const uint32_t bufferElementCount = std::max(1u, getLoadedSampleCount());
    if (!mpDebugSampleBuffer || mpDebugSampleBuffer->getElementCount() < bufferElementCount)
    {
        mpDebugSampleBuffer = mpDevice->createStructuredBuffer(
            sizeof(Sample),
            bufferElementCount,
            ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            nullptr
        );
    }

    if (mVisibleSampleCount > 0u)
    {
        mpDebugSampleBuffer->setBlob(
            mVisibleSamples.data(),
            0,
            static_cast<size_t>(mVisibleSampleCount) * sizeof(Sample)
        );
    }
}

void IrradianceSampleDebugVis::updateIrradianceStats()
{
    mMaxIrradianceComponent = 0.f;
    for (const auto& sample : mLoadedSamples)
    {
        mMaxIrradianceComponent = std::max(
            mMaxIrradianceComponent,
            std::max(sample.irradiance.x, std::max(sample.irradiance.y, sample.irradiance.z))
        );
    }
}

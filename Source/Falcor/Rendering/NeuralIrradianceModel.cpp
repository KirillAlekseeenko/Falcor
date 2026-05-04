#include "NeuralIrradianceModel.h"

#include <array>
#include <fstream>
#include <limits>
#include <numeric>

namespace Falcor
{
namespace
{
constexpr std::array<char, 4> kModelMagic = {'N', 'L', 'G', '1'};

#pragma pack(push, 1)
struct ModelFileHeader
{
    std::array<char, 4> magic = {};
    uint32_t version = 0;
    uint32_t inputDim = 0;
    uint32_t positionFrequencyCount = 0;
    uint32_t directionBinCount = 0;
    uint32_t hiddenWidth = 0;
    uint32_t layerCount = 0;
    uint32_t outputDim = 0;
    uint32_t parameterFloatCount = 0;
    uint32_t sourceSamplePathByteCount = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;
};

struct ModelFileLayerHeader
{
    uint32_t inputDim = 0;
    uint32_t outputDim = 0;
    uint32_t weightCount = 0;
    uint32_t biasCount = 0;
};
#pragma pack(pop)

static_assert(sizeof(ModelFileHeader) == 52);
static_assert(sizeof(ModelFileLayerHeader) == 16);

uint64_t checkedAdd(uint64_t a, uint64_t b)
{
    FALCOR_CHECK(a <= std::numeric_limits<uint64_t>::max() - b, "Neural irradiance model size overflows uint64_t.");
    return a + b;
}

uint64_t checkedMul(uint64_t a, uint64_t b)
{
    if (a == 0 || b == 0)
        return 0;
    FALCOR_CHECK(a <= std::numeric_limits<uint64_t>::max() / b, "Neural irradiance model size overflows uint64_t.");
    return a * b;
}

void readExact(std::ifstream& stream, void* pData, size_t byteCount, const std::filesystem::path& path, std::string_view label)
{
    stream.read(reinterpret_cast<char*>(pData), static_cast<std::streamsize>(byteCount));
    FALCOR_CHECK(stream.good(), "Failed to read {} from neural irradiance model '{}'.", label, path);
}

template<typename T>
void readVector(std::ifstream& stream, std::vector<T>& values, uint32_t count, const std::filesystem::path& path, std::string_view label)
{
    values.resize(count);
    if (count > 0)
        readExact(stream, values.data(), sizeof(T) * values.size(), path, label);
}

std::string readString(std::ifstream& stream, uint32_t byteCount, const std::filesystem::path& path)
{
    std::string value(byteCount, '\0');
    if (byteCount > 0)
        readExact(stream, value.data(), value.size(), path, "source sample path");
    return value;
}

void validateHeader(const ModelFileHeader& header, const std::filesystem::path& path)
{
    FALCOR_CHECK(header.magic == kModelMagic, "Neural irradiance model '{}' has an invalid magic number.", path);
    FALCOR_CHECK(
        header.version == NeuralIrradianceModel::kModelVersion,
        "Neural irradiance model '{}' has unsupported version {}; expected {}.",
        path,
        header.version,
        NeuralIrradianceModel::kModelVersion
    );
    FALCOR_CHECK(header.outputDim == NeuralIrradianceModel::kOutputDim, "Neural irradiance model '{}' has output dimension {}; expected 3.", path, header.outputDim);
    FALCOR_CHECK(header.layerCount > 0, "Neural irradiance model '{}' has no layers.", path);
    FALCOR_CHECK(header.positionFrequencyCount > 0, "Neural irradiance model '{}' has no positional frequencies.", path);
    FALCOR_CHECK(header.directionBinCount > 0, "Neural irradiance model '{}' has no direction bins.", path);

    const uint32_t expectedInputDim = 3u * header.positionFrequencyCount * 2u + 2u * header.directionBinCount;
    FALCOR_CHECK(
        header.inputDim == expectedInputDim,
        "Neural irradiance model '{}' input dimension is {}; expected {} from encoding config.",
        path,
        header.inputDim,
        expectedInputDim
    );
    FALCOR_CHECK(
        header.inputDim <= NeuralIrradianceModel::kMaxActivationDim,
        "Neural irradiance model '{}' input dimension is {}; Slang inference currently supports at most {}.",
        path,
        header.inputDim,
        NeuralIrradianceModel::kMaxActivationDim
    );
    FALCOR_CHECK(
        header.hiddenWidth <= NeuralIrradianceModel::kMaxActivationDim,
        "Neural irradiance model '{}' hidden width is {}; Slang inference currently supports at most {}.",
        path,
        header.hiddenWidth,
        NeuralIrradianceModel::kMaxActivationDim
    );
}

uint64_t computeMinimumFileSize(const ModelFileHeader& header)
{
    return checkedAdd(checkedAdd(sizeof(ModelFileHeader), sizeof(float) * 6ull), header.sourceSamplePathByteCount);
}
}

NeuralIrradianceModel::NeuralIrradianceModel(ref<Device> pDevice)
    : mpDevice(std::move(pDevice))
{
    FALCOR_CHECK(mpDevice != nullptr, "NeuralIrradianceModel requires a valid device.");
}

std::unique_ptr<NeuralIrradianceModel> NeuralIrradianceModel::create(ref<Device> pDevice, const std::filesystem::path& path)
{
    auto pModel = std::make_unique<NeuralIrradianceModel>(std::move(pDevice));
    pModel->loadFromFile(path);
    return pModel;
}

void NeuralIrradianceModel::loadFromFile(const std::filesystem::path& path)
{
    clear();

    FALCOR_CHECK(std::filesystem::exists(path), "Neural irradiance model '{}' does not exist.", path);
    const uint64_t fileSize = std::filesystem::file_size(path);
    FALCOR_CHECK(fileSize >= sizeof(ModelFileHeader), "Neural irradiance model '{}' is too small to contain a header.", path);

    std::ifstream stream(path, std::ios::binary);
    FALCOR_CHECK(stream.good(), "Failed to open neural irradiance model '{}'.", path);

    ModelFileHeader header;
    readExact(stream, &header, sizeof(header), path, "model header");
    validateHeader(header, path);

    uint64_t expectedSize = computeMinimumFileSize(header);
    FALCOR_CHECK(fileSize >= expectedSize, "Neural irradiance model '{}' is truncated: expected at least {} bytes, found {}.", path, expectedSize, fileSize);

    std::array<float, 3> positionMin = {};
    std::array<float, 3> positionExtent = {};
    readExact(stream, positionMin.data(), sizeof(float) * positionMin.size(), path, "position_min");
    readExact(stream, positionExtent.data(), sizeof(float) * positionExtent.size(), path, "position_extent");
    mSourceSamplePath = readString(stream, header.sourceSamplePathByteCount, path);

    mPath = path;
    mConstants = {};
    mConstants.positionMin = float4(positionMin[0], positionMin[1], positionMin[2], 0.f);
    mConstants.positionExtent = float4(positionExtent[0], positionExtent[1], positionExtent[2], 0.f);
    mConstants.inputDim = header.inputDim;
    mConstants.outputDim = header.outputDim;
    mConstants.positionFrequencyCount = header.positionFrequencyCount;
    mConstants.directionBinCount = header.directionBinCount;
    mConstants.hiddenWidth = header.hiddenWidth;
    mConstants.layerCount = header.layerCount;

    mLayers.reserve(header.layerCount);
    uint32_t previousOutputDim = 0;
    uint32_t parameterFloatCount = 0;

    for (uint32_t layerIndex = 0; layerIndex < header.layerCount; ++layerIndex)
    {
        expectedSize = checkedAdd(expectedSize, sizeof(ModelFileLayerHeader));
        FALCOR_CHECK(fileSize >= expectedSize, "Neural irradiance model '{}' is truncated before layer {} header.", path, layerIndex);

        ModelFileLayerHeader layerHeader;
        readExact(stream, &layerHeader, sizeof(layerHeader), path, "layer header");

        const uint32_t expectedLayerInputDim = layerIndex == 0 ? header.inputDim : previousOutputDim;
        FALCOR_CHECK(
            layerHeader.inputDim == expectedLayerInputDim,
            "Neural irradiance model '{}' layer {} input dimension is {}; expected {}.",
            path,
            layerIndex,
            layerHeader.inputDim,
            expectedLayerInputDim
        );
        FALCOR_CHECK(layerHeader.outputDim > 0, "Neural irradiance model '{}' layer {} has zero output dimension.", path, layerIndex);
        FALCOR_CHECK(
            layerHeader.outputDim <= NeuralIrradianceModel::kMaxActivationDim,
            "Neural irradiance model '{}' layer {} output dimension is {}; Slang inference currently supports at most {}.",
            path,
            layerIndex,
            layerHeader.outputDim,
            NeuralIrradianceModel::kMaxActivationDim
        );

        const uint32_t expectedWeightCount = layerHeader.inputDim * layerHeader.outputDim;
        FALCOR_CHECK(
            layerHeader.weightCount == expectedWeightCount,
            "Neural irradiance model '{}' layer {} has {} weights; expected {}.",
            path,
            layerIndex,
            layerHeader.weightCount,
            expectedWeightCount
        );
        FALCOR_CHECK(
            layerHeader.biasCount == layerHeader.outputDim,
            "Neural irradiance model '{}' layer {} has {} bias values; expected {}.",
            path,
            layerIndex,
            layerHeader.biasCount,
            layerHeader.outputDim
        );

        LayerDesc layer;
        layer.inputDim = layerHeader.inputDim;
        layer.outputDim = layerHeader.outputDim;
        layer.weightOffset = static_cast<uint32_t>(mWeights.size());
        layer.biasOffset = static_cast<uint32_t>(mBiases.size());
        mLayers.push_back(layer);

        std::vector<float> layerWeights;
        std::vector<float> layerBiases;
        readVector(stream, layerWeights, layerHeader.weightCount, path, "layer weights");
        readVector(stream, layerBiases, layerHeader.biasCount, path, "layer biases");

        mWeights.insert(mWeights.end(), layerWeights.begin(), layerWeights.end());
        mBiases.insert(mBiases.end(), layerBiases.begin(), layerBiases.end());

        parameterFloatCount += layerHeader.weightCount + layerHeader.biasCount;
        expectedSize = checkedAdd(expectedSize, checkedMul(layerHeader.weightCount + layerHeader.biasCount, sizeof(float)));
        FALCOR_CHECK(fileSize >= expectedSize, "Neural irradiance model '{}' is truncated before layer {} payload.", path, layerIndex);

        previousOutputDim = layerHeader.outputDim;
    }

    FALCOR_CHECK(previousOutputDim == header.outputDim, "Neural irradiance model '{}' final layer output dimension does not match the header.", path);
    FALCOR_CHECK(
        parameterFloatCount == header.parameterFloatCount,
        "Neural irradiance model '{}' stores {} parameter floats; header expected {}.",
        path,
        parameterFloatCount,
        header.parameterFloatCount
    );
    FALCOR_CHECK(fileSize == expectedSize, "Neural irradiance model '{}' has trailing data: expected {} bytes, found {}.", path, expectedSize, fileSize);

    mConstants.weightCount = static_cast<uint32_t>(mWeights.size());
    mConstants.biasCount = static_cast<uint32_t>(mBiases.size());

    uploadBuffers();
    mLoaded = true;

    logInfo(
        "Loaded neural irradiance model '{}' ({} layers, input {}, output {}, {} weights, {} biases).",
        path,
        mConstants.layerCount,
        mConstants.inputDim,
        mConstants.outputDim,
        mConstants.weightCount,
        mConstants.biasCount
    );
}

void NeuralIrradianceModel::bindShaderData(const ShaderVar& rootVar, std::string_view prefix) const
{
    FALCOR_CHECK(mLoaded, "Cannot bind neural irradiance model before loading it.");
    FALCOR_CHECK(rootVar.isValid(), "Cannot bind neural irradiance model to an invalid shader root variable.");

    const std::string prefixString(prefix);
    rootVar[prefixString + "Constants"].setBlob(mConstants);
    rootVar[prefixString + "Layers"].setBuffer(mpLayerBuffer);
    rootVar[prefixString + "Weights"].setBuffer(mpWeightBuffer);
    rootVar[prefixString + "Biases"].setBuffer(mpBiasBuffer);
}

void NeuralIrradianceModel::uploadBuffers()
{
    FALCOR_CHECK(!mLayers.empty(), "Cannot upload neural irradiance model without layers.");
    FALCOR_CHECK(!mWeights.empty(), "Cannot upload neural irradiance model without weights.");
    FALCOR_CHECK(!mBiases.empty(), "Cannot upload neural irradiance model without biases.");

    mpLayerBuffer = mpDevice->createStructuredBuffer(
        sizeof(LayerDesc),
        static_cast<uint32_t>(mLayers.size()),
        ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal,
        mLayers.data(),
        false
    );
    mpWeightBuffer = mpDevice->createStructuredBuffer(
        sizeof(float),
        static_cast<uint32_t>(mWeights.size()),
        ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal,
        mWeights.data(),
        false
    );
    mpBiasBuffer = mpDevice->createStructuredBuffer(
        sizeof(float),
        static_cast<uint32_t>(mBiases.size()),
        ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal,
        mBiases.data(),
        false
    );
}

void NeuralIrradianceModel::clear()
{
    mpLayerBuffer = nullptr;
    mpWeightBuffer = nullptr;
    mpBiasBuffer = nullptr;
    mPath.clear();
    mSourceSamplePath.clear();
    mConstants = {};
    mLayers.clear();
    mWeights.clear();
    mBiases.clear();
    mLoaded = false;
}
} // namespace Falcor

#include "Testing/UnitTest.h"
#include "Rendering/NeuralIrradianceModel.h"
#include "Core/Platform/OS.h"
#include "Utils/Math/Common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace Falcor
{
namespace
{
constexpr std::array<char, 4> kModelMagic = {'N', 'L', 'G', '1'};
constexpr std::array<char, 4> kParityMagic = {'N', 'L', 'P', '1'};
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.f * kPi;
constexpr uint32_t kParityVersion = 1;
constexpr uint32_t kParityFlagHasTargetIrradiance = 1;

#pragma pack(push, 1)
struct TestModelFileHeader
{
    std::array<char, 4> magic = kModelMagic;
    uint32_t version = NeuralIrradianceModel::kModelVersion;
    uint32_t inputDim = 44;
    uint32_t positionFrequencyCount = 6;
    uint32_t directionBinCount = 4;
    uint32_t hiddenWidth = 4;
    uint32_t layerCount = 2;
    uint32_t outputDim = NeuralIrradianceModel::kOutputDim;
    uint32_t parameterFloatCount = 0;
    uint32_t sourceSamplePathByteCount = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;
};

struct TestModelFileLayerHeader
{
    uint32_t inputDim = 0;
    uint32_t outputDim = 0;
    uint32_t weightCount = 0;
    uint32_t biasCount = 0;
};

struct TestParityFileHeader
{
    std::array<char, 4> magic = {};
    uint32_t version = 0;
    uint32_t vectorCount = 0;
    uint32_t inputDim = 0;
    uint32_t outputDim = 0;
    uint32_t positionFrequencyCount = 0;
    uint32_t directionBinCount = 0;
    uint32_t flags = 0;
    uint32_t reserved = 0;
};
#pragma pack(pop)

static_assert(sizeof(TestParityFileHeader) == 36);

struct TestLayer
{
    TestModelFileLayerHeader header;
    std::vector<float> weights;
    std::vector<float> biases;
};

struct TestModelData
{
    TestModelFileHeader header;
    std::array<float, 3> positionMin = {-1.f, -2.f, -3.f};
    std::array<float, 3> positionExtent = {10.f, 20.f, 30.f};
    std::string sourceSamplePath = "unit_test_samples.irradiance-samples.bin";
    std::vector<TestLayer> layers;
};

struct TestParityData
{
    TestParityFileHeader header;
    std::vector<float3> positions;
    std::vector<float3> directions;
    std::vector<float> encodedInputs;
    std::vector<float3> expectedOutputs;
    std::vector<float3> targetIrradiance;
};

struct TestInferenceResult
{
    std::vector<float> encodedOutputs;
    std::vector<float3> irradianceOutputs;
};

std::filesystem::path makeTempPath(std::string_view name)
{
    return std::filesystem::temp_directory_path() / std::filesystem::path(std::string(name));
}

std::filesystem::path getEnvPath(const char* envVarName)
{
    const char* value = std::getenv(envVarName);
    if (value == nullptr || value[0] == '\0')
        return {};

    return std::filesystem::path(value);
}

std::filesystem::path getDefaultCheckpointPath(std::string_view filename)
{
    return getProjectDirectory() / "NeuralLightGridLearning" / "checkpoints" / std::filesystem::path(std::string(filename));
}

uint64_t checkedAdd(uint64_t a, uint64_t b)
{
    FALCOR_CHECK(a <= std::numeric_limits<uint64_t>::max() - b, "Test file size overflows uint64_t.");
    return a + b;
}

uint64_t checkedMul(uint64_t a, uint64_t b)
{
    if (a == 0 || b == 0)
        return 0;

    FALCOR_CHECK(a <= std::numeric_limits<uint64_t>::max() / b, "Test file size overflows uint64_t.");
    return a * b;
}

void removeFile(const std::filesystem::path& path)
{
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

std::vector<float> makeSequence(uint32_t count, float base)
{
    std::vector<float> values(count);
    for (uint32_t i = 0; i < count; ++i)
        values[i] = base + float(i) * 0.01f;
    return values;
}

TestModelData makeValidModelData()
{
    TestModelData data;

    TestLayer layer0;
    layer0.header.inputDim = data.header.inputDim;
    layer0.header.outputDim = data.header.hiddenWidth;
    layer0.header.weightCount = layer0.header.inputDim * layer0.header.outputDim;
    layer0.header.biasCount = layer0.header.outputDim;
    layer0.weights = makeSequence(layer0.header.weightCount, 0.1f);
    layer0.biases = makeSequence(layer0.header.biasCount, 1.0f);

    TestLayer layer1;
    layer1.header.inputDim = layer0.header.outputDim;
    layer1.header.outputDim = data.header.outputDim;
    layer1.header.weightCount = layer1.header.inputDim * layer1.header.outputDim;
    layer1.header.biasCount = layer1.header.outputDim;
    layer1.weights = makeSequence(layer1.header.weightCount, 2.0f);
    layer1.biases = makeSequence(layer1.header.biasCount, 3.0f);

    data.layers = {std::move(layer0), std::move(layer1)};
    data.header.layerCount = static_cast<uint32_t>(data.layers.size());
    data.header.parameterFloatCount = 0;
    for (const TestLayer& layer : data.layers)
        data.header.parameterFloatCount += layer.header.weightCount + layer.header.biasCount;
    data.header.sourceSamplePathByteCount = static_cast<uint32_t>(data.sourceSamplePath.size());

    return data;
}

void writeBytes(std::ofstream& stream, const void* pData, size_t byteCount)
{
    stream.write(reinterpret_cast<const char*>(pData), static_cast<std::streamsize>(byteCount));
}

void readBytes(std::ifstream& stream, void* pData, size_t byteCount, const std::filesystem::path& path, std::string_view label)
{
    stream.read(reinterpret_cast<char*>(pData), static_cast<std::streamsize>(byteCount));
    FALCOR_CHECK(stream.good(), "Failed to read {} from test file '{}'.", label, path);
}

void writeModelFile(const std::filesystem::path& path, const TestModelData& data)
{
    std::ofstream stream(path, std::ios::binary);
    FALCOR_CHECK(stream.good(), "Failed to write test model file '{}'.", path);

    writeBytes(stream, &data.header, sizeof(data.header));
    writeBytes(stream, data.positionMin.data(), data.positionMin.size() * sizeof(float));
    writeBytes(stream, data.positionExtent.data(), data.positionExtent.size() * sizeof(float));
    writeBytes(stream, data.sourceSamplePath.data(), data.sourceSamplePath.size());

    for (const TestLayer& layer : data.layers)
    {
        writeBytes(stream, &layer.header, sizeof(layer.header));
        writeBytes(stream, layer.weights.data(), layer.weights.size() * sizeof(float));
        writeBytes(stream, layer.biases.data(), layer.biases.size() * sizeof(float));
    }
}

void appendTrailingBytes(const std::filesystem::path& path)
{
    std::ofstream stream(path, std::ios::binary | std::ios::app);
    const uint32_t extra = 0xDEADBEEFu;
    writeBytes(stream, &extra, sizeof(extra));
}

void truncateFile(const std::filesystem::path& path, uintmax_t bytesToKeep)
{
    std::ifstream input(path, std::ios::binary);
    std::vector<char> data(bytesToKeep);
    input.read(data.data(), static_cast<std::streamsize>(data.size()));
    input.close();

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
}

TestParityData readParityFile(const std::filesystem::path& path)
{
    FALCOR_CHECK(std::filesystem::exists(path), "Parity file '{}' does not exist.", path);
    const uint64_t fileSize = std::filesystem::file_size(path);
    FALCOR_CHECK(fileSize >= sizeof(TestParityFileHeader), "Parity file '{}' is too small to contain a header.", path);

    std::ifstream stream(path, std::ios::binary);
    FALCOR_CHECK(stream.good(), "Failed to open parity file '{}'.", path);

    TestParityData data;
    readBytes(stream, &data.header, sizeof(data.header), path, "parity header");

    FALCOR_CHECK(data.header.magic == kParityMagic, "Parity file '{}' has an invalid magic number.", path);
    FALCOR_CHECK(data.header.version == kParityVersion, "Parity file '{}' has version {}; expected {}.", path, data.header.version, kParityVersion);
    FALCOR_CHECK(data.header.vectorCount > 0, "Parity file '{}' contains no vectors.", path);
    FALCOR_CHECK(data.header.outputDim == NeuralIrradianceModel::kOutputDim, "Parity file '{}' has output dimension {}; expected 3.", path, data.header.outputDim);
    FALCOR_CHECK(
        data.header.inputDim <= NeuralIrradianceModel::kMaxActivationDim,
        "Parity file '{}' has input dimension {}; Slang inference currently supports at most {}.",
        path,
        data.header.inputDim,
        NeuralIrradianceModel::kMaxActivationDim
    );
    FALCOR_CHECK((data.header.flags & kParityFlagHasTargetIrradiance) != 0, "Parity file '{}' does not contain target irradiance data.", path);

    const uint64_t floatsPerVector = checkedAdd(checkedAdd(6ull, data.header.inputDim), checkedMul(data.header.outputDim, 2ull));
    const uint64_t expectedSize = checkedAdd(sizeof(TestParityFileHeader), checkedMul(checkedMul(data.header.vectorCount, floatsPerVector), sizeof(float)));
    FALCOR_CHECK(fileSize == expectedSize, "Parity file '{}' has invalid size: expected {} bytes, found {}.", path, expectedSize, fileSize);

    data.positions.resize(data.header.vectorCount);
    data.directions.resize(data.header.vectorCount);
    data.encodedInputs.resize(uint64_t(data.header.vectorCount) * data.header.inputDim);
    data.expectedOutputs.resize(data.header.vectorCount);
    data.targetIrradiance.resize(data.header.vectorCount);

    for (uint32_t vectorIndex = 0; vectorIndex < data.header.vectorCount; ++vectorIndex)
    {
        std::array<float, 3> position;
        std::array<float, 3> direction;
        std::array<float, 3> expectedOutput;
        std::array<float, 3> targetIrradiance;

        readBytes(stream, position.data(), sizeof(float) * position.size(), path, "parity position");
        readBytes(stream, direction.data(), sizeof(float) * direction.size(), path, "parity direction");
        readBytes(
            stream,
            data.encodedInputs.data() + uint64_t(vectorIndex) * data.header.inputDim,
            sizeof(float) * data.header.inputDim,
            path,
            "parity encoded input"
        );
        readBytes(stream, expectedOutput.data(), sizeof(float) * expectedOutput.size(), path, "parity expected output");
        readBytes(stream, targetIrradiance.data(), sizeof(float) * targetIrradiance.size(), path, "parity target irradiance");

        data.positions[vectorIndex] = float3(position[0], position[1], position[2]);
        data.directions[vectorIndex] = float3(direction[0], direction[1], direction[2]);
        data.expectedOutputs[vectorIndex] = float3(expectedOutput[0], expectedOutput[1], expectedOutput[2]);
        data.targetIrradiance[vectorIndex] = float3(targetIrradiance[0], targetIrradiance[1], targetIrradiance[2]);
    }

    return data;
}

float component(const float3& value, uint32_t index)
{
    switch (index)
    {
    case 0:
        return value.x;
    case 1:
        return value.y;
    default:
        return value.z;
    }
}

float oneBlobValue(float value01, uint32_t binIndex, uint32_t binCount, bool wrap)
{
    const float binCountF = float(binCount);
    const float center = (float(binIndex) + 0.5f) / binCountF;
    float distance = std::abs(value01 - center);
    if (wrap)
        distance = std::min(distance, 1.f - distance);

    return std::max(1.f - distance * binCountF, 0.f);
}

std::vector<float> encodeInputReference(const TestModelData& data, const float3& position, const float3& direction)
{
    std::vector<float> result(data.header.inputDim, 0.f);

    const float3 positionMin(data.positionMin[0], data.positionMin[1], data.positionMin[2]);
    const float3 positionExtent(
        std::max(data.positionExtent[0], 1e-6f),
        std::max(data.positionExtent[1], 1e-6f),
        std::max(data.positionExtent[2], 1e-6f)
    );
    const float3 position01 = (position - positionMin) / positionExtent;

    uint32_t writeIndex = 0;
    for (uint32_t componentIndex = 0; componentIndex < 3; ++componentIndex)
    {
        const float positionComponent = component(position01, componentIndex);
        for (uint32_t frequencyIndex = 0; frequencyIndex < data.header.positionFrequencyCount; ++frequencyIndex)
        {
            const float frequency = std::exp2(float(frequencyIndex)) * kTwoPi;
            result[writeIndex + frequencyIndex] = std::sin(positionComponent * frequency);
            result[writeIndex + data.header.positionFrequencyCount + frequencyIndex] = std::cos(positionComponent * frequency);
        }
        writeIndex += data.header.positionFrequencyCount * 2u;
    }

    const float3 dir = direction / std::max(length(direction), 1e-8f);
    const float theta = std::acos(std::clamp(dir.z, -1.f, 1.f)) / kPi;
    float phi = std::atan2(dir.y, dir.x) / kTwoPi + 1.f;
    phi -= std::floor(phi);

    for (uint32_t binIndex = 0; binIndex < data.header.directionBinCount; ++binIndex)
        result[writeIndex + binIndex] = oneBlobValue(theta, binIndex, data.header.directionBinCount, false);

    writeIndex += data.header.directionBinCount;
    for (uint32_t binIndex = 0; binIndex < data.header.directionBinCount; ++binIndex)
        result[writeIndex + binIndex] = oneBlobValue(phi, binIndex, data.header.directionBinCount, true);

    return result;
}

float3 evaluateModelReference(const TestModelData& data, const std::vector<float>& input)
{
    std::vector<float> activations = input;

    for (size_t layerIndex = 0; layerIndex < data.layers.size(); ++layerIndex)
    {
        const TestLayer& layer = data.layers[layerIndex];
        std::vector<float> next(layer.header.outputDim, 0.f);

        for (uint32_t outputIndex = 0; outputIndex < layer.header.outputDim; ++outputIndex)
        {
            float value = layer.biases[outputIndex];
            const uint32_t weightRowOffset = outputIndex * layer.header.inputDim;
            for (uint32_t inputIndex = 0; inputIndex < layer.header.inputDim; ++inputIndex)
                value += activations[inputIndex] * layer.weights[weightRowOffset + inputIndex];

            if (layerIndex + 1 < data.layers.size())
                value = std::max(value, 0.f);

            next[outputIndex] = value;
        }

        activations = std::move(next);
    }

    return float3(activations[0], activations[1], activations[2]);
}

void expectNear(
    GPUUnitTestContext& ctx,
    float actual,
    float expected,
    float tolerance,
    uint32_t sampleIndex,
    uint32_t valueIndex,
    std::string_view label
)
{
    EXPECT_LE(std::abs(actual - expected), tolerance)
        << label << " mismatch at sample " << sampleIndex << ", value " << valueIndex << ": expected " << expected << ", got " << actual;
}

TestInferenceResult runInference(GPUUnitTestContext& ctx, const NeuralIrradianceModel& model, const std::vector<float3>& positions, const std::vector<float3>& directions)
{
    FALCOR_CHECK(positions.size() == directions.size(), "Inference test positions/directions size mismatch.");

    const uint32_t sampleCount = static_cast<uint32_t>(positions.size());
    ctx.createProgram(
        "Tests/Samples/NeuralIrradianceModelInference.cs.slang",
        "testNeuralIrradianceInference",
        DefineList(),
        SlangCompilerFlags::FloatingPointModePrecise
    );
    ctx.allocateStructuredBuffer("positions", sampleCount, positions.data(), positions.size() * sizeof(float3));
    ctx.allocateStructuredBuffer("directions", sampleCount, directions.data(), directions.size() * sizeof(float3));
    ctx.allocateStructuredBuffer("encodedOutputs", sampleCount * NeuralIrradianceModel::kMaxActivationDim);
    ctx.allocateStructuredBuffer("irradianceOutputs", sampleCount);
    ctx["CB"]["sampleCount"] = sampleCount;
    model.bindShaderData(ctx.getVars()->getRootVar());
    ctx.runProgram(sampleCount);

    return {ctx.readBuffer<float>("encodedOutputs"), ctx.readBuffer<float3>("irradianceOutputs")};
}
}

GPU_TEST(NeuralIrradianceModel_LoadValidExport)
{
    const auto path = makeTempPath("falcor_neural_irradiance_model_valid.falcor-mlp.bin");
    removeFile(path);
    writeModelFile(path, makeValidModelData());

    NeuralIrradianceModel model(ctx.getDevice());
    model.loadFromFile(path);

    EXPECT(model.isLoaded());
    EXPECT_EQ(model.getPath(), path);
    EXPECT_EQ(model.getSourceSamplePath(), std::string("unit_test_samples.irradiance-samples.bin"));
    EXPECT_EQ(model.getLayers().size(), size_t(2));
    EXPECT_EQ(model.getWeightCount(), uint32_t(188));
    EXPECT_EQ(model.getBiasCount(), uint32_t(7));
    EXPECT(model.getLayerBuffer() != nullptr);
    EXPECT(model.getWeightBuffer() != nullptr);
    EXPECT(model.getBiasBuffer() != nullptr);
    EXPECT_EQ(model.getLayerBuffer()->getElementCount(), uint32_t(2));
    EXPECT_EQ(model.getWeightBuffer()->getElementCount(), uint32_t(188));
    EXPECT_EQ(model.getBiasBuffer()->getElementCount(), uint32_t(7));

    const auto& constants = model.getConstants();
    EXPECT_EQ(constants.inputDim, uint32_t(44));
    EXPECT_EQ(constants.outputDim, uint32_t(3));
    EXPECT_EQ(constants.positionFrequencyCount, uint32_t(6));
    EXPECT_EQ(constants.directionBinCount, uint32_t(4));
    EXPECT_EQ(constants.hiddenWidth, uint32_t(4));
    EXPECT_EQ(constants.layerCount, uint32_t(2));
    EXPECT_EQ(constants.weightCount, uint32_t(188));
    EXPECT_EQ(constants.biasCount, uint32_t(7));
    EXPECT_EQ(constants.positionMin, float4(-1.f, -2.f, -3.f, 0.f));
    EXPECT_EQ(constants.positionExtent, float4(10.f, 20.f, 30.f, 0.f));

    ASSERT_EQ(model.getLayers().size(), size_t(2));
    EXPECT_EQ(model.getLayers()[0].inputDim, uint32_t(44));
    EXPECT_EQ(model.getLayers()[0].outputDim, uint32_t(4));
    EXPECT_EQ(model.getLayers()[0].weightOffset, uint32_t(0));
    EXPECT_EQ(model.getLayers()[0].biasOffset, uint32_t(0));
    EXPECT_EQ(model.getLayers()[1].inputDim, uint32_t(4));
    EXPECT_EQ(model.getLayers()[1].outputDim, uint32_t(3));
    EXPECT_EQ(model.getLayers()[1].weightOffset, uint32_t(176));
    EXPECT_EQ(model.getLayers()[1].biasOffset, uint32_t(4));

    removeFile(path);
}

GPU_TEST(NeuralIrradianceModel_InferenceMatchesReference)
{
    const auto path = makeTempPath("falcor_neural_irradiance_model_inference.falcor-mlp.bin");
    removeFile(path);
    const TestModelData modelData = makeValidModelData();
    writeModelFile(path, modelData);

    NeuralIrradianceModel model(ctx.getDevice());
    model.loadFromFile(path);

    const std::vector<float3> positions = {
        float3(-1.f, -2.f, -3.f),
        float3(4.f, 8.f, 12.f),
        float3(1.5f, 3.25f, -0.75f),
        float3(9.f, 18.f, 27.f),
    };
    const std::vector<float3> directions = {
        float3(1.f, 0.f, 0.f),
        float3(0.f, 1.f, 0.f),
        normalize(float3(-0.25f, 0.75f, 0.6f)),
        normalize(float3(0.4f, -0.5f, -0.75f)),
    };
    const uint32_t sampleCount = static_cast<uint32_t>(positions.size());
    const TestInferenceResult result = runInference(ctx, model, positions, directions);

    for (uint32_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
    {
        const std::vector<float> encodedReference = encodeInputReference(modelData, positions[sampleIndex], directions[sampleIndex]);
        const float3 irradianceReference = evaluateModelReference(modelData, encodedReference);

        const uint32_t encodedBase = sampleIndex * NeuralIrradianceModel::kMaxActivationDim;
        for (uint32_t valueIndex = 0; valueIndex < modelData.header.inputDim; ++valueIndex)
            expectNear(ctx, result.encodedOutputs[encodedBase + valueIndex], encodedReference[valueIndex], 1e-4f, sampleIndex, valueIndex, "Encoded input");

        for (uint32_t valueIndex = modelData.header.inputDim; valueIndex < NeuralIrradianceModel::kMaxActivationDim; ++valueIndex)
            expectNear(ctx, result.encodedOutputs[encodedBase + valueIndex], 0.f, 1e-7f, sampleIndex, valueIndex, "Encoded padding");

        expectNear(ctx, result.irradianceOutputs[sampleIndex].x, irradianceReference.x, 1e-2f, sampleIndex, 0, "Irradiance");
        expectNear(ctx, result.irradianceOutputs[sampleIndex].y, irradianceReference.y, 1e-2f, sampleIndex, 1, "Irradiance");
        expectNear(ctx, result.irradianceOutputs[sampleIndex].z, irradianceReference.z, 1e-2f, sampleIndex, 2, "Irradiance");
    }

    removeFile(path);
}

GPU_TEST(NeuralIrradianceModel_TrainedCheckpointParity, DEVICE_TYPES(Device::Type::D3D12))
{
    std::filesystem::path modelPath = getEnvPath("FALCOR_NLG_MODEL_PATH");
    if (modelPath.empty())
        modelPath = getDefaultCheckpointPath("tiny_irradiance_mlp.falcor-mlp.bin");

    std::filesystem::path parityPath = getEnvPath("FALCOR_NLG_PARITY_PATH");
    if (parityPath.empty())
        parityPath = getDefaultCheckpointPath("tiny_irradiance_mlp.falcor-mlp.parity.bin");

    if (!std::filesystem::exists(modelPath) || !std::filesystem::exists(parityPath))
    {
        std::string message = "Optional trained neural irradiance parity artifacts were not found. Expected model '" + modelPath.string() + "' and parity '" +
                              parityPath.string() + "'. Override with FALCOR_NLG_MODEL_PATH and FALCOR_NLG_PARITY_PATH.";
        ctx.skip(message.c_str());
    }

    NeuralIrradianceModel model(ctx.getDevice());
    model.loadFromFile(modelPath);
    const TestParityData parity = readParityFile(parityPath);

    const auto& constants = model.getConstants();
    ASSERT_EQ(parity.header.inputDim, constants.inputDim);
    ASSERT_EQ(parity.header.outputDim, constants.outputDim);
    ASSERT_EQ(parity.header.positionFrequencyCount, constants.positionFrequencyCount);
    ASSERT_EQ(parity.header.directionBinCount, constants.directionBinCount);

    const TestInferenceResult result = runInference(ctx, model, parity.positions, parity.directions);

    float maxEncodedError = 0.f;
    uint32_t maxEncodedSample = 0;
    uint32_t maxEncodedValue = 0;
    for (uint32_t sampleIndex = 0; sampleIndex < parity.header.vectorCount; ++sampleIndex)
    {
        const uint32_t actualBase = sampleIndex * NeuralIrradianceModel::kMaxActivationDim;
        const uint32_t expectedBase = sampleIndex * parity.header.inputDim;
        for (uint32_t valueIndex = 0; valueIndex < parity.header.inputDim; ++valueIndex)
        {
            const float error = std::abs(result.encodedOutputs[actualBase + valueIndex] - parity.encodedInputs[expectedBase + valueIndex]);
            if (error > maxEncodedError)
            {
                maxEncodedError = error;
                maxEncodedSample = sampleIndex;
                maxEncodedValue = valueIndex;
            }
        }
    }

    float maxOutputError = 0.f;
    uint32_t maxOutputSample = 0;
    uint32_t maxOutputValue = 0;
    for (uint32_t sampleIndex = 0; sampleIndex < parity.header.vectorCount; ++sampleIndex)
    {
        for (uint32_t valueIndex = 0; valueIndex < constants.outputDim; ++valueIndex)
        {
            const float error = std::abs(component(result.irradianceOutputs[sampleIndex], valueIndex) - component(parity.expectedOutputs[sampleIndex], valueIndex));
            if (error > maxOutputError)
            {
                maxOutputError = error;
                maxOutputSample = sampleIndex;
                maxOutputValue = valueIndex;
            }
        }
    }

    EXPECT_LE(maxEncodedError, 5e-4f) << "Max encoded-input parity error at sample " << maxEncodedSample << ", value " << maxEncodedValue;
    EXPECT_LE(maxOutputError, 5e-3f) << "Max MLP-output parity error at sample " << maxOutputSample << ", value " << maxOutputValue;
}

GPU_TEST(NeuralIrradianceModel_RejectsBadVersion)
{
    const auto path = makeTempPath("falcor_neural_irradiance_model_bad_version.falcor-mlp.bin");
    removeFile(path);
    TestModelData data = makeValidModelData();
    data.header.version = 999u;
    writeModelFile(path, data);

    NeuralIrradianceModel model(ctx.getDevice());
    EXPECT_THROW(model.loadFromFile(path));

    removeFile(path);
}

GPU_TEST(NeuralIrradianceModel_RejectsTruncatedFile)
{
    const auto path = makeTempPath("falcor_neural_irradiance_model_truncated.falcor-mlp.bin");
    removeFile(path);
    writeModelFile(path, makeValidModelData());
    truncateFile(path, 64u);

    NeuralIrradianceModel model(ctx.getDevice());
    EXPECT_THROW(model.loadFromFile(path));

    removeFile(path);
}

GPU_TEST(NeuralIrradianceModel_RejectsTrailingBytes)
{
    const auto path = makeTempPath("falcor_neural_irradiance_model_trailing.falcor-mlp.bin");
    removeFile(path);
    writeModelFile(path, makeValidModelData());
    appendTrailingBytes(path);

    NeuralIrradianceModel model(ctx.getDevice());
    EXPECT_THROW(model.loadFromFile(path));

    removeFile(path);
}

GPU_TEST(NeuralIrradianceModel_RejectsLayerDimensionMismatch)
{
    const auto path = makeTempPath("falcor_neural_irradiance_model_bad_layer.falcor-mlp.bin");
    removeFile(path);
    TestModelData data = makeValidModelData();
    data.layers[1].header.inputDim = 5u;
    data.layers[1].header.weightCount = data.layers[1].header.inputDim * data.layers[1].header.outputDim;
    data.layers[1].weights = makeSequence(data.layers[1].header.weightCount, 2.0f);
    data.header.parameterFloatCount = 0;
    for (const TestLayer& layer : data.layers)
        data.header.parameterFloatCount += layer.header.weightCount + layer.header.biasCount;
    writeModelFile(path, data);

    NeuralIrradianceModel model(ctx.getDevice());
    EXPECT_THROW(model.loadFromFile(path));

    removeFile(path);
}

} // namespace Falcor

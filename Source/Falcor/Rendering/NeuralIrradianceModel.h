#pragma once

#include "Falcor.h"
#include "Core/Program/ShaderVar.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace Falcor
{
class FALCOR_API NeuralIrradianceModel
{
public:
    struct ShaderConstants
    {
        float4 positionMin = float4(0.f);
        float4 positionExtent = float4(1.f);
        uint32_t inputDim = 0;
        uint32_t outputDim = 0;
        uint32_t positionFrequencyCount = 0;
        uint32_t directionBinCount = 0;
        uint32_t hiddenWidth = 0;
        uint32_t layerCount = 0;
        uint32_t weightCount = 0;
        uint32_t biasCount = 0;
    };

    struct LayerDesc
    {
        uint32_t inputDim = 0;
        uint32_t outputDim = 0;
        uint32_t weightOffset = 0;
        uint32_t biasOffset = 0;
    };

    static constexpr uint32_t kModelVersion = 1;
    static constexpr uint32_t kOutputDim = 3;
    static constexpr uint32_t kMaxActivationDim = 64;

    explicit NeuralIrradianceModel(ref<Device> pDevice);

    static std::unique_ptr<NeuralIrradianceModel> create(ref<Device> pDevice, const std::filesystem::path& path);

    void loadFromFile(const std::filesystem::path& path);
    void bindShaderData(const ShaderVar& rootVar, std::string_view prefix = "gNeuralIrradiance") const;

    bool isLoaded() const { return mLoaded; }
    const std::filesystem::path& getPath() const { return mPath; }
    const std::string& getSourceSamplePath() const { return mSourceSamplePath; }
    const ShaderConstants& getConstants() const { return mConstants; }
    const std::vector<LayerDesc>& getLayers() const { return mLayers; }
    uint32_t getWeightCount() const { return static_cast<uint32_t>(mWeights.size()); }
    uint32_t getBiasCount() const { return static_cast<uint32_t>(mBiases.size()); }

    ref<Buffer> getLayerBuffer() const { return mpLayerBuffer; }
    ref<Buffer> getWeightBuffer() const { return mpWeightBuffer; }
    ref<Buffer> getBiasBuffer() const { return mpBiasBuffer; }

private:
    void uploadBuffers();
    void clear();

    ref<Device> mpDevice;
    ref<Buffer> mpLayerBuffer;
    ref<Buffer> mpWeightBuffer;
    ref<Buffer> mpBiasBuffer;

    std::filesystem::path mPath;
    std::string mSourceSamplePath;
    ShaderConstants mConstants;
    std::vector<LayerDesc> mLayers;
    std::vector<float> mWeights;
    std::vector<float> mBiases;
    bool mLoaded = false;
};

static_assert(sizeof(NeuralIrradianceModel::ShaderConstants) % 16 == 0);
static_assert(sizeof(NeuralIrradianceModel::LayerDesc) == sizeof(uint32_t) * 4);
} // namespace Falcor

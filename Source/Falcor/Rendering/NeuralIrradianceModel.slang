/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

#include "Utils/Math/MathConstants.slangh"

static const uint kNeuralIrradianceMaxActivationDim = 64;

struct NeuralIrradianceModelConstants
{
    float4 positionMin;
    float4 positionExtent;
    uint inputDim;
    uint outputDim;
    uint positionFrequencyCount;
    uint directionBinCount;
    uint hiddenWidth;
    uint layerCount;
    uint weightCount;
    uint biasCount;
};

struct NeuralIrradianceLayerDesc
{
    uint inputDim;
    uint outputDim;
    uint weightOffset;
    uint biasOffset;
};

struct NeuralIrradianceInput
{
    float values[kNeuralIrradianceMaxActivationDim];
};

ConstantBuffer<NeuralIrradianceModelConstants> gNeuralIrradianceConstants;
StructuredBuffer<NeuralIrradianceLayerDesc> gNeuralIrradianceLayers;
StructuredBuffer<float> gNeuralIrradianceWeights;
StructuredBuffer<float> gNeuralIrradianceBiases;

float neuralIrradianceOneBlobValue(float value01, uint binIndex, uint binCount, bool wrap)
{
    const float binCountF = float(binCount);
    const float center = (float(binIndex) + 0.5f) / binCountF;
    float distance = abs(value01 - center);
    if (wrap)
        distance = min(distance, 1.f - distance);

    return max(1.f - distance * binCountF, 0.f);
}

void neuralIrradianceClearInput(out NeuralIrradianceInput result)
{
    for (uint i = 0; i < kNeuralIrradianceMaxActivationDim; ++i)
        result.values[i] = 0.f;
}

void neuralIrradianceEncodeInput(float3 positionW, float3 directionW, out NeuralIrradianceInput result)
{
    neuralIrradianceClearInput(result);

    const uint frequencyCount = gNeuralIrradianceConstants.positionFrequencyCount;
    const uint directionBinCount = gNeuralIrradianceConstants.directionBinCount;
    const float3 positionExtent = max(gNeuralIrradianceConstants.positionExtent.xyz, float3(1e-6f));
    const float3 position01 = (positionW - gNeuralIrradianceConstants.positionMin.xyz) / positionExtent;

    uint writeIndex = 0;
    for (uint component = 0; component < 3; ++component)
    {
        const float positionComponent = position01[component];
        for (uint frequencyIndex = 0; frequencyIndex < frequencyCount; ++frequencyIndex)
        {
            const float frequency = exp2(float(frequencyIndex)) * float(M_2PI);
            result.values[writeIndex + frequencyIndex] = sin(positionComponent * frequency);
            result.values[writeIndex + frequencyCount + frequencyIndex] = cos(positionComponent * frequency);
        }
        writeIndex += frequencyCount * 2u;
    }

    const float3 direction = directionW / max(length(directionW), 1e-8f);
    const float theta = acos(clamp(direction.z, -1.f, 1.f)) * float(M_1_PI);
    const float phi = frac(atan2(direction.y, direction.x) * float(M_1_2PI) + 1.f);

    for (uint binIndex = 0; binIndex < directionBinCount; ++binIndex)
        result.values[writeIndex + binIndex] = neuralIrradianceOneBlobValue(theta, binIndex, directionBinCount, false);

    writeIndex += directionBinCount;
    for (uint binIndex = 0; binIndex < directionBinCount; ++binIndex)
        result.values[writeIndex + binIndex] = neuralIrradianceOneBlobValue(phi, binIndex, directionBinCount, true);
}

float3 neuralIrradianceEvaluateEncoded(NeuralIrradianceInput input)
{
    float activationsA[kNeuralIrradianceMaxActivationDim];
    float activationsB[kNeuralIrradianceMaxActivationDim];

    for (uint i = 0; i < kNeuralIrradianceMaxActivationDim; ++i)
    {
        activationsA[i] = input.values[i];
        activationsB[i] = 0.f;
    }

    for (uint layerIndex = 0; layerIndex < gNeuralIrradianceConstants.layerCount; ++layerIndex)
    {
        const NeuralIrradianceLayerDesc layer = gNeuralIrradianceLayers[layerIndex];

        for (uint outputIndex = 0; outputIndex < layer.outputDim; ++outputIndex)
        {
            float value = gNeuralIrradianceBiases[layer.biasOffset + outputIndex];
            const uint weightRowOffset = layer.weightOffset + outputIndex * layer.inputDim;

            for (uint inputIndex = 0; inputIndex < layer.inputDim; ++inputIndex)
                value += activationsA[inputIndex] * gNeuralIrradianceWeights[weightRowOffset + inputIndex];

            if (layerIndex + 1u < gNeuralIrradianceConstants.layerCount)
                value = max(value, 0.f);

            activationsB[outputIndex] = value;
        }

        for (uint i = 0; i < kNeuralIrradianceMaxActivationDim; ++i)
            activationsA[i] = (i < layer.outputDim) ? activationsB[i] : 0.f;
    }

    return float3(activationsA[0], activationsA[1], activationsA[2]);
}

float3 neuralIrradianceEvaluate(float3 positionW, float3 directionW)
{
    NeuralIrradianceInput input;
    neuralIrradianceEncodeInput(positionW, directionW, input);
    return neuralIrradianceEvaluateEncoded(input);
}

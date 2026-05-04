/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

import Rendering.NeuralIrradianceModel;

cbuffer CB
{
    uint sampleCount;
};

StructuredBuffer<float3> positions;
StructuredBuffer<float3> directions;
RWStructuredBuffer<float> encodedOutputs;
RWStructuredBuffer<float3> irradianceOutputs;

[numthreads(64, 1, 1)]
void testNeuralIrradianceInference(uint3 threadId: SV_DispatchThreadID)
{
    const uint index = threadId.x;
    if (index >= sampleCount)
        return;

    NeuralIrradianceInput input;
    neuralIrradianceEncodeInput(positions[index], directions[index], input);

    const uint encodedBase = index * kNeuralIrradianceMaxActivationDim;
    for (uint i = 0; i < kNeuralIrradianceMaxActivationDim; ++i)
        encodedOutputs[encodedBase + i] = input.values[i];

    irradianceOutputs[index] = neuralIrradianceEvaluateEncoded(input);
}

# Neural Light Grid Learning

Python-side tooling for inspecting and later training on irradiance samples baked by Falcor's `IrradianceSamplesBaker`.

The first supported data source is the binary `.irradiance-samples.bin` format written by:

`Source/Samples/IrradianceSamplesBaker`

## Quick Start

From the repository root:

```powershell
python NeuralLightGridLearning\inspect_irradiance_samples.py build\windows-vs2022\bin\Debug\baked_samples\BistroExterior.irradiance-samples.bin --no-plot
```

Run a short MLP training smoke test:

```powershell
python NeuralLightGridLearning\train_tiny_mlp.py build\windows-vs2022\bin\Debug\baked_samples\BistroExterior.irradiance-samples.bin --iterations 10 --batch-size 4096
```

Run the default training schedule:

```powershell
python NeuralLightGridLearning\train_tiny_mlp.py build\windows-vs2022\bin\Debug\baked_samples\BistroExterior.irradiance-samples.bin --checkpoint NeuralLightGridLearning\checkpoints\tiny_irradiance_mlp.pt
```

When `--checkpoint` is set, training also writes the Falcor runtime artifacts next to the checkpoint:

- `tiny_irradiance_mlp.falcor-mlp.bin`
- `tiny_irradiance_mlp.falcor-mlp.parity.bin`

Override or disable that behavior with:

```powershell
python NeuralLightGridLearning\train_tiny_mlp.py build\windows-vs2022\bin\Debug\baked_samples\BistroExterior.irradiance-samples.bin `
  --checkpoint NeuralLightGridLearning\checkpoints\tiny_irradiance_mlp.pt `
  --falcor-parity-count 128
```

```powershell
python NeuralLightGridLearning\train_tiny_mlp.py build\windows-vs2022\bin\Debug\baked_samples\BistroExterior.irradiance-samples.bin `
  --checkpoint NeuralLightGridLearning\checkpoints\tiny_irradiance_mlp.pt `
  --skip-falcor-artifacts
```

You can still export an existing checkpoint manually:

```powershell
python NeuralLightGridLearning\export_tiny_mlp.py NeuralLightGridLearning\checkpoints\tiny_irradiance_mlp.pt
python NeuralLightGridLearning\create_inference_parity_vectors.py NeuralLightGridLearning\checkpoints\tiny_irradiance_mlp.pt NeuralLightGridLearning\checkpoints\tiny_irradiance_mlp.falcor-mlp.bin
```

Run the optional Falcor D3D12 parity test against the exported model and parity vectors:

```powershell
build\windows-vs2022\bin\Debug\FalcorTest.exe --device-type d3d12 --test-case NeuralIrradianceModel_TrainedCheckpointParity
```

By default, this test looks for:

- `NeuralLightGridLearning\checkpoints\tiny_irradiance_mlp.falcor-mlp.bin`
- `NeuralLightGridLearning\checkpoints\tiny_irradiance_mlp.falcor-mlp.parity.bin`

Override those paths with `FALCOR_NLG_MODEL_PATH` and `FALCOR_NLG_PARITY_PATH`.

Inside a VS Code Jupyter notebook:

```python
from pathlib import Path
import sys

repo_root = Path.cwd()
if repo_root.name == "NeuralLightGridLearning":
    repo_root = repo_root.parent

sys.path.insert(0, str(repo_root / "NeuralLightGridLearning"))

from irradiance_samples import load_irradiance_samples, plot_samples_overview

sample_path = repo_root / "build/windows-vs2022/bin/Debug/baked_samples/BistroExterior.irradiance-samples.bin"
samples = load_irradiance_samples(sample_path)
samples.summary()
```

```python
plot_samples_overview(samples, max_points=50_000, color_by="irradiance")
```

## Binary Format

### Irradiance Samples

The supported v2 file format matches `IrradianceSampleDebugVis::FileHeader` followed by `IrradianceSampleDebugVis::Sample`:

- header: 32 bytes
- sample v2: `float4 position`, `float4 normal`, `float4 irradiance`, `uint4 meta`

The parsed `meta` fields are:

- `meta.x`: sample flags, where bit `1` marks a surface sample
- `meta.y`: validation probe hit count
- `meta.z`: validation probe backface hit count
- `meta.w`: reserved

### Exported Tiny MLP

The exported model file is intentionally simple:

- header: magic `NLG1`, version, encoding config, network dimensions, and weight count
- normalization: `float3 position_min`, `float3 position_extent`
- source sample path as UTF-8 bytes
- one record per linear layer:
  - `uint32 input_dim`
  - `uint32 output_dim`
  - `uint32 weight_count`
  - `uint32 bias_count`
  - row-major `float32` weights shaped as `[output_dim, input_dim]`
  - `float32` bias shaped as `[output_dim]`

### Inference Parity Vectors

The parity-vector file is for checking that Falcor's Slang implementation matches Python:

- header: magic `NLP1`, version, vector count, dimensions, and encoding config
- one record per vector:
  - `float3 position`
  - `float3 direction`
  - encoded network input as `float32[input_dim]`
  - expected model output as `float32[output_dim]`
  - target baked irradiance as `float32[output_dim]`

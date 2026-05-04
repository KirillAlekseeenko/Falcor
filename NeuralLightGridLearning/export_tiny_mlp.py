from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from .train_tiny_mlp import TinyIrradianceMLP, TrainingConfig
except ImportError:
    from train_tiny_mlp import TinyIrradianceMLP, TrainingConfig


MODEL_MAGIC = b"NLG1"
MODEL_VERSION = 1
OUTPUT_DIM = 3

_HEADER_STRUCT = struct.Struct("<4s12I")
_LAYER_STRUCT = struct.Struct("<4I")


@dataclass(frozen=True)
class LayerExportData:
    input_dim: int
    output_dim: int
    weight: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class ExportedModelData:
    path: Path
    version: int
    input_dim: int
    position_frequencies: int
    direction_bins: int
    hidden_width: int
    layer_count: int
    output_dim: int
    weight_float_count: int
    position_min: np.ndarray
    position_extent: np.ndarray
    sample_path: str
    layers: list[LayerExportData]


def _config_from_checkpoint(checkpoint: dict) -> TrainingConfig:
    config_dict = checkpoint.get("config")
    if not isinstance(config_dict, dict):
        raise ValueError("Checkpoint does not contain a 'config' dictionary.")

    valid_keys = set(TrainingConfig.__dataclass_fields__.keys())
    filtered_config = {key: value for key, value in config_dict.items() if key in valid_keys}
    return TrainingConfig(**filtered_config)


def _load_model_from_checkpoint(checkpoint_path: Path) -> tuple[TinyIrradianceMLP, TrainingConfig, torch.Tensor, torch.Tensor, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"'{checkpoint_path}' is not a supported checkpoint dictionary.")

    config = _config_from_checkpoint(checkpoint)

    position_min = checkpoint.get("position_min")
    position_extent = checkpoint.get("position_extent")
    if position_min is None or position_extent is None:
        raise ValueError("Checkpoint is missing 'position_min' or 'position_extent'.")

    position_min = torch.as_tensor(position_min, dtype=torch.float32).reshape(-1)
    position_extent = torch.as_tensor(position_extent, dtype=torch.float32).reshape(-1)
    if position_min.numel() != 3 or position_extent.numel() != 3:
        raise ValueError("Checkpoint position normalization tensors must contain exactly 3 values.")

    model = TinyIrradianceMLP(
        input_dim=config.input_dim,
        hidden_width=config.hidden_width,
        layer_count=config.layer_count,
        output_dim=OUTPUT_DIM,
    )

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a 'model_state_dict' dictionary.")

    model.load_state_dict(state_dict)
    model.eval()
    return model, config, position_min, position_extent, checkpoint


def _collect_linear_layers(model: TinyIrradianceMLP) -> list[LayerExportData]:
    layers: list[LayerExportData] = []
    for module in model.net:
        if not isinstance(module, torch.nn.Linear):
            continue

        weight = module.weight.detach().cpu().numpy().astype(np.float32, copy=True)
        bias = module.bias.detach().cpu().numpy().astype(np.float32, copy=True)
        layers.append(
            LayerExportData(
                input_dim=int(module.in_features),
                output_dim=int(module.out_features),
                weight=np.ascontiguousarray(weight),
                bias=np.ascontiguousarray(bias),
            )
        )

    return layers


def _validate_layers(layers: list[LayerExportData], config: TrainingConfig) -> None:
    if len(layers) != config.layer_count:
        raise ValueError(f"Checkpoint contains {len(layers)} linear layers, expected {config.layer_count}.")

    if layers[0].input_dim != config.input_dim:
        raise ValueError(f"First layer input dimension is {layers[0].input_dim}, expected {config.input_dim}.")

    if layers[-1].output_dim != OUTPUT_DIM:
        raise ValueError(f"Output layer dimension is {layers[-1].output_dim}, expected {OUTPUT_DIM}.")

    for layer_index, layer in enumerate(layers):
        if layer.weight.shape != (layer.output_dim, layer.input_dim):
            raise ValueError(f"Layer {layer_index} has invalid weight shape {layer.weight.shape}.")
        if layer.bias.shape != (layer.output_dim,):
            raise ValueError(f"Layer {layer_index} has invalid bias shape {layer.bias.shape}.")
        if layer_index > 0 and layer.input_dim != layers[layer_index - 1].output_dim:
            raise ValueError(
                f"Layer {layer_index} input dimension {layer.input_dim} does not match previous output "
                f"dimension {layers[layer_index - 1].output_dim}."
            )


def export_checkpoint(checkpoint_path: Path, output_path: Path) -> None:
    model, config, position_min, position_extent, checkpoint = _load_model_from_checkpoint(checkpoint_path)
    layers = _collect_linear_layers(model)
    _validate_layers(layers, config)

    weight_float_count = sum(layer.weight.size + layer.bias.size for layer in layers)
    sample_path = str(checkpoint.get("sample_path", ""))
    sample_path_bytes = sample_path.encode("utf-8")

    header = _HEADER_STRUCT.pack(
        MODEL_MAGIC,
        MODEL_VERSION,
        config.input_dim,
        config.position_frequencies,
        config.direction_bins,
        config.hidden_width,
        config.layer_count,
        OUTPUT_DIM,
        weight_float_count,
        len(sample_path_bytes),
        0,
        0,
        0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        stream.write(header)
        stream.write(position_min.numpy().astype("<f4", copy=False).tobytes())
        stream.write(position_extent.numpy().astype("<f4", copy=False).tobytes())
        stream.write(sample_path_bytes)

        for layer in layers:
            stream.write(_LAYER_STRUCT.pack(layer.input_dim, layer.output_dim, layer.weight.size, layer.bias.size))
            stream.write(layer.weight.astype("<f4", copy=False).tobytes(order="C"))
            stream.write(layer.bias.astype("<f4", copy=False).tobytes(order="C"))


def load_exported_model(path: Path) -> ExportedModelData:
    file_size = path.stat().st_size

    with path.open("rb") as stream:
        header_bytes = stream.read(_HEADER_STRUCT.size)
        if len(header_bytes) != _HEADER_STRUCT.size:
            raise ValueError(f"'{path}' is too small to contain a neural model header.")

        (
            magic,
            version,
            input_dim,
            position_frequencies,
            direction_bins,
            hidden_width,
            layer_count,
            output_dim,
            weight_float_count,
            sample_path_byte_count,
            _reserved0,
            _reserved1,
            _reserved2,
        ) = _HEADER_STRUCT.unpack(header_bytes)

        if magic != MODEL_MAGIC:
            raise ValueError(f"'{path}' has invalid model magic {magic!r}.")

        if version != MODEL_VERSION:
            raise ValueError(f"'{path}' has unsupported model version {version}; expected {MODEL_VERSION}.")

        expected_size = _HEADER_STRUCT.size + 24 + sample_path_byte_count
        if file_size < expected_size:
            raise ValueError(f"'{path}' is truncated: expected at least {expected_size} bytes, found {file_size}.")

        position_min_bytes = stream.read(12)
        position_extent_bytes = stream.read(12)
        if len(position_min_bytes) != 12 or len(position_extent_bytes) != 12:
            raise ValueError(f"'{path}' ended before position normalization was read.")

        position_min = np.frombuffer(position_min_bytes, dtype="<f4").copy()
        position_extent = np.frombuffer(position_extent_bytes, dtype="<f4").copy()
        sample_path = stream.read(sample_path_byte_count).decode("utf-8") if sample_path_byte_count > 0 else ""

        layers: list[LayerExportData] = []
        for _ in range(layer_count):
            expected_size += _LAYER_STRUCT.size
            if file_size < expected_size:
                raise ValueError(f"'{path}' is truncated: expected at least {expected_size} bytes, found {file_size}.")

            layer_bytes = stream.read(_LAYER_STRUCT.size)
            if len(layer_bytes) != _LAYER_STRUCT.size:
                raise ValueError(f"'{path}' ended before all layer headers were read.")

            layer_input_dim, layer_output_dim, layer_weight_count, layer_bias_count = _LAYER_STRUCT.unpack(layer_bytes)
            layer_float_count = layer_weight_count + layer_bias_count
            layer_byte_count = layer_float_count * 4
            expected_size += layer_byte_count
            if file_size < expected_size:
                raise ValueError(f"'{path}' is truncated: expected at least {expected_size} bytes, found {file_size}.")

            layer_payload = stream.read(layer_byte_count)
            if len(layer_payload) != layer_byte_count:
                raise ValueError(f"'{path}' ended before all layer weights were read.")

            if layer_weight_count != layer_input_dim * layer_output_dim:
                raise ValueError(
                    f"Layer has {layer_weight_count} weights, expected {layer_input_dim * layer_output_dim} "
                    f"for dimensions {layer_input_dim}x{layer_output_dim}."
                )
            if layer_bias_count != layer_output_dim:
                raise ValueError(f"Layer has {layer_bias_count} bias values, expected {layer_output_dim}.")

            floats = np.frombuffer(layer_payload, dtype="<f4")
            weight = floats[:layer_weight_count].reshape((layer_output_dim, layer_input_dim)).copy()
            bias = floats[layer_weight_count:].copy()
            layers.append(LayerExportData(input_dim=layer_input_dim, output_dim=layer_output_dim, weight=weight, bias=bias))

        if file_size != expected_size:
            raise ValueError(f"'{path}' has unexpected trailing data: expected {expected_size} bytes, found {file_size}.")

    actual_weight_float_count = sum(layer.weight.size + layer.bias.size for layer in layers)
    if actual_weight_float_count != weight_float_count:
        raise ValueError(f"'{path}' stores {actual_weight_float_count} parameter floats, header expected {weight_float_count}.")

    if len(layers) != layer_count:
        raise ValueError(f"'{path}' stores {len(layers)} layers, header expected {layer_count}.")

    if layers and layers[0].input_dim != input_dim:
        raise ValueError(f"First layer input dimension is {layers[0].input_dim}, header expected {input_dim}.")

    if layers and layers[-1].output_dim != output_dim:
        raise ValueError(f"Last layer output dimension is {layers[-1].output_dim}, header expected {output_dim}.")

    for layer_index in range(1, len(layers)):
        if layers[layer_index].input_dim != layers[layer_index - 1].output_dim:
            raise ValueError(
                f"Layer {layer_index} input dimension is {layers[layer_index].input_dim}, previous layer output "
                f"dimension is {layers[layer_index - 1].output_dim}."
            )

    return ExportedModelData(
        path=path,
        version=version,
        input_dim=input_dim,
        position_frequencies=position_frequencies,
        direction_bins=direction_bins,
        hidden_width=hidden_width,
        layer_count=layer_count,
        output_dim=output_dim,
        weight_float_count=weight_float_count,
        position_min=position_min,
        position_extent=position_extent,
        sample_path=sample_path,
        layers=layers,
    )


def read_export_summary(path: Path) -> dict[str, object]:
    model = load_exported_model(path)
    return {
        "path": str(model.path),
        "version": model.version,
        "input_dim": model.input_dim,
        "position_frequencies": model.position_frequencies,
        "direction_bins": model.direction_bins,
        "hidden_width": model.hidden_width,
        "layer_count": model.layer_count,
        "output_dim": model.output_dim,
        "weight_float_count": model.weight_float_count,
        "position_min": model.position_min.tolist(),
        "position_extent": model.position_extent.tolist(),
        "sample_path": model.sample_path,
        "layers": [
            {
                "input_dim": layer.input_dim,
                "output_dim": layer.output_dim,
                "weight_count": layer.weight.size,
                "bias_count": layer.bias.size,
            }
            for layer in model.layers
        ],
    }


def print_export_summary(summary: dict[str, object]) -> None:
    for key, value in summary.items():
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a tiny irradiance MLP PyTorch checkpoint to a Falcor-readable binary file.")
    parser.add_argument("checkpoint", type=Path, help="Input .pt checkpoint produced by train_tiny_mlp.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output binary path. Defaults to the checkpoint stem with .falcor-mlp.bin next to the checkpoint.",
    )
    parser.add_argument("--summary-only", action="store_true", help="Read and print an already exported Falcor model file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.summary_only:
        print_export_summary(read_export_summary(args.checkpoint))
        return 0

    output_path = args.output
    if output_path is None:
        output_path = args.checkpoint.with_suffix(".falcor-mlp.bin")

    export_checkpoint(args.checkpoint, output_path)
    print_export_summary(read_export_summary(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

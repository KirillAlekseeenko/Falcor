from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from .export_tiny_mlp import MODEL_VERSION, ExportedModelData, _load_model_from_checkpoint, load_exported_model
    from .irradiance_samples import load_irradiance_samples
    from .train_tiny_mlp import TrainingConfig, encode_network_inputs_with_normalization
except ImportError:
    from export_tiny_mlp import MODEL_VERSION, ExportedModelData, _load_model_from_checkpoint, load_exported_model
    from irradiance_samples import load_irradiance_samples
    from train_tiny_mlp import TrainingConfig, encode_network_inputs_with_normalization


PARITY_MAGIC = b"NLP1"
PARITY_VERSION = 1
PARITY_FLAG_HAS_TARGET_IRRADIANCE = 1

_PARITY_HEADER_STRUCT = struct.Struct("<4s8I")


@dataclass(frozen=True)
class ParityVectors:
    positions: np.ndarray
    directions: np.ndarray
    encoded_inputs: np.ndarray
    expected_outputs: np.ndarray
    target_irradiance: np.ndarray

    @property
    def vector_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.encoded_inputs.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.expected_outputs.shape[1])


def _config_from_exported_model(model: ExportedModelData) -> TrainingConfig:
    return TrainingConfig(
        position_frequencies=model.position_frequencies,
        direction_bins=model.direction_bins,
        hidden_width=model.hidden_width,
        layer_count=model.layer_count,
    )


def _validate_checkpoint_matches_export(
    checkpoint_config: TrainingConfig,
    checkpoint_position_min: torch.Tensor,
    checkpoint_position_extent: torch.Tensor,
    exported_model: ExportedModelData,
) -> None:
    if exported_model.version != MODEL_VERSION:
        raise ValueError(f"Exported model version {exported_model.version} does not match expected version {MODEL_VERSION}.")

    exported_config = _config_from_exported_model(exported_model)
    fields = ["position_frequencies", "direction_bins", "hidden_width", "layer_count", "input_dim"]
    for field in fields:
        checkpoint_value = getattr(checkpoint_config, field)
        exported_value = getattr(exported_config, field)
        if checkpoint_value != exported_value:
            raise ValueError(f"Checkpoint/export config mismatch for {field}: {checkpoint_value} != {exported_value}.")

    exported_position_min = torch.from_numpy(exported_model.position_min)
    exported_position_extent = torch.from_numpy(exported_model.position_extent)
    if not torch.allclose(checkpoint_position_min.cpu(), exported_position_min, rtol=0.0, atol=0.0):
        raise ValueError("Checkpoint/export position_min mismatch.")
    if not torch.allclose(checkpoint_position_extent.cpu(), exported_position_extent, rtol=0.0, atol=0.0):
        raise ValueError("Checkpoint/export position_extent mismatch.")


def _resolve_sample_path(checkpoint: dict, explicit_sample_path: Path | None) -> Path:
    if explicit_sample_path is not None:
        return explicit_sample_path

    sample_path = checkpoint.get("sample_path")
    if not sample_path:
        raise ValueError("Checkpoint does not contain a sample_path. Pass --sample-path explicitly.")

    path = Path(str(sample_path))
    if path.exists():
        return path

    raise FileNotFoundError(f"Checkpoint sample_path '{path}' does not exist. Pass --sample-path explicitly.")


def _choose_sample_indices(sample_count: int, vector_count: int, seed: int) -> np.ndarray:
    if vector_count <= 0:
        raise ValueError("vector_count must be positive.")
    if sample_count <= 0:
        raise ValueError("Cannot create parity vectors from an empty sample set.")

    rng = np.random.default_rng(seed)
    if vector_count <= sample_count:
        return np.sort(rng.choice(sample_count, size=vector_count, replace=False))

    return rng.choice(sample_count, size=vector_count, replace=True)


def run_exported_model_numpy(exported_model: ExportedModelData, encoded_inputs: np.ndarray) -> np.ndarray:
    x = np.asarray(encoded_inputs, dtype=np.float32)
    for layer_index, layer in enumerate(exported_model.layers):
        x = x @ layer.weight.T + layer.bias
        if layer_index + 1 < len(exported_model.layers):
            x = np.maximum(x, 0.0)
        x = np.asarray(x, dtype=np.float32)
    return x


def create_parity_vectors(
    checkpoint_path: Path,
    exported_model_path: Path,
    sample_path: Path | None,
    vector_count: int,
    seed: int,
    tolerance: float,
) -> tuple[ParityVectors, float]:
    torch_model, checkpoint_config, position_min, position_extent, checkpoint = _load_model_from_checkpoint(checkpoint_path)
    exported_model = load_exported_model(exported_model_path)
    _validate_checkpoint_matches_export(checkpoint_config, position_min, position_extent, exported_model)

    samples = load_irradiance_samples(_resolve_sample_path(checkpoint, sample_path))
    indices = _choose_sample_indices(samples.sample_count, vector_count, seed)

    positions_np = np.ascontiguousarray(samples.positions[indices], dtype=np.float32)
    directions_np = np.ascontiguousarray(samples.normals[indices], dtype=np.float32)
    targets_np = np.ascontiguousarray(samples.irradiance[indices], dtype=np.float32)

    positions = torch.from_numpy(positions_np)
    directions = torch.from_numpy(directions_np)
    encoded_inputs = encode_network_inputs_with_normalization(
        positions,
        directions,
        checkpoint_config,
        position_min,
        position_extent,
    )

    with torch.no_grad():
        torch_outputs = torch_model(encoded_inputs).cpu().numpy().astype(np.float32, copy=True)

    encoded_np = np.ascontiguousarray(encoded_inputs.cpu().numpy(), dtype=np.float32)
    exported_outputs = run_exported_model_numpy(exported_model, encoded_np)
    max_abs_error = float(np.max(np.abs(exported_outputs - torch_outputs))) if torch_outputs.size else 0.0
    if max_abs_error > tolerance:
        raise ValueError(f"Exported model inference differs from checkpoint by {max_abs_error:.8g}, tolerance {tolerance:.8g}.")

    return (
        ParityVectors(
            positions=positions_np,
            directions=directions_np,
            encoded_inputs=encoded_np,
            expected_outputs=torch_outputs,
            target_irradiance=targets_np,
        ),
        max_abs_error,
    )


def write_parity_vectors(path: Path, vectors: ParityVectors, position_frequencies: int, direction_bins: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    header = _PARITY_HEADER_STRUCT.pack(
        PARITY_MAGIC,
        PARITY_VERSION,
        vectors.vector_count,
        vectors.input_dim,
        vectors.output_dim,
        position_frequencies,
        direction_bins,
        PARITY_FLAG_HAS_TARGET_IRRADIANCE,
        0,
    )

    with path.open("wb") as stream:
        stream.write(header)
        for i in range(vectors.vector_count):
            stream.write(vectors.positions[i].astype("<f4", copy=False).tobytes())
            stream.write(vectors.directions[i].astype("<f4", copy=False).tobytes())
            stream.write(vectors.encoded_inputs[i].astype("<f4", copy=False).tobytes())
            stream.write(vectors.expected_outputs[i].astype("<f4", copy=False).tobytes())
            stream.write(vectors.target_irradiance[i].astype("<f4", copy=False).tobytes())


def read_parity_vectors(path: Path) -> tuple[dict[str, int], ParityVectors]:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        header_bytes = stream.read(_PARITY_HEADER_STRUCT.size)
        if len(header_bytes) != _PARITY_HEADER_STRUCT.size:
            raise ValueError(f"'{path}' is too small to contain a parity-vector header.")

        (
            magic,
            version,
            vector_count,
            input_dim,
            output_dim,
            position_frequencies,
            direction_bins,
            flags,
            _reserved,
        ) = _PARITY_HEADER_STRUCT.unpack(header_bytes)

        if magic != PARITY_MAGIC:
            raise ValueError(f"'{path}' has invalid parity-vector magic {magic!r}.")
        if version != PARITY_VERSION:
            raise ValueError(f"'{path}' has unsupported parity-vector version {version}; expected {PARITY_VERSION}.")
        if (flags & PARITY_FLAG_HAS_TARGET_IRRADIANCE) == 0:
            raise ValueError(f"'{path}' does not contain target irradiance vectors.")

        floats_per_vector = 3 + 3 + input_dim + output_dim + output_dim
        expected_size = _PARITY_HEADER_STRUCT.size + vector_count * floats_per_vector * 4
        if file_size != expected_size:
            raise ValueError(f"'{path}' has invalid size: expected {expected_size} bytes, found {file_size}.")

        payload = np.fromfile(stream, dtype="<f4", count=vector_count * floats_per_vector)
        if payload.size != vector_count * floats_per_vector:
            raise ValueError(f"Failed to read {vector_count} parity vectors from '{path}'.")

    payload = payload.reshape((vector_count, floats_per_vector))
    offset = 0
    positions = payload[:, offset : offset + 3].copy()
    offset += 3
    directions = payload[:, offset : offset + 3].copy()
    offset += 3
    encoded_inputs = payload[:, offset : offset + input_dim].copy()
    offset += input_dim
    expected_outputs = payload[:, offset : offset + output_dim].copy()
    offset += output_dim
    target_irradiance = payload[:, offset : offset + output_dim].copy()

    header = {
        "version": version,
        "vector_count": vector_count,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "position_frequencies": position_frequencies,
        "direction_bins": direction_bins,
        "flags": flags,
    }
    return header, ParityVectors(positions, directions, encoded_inputs, expected_outputs, target_irradiance)


def print_parity_summary(path: Path) -> None:
    header, vectors = read_parity_vectors(path)
    print(f"path: {path}")
    for key, value in header.items():
        print(f"{key}: {value}")
    print(f"position_min: {vectors.positions.min(axis=0).tolist()}")
    print(f"position_max: {vectors.positions.max(axis=0).tolist()}")
    print(f"expected_output_min: {vectors.expected_outputs.min(axis=0).tolist()}")
    print(f"expected_output_max: {vectors.expected_outputs.max(axis=0).tolist()}")
    print(f"target_irradiance_min: {vectors.target_irradiance.min(axis=0).tolist()}")
    print(f"target_irradiance_max: {vectors.target_irradiance.max(axis=0).tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Python/Falcor neural irradiance inference parity vectors.")
    parser.add_argument("checkpoint", type=Path, nargs="?", help="Input .pt checkpoint produced by train_tiny_mlp.py.")
    parser.add_argument("exported_model", type=Path, nargs="?", help="Input .falcor-mlp.bin produced by export_tiny_mlp.py.")
    parser.add_argument("--sample-path", type=Path, default=None, help="Optional irradiance sample file. Defaults to checkpoint sample_path.")
    parser.add_argument("--output", type=Path, default=None, help="Output .parity.bin path.")
    parser.add_argument("--count", type=int, default=64, help="Number of parity vectors to export.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--summary-only", action="store_true", help="Read and summarize an existing parity-vector file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.summary_only:
        parity_path = args.exported_model or args.checkpoint
        if parity_path is None:
            raise ValueError("Pass a parity-vector file path when using --summary-only.")
        print_parity_summary(parity_path)
        return 0

    if args.checkpoint is None or args.exported_model is None:
        raise ValueError("Pass both checkpoint and exported_model paths when creating parity vectors.")

    output_path = args.output
    if output_path is None:
        output_path = args.exported_model.with_suffix(".parity.bin")

    exported_model = load_exported_model(args.exported_model)
    vectors, max_abs_error = create_parity_vectors(
        checkpoint_path=args.checkpoint,
        exported_model_path=args.exported_model,
        sample_path=args.sample_path,
        vector_count=args.count,
        seed=args.seed,
        tolerance=args.tolerance,
    )
    write_parity_vectors(
        output_path,
        vectors,
        position_frequencies=exported_model.position_frequencies,
        direction_bins=exported_model.direction_bins,
    )

    print(f"wrote: {output_path}")
    print(f"vector_count: {vectors.vector_count}")
    print(f"input_dim: {vectors.input_dim}")
    print(f"output_dim: {vectors.output_dim}")
    print(f"max_abs_checkpoint_vs_exported_error: {max_abs_error:.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

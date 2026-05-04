from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .irradiance_samples import load_irradiance_samples
except ImportError:
    from irradiance_samples import load_irradiance_samples


@dataclass(frozen=True)
class TrainingConfig:
    position_frequencies: int = 6
    direction_bins: int = 4
    hidden_width: int = 64
    layer_count: int = 5
    learning_rate: float = 1e-2
    reduced_learning_rate: float = 1e-4
    lr_drop_iteration: int = 10_000
    batch_size: int = 1 << 16
    loss_epsilon: float = 0.01

    @property
    def input_dim(self) -> int:
        position_dim = 3 * self.position_frequencies * 2
        direction_dim = 2 * self.direction_bins
        return position_dim + direction_dim


class TinyIrradianceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_width: int = 64, layer_count: int = 5, output_dim: int = 3):
        super().__init__()

        if layer_count < 2:
            raise ValueError("layer_count must include at least one hidden layer and one output layer.")

        layers: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(layer_count - 1):
            layers.append(nn.Linear(last_dim, hidden_width))
            layers.append(nn.ReLU())
            last_dim = hidden_width

        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


@dataclass(frozen=True)
class EncodedTrainingData:
    inputs: torch.Tensor
    targets: torch.Tensor
    position_min: torch.Tensor
    position_extent: torch.Tensor


@dataclass(frozen=True)
class FalcorArtifactConfig:
    model_output_path: Path | None = None
    parity_output_path: Path | None = None
    parity_count: int = 64
    parity_seed: int = 1
    parity_tolerance: float = 1e-5


def normalize_positions(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position_min = positions.amin(dim=0)
    position_max = positions.amax(dim=0)
    position_extent = torch.clamp(position_max - position_min, min=1e-6)
    return (positions - position_min) / position_extent, position_min, position_extent


def normalize_positions_with_bounds(positions: torch.Tensor, position_min: torch.Tensor, position_extent: torch.Tensor) -> torch.Tensor:
    return (positions - position_min) / torch.clamp(position_extent, min=1e-6)


def encode_positions(positions01: torch.Tensor, frequency_count: int = 6) -> torch.Tensor:
    frequencies = (2.0 ** torch.arange(frequency_count, device=positions01.device, dtype=positions01.dtype)) * (2.0 * math.pi)
    values = positions01[..., None] * frequencies
    encoded = torch.cat([torch.sin(values), torch.cos(values)], dim=-1)
    return encoded.flatten(start_dim=-2)


def directions_to_spherical01(directions: torch.Tensor) -> torch.Tensor:
    directions = nn.functional.normalize(directions, dim=-1, eps=1e-8)
    theta = torch.acos(torch.clamp(directions[:, 2], -1.0, 1.0)) / math.pi
    phi = torch.atan2(directions[:, 1], directions[:, 0]) / (2.0 * math.pi)
    phi = torch.remainder(phi + 1.0, 1.0)
    return torch.stack([theta, phi], dim=-1)


def one_blob_encode(values01: torch.Tensor, bin_count: int = 4, wrap: bool = False) -> torch.Tensor:
    centers = (torch.arange(bin_count, device=values01.device, dtype=values01.dtype) + 0.5) / float(bin_count)
    distances = torch.abs(values01[..., None] - centers)
    if wrap:
        distances = torch.minimum(distances, 1.0 - distances)

    bin_width = 1.0 / float(bin_count)
    return torch.clamp(1.0 - distances / bin_width, min=0.0)


def encode_directions(directions: torch.Tensor, bin_count: int = 4) -> torch.Tensor:
    spherical = directions_to_spherical01(directions)
    theta = one_blob_encode(spherical[:, 0], bin_count=bin_count, wrap=False)
    phi = one_blob_encode(spherical[:, 1], bin_count=bin_count, wrap=True)
    return torch.cat([theta, phi], dim=-1)


def encode_network_inputs(positions: torch.Tensor, directions: torch.Tensor, config: TrainingConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions01, position_min, position_extent = normalize_positions(positions)
    encoded_inputs = encode_network_inputs_with_bounds(positions01, directions, config)
    return encoded_inputs, position_min, position_extent


def encode_network_inputs_with_bounds(positions01: torch.Tensor, directions: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    encoded_positions = encode_positions(positions01, frequency_count=config.position_frequencies)
    encoded_directions = encode_directions(directions, bin_count=config.direction_bins)
    return torch.cat([encoded_positions, encoded_directions], dim=-1)


def encode_network_inputs_with_normalization(
    positions: torch.Tensor,
    directions: torch.Tensor,
    config: TrainingConfig,
    position_min: torch.Tensor,
    position_extent: torch.Tensor,
) -> torch.Tensor:
    positions01 = normalize_positions_with_bounds(positions, position_min, position_extent)
    return encode_network_inputs_with_bounds(positions01, directions, config)


def load_training_data(sample_path: Path, device: torch.device, config: TrainingConfig) -> EncodedTrainingData:
    samples = load_irradiance_samples(sample_path)

    positions = torch.from_numpy(np.ascontiguousarray(samples.positions)).to(device=device, dtype=torch.float32)
    directions = torch.from_numpy(np.ascontiguousarray(samples.normals)).to(device=device, dtype=torch.float32)
    targets = torch.from_numpy(np.ascontiguousarray(samples.irradiance)).to(device=device, dtype=torch.float32)

    inputs, position_min, position_extent = encode_network_inputs(positions, directions, config)

    return EncodedTrainingData(
        inputs=inputs,
        targets=targets,
        position_min=position_min.detach().cpu(),
        position_extent=position_extent.detach().cpu(),
    )


def relative_squared_irradiance_loss(predicted: torch.Tensor, target: torch.Tensor, epsilon: float = 0.01) -> torch.Tensor:
    denominator = predicted.square().detach() + epsilon
    return ((predicted - target).square() / denominator).mean()


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def write_falcor_artifacts(checkpoint_path: Path, sample_path: Path, artifact_config: FalcorArtifactConfig) -> tuple[Path, Path]:
    try:
        from .create_inference_parity_vectors import create_parity_vectors, write_parity_vectors
        from .export_tiny_mlp import export_checkpoint, load_exported_model
    except ImportError:
        from create_inference_parity_vectors import create_parity_vectors, write_parity_vectors
        from export_tiny_mlp import export_checkpoint, load_exported_model

    model_output_path = artifact_config.model_output_path or checkpoint_path.with_suffix(".falcor-mlp.bin")
    parity_output_path = artifact_config.parity_output_path or model_output_path.with_suffix(".parity.bin")

    export_checkpoint(checkpoint_path, model_output_path)
    exported_model = load_exported_model(model_output_path)
    vectors, max_abs_error = create_parity_vectors(
        checkpoint_path=checkpoint_path,
        exported_model_path=model_output_path,
        sample_path=sample_path,
        vector_count=artifact_config.parity_count,
        seed=artifact_config.parity_seed,
        tolerance=artifact_config.parity_tolerance,
    )
    write_parity_vectors(
        parity_output_path,
        vectors,
        position_frequencies=exported_model.position_frequencies,
        direction_bins=exported_model.direction_bins,
    )

    print(f"Saved Falcor model to {model_output_path}")
    print(f"Saved Falcor parity vectors to {parity_output_path}")
    print(f"Falcor parity vectors: {vectors.vector_count}")
    print(f"Max checkpoint/export parity error: {max_abs_error:.8g}")
    return model_output_path, parity_output_path


def train(
    config: TrainingConfig,
    sample_path: Path,
    iterations: int,
    device: torch.device,
    checkpoint_path: Path | None,
    log_interval: int,
    falcor_artifacts: FalcorArtifactConfig | None = None,
) -> TinyIrradianceMLP:
    data = load_training_data(sample_path, device=device, config=config)

    model = TinyIrradianceMLP(
        input_dim=config.input_dim,
        hidden_width=config.hidden_width,
        layer_count=config.layer_count,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    sample_count = data.inputs.shape[0]

    print(f"Loaded {sample_count:,} samples from {sample_path}")
    print(f"Device: {device}")
    print(f"Input dimension: {config.input_dim}")
    print(f"Batch size: {config.batch_size:,}")

    model.train()
    for iteration in range(1, iterations + 1):
        if iteration == config.lr_drop_iteration + 1:
            set_learning_rate(optimizer, config.reduced_learning_rate)

        indices = torch.randint(sample_count, (config.batch_size,), device=device)
        batch_inputs = data.inputs[indices]
        batch_targets = data.targets[indices]

        optimizer.zero_grad(set_to_none=True)
        predicted = model(batch_inputs)
        loss = relative_squared_irradiance_loss(predicted, batch_targets, epsilon=config.loss_epsilon)
        loss.backward()
        optimizer.step()

        if iteration == 1 or iteration % log_interval == 0 or iteration == iterations:
            with torch.no_grad():
                mean_abs_error = (predicted - batch_targets).abs().mean().item()
                lr = optimizer.param_groups[0]["lr"]
            print(f"iter {iteration:>7} | loss {loss.item():.6f} | mae {mean_abs_error:.6f} | lr {lr:.2e}")

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "position_min": data.position_min,
                "position_extent": data.position_extent,
                "sample_path": str(sample_path),
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")
        if falcor_artifacts is not None:
            write_falcor_artifacts(checkpoint_path, sample_path, falcor_artifacts)

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny MLP to predict RGB irradiance from position and direction.")
    parser.add_argument("sample_path", type=Path, help="Path to a v2 .irradiance-samples.bin file.")
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=1 << 16)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--reduced-learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-drop-iteration", type=int, default=10_000)
    parser.add_argument("--loss-epsilon", type=float, default=0.01)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional output .pt checkpoint path.")
    parser.add_argument(
        "--skip-falcor-artifacts",
        action="store_true",
        help="Only save the PyTorch checkpoint. By default, --checkpoint also writes .falcor-mlp.bin and .falcor-mlp.parity.bin.",
    )
    parser.add_argument("--falcor-model-output", type=Path, default=None, help="Optional output path for the exported .falcor-mlp.bin file.")
    parser.add_argument("--falcor-parity-output", type=Path, default=None, help="Optional output path for the generated .falcor-mlp.parity.bin file.")
    parser.add_argument("--falcor-parity-count", type=int, default=64, help="Number of parity vectors to generate after checkpoint export.")
    parser.add_argument("--falcor-parity-seed", type=int, default=None, help="Random seed for parity-vector sample selection. Defaults to --seed.")
    parser.add_argument("--falcor-parity-tolerance", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = TrainingConfig(
        hidden_width=args.hidden_width,
        learning_rate=args.learning_rate,
        reduced_learning_rate=args.reduced_learning_rate,
        lr_drop_iteration=args.lr_drop_iteration,
        batch_size=args.batch_size,
        loss_epsilon=args.loss_epsilon,
    )

    train(
        config=config,
        sample_path=args.sample_path,
        iterations=args.iterations,
        device=torch.device(args.device),
        checkpoint_path=args.checkpoint,
        log_interval=max(1, args.log_interval),
        falcor_artifacts=None
        if args.checkpoint is None or args.skip_falcor_artifacts
        else FalcorArtifactConfig(
            model_output_path=args.falcor_model_output,
            parity_output_path=args.falcor_parity_output,
            parity_count=args.falcor_parity_count,
            parity_seed=args.seed if args.falcor_parity_seed is None else args.falcor_parity_seed,
            parity_tolerance=args.falcor_parity_tolerance,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

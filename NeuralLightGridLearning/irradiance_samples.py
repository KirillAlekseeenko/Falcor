from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

SURFACE_SAMPLE_FLAG = 1
IRRADIANCE_SAMPLE_MAGIC = 0x31534249

_HEADER_DTYPE = np.dtype(
    [
        ("magic", "<u4"),
        ("version", "<u4"),
        ("sample_count", "<u4"),
        ("surface_sample_count", "<u4"),
        ("surface_sample_ratio", "<f4"),
        ("backface_threshold", "<f4"),
        ("surface_offset", "<f4"),
        ("reserved", "<u4"),
    ],
    align=False,
)

_SAMPLE_V2_DTYPE = np.dtype(
    [
        ("position", "<f4", (4,)),
        ("normal", "<f4", (4,)),
        ("irradiance", "<f4", (4,)),
        ("meta", "<u4", (4,)),
    ],
    align=False,
)


@dataclass(frozen=True)
class IrradianceSampleHeader:
    magic: int
    version: int
    sample_count: int
    surface_sample_count: int
    surface_sample_ratio: float
    backface_threshold: float
    surface_offset: float
    reserved: int

    @classmethod
    def from_record(cls, record: np.void) -> "IrradianceSampleHeader":
        return cls(
            magic=int(record["magic"]),
            version=int(record["version"]),
            sample_count=int(record["sample_count"]),
            surface_sample_count=int(record["surface_sample_count"]),
            surface_sample_ratio=float(record["surface_sample_ratio"]),
            backface_threshold=float(record["backface_threshold"]),
            surface_offset=float(record["surface_offset"]),
            reserved=int(record["reserved"]),
        )


@dataclass(frozen=True)
class IrradianceSampleSet:
    source_path: Path
    header: IrradianceSampleHeader
    positions: np.ndarray
    normals: np.ndarray
    irradiance: np.ndarray
    meta: np.ndarray

    def __post_init__(self) -> None:
        count = self.header.sample_count
        arrays = {
            "positions": self.positions,
            "normals": self.normals,
            "irradiance": self.irradiance,
            "meta": self.meta,
        }
        for name, value in arrays.items():
            if value.shape[0] != count:
                raise ValueError(f"{name} contains {value.shape[0]} rows, expected {count}.")

    @property
    def sample_count(self) -> int:
        return self.header.sample_count

    @property
    def surface_mask(self) -> np.ndarray:
        return (self.meta[:, 0] & SURFACE_SAMPLE_FLAG) != 0

    @property
    def volume_mask(self) -> np.ndarray:
        return ~self.surface_mask

    @property
    def hit_count(self) -> np.ndarray:
        return self.meta[:, 1]

    @property
    def backface_count(self) -> np.ndarray:
        return self.meta[:, 2]

    @property
    def backface_ratio(self) -> np.ndarray:
        hit_count = self.hit_count.astype(np.float32)
        backface_count = self.backface_count.astype(np.float32)
        return np.divide(backface_count, hit_count, out=np.zeros_like(hit_count), where=hit_count > 0)

    @property
    def irradiance_luminance(self) -> np.ndarray:
        return self.irradiance @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    def choose_indices(self, max_points: int | None = 50_000, seed: int = 1) -> np.ndarray:
        if max_points is None or self.sample_count <= max_points:
            return np.arange(self.sample_count)

        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(self.sample_count, size=max_points, replace=False))

    def summary(self) -> dict[str, Any]:
        surface_count = int(np.count_nonzero(self.surface_mask))
        volume_count = self.sample_count - surface_count

        if self.sample_count == 0:
            position_min = position_max = irradiance_min = irradiance_max = [0.0, 0.0, 0.0]
            irradiance_mean = [0.0, 0.0, 0.0]
            luminance_min = luminance_max = luminance_mean = 0.0
        else:
            position_min = self.positions.min(axis=0).tolist()
            position_max = self.positions.max(axis=0).tolist()
            irradiance_min = self.irradiance.min(axis=0).tolist()
            irradiance_max = self.irradiance.max(axis=0).tolist()
            irradiance_mean = self.irradiance.mean(axis=0).tolist()
            luminance = self.irradiance_luminance
            luminance_min = float(luminance.min())
            luminance_max = float(luminance.max())
            luminance_mean = float(luminance.mean())

        return {
            "path": str(self.source_path),
            "version": self.header.version,
            "sample_count": self.sample_count,
            "surface_count": surface_count,
            "volume_count": volume_count,
            "header_surface_count": self.header.surface_sample_count,
            "surface_sample_ratio": self.header.surface_sample_ratio,
            "backface_threshold": self.header.backface_threshold,
            "surface_offset": self.header.surface_offset,
            "position_min": position_min,
            "position_max": position_max,
            "irradiance_min": irradiance_min,
            "irradiance_max": irradiance_max,
            "irradiance_mean": irradiance_mean,
            "luminance_min": luminance_min,
            "luminance_max": luminance_max,
            "luminance_mean": luminance_mean,
        }


def load_irradiance_samples(path: str | Path) -> IrradianceSampleSet:
    path = Path(path)
    file_size = path.stat().st_size

    with path.open("rb") as stream:
        header_records = np.fromfile(stream, dtype=_HEADER_DTYPE, count=1)
        if header_records.size != 1:
            raise ValueError(f"'{path}' is too small to contain an irradiance sample header.")

        header = IrradianceSampleHeader.from_record(header_records[0])
        if header.magic != IRRADIANCE_SAMPLE_MAGIC:
            raise ValueError(f"'{path}' has invalid magic 0x{header.magic:08X}; expected 0x{IRRADIANCE_SAMPLE_MAGIC:08X}.")

        if header.version != 2:
            raise ValueError(f"'{path}' has unsupported irradiance sample version {header.version}; expected v2 with irradiance data.")

        expected_size = _HEADER_DTYPE.itemsize + header.sample_count * _SAMPLE_V2_DTYPE.itemsize
        if file_size < expected_size:
            raise ValueError(f"'{path}' is truncated: expected at least {expected_size} bytes, found {file_size}.")

        records = np.fromfile(stream, dtype=_SAMPLE_V2_DTYPE, count=header.sample_count)
        if records.size != header.sample_count:
            raise ValueError(f"Failed to read {header.sample_count} samples from '{path}'.")

    positions = np.ascontiguousarray(records["position"][:, :3], dtype=np.float32)
    normals = np.ascontiguousarray(records["normal"][:, :3], dtype=np.float32)
    meta = np.ascontiguousarray(records["meta"], dtype=np.uint32)

    irradiance = np.ascontiguousarray(records["irradiance"][:, :3], dtype=np.float32)

    return IrradianceSampleSet(
        source_path=path,
        header=header,
        positions=positions,
        normals=normals,
        irradiance=irradiance,
        meta=meta,
    )


def print_summary(samples: IrradianceSampleSet) -> None:
    summary = samples.summary()
    for key, value in summary.items():
        print(f"{key}: {value}")


def _set_axes_equal(ax: Any, points: np.ndarray) -> None:
    if points.size == 0:
        return

    center = points.mean(axis=0)
    radius = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), float(np.ptp(points[:, 2])), 1e-6) * 0.5
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_sample_positions(
    samples: IrradianceSampleSet,
    *,
    max_points: int | None = 50_000,
    color_by: Literal["irradiance", "surface", "backface"] = "irradiance",
    seed: int = 1,
    figsize: tuple[float, float] = (7.0, 6.0),
    point_size: float = 1.0,
    alpha: float = 0.65,
    ax: Any | None = None,
) -> Any:
    import matplotlib.pyplot as plt

    indices = samples.choose_indices(max_points=max_points, seed=seed)
    positions = samples.positions[indices]

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)

    if color_by == "surface":
        colors = np.where(samples.surface_mask[indices], "tab:orange", "tab:cyan")
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=point_size, alpha=alpha, linewidths=0)
    elif color_by == "backface":
        values = samples.backface_ratio[indices]
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=values,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            cmap="viridis",
            vmin=0.0,
            vmax=max(float(samples.header.backface_threshold), 1e-6),
        )
        plt.colorbar(scatter, ax=ax, shrink=0.65, label="Backface ratio")
    elif color_by == "irradiance":
        values = samples.irradiance_luminance[indices]
        positive = values[values > 0]
        vmax = float(np.percentile(positive, 99.0)) if positive.size else 1.0
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=values,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            cmap="inferno",
            vmin=0.0,
            vmax=max(vmax, 1e-6),
        )
        plt.colorbar(scatter, ax=ax, shrink=0.65, label="Irradiance luminance")
    else:
        raise ValueError(f"Unsupported color_by mode '{color_by}'.")

    ax.set_title(f"{len(indices):,} displayed / {samples.sample_count:,} samples")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_axes_equal(ax, positions)
    return ax


def plot_irradiance_histogram(samples: IrradianceSampleSet, *, ax: Any | None = None, bins: int = 100) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    luminance = samples.irradiance_luminance
    if luminance.size > 0:
        upper = float(np.percentile(luminance, 99.5))
        luminance = luminance[luminance <= upper]

    ax.hist(luminance, bins=bins, color="tab:blue", alpha=0.85)
    ax.set_title("Irradiance luminance")
    ax.set_xlabel("Luminance")
    ax.set_ylabel("Samples")
    return ax


def plot_backface_histogram(samples: IrradianceSampleSet, *, ax: Any | None = None, bins: int = 64) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.hist(samples.backface_ratio, bins=bins, color="tab:green", alpha=0.85)
    ax.axvline(samples.header.backface_threshold, color="tab:red", linestyle="--", linewidth=1.5)
    ax.set_title("Backface validation")
    ax.set_xlabel("Backface ratio")
    ax.set_ylabel("Samples")
    return ax


def plot_samples_overview(
    samples: IrradianceSampleSet,
    *,
    max_points: int | None = 50_000,
    seed: int = 1,
    color_by: Literal["irradiance", "surface", "backface"] = "irradiance",
) -> Any:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 4.8))
    ax_positions = fig.add_subplot(1, 3, 1, projection="3d")
    ax_irradiance = fig.add_subplot(1, 3, 2)
    ax_backfaces = fig.add_subplot(1, 3, 3)

    plot_sample_positions(samples, max_points=max_points, color_by=color_by, seed=seed, ax=ax_positions)
    plot_irradiance_histogram(samples, ax=ax_irradiance)
    plot_backface_histogram(samples, ax=ax_backfaces)

    fig.tight_layout()
    return fig

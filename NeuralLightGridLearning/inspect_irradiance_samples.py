from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .irradiance_samples import load_irradiance_samples, plot_samples_overview, print_summary
except ImportError:
    from irradiance_samples import load_irradiance_samples, plot_samples_overview, print_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a Falcor irradiance sample bake.")
    parser.add_argument("path", type=Path, help="Path to a .irradiance-samples.bin file.")
    parser.add_argument("--max-points", type=int, default=50_000, help="Maximum points to show in the scatter plot.")
    parser.add_argument("--color-by", choices=["irradiance", "surface", "backface"], default="irradiance")
    parser.add_argument("--save-plot", type=Path, default=None, help="Optional path for a PNG overview plot.")
    parser.add_argument("--no-plot", action="store_true", help="Only print the parsed file summary.")
    args = parser.parse_args()

    samples = load_irradiance_samples(args.path)
    print_summary(samples)

    if not args.no_plot:
        import matplotlib.pyplot as plt

        fig = plot_samples_overview(samples, max_points=args.max_points, color_by=args.color_by)
        if args.save_plot is not None:
            args.save_plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save_plot, dpi=180)
        else:
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

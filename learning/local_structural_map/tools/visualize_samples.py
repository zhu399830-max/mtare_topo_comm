import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from learning.local_structural_map.tools.io_utils import load_map_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sample in args.samples:
        data = load_map_npz(sample)
        tensor = data["tensor"]
        names = data["channel_names"]
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
        for i, ax in enumerate(axes.flat):
            if i < len(names):
                ax.imshow(tensor[i], cmap="viridis", vmin=0.0, vmax=1.0, origin="upper")
                ax.set_title(names[i], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{data['source']} {Path(sample).name}")
        fig.savefig(out / f"{Path(sample).stem}.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()

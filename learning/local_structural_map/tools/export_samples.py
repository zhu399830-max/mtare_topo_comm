import argparse
from pathlib import Path

from learning.local_structural_map.builder import LocalStructuralMapBuilder
from learning.local_structural_map.config import LocalMapConfig
from learning.local_structural_map.datasets.lamp import LAMPDatasetAdapter
from learning.local_structural_map.datasets.mtare_bag import MTAREBagAdapter
from learning.local_structural_map.tools.io_utils import sample_stats, save_map_npz, write_json


def export_source(name: str, frames, out_dir: Path, max_samples: int) -> dict:
    builder = LocalStructuralMapBuilder(LocalMapConfig())
    saved = []
    for idx, frame in enumerate(frames):
        item = builder.update(frame)
        if item.tensor.shape != (len(item.channel_names), 100, 100):
            raise RuntimeError(f"bad tensor shape {item.tensor.shape}")
        path = out_dir / "samples" / f"{name}_{idx:04d}.npz"
        save_map_npz(path, item)
        saved.append(path)
        if len(saved) >= max_samples:
            break
    stats = sample_stats(saved)
    stats["files"] = [str(p) for p in saved]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", choices=["lamp", "mtare"], required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--lamp-bag", default="/mnt/nas_znfy/Shared-2/dataset/SLAM/subt/LAMP/tunnel/rosbag/husky3.bag")
    parser.add_argument("--mtare-bag", default="")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    if args.source == "lamp":
        frames = LAMPDatasetAdapter(args.lamp_bag, max_samples=args.max_samples, stride=args.stride)
    else:
        if not args.mtare_bag:
            raise SystemExit("--mtare-bag is required for M-TARE export")
        frames = MTAREBagAdapter(args.mtare_bag, max_samples=args.max_samples, stride=args.stride)
    stats = export_source(args.source, frames, out_dir, args.max_samples)
    write_json(out_dir / f"{args.source}_export_stats.json", stats)


if __name__ == "__main__":
    main()

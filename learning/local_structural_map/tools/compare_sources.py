import argparse
import glob
from pathlib import Path

from learning.local_structural_map.config import LocalMapConfig
from learning.local_structural_map.tools.io_utils import sample_stats, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--lamp-glob", required=True)
    parser.add_argument("--mtare-glob", required=True)
    args = parser.parse_args()
    lamp = [Path(p) for p in sorted(glob.glob(args.lamp_glob))]
    mtare = [Path(p) for p in sorted(glob.glob(args.mtare_glob))]
    data = {
        "contract": LocalMapConfig().to_dict(),
        "lamp": sample_stats(lamp),
        "mtare": sample_stats(mtare),
    }
    write_json(args.output, data)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from learning.local_structural_map.builder import LocalStructuralMapBuilder
from learning.local_structural_map.config import LocalMapConfig
from learning.local_structural_map.datasets.mtare_bag import MTAREBagAdapter
from learning.local_structural_map.runtime.mtare_runtime import MTARERuntimeAdapter
from learning.local_structural_map.tools.io_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()
    cfg = LocalMapConfig()
    offline_builder = LocalStructuralMapBuilder(cfg)
    offline = []
    for frame in MTAREBagAdapter(args.bag, max_samples=args.max_samples):
        offline.append(offline_builder.update(frame))
    runtime_builder = LocalStructuralMapBuilder(cfg)
    runtime = MTARERuntimeAdapter(runtime_builder)
    online = []
    with AnyReader([Path(args.bag)]) as reader:
        conns = [c for c in reader.connections if c.topic in ["/registered_scan", "/state_estimation_at_scan"]]
        for conn, timestamp_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == "/state_estimation_at_scan":
                runtime.handle_odom(msg, int(timestamp_ns))
            else:
                item = runtime.handle_scan(msg, int(timestamp_ns))
                if item is not None:
                    online.append(item)
                    if len(online) >= args.max_samples:
                        break
    n = min(len(offline), len(online))
    if n == 0:
        result = {"status": "FAILED", "reason": "no comparable samples", "offline_count": len(offline), "online_count": len(online)}
    else:
        diffs = np.stack([np.abs(offline[i].tensor - online[i].tensor) for i in range(n)], axis=0)
        names = offline[0].channel_names
        result = {
            "status": "PASS" if float(np.max(diffs)) == 0.0 else "DIFF",
            "count": n,
            "channel_names": names,
            "per_channel_max_abs_error": {name: float(np.max(diffs[:, i])) for i, name in enumerate(names)},
            "per_channel_mean_abs_error": {name: float(np.mean(diffs[:, i])) for i, name in enumerate(names)},
            "mask_mismatch_pixels": {},
        }
        result["mask_mismatch_pixels"] = {}
        for i, name in enumerate(names):
            if "mask" in name or "occupancy" in name:
                result["mask_mismatch_pixels"][name] = int(
                    sum(np.count_nonzero((offline[j].tensor[i] > 0.5) != (online[j].tensor[i] > 0.5)) for j in range(n))
                )
    write_json(args.output, result)


if __name__ == "__main__":
    main()

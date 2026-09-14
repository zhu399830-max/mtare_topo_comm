"""Diagnostic-only SubT-MRS raw Velodyne packet inspection.

This script is not a training data path. It uses unverified local model
assumptions when official sensor model, calibration and velodyne_pointcloud
conversion are unavailable.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from learning.local_structural_map.tools.io_utils import write_json


HDL32E_VERTICAL_DEG = np.asarray(
    [-30.67, -9.33, -29.33, -8.0, -28.0, -6.66, -26.66, -5.33,
     -25.33, -4.0, -24.0, -2.67, -22.67, -1.33, -21.33, 0.0,
     -20.0, 1.33, -18.67, 2.67, -17.33, 4.0, -16.0, 5.33,
     -14.67, 6.67, -13.33, 8.0, -12.0, 9.33, -10.67, 10.67],
    dtype=np.float32,
)

VLP16_VERTICAL_DEG = np.asarray([-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15], dtype=np.float32)


def decode_packet(data: bytes, model: str) -> np.ndarray:
    points = []
    if model == "hdl32e":
        vertical = np.deg2rad(HDL32E_VERTICAL_DEG)
        for block in range(12):
            off = block * 100
            if data[off : off + 2] != b"\xff\xee":
                continue
            az = int.from_bytes(data[off + 2 : off + 4], "little") / 100.0
            az_rad = math.radians(az)
            for ch in range(32):
                base = off + 4 + ch * 3
                dist = int.from_bytes(data[base : base + 2], "little") * 0.002
                if dist <= 0.1:
                    continue
                v = float(vertical[ch])
                xy = dist * math.cos(v)
                points.append((xy * math.sin(az_rad), xy * math.cos(az_rad), dist * math.sin(v)))
    elif model == "vlp16":
        vertical = np.deg2rad(VLP16_VERTICAL_DEG)
        for block in range(12):
            off = block * 100
            if data[off : off + 2] != b"\xff\xee":
                continue
            az = int.from_bytes(data[off + 2 : off + 4], "little") / 100.0
            next_az = az
            if block < 11:
                next_az = int.from_bytes(data[off + 102 : off + 104], "little") / 100.0
                if next_az < az:
                    next_az += 360.0
            for firing in range(2):
                firing_az = az + firing * (next_az - az) / 2.0
                az_rad = math.radians(firing_az % 360.0)
                for ch in range(16):
                    base = off + 4 + (firing * 16 + ch) * 3
                    dist = int.from_bytes(data[base : base + 2], "little") * 0.002
                    if dist <= 0.1:
                        continue
                    v = float(vertical[ch])
                    xy = dist * math.cos(v)
                    points.append((xy * math.sin(az_rad), xy * math.cos(az_rad), dist * math.sin(v)))
    else:
        raise ValueError(model)
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=["hdl32e", "vlp16"], default="hdl32e")
    parser.add_argument("--max-scans", type=int, default=20)
    args = parser.parse_args()
    scans = []
    first_stamp = None
    last_stamp = None
    frame_id = None
    factory = {}
    with AnyReader([Path(args.bag)]) as reader:
        conns = [c for c in reader.connections if c.topic == "/velodyne_packets"]
        for conn, timestamp_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            frame_id = msg.header.frame_id
            stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            first_stamp = stamp if first_stamp is None else first_stamp
            last_stamp = stamp
            pts = []
            for packet in msg.packets:
                data = bytes(packet.data)
                key = f"0x{data[-2]:02x}_0x{data[-1]:02x}"
                factory[key] = factory.get(key, 0) + 1
                pts.append(decode_packet(data, args.model))
            cloud = np.concatenate(pts, axis=0) if pts else np.empty((0, 3), dtype=np.float32)
            finite = np.isfinite(cloud).all(axis=1)
            scans.append({
                "stamp_ns": stamp,
                "points": int(len(cloud)),
                "finite_points": int(finite.sum()),
                "nan_inf_points": int((~finite).sum()),
                "xyz_min": cloud[finite].min(axis=0).tolist() if finite.any() else None,
                "xyz_max": cloud[finite].max(axis=0).tolist() if finite.any() else None,
            })
            if len(scans) >= args.max_scans:
                break
    duration = (last_stamp - first_stamp) / 1e9 if first_stamp is not None and last_stamp is not None and last_stamp > first_stamp else 0.0
    write_json(args.output, {
        "bag": args.bag,
        "topic": "/velodyne_packets",
        "input_type": "velodyne_msgs/VelodyneScan",
        "decoder": "diagnostic_only; not_for_training; local deterministic packet decoder; sensor model is UNCONFIRMED without dataset calibration",
        "training_use": "not_for_training",
        "model_assumption": args.model,
        "frame_id": frame_id,
        "scans": scans,
        "scan_count": len(scans),
        "duration_sec": duration,
        "estimated_frequency_hz": (len(scans) - 1) / duration if duration > 0 and len(scans) > 1 else None,
        "factory_byte_counts": factory,
        "mean_points": float(np.mean([s["points"] for s in scans])) if scans else 0.0,
    })


if __name__ == "__main__":
    main()

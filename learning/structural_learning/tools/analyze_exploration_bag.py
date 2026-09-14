#!/usr/bin/env python3
"""Compute planner-level exploration metrics from a ROS1 bag.

The 1 Hz trajectory removes high-rate odometry jitter so path efficiency is
comparable between planners.  Visited cells measure spatial coverage of the
robot path, not the number of messages or model-created topology nodes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import rosbag


def observed_surface_metrics(path: Path, start_time: float, limit_sec: float | None,
                             grid_m: float) -> dict[str, float | int | list[float]]:
    """Count registered-scan surface cells using at most one scan per second."""
    from sensor_msgs import point_cloud2

    cells: set[tuple[int, int]] = set()
    last_bucket = -1
    with rosbag.Bag(str(path)) as bag:
        for _, msg, stamp in bag.read_messages(topics=['/registered_scan']):
            t = stamp.to_sec()
            if t < start_time:
                continue
            elapsed = t - start_time
            if limit_sec is not None and elapsed > limit_sec:
                break
            bucket = int(math.floor(elapsed))
            if bucket == last_bucket:
                continue
            last_bucket = bucket
            # Registered scans can be dense.  Deterministic stride sampling is
            # sufficient for a planner-level surface coverage comparison.
            for index, point in enumerate(point_cloud2.read_points(msg, field_names=('x', 'y'), skip_nans=True)):
                if index % 12:
                    continue
                x, y = float(point[0]), float(point[1])
                cells.add((int(math.floor(x / grid_m)), int(math.floor(y / grid_m))))
    if not cells:
        return {'observed_surface_cells': 0, 'observed_surface_grid_m': grid_m}
    xs = [cell[0] for cell in cells]
    ys = [cell[1] for cell in cells]
    return {
        'observed_surface_cells': len(cells), 'observed_surface_grid_m': grid_m,
        'observed_surface_bounds_xy': [min(xs) * grid_m, (max(xs) + 1) * grid_m,
                                       min(ys) * grid_m, (max(ys) + 1) * grid_m],
    }


def metrics(path: Path, limit_sec: float | None, grid_m: float,
            scan_surface_grid_m: float | None = None,
            start_on_motion: bool = False) -> dict[str, float | int | list[float]]:
    all_samples: list[tuple[float, float, float]] = []
    with rosbag.Bag(str(path)) as bag:
        for _, msg, stamp in bag.read_messages(topics=['/state_estimation_at_scan']):
            t = stamp.to_sec()
            p = msg.pose.pose.position
            all_samples.append((t, float(p.x), float(p.y)))
    if not all_samples:
        raise RuntimeError(f'no /state_estimation_at_scan messages in {path}')
    start_index = 0
    if start_on_motion:
        x0, y0 = all_samples[0][1:]
        start_index = next((i for i, (_, x, y) in enumerate(all_samples)
                            if math.hypot(x - x0, y - y0) >= .5), 0)
    start = all_samples[start_index][0]
    samples = [sample for sample in all_samples[start_index:]
               if limit_sec is None or sample[0] - start <= limit_sec]

    # Select the first sample in each elapsed one-second bucket.
    one_hz: list[tuple[float, float, float]] = []
    last_bucket = -1
    t0 = samples[0][0]
    for sample in samples:
        bucket = int(math.floor(sample[0] - t0))
        if bucket != last_bucket:
            one_hz.append(sample)
            last_bucket = bucket
    path_m = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(one_hz, one_hz[1:]))
    sx, sy = one_hz[0][1:]
    net_m = math.hypot(one_hz[-1][1] - sx, one_hz[-1][2] - sy)
    max_radius_m = max(math.hypot(x - sx, y - sy) for _, x, y in one_hz)
    cells = {(int(math.floor(x / grid_m)), int(math.floor(y / grid_m))) for _, x, y in one_hz}
    revisits = max(len(one_hz) - len(cells), 0)
    xs = [x for _, x, _ in one_hz]
    ys = [y for _, _, y in one_hz]
    result = {
        'bag': str(path), 'duration_sec': one_hz[-1][0] - t0,
        'window_start': 'first_motion_0.5m' if start_on_motion else 'first_pose',
        'raw_pose_count': len(samples), 'trajectory_samples_1hz': len(one_hz),
        'path_m_1hz': path_m, 'net_displacement_m': net_m,
        'max_radius_m': max_radius_m, 'path_efficiency_net_over_path': net_m / max(path_m, 1e-9),
        'visited_cells': len(cells), 'visited_grid_m': grid_m,
        'revisit_fraction_1hz': revisits / max(len(one_hz), 1),
        'bounds_xy': [min(xs), max(xs), min(ys), max(ys)],
        'final_xy': [one_hz[-1][1], one_hz[-1][2]],
        'trajectory_1hz_xy': [[sample[1], sample[2]] for sample in one_hz],
    }
    if scan_surface_grid_m is not None:
        result.update(observed_surface_metrics(path, start, limit_sec, scan_surface_grid_m))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', type=Path)
    parser.add_argument('--limit-sec', type=float)
    parser.add_argument('--grid-m', type=float, default=2.0)
    parser.add_argument('--scan-surface-grid-m', type=float,
                        help='also count registered-scan surface cells at this resolution')
    parser.add_argument('--start-on-motion', action='store_true',
                        help='start the time window after the robot first moves 0.5 m')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = metrics(args.bag, args.limit_sec, args.grid_m, args.scan_surface_grid_m,
                     args.start_on_motion)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()

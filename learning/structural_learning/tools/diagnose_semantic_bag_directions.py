#!/usr/bin/env python3
"""Audit semantic direction predictions against actual odometry motion."""
import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch

from learning.local_structural_map.datasets.common import pointcloud2_xyz, pose_from_odometry, stamp_to_ns
from learning.structural_learning.surface_evidence import SurfaceEvidenceBuilder, SurfaceEvidenceConfig
from learning.structural_learning.tools.run_semantic_bottleneck_role import Semantic


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def frames_from_ros1_bag(path: Path, samples: int, stride: int):
    import rosbag
    bag = rosbag.Bag(str(path)); odom = {}
    for _, msg, stamp in bag.read_messages(topics=['/state_estimation_at_scan']):
        odom[stamp_to_ns(msg.header.stamp) or int(stamp.to_nsec())] = msg
    seen = emitted = 0
    for _, msg, stamp in bag.read_messages(topics=['/registered_scan']):
        ns = stamp_to_ns(msg.header.stamp) or int(stamp.to_nsec()); pose_msg = odom.get(ns)
        if pose_msg is None: continue
        seen += 1
        if (seen - 1) % stride: continue
        pose = pose_from_odometry(pose_msg)
        yield ns, pose, pointcloud2_xyz(msg)
        emitted += 1
        if emitted >= samples: break
    bag.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--stride', type=int, default=10)
    args = ap.parse_args()
    model = Semantic()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state'])
    model.eval(); builder = SurfaceEvidenceBuilder(SurfaceEvidenceConfig()); hist = deque(maxlen=8); rows = []
    for timestamp_ns, pose, points_world in frames_from_ros1_bag(args.bag, args.samples, args.stride):
        cur, _ = builder.build([points_world], pose); hist.append(cur[[0, 2, 3]])
        hh = [np.zeros((3, 100, 100), np.float32)] * (8-len(hist)) + list(hist)
        valid = [0.] * (8-len(hist)) + [1.] * len(hist)
        batch = {'cur': torch.from_numpy(cur[[0,2,3]][None]), 'hist': torch.from_numpy(np.stack(hh)[None]),
                 'valid': torch.tensor([valid], dtype=torch.float32)}
        with torch.no_grad(): pred = model(batch)
        direction = pred['direction'][0].numpy(); distance = pred['distance'][0].numpy(); sector = int(np.argmax(direction * distance))
        rows.append({'time':timestamp_ns/1e9,'x':pose.x,'y':pose.y,'yaw':pose.yaw,
                     'sector':sector,'predicted_world_angle':wrap(pose.yaw+sector*2*math.pi/32),
                     'direction_max':float(direction[sector]),'distance':float(distance[sector]),
                     'surface_coverage':float(cur[0].mean())})
    motion = []
    for a,b in zip(rows, rows[1:]):
        dx,dy=b['x']-a['x'],b['y']-a['y']; d=math.hypot(dx,dy)
        if d>.2: motion.append({'distance':d,'actual_world_angle':math.atan2(dy,dx),
                                'prediction_error_rad':abs(wrap(a['predicted_world_angle']-math.atan2(dy,dx)))})
    out={'bag':str(args.bag),'samples':len(rows),'rows':rows,'motion_segments':len(motion),
         'mean_prediction_motion_angle_error_deg':float(np.degrees(np.mean([m['prediction_error_rad'] for m in motion]))) if motion else None,
         'motion_distance':float(sum(m['distance'] for m in motion))}
    print(json.dumps(out, indent=2))


if __name__ == '__main__': main()

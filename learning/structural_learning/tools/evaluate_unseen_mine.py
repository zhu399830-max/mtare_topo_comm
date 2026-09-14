#!/usr/bin/env python3
"""Evaluate and plot trajectory coverage in the frozen unseen_mine world."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEGMENTS = {
    'main_drift': ((0., 0.), (90., 0.)),
    'west_loop_leg': ((25., 0.), (25., 30.)),
    'north_loop_drift': ((25., 30.), (60., 30.)),
    'east_loop_leg': ((60., 30.), (60., 0.)),
    'south_dead_end': ((60., 0.), (60., -35.)),
}


def sample_segment(start: tuple[float, float], end: tuple[float, float]) -> np.ndarray:
    length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    alpha = np.linspace(0., 1., int(np.ceil(length)) + 1)
    return np.column_stack((start[0] + alpha * (end[0] - start[0]),
                            start[1] + alpha * (end[1] - start[1])))


def coverage(trajectory: np.ndarray, radius_m: float) -> tuple[dict[str, float], float]:
    per_segment: dict[str, float] = {}
    all_points = []
    all_covered = []
    for name, (start, end) in SEGMENTS.items():
        points = sample_segment(start, end)
        distance = np.sqrt(((points[:, None, :] - trajectory[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        covered = distance <= radius_m
        per_segment[name] = float(covered.mean())
        all_points.append(points)
        all_covered.append(covered)
    return per_segment, float(np.concatenate(all_covered).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--semantic', type=Path, required=True)
    parser.add_argument('--tare', type=Path, required=True)
    parser.add_argument('--topology', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--visit-radius-m', type=float, default=4.)
    args = parser.parse_args()
    rows = {'semantic_topology': json.loads(args.semantic.read_text()),
            'original_tare': json.loads(args.tare.read_text())}
    trajectories = {name: np.asarray(row['trajectory_1hz_xy'], dtype=np.float32)
                    for name, row in rows.items()}
    report = {'protocol': {'time_window_sec': 180, 'window_start': 'first_motion_0.5m',
                           'centerline_visit_radius_m': args.visit_radius_m}, 'methods': {}}
    for name, trajectory in trajectories.items():
        segments, overall = coverage(trajectory, args.visit_radius_m)
        report['methods'][name] = {
            'centerline_coverage': overall, 'segment_coverage': segments,
            'path_m_1hz': rows[name]['path_m_1hz'],
            'visited_2m_cells': rows[name]['visited_cells'],
            'revisit_fraction_1hz': rows[name]['revisit_fraction_1hz'],
            'observed_surface_cells_1m': rows[name]['observed_surface_cells'],
        }

    fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    for name, (start, end) in SEGMENTS.items():
        points = sample_segment(start, end)
        ax.plot(points[:, 0], points[:, 1], '--', color='.72', lw=5, solid_capstyle='round')
        midpoint = points[len(points)//2]
        ax.text(midpoint[0], midpoint[1], name.replace('_', ' '), fontsize=8, color='.3')
    colors = {'semantic_topology': '#168c8c', 'original_tare': '#7a4ba3'}
    for name, trajectory in trajectories.items():
        ax.plot(trajectory[:, 0], trajectory[:, 1], lw=2.2, color=colors[name], label=name.replace('_', ' '))
        ax.scatter(trajectory[0, 0], trajectory[0, 1], marker='s', s=45, color=colors[name])
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='*', s=90, color=colors[name])
    if args.topology:
        topo = json.loads(args.topology.read_text())['topology']['nodes']
        xy = np.asarray([[node['x'], node['y']] for node in topo])
        ax.scatter(xy[:, 0], xy[:, 1], s=22, facecolor='white', edgecolor=colors['semantic_topology'],
                   linewidth=.9, label='semantic nodes')
    ax.set_aspect('equal'); ax.set_xlabel('world x (m)'); ax.set_ylabel('world y (m)')
    ax.set_title('Frozen-model unseen mine exploration: first 180 s of motion')
    ax.grid(alpha=.2); ax.legend(loc='best')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / 'unseen_mine_trajectory_comparison.png', dpi=220, bbox_inches='tight')
    fig.savefig(args.output_dir / 'unseen_mine_trajectory_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    (args.output_dir / 'unseen_mine_180s_comparison.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

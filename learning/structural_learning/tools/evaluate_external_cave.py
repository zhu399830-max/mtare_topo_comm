#!/usr/bin/env python3
"""Create reproducible figures for the frozen external Cave World comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_vertices(obj: Path, stride: int = 12) -> np.ndarray:
    vertices = []
    with obj.open(encoding='utf-8', errors='ignore') as stream:
        for index, line in enumerate(stream):
            if not line.startswith('v '):
                continue
            if index % stride:
                continue
            fields = line.split()
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
    xyz = np.asarray(vertices, dtype=np.float32)
    # cave_world.world: model pose=(5.66804,-21.9271,0), link roll=pi/2.
    return np.column_stack((xyz[:, 0] + 5.66804, -xyz[:, 2] - 21.9271))


def coverage_curve(xy: np.ndarray, grid_m: float = 2.) -> np.ndarray:
    cells: set[tuple[int, int]] = set()
    values = []
    for x, y in xy:
        cells.add((math.floor(float(x) / grid_m), math.floor(float(y) / grid_m)))
        values.append(len(cells))
    return np.asarray(values)


def radius_curve(xy: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(xy - xy[0], axis=1)
    return np.maximum.accumulate(distance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--semantic', type=Path, required=True)
    parser.add_argument('--tare', type=Path, required=True)
    parser.add_argument('--topology', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = {
        'Semantic topology': json.loads(args.semantic.read_text()),
        'TARE': json.loads(args.tare.read_text()),
    }
    trajectories = {name: np.asarray(row['trajectory_1hz_xy'], dtype=np.float32)
                    for name, row in rows.items()}
    colors = {'Semantic topology': '#008b8b', 'TARE': '#7b4ab0'}
    cave_xy = load_vertices(args.mesh)
    topology = json.loads(args.topology.read_text())['topology']
    node_xy = np.asarray([[node['x'], node['y']] for node in topology['nodes']])

    fig, ax = plt.subplots(figsize=(9.0, 8.2), constrained_layout=True)
    ax.scatter(cave_xy[:, 0], cave_xy[:, 1], s=.08, c='#c7c7c7', alpha=.3,
               linewidths=0, rasterized=True, label='external cave mesh')
    for name, xy in trajectories.items():
        ax.plot(xy[:, 0], xy[:, 1], lw=2., color=colors[name], label=name)
        ax.scatter(*xy[0], marker='s', s=55, color=colors[name], edgecolor='white', zorder=4)
        ax.scatter(*xy[-1], marker='*', s=100, color=colors[name], edgecolor='white', zorder=4)
    for edge in topology['edges']:
        a, b = node_xy[edge['from']], node_xy[edge['to']]
        ax.plot([a[0], b[0]], [a[1], b[1]], color='#006f6f', lw=.65, alpha=.55)
    ax.scatter(node_xy[:, 0], node_xy[:, 1], s=16, facecolor='white', edgecolor='#006f6f',
               linewidth=.8, zorder=3, label='semantic topology nodes')
    ax.set(title='Frozen-model exploration in an unseen external cave (600 s)',
           xlabel='world x (m)', ylabel='world y (m)')
    ax.set_aspect('equal'); ax.grid(alpha=.18); ax.legend(loc='best')
    fig.savefig(args.output_dir / 'external_cave_trajectory_topology.png', dpi=240, bbox_inches='tight')
    fig.savefig(args.output_dir / 'external_cave_trajectory_topology.pdf', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)
    for name, xy in trajectories.items():
        seconds = np.arange(len(xy))
        axes[0].plot(seconds, coverage_curve(xy), color=colors[name], lw=2, label=name)
        axes[1].plot(seconds, radius_curve(xy), color=colors[name], lw=2, label=name)
    axes[0].set(title='Unique trajectory coverage', xlabel='motion time (s)', ylabel='visited 2 m cells')
    axes[1].set(title='Exploration reach', xlabel='motion time (s)', ylabel='maximum radius (m)')
    metrics = ['visited_cells', 'revisit_fraction_1hz', 'observed_surface_cells']
    labels = ['visited cells', 'revisit fraction', 'observed surface\ncells / 10']
    values = {name: [row[metrics[0]], row[metrics[1]], row[metrics[2]] / 10]
              for name, row in rows.items()}
    x = np.arange(3); width = .36
    for index, (name, value) in enumerate(values.items()):
        axes[2].bar(x + (index - .5) * width, value, width, color=colors[name], label=name)
    axes[2].set(title='End-of-run metrics', xticks=x, xticklabels=labels)
    axes[2].set_yscale('symlog', linthresh=1.)
    for ax in axes:
        ax.grid(axis='y', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(loc='upper left'); axes[2].legend(loc='upper left', fontsize=8)
    fig.savefig(args.output_dir / 'external_cave_efficiency_curves.png', dpi=240, bbox_inches='tight')
    fig.savefig(args.output_dir / 'external_cave_efficiency_curves.pdf', bbox_inches='tight')
    plt.close(fig)

    semantic, tare = rows['Semantic topology'], rows['TARE']
    report = {
        'protocol': {
            'world': 'LTU-RAI gazebo_cave_world', 'training_use': 'forbidden',
            'window_start': 'first_motion_0.5m', 'nominal_budget_sec': 600,
            'same_spawn_xy': [10., -21.], 'robot_count': 1,
        },
        'semantic_topology': {key: semantic[key] for key in semantic if key != 'trajectory_1hz_xy'},
        'tare': {key: tare[key] for key in tare if key != 'trajectory_1hz_xy'},
        'ratios_tare_over_semantic': {
            'visited_cells': tare['visited_cells'] / semantic['visited_cells'],
            'maximum_radius': tare['max_radius_m'] / semantic['max_radius_m'],
            'observed_surface_cells': tare['observed_surface_cells'] / semantic['observed_surface_cells'],
        },
        'semantic_topology_graph': {
            'nodes': topology['node_count'], 'edges': topology['edge_count'],
            'edges_per_node': topology['edge_count'] / topology['node_count'],
        },
        'conclusion': 'External cave generalization executes, but semantic frontier identity/data association causes severe revisiting and lower exploration efficiency than TARE.',
    }
    (args.output_dir / 'comparison_summary.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

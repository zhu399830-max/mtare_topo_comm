#!/usr/bin/env python3
"""Render a semantic-topology snapshot into a human-readable top-down PNG."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('snapshot', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.snapshot.read_text())['topology']
    nodes, edges = data['nodes'], data['edges']
    if not nodes:
        raise SystemExit('snapshot has no topology nodes')
    xy = np.array([(n['x'], n['y']) for n in nodes], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    map_path = args.snapshot.parent / 'accumulated_map_xy.npy'
    if map_path.exists():
        map_xy = np.load(map_path)
        if len(map_xy):
            ax.scatter(map_xy[:, 0], map_xy[:, 1], s=.35, color='#b6b6b6', alpha=.38,
                       linewidths=0, rasterized=True, label='accumulated surface map', zorder=0)
    for edge in edges:
        a, b = xy[edge['from']], xy[edge['to']]
        ax.plot([a[0], b[0]], [a[1], b[1]], color='#54708c', linewidth=1.4, zorder=1)
    ax.plot(xy[:, 0], xy[:, 1], color='#b7c6d4', linewidth=0.8, alpha=.8, zorder=1)
    pts = ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(len(nodes)), cmap='viridis', s=32, zorder=2, label='topology node')
    for i, node in enumerate(nodes):
        ax.text(node['x'], node['y'], str(i), fontsize=6, ha='left', va='bottom')
    cur = int(data['current_node'])
    n = nodes[cur]
    ax.scatter([n['x']], [n['y']], s=150, marker='*', color='crimson', edgecolors='black', zorder=4, label='current node')
    ax.arrow(n['x'], n['y'], 2*np.cos(n['yaw']), 2*np.sin(n['yaw']), width=.08,
             head_width=.5, color='crimson', length_includes_head=True, zorder=4)
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=90, marker='s', color='white', edgecolors='black', zorder=3, label='start')
    fig.colorbar(pts, ax=ax, label='node creation order')
    ax.set(title=f"Semantic topology: {len(nodes)} nodes, {len(edges)} traversed edges", xlabel='world x (m)', ylabel='world y (m)')
    ax.axis('equal'); ax.grid(alpha=.25); ax.legend(loc='best')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Render a compact paper-style comparison from bag metric JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', action='append', type=Path, required=True)
    parser.add_argument('--label', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', default='Exploration comparison (1 Hz trajectory metrics)')
    args = parser.parse_args()
    if len(args.metric) != len(args.label):
        raise SystemExit('--metric and --label counts must match')
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in args.metric]
    panels = [
        ('max_radius_m', 'Maximum exploration radius', 'm', False),
        ('visited_cells', 'Visited 2 m cells', 'cells', False),
        ('path_efficiency_net_over_path', 'Path efficiency', 'net / path', False),
        ('revisit_fraction_1hz', 'Trajectory revisit fraction', 'fraction (lower is better)', True),
    ]
    colors = plt.cm.viridis(np.linspace(.18, .82, len(rows)))
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.45), constrained_layout=True)
    for ax, (key, title, ylabel, lower) in zip(axes, panels):
        values = [float(row[key]) for row in rows]
        bars = ax.bar(np.arange(len(rows)), values, color=colors, width=.72)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(np.arange(len(rows)), args.label, rotation=20, ha='right', fontsize=8.5)
        ax.grid(axis='y', alpha=.22)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0., max(values) * 1.22 + 1e-6)
        for bar, value in zip(bars, values):
            label = f'{value:.2f}' if value < 2 else f'{value:.0f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label,
                    ha='center', va='bottom', fontsize=8)
        if lower:
            ax.text(.98, .97, 'lower is better', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7.5, color='.35')
    fig.suptitle(args.title, fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()

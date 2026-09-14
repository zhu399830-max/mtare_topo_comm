"""Plot the sealed current original-system baseline; never edits run evidence."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/gate6_single_robot/gate6_20260913_gse_baseline_restart_v1r1_seed11'
OUT = ROOT / 'docs/figures/gse_baseline_restart_v1r1'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert json.loads((RUN / 'RUN_STATE.json').read_text())['state'] == 'COMPLETED'
    summary = RUN / 'metrics/summary.json'
    evidence = RUN / 'artifacts/cases/tunnel_seed11_original_mtare/evidence'
    trajectory = evidence / 'trajectory.jsonl'
    coverage = evidence / 'coverage_curve.jsonl'
    seal = dict(line.split('  ', 1)[::-1] for line in (RUN / 'artifacts/evidence_sha256.txt').read_text().splitlines())
    sources = (summary, trajectory, coverage)
    for source in sources:
        assert digest(source) == seal[str(source.relative_to(ROOT))]
    poses = [json.loads(line) for line in trajectory.read_text().splitlines()]
    curve = [json.loads(line) for line in coverage.read_text().splitlines()]
    assert poses and curve
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/gse-baseline-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    p = np.array([row['xyz_m'] for row in poses])
    t = np.array([row['elapsed_sec'] for row in curve])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, dimension, label in ((axes[0, 0], 1, 'Y'), (axes[0, 1], 2, 'Z')):
        ax.plot(p[:, 0], p[:, dimension], lw=1, color='steelblue')
        ax.scatter(p[0, 0], p[0, dimension], marker='o', color='green', label='Start')
        ax.scatter(p[-1, 0], p[-1, dimension], marker='s', color='red', label='End')
        ax.set(xlabel='X (m)', ylabel=label + ' (m)', title='Recorded robot trajectory')
        ax.legend()
    axes[1, 0].plot(t, [r['explored_volume_m3'] for r in curve])
    axes[1, 0].set(xlabel='Simulation time (s)', ylabel='Observed voxel volume (m3)', title='Scan-derived coverage proxy, NOT coverage percentage')
    axes[1, 1].plot(t, [r['traveling_distance_m'] for r in curve])
    axes[1, 1].set(xlabel='Simulation time (s)', ylabel='Cumulative path length (m)', title='Measured travel')
    for ax in axes.flat:
        ax.grid(alpha=.25)
    fig.suptitle('Original M-TARE | tunnel / seed 11 / one robot\nDevelopment baseline only; no learned-method comparison')
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=False)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / ('baseline_execution.' + suffix), dpi=160)
    plt.close(fig)
    provenance = dict(source_run=str(RUN.relative_to(ROOT)), source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                      trajectory_rows=len(poses), coverage_rows=len(curve), final_curve=curve[-1],
                      learned_method_comparison=False, complete_exploration_verified=False,
                      plot_source_sha256=digest(Path(__file__)))
    with (OUT / 'provenance.json').open('x') as f:
        json.dump(provenance, f, indent=2)
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()

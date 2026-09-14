"""Read-only explanation of saved hybrid decisions, not a detection benchmark.

All 30 three-sector observations of the sealed constrained_execution_v1 run.
No model invocation, labels, new sampling, policy changes or raw-data export.
"""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def alignment(axes, direction):
    """Unoriented 3-D segment angles, including slope; zero segments invalid."""
    v = np.diff(np.asarray(axes, float), axis=-2)
    length = np.linalg.norm(v, axis=-1)
    cosine = np.divide(np.abs(v @ direction), length,
                       out=np.zeros_like(length), where=length > 0)
    angle = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
    return np.where(length > 0, angle, np.inf)


def main():
    import rosbag
    from sensor_msgs import point_cloud2
    from mtare_topo.integration.aee_organized_scan_adapter import aee_organized_pointcloud2_to_range_image
    from mtare_topo.integration.geometry_constrained_tasks import constrained_tasks
    from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline
    from mtare_topo.semantics.primitive_relation_nonlearning import ELEVATION_DEG, _sector_mask
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    run = ROOT/'results/gate6_single_robot/gate6_20260910_gse_learned_constrained_execution_v1_seed11'
    seal = {p: h for h, p in (s.split('  ', 1) for s in
            (run/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    checked = {}

    def verify(path):
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8388608), b''):
                digest.update(block)
        relative = str(path.relative_to(ROOT))
        if digest.hexdigest() != seal[relative]:
            raise ValueError('sealed input drift: '+relative)
        checked[relative] = digest.hexdigest()
        return path

    trace = verify(run/'artifacts/geometry/live_geometry.jsonl')
    rows = [r['result'] for r in map(json.loads, trace.read_text().splitlines()) if r.get('result')]
    if len(rows) != 289:
        raise ValueError('original population drift')
    selected = [r for r in rows if r['geometry']['task_constraint_audit']['observed_candidates'] == 3]
    if len(selected) != 30:
        raise ValueError('expected all 30 three-direction observations')
    wanted = {r['geometry']['source_frame_keys'][-1]: r for r in selected}
    panels = set(range(0, 30, 5))  # Fixed chronological spacing, not scores.
    scans = {}
    with rosbag.Bag(str(verify(run/'artifacts/sensors.bag'))) as bag:
        for _, msg, _ in bag.read_messages(topics=['/velodyne_points']):
            key = msg.header.frame_id+':'+str(msg.header.stamp.to_nsec())
            if key in wanted:
                if key in scans:
                    raise ValueError('duplicate source scan')
                ranges, valid, _ = aee_organized_pointcloud2_to_range_image(msg, point_cloud2)
                points = np.asarray(list(point_cloud2.read_points(msg, field_names=('x','y','z'), skip_nans=True)))
                scans[key] = (ranges, valid, points)
    if set(scans) != set(wanted):
        raise ValueError('missing original scans')
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    records = []
    for index, row in enumerate(selected):
        g = row['geometry']; key = g['source_frame_keys'][-1]
        ranges, valid, points = scans[key]
        pose = np.asarray(g['sensor_to_local_odometry'])
        sectors = []
        for s in RangeExitBaseline().predict(ranges, valid, np.asarray(ELEVATION_DEG))['sectors']:
            mask = (valid.astype(bool) & (ranges >= 4)
                    & (np.abs(np.asarray(ELEVATION_DEG)) <= 5)[:, None]
                    & _sector_mask(s['heading_robot_deg'], s['angular_width_deg'])[None, :])
            refs = [key+'/ray:'+str(int(i)) for i in np.flatnonzero(mask)]
            if refs:
                sectors.append(dict(heading_robot_deg=s['heading_robot_deg'], source_refs=refs))
        proposals, audit = constrained_tasks(g, sectors, lookahead_m=4, direction_tolerance_deg=15)
        if audit != g['task_constraint_audit'] or proposals != row['decision']['proposals']:
            raise ValueError('reconstructed decision differs from sealed decision')
        raw_path = verify(run/'artifacts/model'/g['inference_evidence'])
        with np.load(io.BytesIO(raw_path.read_bytes()), allow_pickle=False) as raw:
            if raw['source_frame_keys'].tolist() != g['source_frame_keys'] or float(raw['timestamp']) != g['timestamp']:
                raise ValueError('model source mismatch')
            axes = raw['axis_control_current_sensor_m'].astype(float)
            logits = raw['existence_logits'].astype(float)
        active = [(np.asarray(p['axis_controls_world_m'])-pose[:3, 3]) @ pose[:3, :3] for p in g['primitives']]
        sector_records = []
        for j, s in enumerate(sectors):
            rad = np.radians(s['heading_robot_deg']); direction = np.array([np.cos(rad), np.sin(rad), 0.])
            angles = alignment(axes, direction)
            best_slot, best_seg = np.unravel_index(np.argmin(angles), angles.shape)
            kept_angles = [float(np.min(alignment(a, direction))) for a in active]
            best_kept = min(kept_angles) if kept_angles else None
            accepted = any(p['primitive_index'] == j for p in proposals)
            if accepted != (best_kept is not None and best_kept <= 15+1e-10):
                raise ValueError('independent angle check differs')
            if accepted:
                reason = 'ACCEPTED'
            elif not np.any(angles <= 15):
                reason = 'NO_RAW_SEGMENT_WITHIN_15_DEG'
            elif not np.any((angles <= 15) & (logits >= 0)[:, None]):
                reason = 'ALIGNED_RAW_SEGMENTS_ALL_BELOW_EXISTENCE_THRESHOLD'
            else:
                reason = 'ALIGNED_ACTIVE_RAW_SEGMENT_REMOVED_BY_DOMAIN_OR_DEGENERACY'
            sector_records.append(dict(sector_index=j, heading_robot_deg=s['heading_robot_deg'],
                supporting_range_cells=len(s['source_refs']), accepted=accepted, reason=reason,
                best_retained_angle_deg=best_kept, best_raw_angle_deg=float(angles[best_slot, best_seg]),
                best_raw_slot=int(best_slot), best_raw_segment=int(best_seg),
                best_raw_probability=float(1/(1+np.exp(-logits[best_slot])))))
        records.append(dict(timestamp=g['timestamp'], source_frame_keys=g['source_frame_keys'], sectors=sector_records))
        if index in panels:
            ax = axs.flat[index//5]
            pts = points[np.linalg.norm(points, axis=1) <= 10]
            ax.scatter(pts[:, 0], pts[:, 1], c='0.65', s=1, alpha=.35)
            for a in active:
                ax.plot(a[:, 0], a[:, 1], color='royalblue', linewidth=1.2, alpha=.75)
            ax.scatter([0], [0], color='black', marker='^', s=40)
            for s in sector_records:
                rad = np.radians(s['heading_robot_deg']); d = 4*np.array([np.cos(rad), np.sin(rad)])
                color = 'forestgreen' if s['accepted'] else 'crimson'
                ax.arrow(0, 0, *d, color=color, width=.06, length_includes_head=True)
                angle = s['best_retained_angle_deg']
                ax.text(*(d*1.15), '%d: %.1f deg'%(s['sector_index'], angle) if angle is not None else 'none', color=color, fontsize=8)
            ax.set(title='t=%.3fs, accepted %d/3'%(g['timestamp'], audit['accepted']),
                   xlabel='sensor X (m)', ylabel='sensor Y (m)', xlim=(-10, 10), ylim=(-10, 10))
            ax.set_aspect('equal'); ax.grid(alpha=.2)
    counts = dict(Counter(s['reason'] for r in records for s in r['sectors']))
    report = dict(scope='all 30 three-sector windows of one sealed development trajectory; 90 candidates',
        source_sha256=checked, source_card='configs/v3/gate6/data_cards/gse_learned_constrained_execution_v1.json',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_decisions_reproduced=30, counts=counts, observations=records,
        figure_indices=sorted(panels), figure_input='actual current scan returns; learned axes use original five causal scans',
        training_steps=0, model_calls=0, threshold_changes=False,
        limitation='Observed sectors are not GT branches; angular agreement alone is not spatial association or traversability. No precision/recall claim.')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'direction_diagnostic.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    fig.suptitle('Gray: current LiDAR returns | Blue: retained learned axes | Green/red: accepted/rejected observed direction\nNumbers: best 3D unoriented angle to retained model segment; NOT detection accuracy')
    if (out/'direction_diagnostic.png').exists():
        raise FileExistsError('preserve previous figure')
    fig.savefig(out/'direction_diagnostic.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(observations=30, candidates=90, counts=counts, decisions_reproduced=30)))


if __name__ == '__main__':
    main()

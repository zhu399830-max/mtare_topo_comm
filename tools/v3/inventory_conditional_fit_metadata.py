"""Read existing sealed C01-C06 selection identities, never sensor/teacher payloads."""
from _bootstrap import PROJECT_ROOT as ROOT
from collections import Counter
import hashlib
import json

SOURCE = 'results/gate3_semantics/gate3_20260907_gse_surface_identity_selection_v1_seed20260906'
OUT = 'docs/figures/gse_conditional_geometry_fit_v1/fit_population_metadata.json'


def main():
    seal = ROOT / SOURCE / 'artifacts/evidence_sha256.txt'
    pins = {p: h for h, p in (line.split('  ', 1) for line in seal.read_text().splitlines())}
    # Discover names only. Never open C07 or held-out payloads here.
    paths = sorted(p for p in pins if p.startswith(SOURCE + '/artifacts/')
                   and any(p.endswith(f'_C{i:02}_selection.json') for i in range(1, 7)))
    assert len(paths) == 60
    parents = []; frames = set(); tasks = set(); edges = set(); positions = Counter()
    hashes = {}; total = 0; per_parent = []
    for name in paths:
        raw = (ROOT / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pins[name]
        hashes[name] = pins[name]
        d = json.loads(raw); rows = d['observations']
        assert d['split'] == 'fit' and len(rows) == 48 and d['selected_physical_edges'] == 16
        parent = d['parent_id']; parents.append(parent); grouped = {}
        for row in rows:
            assert row['split'] == 'fit' and row['parent_id'] == parent
            assert row['frame_rows'] == sorted(set(row['frame_rows'])) and len(row['frame_rows']) == 5
            assert row['frame_rows'][-1] == row['decision_frame_row']
            assert not row['continuous_route_evidence']
            grouped.setdefault(row['physical_edge_id'], []).append(row)
            frames.update((row['task'], f) for f in row['frame_rows'])
            tasks.add(row['task']); edges.add(row['physical_edge_id'])
            positions[row['decision_position_policy']] += 1
        assert len(grouped) == 16
        for group in grouped.values():
            assert len(group) == 3 and {r['variant'] for r in group} == {'ellipse', 'rounded_rectangle', 'c1_mixed'}
            assert len({r['traversal_id'] for r in group}) == 1
        total += len(rows)
        per_parent.append(dict(parent=parent, observations=48, physical_edges=16,
                               available_physical_edges=d['available_physical_edges']))
    assert total == 2880 and len(edges) == 960 and len(tasks) == 180
    result = dict(status='METADATA_ONLY_NOT_TRAINING_AUTHORIZATION', parents=60,
                  observations=total, physical_edges=len(edges), tasks=len(tasks),
                  frame_exposures=total * 5, unique_variant_frames=len(frames),
                  position_policy_counts=dict(positions), parent_ids=parents, per_parent=per_parent,
                  known_relation_count=None, independent_structure_count=None,
                  actual_patch_count=None, label_eligibility='not established by identities',
                  continuous_route=False, history_spacing_m=None, acquisition_duration_s=None,
                  extrapolation_only=dict(reference_basis_observations=240,
                      feature_wall_s=72.654753 * 12,
                      reference_wall_s_same_four_workers=4448.142895845929 * 12,
                      feature_bytes=1066615359 * 12, reference_bytes=350753620 * 12,
                      warning='Linear development throughput extrapolation, not measured fit cost or resource bound'),
                  source_sha256=hashes, seal_sha256=hashlib.sha256(seal.read_bytes()).hexdigest(),
                  payload_reads=0, targets_generated=0, training_steps=0,
                  next='Design bounded fit-only proposal and exact card before target export; no automatic full export or training')
    out = ROOT / OUT
    if out.exists():
        assert json.loads(out.read_text()) == result, 'Do not overwrite differing evidence'
    else:
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_sha256', 'parent_ids', 'per_parent')}, indent=2))


if __name__ == '__main__':
    main()

"""Metadata-only opposite traversals for the fixed141 fit observations."""
import hashlib
import json
from pathlib import Path
from collections import Counter
from calibration_route_inventory_v1 import ORIGINAL, ORIGINAL_SEAL, SUPPLEMENT, SUPPLEMENT_SEAL, INVENTORY, SEAL_SHA256

ROOT = Path(__file__).resolve().parents[2]
AUDIT = 'docs/figures/gse_graph/v8_saved_paired_audit_20260909.json'
AUDIT_SHA = '42026d9bf0c38af1c185ae1a2d291842dbd48e993e9fe414d657668ea9b23d65'


def inventory():
    opened = {}
    def read(path, expected):
        raw = (ROOT / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('metadata drift: ' + path)
        opened[path] = expected
        return raw
    def seal(run, expected):
        raw = read(run + '/artifacts/evidence_sha256.txt', expected)
        return {p: h for h, p in (line.split('  ', 1) for line in raw.decode().splitlines())}
    audit = json.loads(read(AUDIT, AUDIT_SHA))
    sources = [r['source'] for r in audit['rows']]
    if len(sources) != 141 or any(s['split'] != 'fit' for s in sources):
        raise ValueError('fixed fit population required')
    pins = seal(INVENTORY, SEAL_SHA256)
    reports = {}
    for parent in sorted({s['parent_id'] for s in sources}):
        if parent.rsplit('_', 1)[-1] not in {'C01','C02','C03','C04','C05','C06'}:
            raise ValueError('nonfit parent')
        p = INVENTORY + '/artifacts/' + parent + '_identity_intervals.json'
        reports[parent] = json.loads(read(p, pins[p]))
    chosen = {}
    for s in sources:
        intervals = reports[s['parent_id']]['intervals']
        matches = [(i, v) for i in intervals for v in i['variants'] if v['task'] == s['task']
                   and any(sid == s['source_sequence_id'] and fs == s['frame_rows']
                           for sid, fs in zip(v['source_sequence_ids'], v['frame_rows'], strict=True))]
        if len(matches) != 1 or matches[0][0]['traversal_id'] != s['traversal_id']:
            raise ValueError('original source interval mismatch')
        chosen[(s['task'], s['traversal_id'])] = s
    if len(chosen) != 13 or len(reports) != 5:
        raise ValueError('fixed13 variants/5 parents required')
    rows = []; pairs = []
    for (task, traversal), s in sorted(chosen.items()):
        prefix, direction = traversal.rsplit(':', 1)
        if direction not in {'d0', 'd1'}: raise ValueError('direction encoding')
        opposite = prefix + (':d1' if direction == 'd0' else ':d0')
        matches = [v for i in reports[s['parent_id']]['intervals'] if i['traversal_id'] == opposite
                   for v in i['variants'] if v['task'] == task]
        if len(matches) != 1: raise ValueError('missing or ambiguous opposite variant')
        v = matches[0]
        pairs.append(dict(task=task, original_traversal=traversal, opposite_traversal=opposite))
        for sid, frames, sequence, arc in zip(v['source_sequence_ids'], v['frame_rows'], v['sequence_rows'], v['decision_arc_m'], strict=True):
            rows.append(dict(task=task, parent_id=s['parent_id'], variant=s['variant'], traversal_id=opposite,
                             source_sequence_id=sid, frame_rows=frames, sequence_row=sequence,
                             decision_route_arc_m=arc, split='fit'))
    key = lambda r: (r['task'], r['source_sequence_id'], tuple(r['frame_rows']))
    wanted = {key(r): r for r in rows}
    if len(wanted) != len(rows) or set(wanted) & {key(s) for s in sources}:
        raise ValueError('duplicate or original-direction overlap')
    cached = {}
    for run, expected in ((ORIGINAL, ORIGINAL_SEAL), (SUPPLEMENT, SUPPLEMENT_SEAL)):
        pns = seal(run, expected); p = run + '/artifacts/input_manifest.json'
        manifest = json.loads(read(p, pns[p])); shards = {s['task']: s for s in manifest['task_shards']}; offsets = Counter()
        for item in manifest['observations']:
            task = item['task']; offset = offsets[task]; offsets[task] += 1
            if key(item) not in wanted: continue
            if item['split'] != 'fit': raise ValueError('cache split mismatch')
            shard = shards[task]; p = run + '/' + shard['path']
            if pns[p] != shard['sha256']: raise ValueError('cache pin mismatch')
            cached.setdefault(key(item), dict(path=p, sha256=pns[p], row=offset))
    return dict(status='METADATA_ONLY_NOT_EXPORT_OR_TRAINING_AUTHORITY',
                counts=dict(observations=len(rows), parents=len(reports), variant_traversals=len(pairs),
                            physical_traversals=len({r['traversal_id'] for r in rows}),
                            unique_variant_frames=len({(r['task'],f) for r in rows for f in r['frame_rows']}),
                            original_supplement_cached=len(cached), other_cache_reuse_not_checked=True),
                pairs=pairs, rows=[dict(r, cached_input=cached.get(key(r))) for r in rows],
                metadata_sha256=opened, payload_reads=0, teacher_calls=0, training_updates=0,
                warning='Same historical positive-conditioned entities; no claim of unbiased population or valid labels.')


if __name__ == '__main__':
    print(json.dumps(inventory(), sort_keys=True))

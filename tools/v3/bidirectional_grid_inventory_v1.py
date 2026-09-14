"""Metadata-only join of existing source cards and observed-grid reuse.

This inventory grants no export authority. Original source scopes remain pinned;
grid geometry/binding must be checked by the eventual authorized reader.
"""
import json
from collections import Counter
from pathlib import Path
from bidirectional_feature_inventory_v1 import INPUT_RUNS, BASE
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.bidirectional_feature_scope_v1 import INVENTORY, INVENTORY_SHA

GRID_RUN = 'results/gate3_semantics/gate3_20260909_gse_development_grids_v1_seed20260906'
GRID_SHA = 'd3d3ddf06a56b2942dba93176d1808919850377e95cb044a53508c95980c8909'


def identity(source):
    return source['task'], source['source_sequence_id'], tuple(source['frame_rows'])


def compile_inventory(root):
    root = Path(root)
    opened = {}
    def read(path, sha):
        payload = read_pinned(root, path, sha)
        opened[path] = sha
        return payload
    inv = json.loads(read(INVENTORY, INVENTORY_SHA))
    expected = {identity(r['source']): r for r in inv['rows']}
    if len(expected) != len(inv['rows']) or len(expected) != 860:
        raise ValueError('exact unique860 inventory required')
    gm = json.loads(read(GRID_RUN+'/artifacts/grid_manifest.json', GRID_SHA))
    if gm['status'] != 'COMPLETE':
        raise ValueError('incomplete old grid manifest')
    grids = {identity(r['binding']['source']): r for r in gm['observations']}
    if len(grids) != len(gm['observations']):
        raise ValueError('duplicate old grid source')
    rows, cards, seen, files, inputs = [], [], set(), {}, {}
    for name, seal_sha in INPUT_RUNS.items():
        run = BASE+name
        seal = read(run+'/artifacts/evidence_sha256.txt', seal_sha)
        pins = {p: h for h, p in (line.split('  ', 1) for line in seal.decode().splitlines())}
        path = run+'/config/data_card.json'
        card = json.loads(read(path, pins[path]))
        scope = card['scope']
        cards.append(dict(path=path, sha256=pins[path], counts=scope['counts']))
        for mapping, merged in ((scope['file_sha256'], files), (scope['input_sha256'], inputs)):
            for p, h in mapping.items():
                if p in merged and merged[p] != h:
                    raise ValueError('source hash conflict')
                merged[p] = h
        for entry in scope['entries']:
            for ref in entry['input_references']:
                s = ref['source']; key = identity(s)
                if key not in expected or key in seen or ref != expected[key]['input_reference']:
                    raise ValueError('source/reference drift')
                seen.add(key)
                old = grids.get(key)
                if old and old['split'] != s['split']:
                    raise ValueError('grid split mismatch')
                reuse = None
                if old:
                    name = old['file']
                    if '/' in name or name in ('', '.', '..'):
                        raise ValueError('invalid grid filename')
                    reuse = dict(path=GRID_RUN+'/artifacts/grids/'+name,
                                 sha256=old['sha256'], binding=old['binding'])
                rows.append(dict(source={k:s[k] for k in ('task','source_sequence_id','frame_rows')},
                    split=s['split'], source_card=path, construction_path=entry['construction_path'],
                    construction_file_sha256=scope['file_sha256'][entry['construction_path']],
                    reused_grid=reuse))
    missing = [r for r in rows if r['reused_grid'] is None]
    if seen != set(expected) or len(missing) != 833:
        raise ValueError('grid population drift')
    return dict(schema='bidirectional_grid_inventory_v1', source_cards=cards,
        observations=sorted(rows, key=lambda r:identity(r['source'])), metadata_sha256=opened,
        source_file_sha256=files, input_sha256=inputs,
        counts=dict(observations=860, reused=27, missing=833,
            missing_splits=dict(Counter(r['split'] for r in missing)),
            source_files=len(files), input_packages=len(inputs),
            missing_unique_variant_frames=len({(r['source']['task'],f) for r in missing for f in r['source']['frame_rows']})),
        status='METADATA_ONLY_BINDING_AND_PAYLOAD_CHECKS_PENDING',
        restrictions=['no_export_authority', 'no_new_teacher', 'unknown_not_negative',
                      'fit_calibration_only_not_independent_evaluation'])


if __name__ == '__main__':
    print(json.dumps(compile_inventory(Path(__file__).resolve().parents[2]), separators=(',', ':')))

"""Existing feature identity and reference-incidence coverage, not labels."""
import base64
import collections
import gzip
import hashlib
import json
from pathlib import Path
from mtare_topo.teacher.gse_construction_paths_v3 import construction_incident_paths


def check(root):
    wrapper = json.loads((root / 'configs/v3/gate3/saved_branch_binding_scope_v1.json').read_text())
    data = gzip.decompress(base64.b64decode(wrapper['payload'], validate=True))
    if hashlib.sha256(data).hexdigest() != '9965eeac86b827266b76b0318c3bf99f3dee39641cba47ccab3a80c8b25b2bcb':
        raise ValueError('branch scope drift')
    scope = json.loads(data)
    feature_run = root / 'results/gate3_semantics/gate3_20260908_gse_supplement_features_v1_seed0'
    raw = (feature_run / 'artifacts/feature_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != '9e9d12c056c8ad138a66710da93e58554b430af59f717a55eabfe11b6cd051bc':
        raise ValueError('feature manifest drift')
    features = {}
    for f in json.loads(raw)['observations']:
        s = f['source']; key = (s['task'], s['source_sequence_id'], tuple(s['frame_rows']))
        if key in features:
            raise ValueError('duplicate feature identity')
        features[key] = f
    def read(path):
        raw = (root / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != scope['file_sha256'][path]:
            raise ValueError('source drift: ' + path)
        return raw
    groups = {}
    stats = collections.defaultdict(collections.Counter)
    records = []
    for row in scope['rows']:
        source = row['source']
        key = (source['task'], source['source_sequence_id'], tuple(source['frame_rows']))
        feature = features[key]
        if not (feature_run / feature['path']).is_file():
            raise ValueError('feature file missing')
        path = row['construction_path']
        if path not in groups:
            entries = construction_incident_paths(json.loads(read(path)))
            groups[path] = {g['node_id_teacher_only']: g for g in entries}
            if len(groups[path]) != len(entries):
                raise ValueError('duplicate construction node')
        target = json.loads(gzip.decompress(read(row['target_path'])))['produced_targets']
        counts = collections.Counter(observations=1, feature_identity_matches=1)
        for anchor in target['teacher_provenance']['anchors']:
            group = groups[path][anchor['node_id_teacher_only']]
            full = len(group['paths']); known = len(anchor['interface_ids'])
            # The prior frozen binding check establishes incidence membership;
            # here only cardinality is compared, not inferred physical geometry.
            if full < known or known != len(set(anchor['interface_ids'])):
                raise ValueError('incidence cardinality inconsistent')
            counts.update(anchors=1, reference_branches=full, witnessed_branches=known,
                          unwitnessed_reference_branches=full-known,
                          reference_set_fully_witnessed=int(full==known))
        stats[row['split']].update(counts)
        records.append(dict(source=source, split=row['split'], parent=row['parent'],
                            counts=counts, feature_path=str((feature_run / feature['path']).relative_to(root)),
                            feature_sha256=feature['sha256']))
    return dict(status='METADATA_ALIGNMENT_AND_REFERENCE_COVERAGE_ONLY',
                by_split=stats, feature_manifest_population=len(features),
                feature_payload_hashes_checked=False, feature_payloads_decoded=False,
                all_physical_branches_complete=False, training_eligible=False, rows=records)


if __name__ == '__main__':
    result = check(Path(__file__).resolve().parents[2])
    result.pop('rows')
    print(json.dumps(result, indent=2))

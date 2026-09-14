"""Read-only binding check of frozen existing branch evidence; no label export."""
import base64
import collections
import gzip
import hashlib
import json
from pathlib import Path
import zarr
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.teacher.gse_construction_paths_v3 import construction_incident_paths
from mtare_topo.teacher.gse_directed_interface_binding_v1 import bind_axes
from mtare_topo.teacher.saved_branch_evidence import read_saved_junction_branches


def check(root):
    wrapper = json.loads((root / 'configs/v3/gate3/saved_branch_binding_scope_v1.json').read_text())
    raw_scope = gzip.decompress(base64.b64decode(wrapper['payload'], validate=True))
    if hashlib.sha256(raw_scope).hexdigest() != '9965eeac86b827266b76b0318c3bf99f3dee39641cba47ccab3a80c8b25b2bcb':
        raise ValueError('scope drift')
    scope = json.loads(raw_scope)
    def read(path):
        if Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError('invalid frozen path')
        data = (root / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != scope['file_sha256'][path]:
            raise ValueError('frozen input drift: ' + path)
        return data
    # Only pinned Zarr keys are made visible; no automatic directory traversal.
    yaw = {}
    for name, info in scope['arrays'].items():
        store = {'.zarray': read(name + '/.zarray')}
        for key in info['chunk_keys']:
            store[key] = read(name + '/' + key)
        array = zarr.open_array(store=store, mode='r')
        if list(array.shape) != info['shape'] or list(array.chunks) != info['chunks']:
            raise ValueError('array metadata mismatch')
        yaw[name] = {i: float(array[i]) for i in info['selected_rows']}
    documents = {}
    counts = collections.Counter()
    by_split = collections.defaultdict(collections.Counter)
    for row in scope['rows']:
        target = json.loads(gzip.decompress(read(row['target_path'])))['produced_targets']
        original = json.loads(gzip.decompress(read(row['reference_chain'][-1])))['raw_interfaces']
        source = row['source']
        if (target['source_binding']['source'] != source
                or any(original['source'][k] != source[k] for k in source)):
            raise ValueError('source identity drift')
        name = row['construction_path']
        if name not in documents:
            doc = json.loads(read(name))
            documents[name] = (canonical_sha(doc), construction_incident_paths(doc))
        digest, groups = documents[name]
        if digest != target['source_binding']['construction_sha256']:
            raise ValueError('construction semantic digest differs')
        axes = bind_axes(groups, original['interfaces_teacher_only'])
        branches = read_saved_junction_branches(target['record'], target['teacher_provenance'], axes,
                      current_yaw_deg=yaw[row['yaw_array']][row['yaw_row']], ray_count=57600)
        if len(branches) != row['branch_occurrences'] or not all(b.supported for b in branches):
            raise ValueError('saved branch count/support differs')
        if any(b.training_eligible or b.aperture_position_m is not None for b in branches):
            raise ValueError('reader unexpectedly generated qualified aperture targets')
        stat = dict(observations=1, bound_branches=len(branches))
        counts.update(stat); by_split[row['split']].update(stat)
    if counts != dict(observations=307, bound_branches=951):
        raise ValueError('population mismatch')
    return dict(status='EXISTING_DIRECTION_BINDING_PASS_NOT_TRAINING_QUALIFICATION',
                counts=counts, by_split=by_split, yaw_arrays=len(yaw), construction_files=len(documents),
                new_teacher_execution=False, labels_exported=0, optimizer_steps=0)


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[2]), indent=2))

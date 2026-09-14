"""Single sealed C01 paired check, not a general population selector."""
import gzip
import json
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_joint_cached_interfaces_v1 import read_cached
from .gse_joint_reference_targets_v6 import produce_joint_reference_targets

PATH = 'results/gate3_semantics/gate3_20260907_gse_supplement_joint_v2_seed20260906/artifacts/S01_flat_tree_small_C01__c1_mixed_225.json.gz'
SHA = '7900a4b3f73b6ac2a41d58686f8ee2691d6149830de70f8b7d0d8e7722adf718'


def inspect_pair(bundle, root, opened):
    s = bundle['source']
    if (s['task'], s['source_sequence_id'], s['frame_rows']) != ('S01_flat_tree_small_C01__c1_mixed', 225, [289,290,291,292,293]):
        raise ValueError('exact sealed225 only')
    archived = json.loads(gzip.decompress(read_pinned(root, PATH, SHA)))
    opened[PATH] = SHA
    raw, _ = read_cached(root, archived['raw_interfaces_reference'], s, opened)
    old = archived['produced_targets']
    new = produce_joint_reference_targets(bundle, raw)
    if old['source_binding'] != new['source_binding']:
        raise ValueError('source binding changed')
    for key in old['record']:
        if key != 'membership' and old['record'][key] != new['record'][key]:
            raise ValueError('unexpected geometry or score-region change: '+key)
    if old['record']['membership'] != [[None]] or new['record']['membership'] != [[True]]:
        raise ValueError('witnessed single terminal correspondence not recovered')
    return dict(source=s, sealed_v4=dict(path=PATH, sha256=SHA),
        raw_interfaces_reference=archived['raw_interfaces_reference'], old=old, new=new,
        new_positive_memberships=1, geometry_unchanged=True, qualified_complete_labels=0,
        optimizer_steps=0, scientific_gate_pass=False)

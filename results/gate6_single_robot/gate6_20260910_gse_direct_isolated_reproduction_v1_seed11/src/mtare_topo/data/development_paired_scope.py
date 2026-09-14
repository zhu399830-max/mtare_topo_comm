"""Metadata-only exact join of frozen development assets for paired training."""
import base64
import gzip
import hashlib
import json
from .development_partition_scope import compile_scope as partition_scope, GRID_RUN
from mtare_topo.governance_surface_material import read_pinned

PARTITION_RUN='results/gate3_semantics/gate3_20260909_gse_development_partitions_v1_seed0'
PINS={
    PARTITION_RUN+'/artifacts/partition_manifest.json':'0cd19b29b06815cc62b26a8e2ec09a36498c844de7fa1be8881ff0fc7693da81',
    GRID_RUN+'/artifacts/grid_manifest.json':'d3d3ddf06a56b2942dba93176d1808919850377e95cb044a53508c95980c8909',
    'configs/v3/gate3/saved_branch_binding_scope_v1.json':'f859cb8436974d63350dda8856a50fee68b7921c410cbbaaf06540e2482b485a',
    'configs/v3/gate3/data_cards/gse_supplement_joint_v3.json':'6025aee23b8382617427316df60304f6084568f127e315c873c9bc0c30118c22',
}


def identity(source):
    return source['task'],source['source_sequence_id'],tuple(source['frame_rows'])


def unique_index(rows, source):
    result={identity(source(r)):r for r in rows}
    if len(result)!=len(rows):raise ValueError('duplicate joined observation')
    return result


def compile_scope(root):
    assets=partition_scope(root)
    docs={p:json.loads(read_pinned(root,p,h)) for p,h in PINS.items()}
    encoded=docs['configs/v3/gate3/saved_branch_binding_scope_v1.json']
    raw=gzip.decompress(base64.b64decode(encoded['payload'],validate=True))
    if hashlib.sha256(raw).hexdigest()!='9965eeac86b827266b76b0318c3bf99f3dee39641cba47ccab3a80c8b25b2bcb':raise ValueError('selection drift')
    selection=json.loads(raw)
    partitions=docs[PARTITION_RUN+'/artifacts/partition_manifest.json']
    grids=docs[GRID_RUN+'/artifacts/grid_manifest.json']
    if partitions['status']!='COMPLETE' or grids['status']!='COMPLETE':raise ValueError('incomplete assets')
    pi=unique_index(partitions['observations'],lambda r:r['source'])
    gi=unique_index(grids['observations'],lambda r:r['binding']['source'])
    ti=unique_index(selection['rows'],lambda r:r['source'])
    ai=unique_index(assets['observations'],lambda r:r['source'])
    if set(ai)!=set(pi) or set(ai)!=set(gi) or set(ai)!=set(ti) or len(ai)!=307:
        raise ValueError('asset populations differ')
    rows=[]
    for key,a in ai.items():
        p,g,t=pi[key],gi[key],ti[key]
        if a['split']!=p['split'] or a['split']!=g['split'] or a['split']!=t['split'] or p['cache_sha256']!=a['cache_sha256']:
            raise ValueError('paired split/cache mismatch')
        for name in (p['file'],g['file']):
            if '/' in name or name in ('','.','..'):raise ValueError('invalid artifact filename')
        rows.append(dict(**a,partition_path=PARTITION_RUN+'/artifacts/partitions/'+p['file'],partition_sha256=p['sha256'],
                         grid_path=GRID_RUN+'/artifacts/grids/'+g['file'],grid_sha256=g['sha256'],source_binding=g['binding'],
                         target_path=t['target_path'],target_sha256=selection['file_sha256'][t['target_path']],
                         raw_interfaces_path=t['reference_chain'][-1],raw_interfaces_sha256=selection['file_sha256'][t['reference_chain'][-1]]))
    return dict(schema='development_paired_scope_v1',manifest_sha256={**assets['manifests_sha256'],**PINS},
                observations=rows,counts=assets['counts'],original_reader_scope=docs['configs/v3/gate3/data_cards/gse_supplement_joint_v3.json']['scope'],
                decoded_task_observations=2184,collateral_observations_not_targets=1877,
                restrictions=['C01_C07_only','no_new_scan_or_encoding','teacher_never_model_input','unknown_never_negative','reference_component_not_full_detection'])

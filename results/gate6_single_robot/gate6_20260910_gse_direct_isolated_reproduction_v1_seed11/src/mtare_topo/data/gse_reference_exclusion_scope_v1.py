"""Metadata-only scope for the fixed30 archived-grid exclusion check."""
import hashlib
import json
from pathlib import Path

SOURCE_CARD='configs/v3/gate3/data_cards/gse_surface_junction_interfaces_v1.json'
MATERIAL='results/gate3_semantics/gate3_20260907_gse_surface_observed_material_v1_seed20260906'


def compile_scope(root):
    root=Path(root)
    card_path=root/SOURCE_CARD
    card_raw=card_path.read_bytes();card=json.loads(card_raw)
    old_scope=card['scope']; selected=old_scope['selected']
    if len(selected)!=30 or any(not r['task'].split('__')[0].endswith('_C01') for r in selected):
        raise ValueError('only original fixed30 C01 observations allowed')
    seal_path=MATERIAL+'/artifacts/evidence_sha256.txt'
    seal_raw=(root/seal_path).read_bytes()
    sealed={}
    for line in seal_raw.decode().splitlines():
        digest,path=line.split('  ',1)
        if path in sealed or len(digest)!=64 or Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError('invalid/duplicate archive seal path')
        sealed[path]=digest
    manifest_path=MATERIAL+'/artifacts/material_manifest.json'
    raw=(root/manifest_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sealed[manifest_path]:
        raise ValueError('material manifest differs from seal')
    material=json.loads(raw)['observations']
    by_task={row['selection']['task']:row for row in material}
    if len(by_task)!=30:raise ValueError('archived material population mismatch')
    rows=[];frames=set();parents=set();traversals=set()
    for row in selected:
        task=row['task']; item=by_task[task];source=row['source']
        if item['selection']['source']!=source:
            raise ValueError('material/source observation drift')
        numeric=MATERIAL+'/'+item['numeric_file']
        if not numeric.startswith(MATERIAL+'/artifacts/') or '..' in Path(numeric).parts or numeric not in sealed:
            raise ValueError('numeric artifact outside sealed scope')
        parents.add(source['parent_id']);traversals.add(source['traversal_id'])
        frames.update((task,f) for f in source['frame_rows'])
        rows.append(dict(task=task,source=source,numeric_path=numeric,numeric_sha256=sealed[numeric],
            grid_content_sha256=item['metrics']['grid_content_sha256'],
            grid_source_sha256=item['metrics']['grid_source_sha256']))
    if len(parents)!=10 or len(traversals)!=10 or len(frames)!=150:
        raise ValueError('independent units/frame population drift')
    return dict(schema='gse_reference_exclusion_scope_v1',observations=rows,
        source_card=SOURCE_CARD,source_card_sha256=hashlib.sha256(card_raw).hexdigest(),
        material_seal=seal_path,material_seal_sha256=hashlib.sha256(seal_raw).hexdigest(),
        material_manifest=manifest_path,material_manifest_sha256=sealed[manifest_path],
        teacher_source_scope=old_scope,parents=sorted(parents),independent_traversals=10,
        unique_variant_frames=150,query_policy=dict(coordinates_per_axis_m=[-5.625,-1.875,1.875,5.625],
            queries_per_observation=64,selection='fixed Cartesian lattice; no model or observation-based selection',
            matching_radius_m=4.,maximum_query_radius_m=9.742785792574935),
        real_training=False,semantic_label_export=False,
        status='SCOPE_PREPARATION_NOT_EXECUTION_OR_LABEL_AUTHORITY')

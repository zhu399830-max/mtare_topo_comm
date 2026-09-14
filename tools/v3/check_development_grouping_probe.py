"""Fixed archived-cache grouping integration check; no training or export."""
import argparse
import hashlib
import io
import json
import resource
import time
from pathlib import Path
import numpy as np


def check(root, native=False):
    start=time.monotonic()
    run=root/'results/gate3_semantics/gate3_20260908_gse_supplement_features_v1_seed0'
    raw=(run/'artifacts/feature_manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!='9e9d12c056c8ad138a66710da93e58554b430af59f717a55eabfe11b6cd051bc':
        raise ValueError('manifest drift')
    source=dict(task='S01_flat_tree_small_C01__c1_mixed',source_sequence_id=598,frame_rows=[806,807,808,809,810])
    rows=[r for r in json.loads(raw)['observations'] if all(r['source'][k]==v for k,v in source.items())]
    if len(rows)!=1:raise ValueError('unique fixed observation required')
    row=rows[0];payload=(run/row['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!='ee7a80456b1e60d71ba822058fd4e1e2e2ba1f7bb2d6ca39a8b679c24e0d8956':
        raise ValueError('cache drift')
    if native:
        # Native SPG is Python3.12; no Torch or inference is needed here.
        import libply_c
        import libcp
        from mtare_topo.representation.gse_spg_extractor import extract_spg
        from mtare_topo.representation.gse_block_points import bind_block_points
        from mtare_topo.representation.gse_block_context import pool_block_context
        with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
            xyz=archive['points_xyz_m'][0];valid=archive['valid'][0]
            memory=archive['frozen_sensor_context'][0]
        indices=np.flatnonzero(valid & (np.linalg.norm(xyz.astype(float),axis=1)<=10.))
        points=xyz[indices]
        out=extract_spg(points,indices//11520,geof_backend=libply_c.compute_geof,partition_backend=libcp.cutpursuit)
        if out['original_point_to_component'] is None:raise ValueError(out['status'])
        assignment=out['original_point_to_component']
        if len(assignment)!=len(points):raise ValueError('native source count differs')
        blocks=bind_block_points(points,indices//11520,assignment)
        pooled=pool_block_context(blocks,indices,memory,valid)
        if int(pooled['point_count'].sum())!=len(points):raise ValueError('context mapping lost')
        methods={'r2':dict(blocks=len(np.unique(assignment)),points=len(assignment),context_shape=list(pooled['context'].shape))}
    else:
        from mtare_topo.data.development_compact_blocks import read_compact_points,sensor_layout_partition,bind_partition
        from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches
        cache=read_compact_points(payload,manifest_row=row,expected_source=source,
                                 encoder_sha256='200f5c2fbe66d68961cf2aea06e21f747cb5b8d419536008491bb58d3641a8cb')
        points=cache.points_xyz_m;indices=cache.source_flat_ray_index
        patches=extract_surface_patches(points,np.ones(len(points),bool),indices//11520)
        methods={}
        for name,bound in [('r0',sensor_layout_partition(cache)),('r1',bind_partition(cache,patches.point_patch_index))]:
            if len(bound['blocks'].xyz_m)!=len(points):raise ValueError('point mapping lost')
            methods[name]=dict(blocks=len(bound['blocks'].block_ids),points=len(points),context_shape=list(bound['context'].shape))
    if any(m['blocks']>4096 for m in methods.values()):raise OverflowError('block capacity exceeded')
    return dict(status='FIXED_CACHE_GROUPING_CHECK_NOT_RESEARCH_RESULT',source=source,methods=methods,
                points_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
                ray_indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                optimizer_steps=0,labels_exported=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--native-spg',action='store_true')
    print(json.dumps(check(Path(__file__).resolve().parents[2],parser.parse_args().native_spg),indent=2))

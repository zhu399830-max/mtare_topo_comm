"""Read-only feasibility of fixed-k relations on the existing sealed307 assets.

No teacher payload, sensor export, checkpoint, model forward or optimizer.
The original point/partition/context binders are retained without substitution.
"""
from _bootstrap import PROJECT_ROOT as ROOT
from collections import Counter
import json
import time
import numpy as np
import torch
from mtare_topo.data.development_paired_scope import compile_scope
from mtare_topo.data.development_partition_scope import ENCODER
from mtare_topo.data.development_compact_blocks import read_compact_points,bind_partition_payload
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.representation.block_relation_increment_v1 import block_relations


def main():
    scope=compile_scope(ROOT);start=time.monotonic();rows=[];counts=Counter();points=0
    for row in scope['observations']:
        compact=read_compact_points(read_pinned(ROOT,row['cache_path'],row['cache_sha256']),
            manifest_row=row['feature_entry'],expected_source=row['source'],encoder_sha256=ENCODER)
        bound=bind_partition_payload(compact,read_pinned(ROOT,row['partition_path'],row['partition_sha256']),
            expected_sha256=row['partition_sha256'],expected_source=row['source'])['r2']
        b=bound['blocks'];c=torch.tensor(b.centers_m);s=torch.tensor(b.extent_m);d=torch.tensor(b.degenerate)
        relation=block_relations(c,s,d);m=len(c);valid=relation.neighbor_valid[0]
        if not torch.isfinite(relation.relation).all():raise ValueError('nonfinite actual population relation')
        if not torch.all(valid.sum(1)==min(8,max(0,m-1))):raise ValueError('neighbor count drift')
        for i in range(m):
            ids=relation.neighbor_index[0,i,valid[i]].tolist()
            if i in ids or len(set(ids))!=len(ids):raise ValueError('self/duplicate neighbor')
        # Nontrivial reversal tests full input ordering without new random sampling.
        p=torch.arange(m-1,-1,-1);reversed_relation=block_relations(c[p],s[p],d[p])
        for new,old in enumerate(p):
            mask=reversed_relation.neighbor_valid[0,new]
            if not torch.equal(p[reversed_relation.neighbor_index[0,new,mask]],
                               relation.neighbor_index[0,old,valid[old]]):raise ValueError('order-sensitive neighborhood')
            if not torch.equal(reversed_relation.relation[0,new],relation.relation[0,old]):
                raise ValueError('order-sensitive attributes')
        rows.append(dict(source=row['source'],split=row['split'],blocks=m,
            directed_message_neighbors=int(valid.sum()),degenerate_blocks=int(d.sum())))
        counts[row['split']]+=1;points+=len(b.xyz_m)
    if counts!=Counter(fit=250,calibration=27,development=30):raise ValueError('population changed')
    print(json.dumps(dict(status='EXISTING_POPULATION_RELATIONS_FEASIBLE_NOT_TRAINED',
        observations=len(rows),split_counts=dict(counts),scope_sha256=digest(scope),
        points_checked=points,min_blocks=min(r['blocks'] for r in rows),max_blocks=max(r['blocks'] for r in rows),
        directed_message_neighbors=sum(r['directed_message_neighbors'] for r in rows),
        degenerate_blocks=sum(r['degenerate_blocks'] for r in rows),
        permutation_checks=len(rows),coincident_centers=0,elapsed_s=time.monotonic()-start,
        model_invocations=0,optimizer_steps=0,teacher_payload_reads=0),indent=2))


if __name__=='__main__':main()

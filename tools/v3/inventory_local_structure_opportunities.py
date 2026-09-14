"""Existing construction metadata only; opportunities are NOT observed labels."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np


def continuous_arms(node_id, nodes, primitives):
    """Follow degree-2 construction cuts, stopping at the next actual event."""
    pending=[m['primitive_id'] for m in nodes[node_id]['member_endpoints']]
    seen=set()
    while pending:
        pid=pending.pop()
        if pid in seen:continue
        seen.add(pid)
        for endpoint in primitives[pid]['endpoints']:
            n=nodes[endpoint['node_id']]
            if n['degree']==2:
                pending.extend(m['primitive_id'] for m in n['member_endpoints'])
    return seen


def finite_distance(point, controls):
    a=np.asarray(controls,float);p=np.asarray(point,float)
    v=np.diff(a,axis=0);den=np.einsum('ij,ij->i',v,v)
    if len(a)<2 or not np.isfinite(a).all() or np.any(den<=0):
        raise ValueError('nondegenerate finite construction polyline required')
    t=np.clip(np.einsum('ij,ij->i',p-a[:-1],v)/den,0,1)
    return float(np.min(np.linalg.norm(a[:-1]+t[:,None]*v-p,axis=1)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    card_path=ROOT/'configs/v3/gate3/data_cards/gse_supplement_joint_v3.json'
    scope=json.loads(card_path.read_text())['scope'];by_parent=defaultdict(list)
    for entry in scope['entries']:
        parents={o['parent_id'] for o in entry['observations']}
        splits={o['split'] for o in entry['observations']}
        if len(parents)!=1 or len(splits)!=1:raise ValueError('mixed parent/split')
        parent=next(iter(parents))
        if parent.rsplit('_',1)[-1] not in {'C01','C02','C03','C04','C05','C06','C07'}:
            raise ValueError('protected/out-of-scope parent')
        by_parent[parent].append((entry,next(iter(splits))))
    if len(by_parent)!=70:raise ValueError('parent count drift')
    rows=[];hashes={}
    for parent,entries in sorted(by_parent.items()):
        # Geometry variants share base construction (verified previously).
        # One canonical construction per parent; do not multiply independence.
        entry,split=min(entries,key=lambda e:e[0]['task'])
        path=entry['construction_path'];raw=(ROOT/path).read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=scope['file_sha256'][path]:raise ValueError('source drift')
        hashes[path]=digest;base=json.loads(raw)['base_construction']
        nodes={n['node_id']:n for n in base['composition_operations']}
        primitives={p['primitive_id']:p for p in base['primitives']}
        junctions=[n for n in nodes.values() if n['degree']>=3]
        terminals=[n for n in nodes.values() if n['degree']==1]
        jt=[];tt=[];nonincident=[]
        for a,b in itertools.product(junctions,terminals):
            distance=math.dist(a['anchor_xyz_m'],b['anchor_xyz_m'])
            if distance<=20:jt.append(dict(junction=a['node_id'],terminal=b['node_id'],distance_m=distance))
        for a,b in itertools.combinations(terminals,2):
            distance=math.dist(a['anchor_xyz_m'],b['anchor_xyz_m'])
            if distance<=20:tt.append(dict(first=a['node_id'],second=b['node_id'],distance_m=distance))
        for node in junctions:
            arms=continuous_arms(node['node_id'],nodes,primitives)
            for pid,p in primitives.items():
                if pid in arms:continue
                distance=finite_distance(node['anchor_xyz_m'],p['centerline_xyz_m'])
                if distance<=20:nonincident.append(dict(junction=node['node_id'],primitive=pid,distance_m=distance))
        rows.append(dict(parent=parent,split=split,junctions=len(junctions),terminals=len(terminals),
                         junction_terminal=jt,terminal_terminal=tt,nonincident_near_junction=nonincident))
    totals={}
    for split in sorted({r['split'] for r in rows}):
        part=[r for r in rows if r['split']==split]
        totals[split]=dict(parents=len(part),junctions=sum(r['junctions'] for r in part),terminals=sum(r['terminals'] for r in part))
        for key in ('junction_terminal','terminal_terminal','nonincident_near_junction'):
            totals[split][key]=dict(records=sum(len(r[key]) for r in part),parents=sum(bool(r[key]) for r in part))
    result=dict(status='METADATA_OPPORTUNITIES_NOT_LABELS',source_card_sha256=hashlib.sha256(card_path.read_bytes()).hexdigest(),
        source_sha256=hashes,summary=totals,parents=rows,scan_reads=0,labels_generated=0,
        limitations=['20m is geometric neighborhood opportunity only, not observed co-visibility or traversability',
            'Nonincident construction cuts collapsed through degree-2 nodes, not tunnel-ID grouping',
            'Counts are metadata pairs, not independent examples or valid negatives; no samples selected',
            'Centerline distance is to stored finite polyline, not distance to tunnel surface'])
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(summary=totals,source_files=len(hashes),labels_generated=0),indent=2))


if __name__=='__main__':main()

"""Authenticate existing ten partial references and test actual loss masks.

No teacher regeneration, model forward, scan read or training. Only fixed
sealed target records, RUN_STATE and summary are read.
"""
import gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets
ROOT=Path(__file__).resolve().parents[2]
RUN='results/gate3_semantics/gate3_20260908_gse_original_ten_precision_v1_seed20260906'
SEAL='36160f89c77382ff7ec7456634626a9ab78bd7d50dd7191349d81a263ab1b4a8'

def main():
    raw=(ROOT/RUN/'artifacts/evidence_sha256.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SEAL
    pins={p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
    read=[]
    def checked(path):
        payload=(ROOT/path).read_bytes();assert hashlib.sha256(payload).hexdigest()==pins[path],path
        read.append(path);return payload
    summary=json.loads(checked(RUN+'/metrics/summary.json'))
    assert summary['completed_observations']==10 and not summary['full_label_qualification']
    records=[];missing={};parents=set();routes=set();frames=set();counts={'positive':0,'negative':0,'unknown':0}
    for row in summary['observations']:
        s=row['source'];assert s['split']=='fit' and ('_C01' in s['task'] or '_C03' in s['task'])
        path=f"{RUN}/artifacts/{s['task']}_{s['source_sequence_id']}.json.gz"
        payload=checked(path);assert hashlib.sha256(payload).hexdigest()==row['storage']['compressed_sha256']
        expanded=gzip.decompress(payload);assert hashlib.sha256(expanded).hexdigest()==row['storage']['raw_sha256']
        item=json.loads(expanded)['produced_targets'];record=item['record'];records.append(record)
        assert item['full_training_gate_eligible'] is False
        for task in item['missing_tasks']:missing[task]=missing.get(task,0)+1
        for members in record['membership']:
            for value in members:counts['unknown' if value is None else 'positive' if value else 'negative']+=1
        parents.add(s['parent_id']);routes.add(s['traversal_id'])
        frames.update((s['task'],f) for f in s['frame_rows'])
    targets=observed_targets(records)
    assert counts=={'positive':20,'negative':20,'unknown':8},counts
    assert int(targets.membership_valid.sum())==40
    assert int(targets.membership[targets.membership_valid].sum())==20
    assert not targets.anchor_region_complete.any() and not targets.opening_region_complete.any()
    assert not targets.reachability_valid.any() and not targets.physical_reference_valid.any()
    print(json.dumps(dict(status='SEALED_TEN_PARTIAL_TARGET_MASKS_CONFIRMED',seal_sha256=SEAL,
        verified_files=len(read),observations=len(records),parents=len(parents),traversals=len(routes),
        unique_variant_frames=len(frames),counts=counts,missing_tasks=missing,
        anchors=int(targets.anchor_valid.sum()),openings=int(targets.opening_valid.sum()),
        complete_detection_regions=0,physical_reach_targets=0,
        allowed_conclusion='partial-reference mask integration only; not teacher correctness or detection qualification',
        model_forwards=0,optimizer_steps=0,scan_reads=0)))

if __name__=='__main__':main()

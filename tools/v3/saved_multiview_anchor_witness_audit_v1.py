"""Read sealed evidence only; no ray casting, teacher calls or relabeling."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/gate3_semantics/gate3_20260909_gse_multiview_historical_teacher_v1r_seed20260906'
SEAL_SHA='2daea1053cc2f6b00f132bc21496ba77b7091e092e2a882cb771aa6d1ff06d27'
OUT=ROOT/'docs/figures/gse_graph/multiview_anchor_witness_audit_20260909'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def frame_counts(rays):
    if any(type(i) is not int or not 0<=i<57600 for i in rays):raise ValueError('ray index outside five frames')
    return [sum(i//11520==slot for i in rays) for slot in range(5)]


def main():
    if OUT.exists():raise FileExistsError('immutable derived audit')
    seal=RUN/'artifacts/evidence_sha256.txt';raw=seal.read_bytes()
    if sha(raw)!=SEAL_SHA:raise ValueError('seal drift')
    pins={p:h for h,p in (s.split('  ',1) for s in raw.decode().splitlines())};opened={str(seal.relative_to(ROOT)):SEAL_SHA}
    def read(path):
        key=str(path.relative_to(ROOT));raw=path.read_bytes()
        if sha(raw)!=pins[key]:raise ValueError('evidence drift: '+key)
        opened[key]=pins[key];return raw
    manifest=json.loads(read(RUN/'artifacts/target_manifest.json'));rows=manifest['observations']
    assert len(rows)==2259
    lookup={(r['source']['task'],r['source']['decision_index']):r for r in rows}
    pairs=[];calibration=[]
    for row in rows:
        if not row['junction_positions_m'] and row['split']!='calibration':continue
        data=json.loads(gzip.decompress(read(RUN/'artifacts'/row['evidence_file'])))
        if data['raw_interfaces']['source']!=row['source']:raise ValueError('source mismatch')
        if row['split']=='calibration':
            raw=data['raw_interfaces']
            calibration.append(dict(source=row['source'],nearby_junction_references=raw['nearby_junction_references'],
                interface_nodes=sorted({i['node_id_teacher_only'] for i in raw['interfaces_teacher_only']})))
        if not row['junction_positions_m']:continue
        next_row=lookup.get((row['source']['task'],row['source']['decision_index']+1))
        if next_row is None:raise ValueError('positive has no next window')
        if row['source']['frame_rows'][1:]!=next_row['source']['frame_rows'][:-1]:raise ValueError('not overlapping windows')
        following=json.loads(gzip.decompress(read(RUN/'artifacts'/next_row['evidence_file'])))
        for anchor in data['produced_targets']['teacher_provenance']['anchors']:
            node=anchor['node_id_teacher_only'];witnesses=[]
            for index,interface in enumerate(anchor['interface_ids']):
                witness=dict(interface=interface)
                for field in ('witness_ray_indices','entering_witness_ray_indices','interior_witness_ray_indices'):
                    witness[field]=frame_counts(anchor[field][index])
                witness['only_oldest_frame']=bool(witness['witness_ray_indices'][0] and not sum(witness['witness_ray_indices'][1:]))
                witnesses.append(witness)
            pairs.append(dict(source=row['source'],node_teacher_only=node,witnesses=witnesses,
                next_source=next_row['source'],
                next_node_still_in_raw_interfaces=node in {i['node_id_teacher_only'] for i in following['raw_interfaces']['interfaces_teacher_only']},
                next_unknown=[r for r in following['produced_targets']['unknown_candidates']['anchors'] if r['node_id_teacher_only']==node],
                next_positive=node in {r['node_id_teacher_only'] for r in following['produced_targets']['teacher_provenance']['anchors']}))
    summary=dict(positive_pairs=len(pairs),pairs_with_oldest_only_branch=sum(any(w['only_oldest_frame'] for w in p['witnesses']) for p in pairs),
        next_same_node_candidate=sum(p['next_node_still_in_raw_interfaces'] for p in pairs),
        next_same_node_unknown=sum(bool(p['next_unknown']) for p in pairs),next_same_node_positive=sum(p['next_positive'] for p in pairs),
        calibration_windows=len(calibration),calibration_parents=len({r['source']['parent_id'] for r in calibration}),
        calibration_nearby_reference_counts=dict(collections.Counter(r['nearby_junction_references'] for r in calibration)),
        calibration_windows_with_interface_nodes=sum(bool(r['interface_nodes']) for r in calibration))
    result=dict(status='SAVED_WITNESS_ATTRITION_AUDIT_NOT_NEW_LABELS',summary=summary,pairs=pairs,calibration=calibration,
        source_sha256=opened,teacher_calls=0,model_calls=0,
        limitation='Witness-window attrition and candidate absence do not certify what a human or alternative method can observe; no thresholds changed.')
    OUT.mkdir(parents=True)
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
    (OUT/'code_sha256.json').write_text(json.dumps({str(Path(__file__).relative_to(ROOT)):sha(Path(__file__).read_bytes())})+'\n')
    (OUT/'evidence_sha256.txt').write_text(''.join(sha(p.read_bytes())+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(OUT.iterdir()) if p.is_file()))
    print(json.dumps(summary))


if __name__=='__main__':main()

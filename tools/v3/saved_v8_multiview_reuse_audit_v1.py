"""Pinned saved manifest intersection and source drift; no dataset payload."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OLD='results/gate3_semantics/gate3_20260908_gse_original_ten_precision_v1_seed20260906'
NEW='results/gate3_semantics/gate3_20260909_gse_multiview_historical_teacher_v1r_seed20260906'


def main():
    opened={}
    def reader(run, expected):
        path=run+'/artifacts/evidence_sha256.txt'
        raw=(ROOT/path).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==expected, 'seal drift'
        opened[path]=expected
        pins={p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
        def read(relative):
            path=run+'/'+relative
            raw=(ROOT/path).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==pins[path], path
            opened[path]=pins[path]
            return raw
        return read
    oldread=reader(OLD,'36160f89c77382ff7ec7456634626a9ab78bd7d50dd7191349d81a263ab1b4a8')
    newread=reader(NEW,'2daea1053cc2f6b00f132bc21496ba77b7091e092e2a882cb771aa6d1ff06d27')
    old=json.loads(oldread('metrics/summary.json'))['observations']
    new=json.loads(newread('artifacts/target_manifest.json'))['observations']
    assert len(old)==10 and len(new)==2259
    def key(row):
        s=row['source']; return (s['task'],s['source_sequence_id'])
    lookup={key(r):r for r in new}
    assert len(lookup)==2259
    overlaps=[]
    for row in old:
        match=lookup.get(key(row))
        if match is not None:
            for field in ('frame_rows','traversal_id','decision_route_arc_m'):
                assert row['source'][field]==match['source'][field], 'same ID different observation'
            overlaps.append(dict(source=row['source'],multiview_evidence=match['evidence_file']))
    with zipfile.ZipFile(io.BytesIO(oldread('artifacts/source_snapshot.zip'))) as archive:
        changed=[]; checked=[]; settings=[]
        for name in archive.namelist():
            if name.startswith('src/') and name.endswith('.py'):
                raw=archive.read(name); live=ROOT/name
                record=dict(path=name,archive_sha256=hashlib.sha256(raw).hexdigest(),
                    current_sha256=hashlib.sha256(live.read_bytes()).hexdigest() if live.is_file() else None)
                checked.append(record)
                if record['archive_sha256']!=record['current_sha256']: changed.append(record)
            if name.startswith('tools/') and name.endswith('.py') and 'precision' in name:
                for i,line in enumerate(archive.read(name).decode().splitlines(),1):
                    if any(k in line for k in ('axial_spacing_m','angular_segments','field_spacing_m','qualify_cap_precision')):
                        settings.append(dict(path=name,line=i,text=line.strip()))
    old_routes={(r['source']['task'],r['source']['traversal_id']) for r in old}
    same_routes=[r['source'] for r in new if (r['source']['task'],r['source']['traversal_id']) in old_routes]
    print(json.dumps(dict(status='SAVED_V8_MULTIVIEW_REUSE_AUDIT',old_observations=10,
        new_observations=2259,exact_overlap=len(overlaps),overlaps=overlaps,
        same_route_observations=len(same_routes),same_route_sources=same_routes,
        source_files_checked=len(checked),changed_source_files=changed,
        bound_runner_settings=settings,source_sha256=opened,
        teacher_calls=0,model_calls=0,raw_scan_reads=0),indent=2))


if __name__=='__main__': main()

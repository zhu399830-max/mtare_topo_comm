"""Fixed mechanism-diagnostic population, metadata only, not training approval."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
V8='results/gate3_semantics/gate3_20260908_gse_original_ten_precision_v1_seed20260906'
MULTI='results/gate3_semantics/gate3_20260909_gse_multiview_historical_teacher_v1r_seed20260906'


def compile_manifest():
    opened={}
    def seal(run, expected):
        path=run+'/artifacts/evidence_sha256.txt';raw=(ROOT/path).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==expected, 'seal drift'
        opened[path]=expected
        return {p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
    a=seal(V8,'36160f89c77382ff7ec7456634626a9ab78bd7d50dd7191349d81a263ab1b4a8')
    b=seal(MULTI,'2daea1053cc2f6b00f132bc21496ba77b7091e092e2a882cb771aa6d1ff06d27')
    def read(path,pins):
        raw=(ROOT/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==pins[path], path
        opened[path]=pins[path];return json.loads(raw)
    card=read(V8+'/config/data_card.json',a)
    settings=card['scope']['geometry_settings']
    assert settings==dict(axial_spacing_m=.05,angular_segments=64,field_spacing_m=.025), settings
    manifest_path=MULTI+'/artifacts/target_manifest.json'
    rows=read(manifest_path,b)['observations'];assert len(rows)==2259
    tasks=sorted({r['source']['task'] for r in rows if r['junction_positions_m']})
    selected=[]
    for index,row in enumerate(rows):
        s=row['source']
        if s['task'] not in tasks:continue
        assert row['split']=='fit' and s['split']=='fit'
        assert s['parent_id'].rsplit('_',1)[-1] in {'C01','C02','C03','C04','C05','C06'}
        path=MULTI+'/artifacts/'+row['evidence_file']
        selected.append(dict(manifest_index=index,task=s['task'],source_sequence_id=s['source_sequence_id'],
                             evidence_path=path,evidence_sha256=b[path]))
    chosen=[rows[r['manifest_index']]['source'] for r in selected]
    counts=dict(observations=len(selected),tasks=len(tasks),parents=len({r['parent_id'] for r in chosen}),
        unique_variant_frames=len({(r['task'],f) for r in chosen for f in r['frame_rows']}))
    assert counts==dict(observations=141,tasks=13,parents=5,unique_variant_frames=193)
    archive=V8+'/artifacts/source_snapshot.zip'
    return dict(schema='v8_multiview_mechanism_manifest_v1',status='METADATA_BOUND_NOT_EXECUTION_APPROVAL',
        selection='all_windows_of_all_tasks_with_prior_V6_positive_in_fixed2259; no model selection',
        observations=selected,counts=counts,geometry_settings=settings,qualify_cap_precision=True,
        archive_path=archive,archive_sha256=a[archive],source_manifest_path=manifest_path,
        source_manifest_sha256=b[manifest_path],source_sha256=opened,
        restrictions=['diagnostic_selection_not_unbiased_evaluation','no_training','no_calibration',
                      'no_C08_C10','all_selected_task_windows_retained','no_real_payload_reads'])


if __name__=='__main__':print(json.dumps(compile_manifest(),indent=2))

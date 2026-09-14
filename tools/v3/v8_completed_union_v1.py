"""Read sealed evidence, bind reusable observations and exact missing population."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUNS=(
 ('results/gate3_semantics/gate3_20260909_gse_v8_multiview_mechanism_v1_seed20260906',
  '2e5ddeef503cbe00fd851e2342d2480a4cd4230b831233202a809f04580cc9f6'),
 ('results/gate3_semantics/gate3_20260909_gse_v8_lifetime_pair_v1_seed20260906',
  '80b80a311372ea2a8477953f4431ef7c1bb7e3319ce165ccc2889f6732cb0674'))
ARCHIVE='b963b3207c65b2ed69504f4d6040e9b94a05d18f9305dc35ca0ca2a3ff04efea'
SETTINGS=dict(angular_segments=64,axial_spacing_m=.05,field_spacing_m=.025,
              cap_precision_policy='source_float64_float32_interval_v1',
              intersections='original_float32_rays_complete_operand_surfaces',recast_first_returns=False)


def audit(root=ROOT):
    pins_all={};completed={};duplicate=[];expected=None;run_counts=[]
    def read(path,sha):
        raw=(root/path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('evidence drift: '+path)
        return raw
    def key(source):return source['task'],source['source_sequence_id']
    for run,seal_sha in RUNS:
        seal_path=run+'/artifacts/evidence_sha256.txt'
        raw=read(seal_path,seal_sha)
        pins={p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
        if len(pins)!=len(raw.decode().splitlines()):raise ValueError('duplicate seal path')
        files={str(p.relative_to(root)) for p in (root/run).rglob('*') if p.is_file() and str(p.relative_to(root))!=seal_path}
        if files!=set(pins):raise ValueError('sealed file set mismatch')
        for p,h in pins.items():read(p,h)
        pins_all[seal_path]=seal_sha
        def saved(suffix):
            p=run+'/'+suffix;return json.loads(read(p,pins[p]))
        card=saved('config/data_card.json')
        if expected is None:
            expected=[s for e in card['scope']['entries'] for s in e['observations']]
            if len(expected)!=141 or len({key(s) for s in expected})!=141:raise ValueError('original population mismatch')
        expected_by_key={key(s):s for s in expected}
        p=run+'/logs/observations.jsonl'
        logs=[json.loads(line) for line in read(p,pins[p]).decode().splitlines()]
        local=set()
        for log in logs:
            if log['state']!='COMPLETED':continue
            s=log['source'];k=key(s)
            if k in local or expected_by_key.get(k)!=s:raise ValueError('source mismatch or duplicate')
            local.add(k)
            p=run+'/artifacts/'+log['evidence_file']
            d=json.loads(gzip.decompress(read(p,pins[p])))
            if d['archive_sha256']!=ARCHIVE:raise ValueError('algorithm archive drift')
            t=d['produced_targets']
            if t['geometry_evidence_settings']!=SETTINGS:raise ValueError('geometry parameter drift')
            if t['producer_version']!='joint_partial_reference_v8_lateral_terminal_exclusion_source_precision':raise ValueError('teacher version drift')
            record=dict(source=s,path=p,sha256=pins[p],target_sha256=t['target_record_sha256'],
                        anchors=len(t['record']['anchors']),openings=len(t['record']['openings']))
            if k in completed:
                old=completed[k];old_d=json.loads(gzip.decompress(read(old['path'],old['sha256'])))
                if any(old_d[field]!=d[field] for field in ('raw_interfaces','produced_targets')):
                    raise ValueError('duplicate result not equivalent')
                duplicate.append(dict(source=s,retained=old['path'],equivalent=p))
                del old_d
            else:completed[k]=record
            del d,t
        run_counts.append(dict(run=run,completed=len(local),seal_files=len(pins)))
    missing=[s for s in expected if key(s) not in completed]
    reused=[completed[key(s)] for s in expected if key(s) in completed]
    if len(reused)!=122 or len(missing)!=19 or len(duplicate)!=1:raise ValueError('unexpected coverage')
    return dict(schema='v8_completed_union_v1',status='PARTIAL_EVIDENCE_NOT_TRAINING_QUALIFICATION',
        archive_sha256=ARCHIVE,geometry_settings=SETTINGS,runs=run_counts,seal_sha256=pins_all,
        counts=dict(original=141,reused=len(reused),missing=len(missing),equivalent_duplicates=len(duplicate),
                    missing_tasks=len({s['task'] for s in missing}),missing_parents=len({s['parent_id'] for s in missing}),
                    missing_unique_variant_frames=len({(s['task'],f) for s in missing for f in s['frame_rows']})),
        reused=reused,equivalent_duplicates=duplicate,missing=missing,
        restrictions=['no_old_run_modification','no_training','no_new_teacher_calls','remaining_scope_requires_separate_spec'])


if __name__=='__main__':print(json.dumps(audit(),indent=2))

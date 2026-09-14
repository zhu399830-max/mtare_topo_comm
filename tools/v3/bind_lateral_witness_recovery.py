"""Metadata-only exact scope for missing saved lateral intersections."""
from _bootstrap import PROJECT_ROOT as ROOT
import json,gzip,hashlib
OLD='results/gate3_semantics/gate3_20260910_gse_local_pair_support_pilot_v1_seed20260906'
INPUT='results/gate3_semantics/gate3_20260910_gse_local_pair_pilot_inputs_v1_seed20260906'

def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()

def main():
    card=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_acquisition_junction_review_v1.json').read_text())
    original=json.loads((ROOT/OLD/'config/data_card.json').read_text())['scope']
    manifest=json.loads((ROOT/INPUT/'artifacts/manifest.json').read_text())
    pins={p:h for h,p in (x.split('  ',1) for x in (ROOT/INPUT/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    assert sha(INPUT+'/artifacts/manifest.json')==pins[INPUT+'/artifacts/manifest.json']
    a=original['archive'];assert sha(a['path'])==a['sha256']
    rows=[]
    for e in card['entries']:
        assert sha(e['target_path'])==e['target_sha256']
        x=json.loads(gzip.decompress((ROOT/e['target_path']).read_bytes()))
        assert x['archive_sha256']==a['sha256']
        supports=x['produced_targets']['teacher_provenance']['anchors'];rays=set()
        for support in supports:
            for field in ('surface_entry_witness_ray_indices','surface_departure_witness_ray_indices'):
                for group in support[field]:rays.update(group)
        match=[r for r in manifest if r['source']['task']==e['source']['task'] and r['source']['frame_rows']==e['frame_rows']]
        assert len(match)==1;r=match[0]
        paths=[INPUT+'/'+r[k] for k in ('student_path','source_evidence_path')]
        paths += [INPUT+'/artifacts/source_evidence/'+e['source']['task']+'_'+k+'.json' for k in ('constructions','codebooks')]
        rows.append(dict(source=e['source'],original_ray_indices=sorted(rays),ray_count=len(rays),
                         reference=dict(path=e['target_path'],sha256=e['target_sha256']),files={p:pins[p] for p in paths}))
    scope=dict(schema='gse_lateral_witness_recovery_scope_v1',status='BOUND_NOT_EXECUTED_OR_LABEL_QUALIFIED',
        entries=rows,archive=a,geometry_settings=original['geometry_settings'],observations=3,parents=3,
        selected_original_rays=sum(r['ray_count'] for r in rows),original_full_rays=172800,
        proposed_operation='Recover original side-surface entry t/XYZ/triangle/source and rejection evidence for all saved lateral witness rays; preserve global indices',
        constraints=['No teacher target recomputation','No changed radii/mesh/source definitions','No training or new labels','Missing witness remains failure/unknown, not silently dropped',
            'Use archived owner decoding; current named_sources change is not presumed equivalent','All source surfaces potentially intersecting each selected ray retained'],
        additional_source_payload_reads=0,teacher_calls=0)
    p=ROOT/'configs/v3/gate3/gse_lateral_witness_recovery_scope_v1.json'
    with p.open('x') as f:json.dump(scope,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(ray_counts=[r['ray_count'] for r in rows],total=scope['selected_original_rays'],executed=False)))

if __name__=='__main__':main()

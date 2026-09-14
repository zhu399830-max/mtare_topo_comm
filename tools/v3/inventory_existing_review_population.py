"""Read sealed reference metadata for fixed30 review population; no relabeling."""
from _bootstrap import PROJECT_ROOT as ROOT
import json,hashlib
from preview_supervision_acquisition import RUN,OUT,bound_metadata

def main():
    bound_metadata() # verifies known seal and original manifest/card
    seals={p:h for h,p in (line.split('  ',1) for line in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    m=json.loads((RUN/'artifacts/review_manifest.json').read_text())['observations']
    entries=[]
    for e in m:
        p=str((RUN/e['reference']).relative_to(ROOT))
        entries.append(dict(task=e['task'],view_id=e['view_id'],path=p,sha256=seals[p]))
    scope=dict(schema='gse_saved_review_metadata_scope_v1',operation='read_only_reference_metadata',entries=entries,
        observations=30,parents=10,physical_edge_units=10,implementation_frames=150,
        population='Existing fixed C01 rank0, three variants each; no model-dependent selection',
        authorization='User continue supervised-data acquisition; existing-source metadata only',
        limitation='Reference counts are not observed labels. Reading metadata invalidates subsequent claims of fully blind review by this same agent.',
        new_labels=0,training_steps=0)
    card=ROOT/'configs/v3/gate3/data_cards/gse_existing_review_population_metadata_v1.json'
    with card.open('x') as f:json.dump(scope,f,ensure_ascii=False,indent=2)
    rows=[]
    for e in entries:
        raw=(ROOT/e['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=e['sha256']:raise ValueError('reference drift')
        r=json.loads(raw)['construction_reference']
        rows.append(dict(task=e['task'],view_id=e['view_id'],local_axis_anchor_count=r['local_anchor_count'],
            reference_junctions=r['reference_junctions'],supported_reference_junctions=r['supported_reference_junctions'],
            teacher_complete=r['teacher_complete']))
    result=dict(observations=30,parents=10,rows=rows,
        junction_positive_observations=sum(r['supported_reference_junctions']>0 for r in rows),
        junction_reference_observations=sum(r['reference_junctions']>0 for r in rows),
        complete_references=sum(r['teacher_complete'] is True for r in rows),new_labels=0,training_steps=0)
    with (OUT/'population_metadata.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()

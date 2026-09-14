"""Compare fixed blind notes with sealed reference summaries; never label."""
from _bootstrap import PROJECT_ROOT as ROOT
import json,hashlib
from preview_supervision_acquisition import RUN,OUT,bound_metadata

def main():
    blind=bound_metadata()
    notes=OUT/'blind_observations.md'
    if not notes.exists():raise ValueError('blind notes must precede reveal')
    seals={p:h for h,p in (x.split('  ',1) for x in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    manifest=RUN/'artifacts/review_manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=seals[str(manifest.relative_to(ROOT))]:raise ValueError('manifest drift')
    entries=[]
    for e in json.loads(manifest.read_text())['observations'][:3]:
        p=str((RUN/e['reference']).relative_to(ROOT))
        entries.append(dict(view_id=e['view_id'],path=p,sha256=seals[p]))
    scope=dict(operation='read_only_reference_comparison',entries=entries,observations=3,independent_parents=1,
        blind_notes_sha256=hashlib.sha256(notes.read_bytes()).hexdigest(),new_labels=0,training_steps=0,
        authorization='User continue explicitly following blind-review and isolated-reference comparison proposal',training_authorized=False)
    card=ROOT/'configs/v3/gate3/data_cards/gse_supervision_acquisition_comparison_v1.json'
    with card.open('x') as f:json.dump(scope,f,ensure_ascii=False,indent=2)
    out=[]
    for entry in entries:
        raw=(ROOT/entry['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('reference drift')
        r=json.loads(raw);ref=r['construction_reference']
        # Preserve only bounded summary; no reference coordinates become labels.
        out.append(dict(view_id=entry['view_id'],reference_keys=list(ref),
                        field_sizes={k:len(v) for k,v in ref.items() if isinstance(v,(dict,list))},
                        scalar_fields={k:v for k,v in ref.items() if isinstance(v,(bool,int,float,str)) or v is None}))
    with (OUT/'reference_summary.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':main()

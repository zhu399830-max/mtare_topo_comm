"""Read-only eligibility of complete old cases; no matcher/target recomputation."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
import numpy as np
from mtare_topo.data.gse_membership_fit_reader import load_student_window
from mtare_topo.data.gse_frozen_observation_roi import validated_frozen_roi
OLD='results/gate3_semantics/gate3_20260911_gse_conditional_fit_geometry_v1_seed0'
OUT='docs/figures/gse_conditional_geometry_fit_v1/completed_reference_reuse.json'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    run=ROOT/OLD;spec=json.loads((run/'config/run_spec.json').read_text())
    card=json.loads((run/'config/data_card.json').read_text())
    summary=json.loads((run/'metrics/summary.json').read_text())
    sealed={p:h for h,p in (l.split('  ',1) for l in (run/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    assert summary['status']=='GATE_FAIL' and summary['completed']==28
    rows=summary['windows'];assert [r['case'] for r in rows]==list(range(28))
    pins={};result=[]
    for row in rows:
        i=row['case'];entry=card['scope']['entries'][i]
        for p in entry['paths'].values():
            if p not in pins:
                assert sha(ROOT/p)==spec['input_sha256'][p];pins[p]=spec['input_sha256'][p]
        files={}
        for name in ('source_records.npz','references.json','targets.npz','summary.json'):
            p=f'{OLD}/artifacts/case_{i:03d}/{name}';assert sha(ROOT/p)==sealed[p];files[p]=sealed[p]
        student=load_student_window(ROOT,entry['student_binding'])
        with np.load(ROOT/entry['paths']['feature'],allow_pickle=False) as f:
            roi=validated_frozen_roi(f['registered_returns_xyz_m'],student.valid_mask.reshape(-1),f['surface_return_indices'])
        with np.load(run/f'artifacts/case_{i:03d}/source_records.npz',allow_pickle=False) as z:
            assert np.array_equal(roi,z['roi_return_indices'])
        with np.load(run/f'artifacts/case_{i:03d}/targets.npz',allow_pickle=False) as z:
            assert np.array_equal(roi,z['original_roi_indices'])
        references=json.loads((run/f'artifacts/case_{i:03d}/references.json').read_text())
        assert references['identity']==entry['identity']
        result.append(dict(case=i,eligible=True,reason='same frozen student ROI, exact inputs and complete sealed original outputs',artifacts=files))
    output=dict(status='REUSE_ELIGIBLE_NOT_COPIED_OR_EXECUTED',completed_cases=result,
                eligible_count=28,incomplete_cases_not_reused=sorted(int(p.name.split('_')[1]) for p in (run/'artifacts').glob('case_*') if p.is_dir() and int(p.name.split('_')[1]) not in {r['case'] for r in rows}),
                input_sha256=pins,original_seal_sha256=sha(run/'artifacts/evidence_sha256.txt'),
                constraints=['Only ROI policy may change; original surface/reference/target algorithms and parameters must remain identical',
                             'New run must bind old artifact hashes and record reuse origin; never overwrite old run',
                             'No partial case reuse; no inference or target recomputation in this check'],
                labels_generated=0,training_steps=0)
    target=ROOT/OUT
    if target.exists():assert json.loads(target.read_text())==output
    else:target.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(eligible=28,new_labels=0,training_steps=0,output=OUT)))

if __name__=='__main__':main()

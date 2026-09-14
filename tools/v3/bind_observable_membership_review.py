"""Bind re-reviewed conservative supports without importing pairwise negatives."""
from _bootstrap import PROJECT_ROOT as ROOT
from observable_membership_review import RUN,CARD,OLD
from ai_branch_review import CACHE
from run_short_observation_chain import write,sha
from mtare_topo.representation.observable_membership import bind_observable_support
import json
import numpy as np

def main():
    out=ROOT/RUN;assert json.loads((out/'RUN_STATE.json').read_text())['state']=='RUNNING'
    card=json.loads((ROOT/CARD).read_text());review=json.loads((out/'artifacts/review.json').read_text())
    old_path=ROOT/OLD/'artifacts/review.json';old=json.loads(old_path.read_text())
    binding=dict(old_regions_sha256=sha(old_path),review_sha256=sha(out/'artifacts/review.json'),
        code_sha256={p:sha(ROOT/p) for p in ['tools/v3/bind_observable_membership_review.py','src/mtare_topo/representation/observable_membership.py']},
        rationale='Reuse already conservative boxes after all12 visual re-review. No expansion to force score improvement; positive support is not exclusion of other directions. Overlap and unselected remain unknown.',
        user_scope='Separate observed membership reference, old labels unchanged, no training',regions=[r['regions'] for r in old['observations']])
    write(out/'config/membership_binding.json',binding);rows=[]
    for i,e in enumerate(card['scope']['entries']):
        assert review['observations'][i]['observation']==i
        with np.load(ROOT/e['student_path'],allow_pickle=False) as d:valid=d['valid_mask'][4].reshape(-1).astype(bool)
        with np.load(ROOT/CACHE/f'artifacts/common_{i:02d}.npz',allow_pickle=False) as d:xyz=d['registered_returns_xyz_m'][4*11520:5*11520]
        ids=np.flatnonzero(valid);r=bind_observable_support(xyz[ids],valid[ids],binding['regions'][i])
        np.savez_compressed(out/f'artifacts/membership_{i:02d}.npz',ray_ids=ids,**r)
        rows.append(dict(observation=i,valid_rays=len(ids),positive_support_per_direction=r['known'].sum(0).tolist(),unsupported_direction_entries=int((r['support']==-1).sum()),unassigned_rays=int(r['unassigned'].sum()),overlap_unknown=int(r['overlap_unresolved'].sum()),negative_labels=0,training_qualified=False))
    write(out/'metrics/membership_binding.json',dict(observations=rows,positive_supports=sum(sum(x['positive_support_per_direction']) for x in rows),unassigned_rays=sum(x['unassigned_rays'] for x in rows),negative_labels=0,newly_resolved_old_unknown_rays=0,training_steps=0,
        limitation='Representation/meaning correction only: old unknowns were not made known. No proof of complete directional membership or full branch qualification.'))
    print(json.dumps(rows))

if __name__=='__main__':main()

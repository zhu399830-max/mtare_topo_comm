"""Frozen adjacent-window correspondence diagnostic, not identity ground truth."""
from _bootstrap import PROJECT_ROOT as ROOT
from explain_local_conflict_replay import OUT, RUN
from plot_conflict_observation_witnesses import CACHE
from run_short_observation_chain import write, sha
import json
import numpy as np
from scipy.spatial import cKDTree

def main():
    entries=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_branch_core_fit_v1.json').read_text())['scope']['entries']
    prior=json.loads((OUT/'unknown_group_support.json').read_text())['cases'];rows=[]
    for c in prior:
        i=c['observation'];j=i+1;e=entries[i]
        if j>=len(entries) or entries[j]['fragment_id']!=e['fragment_id']:
            rows.append(dict(observation=i,successor_available=False,reason='No next observation in the same recorded fragment; do not bridge fragments'));continue
        following=entries[j];assert following['task']==e['task'] and following['frame_rows'][3]==e['frame_rows'][4]
        with np.load(ROOT/e['student_path'],allow_pickle=False) as d:oldranges=d['ranges_m'][4].copy();oldvalid=d['valid_mask'][4].copy()
        with np.load(ROOT/following['student_path'],allow_pickle=False) as d:
            assert np.array_equal(oldranges,d['ranges_m'][3]) and np.array_equal(oldvalid,d['valid_mask'][3])
            valid=d['valid_mask'][4].reshape(-1).astype(bool)
        with np.load(CACHE/f'artifacts/common_{j:02d}.npz',allow_pickle=False) as d:xyz=d['registered_returns_xyz_m'].reshape(5,11520,3)
        old=json.loads((RUN/f'artifacts/groups_{i:02d}.json').read_text());new=json.loads((RUN/f'artifacts/groups_{j:02d}.json').read_text())
        source_ids=np.array(old['groups'][c['group_index']]);current_ids=np.flatnonzero(valid)
        source=xyz[3,source_ids];current=xyz[4,current_ids]
        dist,target=cKDTree(current).query(source);back=cKDTree(source).query(current)[1]
        mutual=back[target]==np.arange(len(source))
        byray=np.full(11520,-1,dtype=int)
        for k,g in enumerate(new['groups']):byray[g]=k
        assigned=byray[current_ids[target]]
        def histogram(mask):
            values,counts=np.unique(assigned[mask],return_counts=True)
            return {str(int(a)):int(b) for a,b in zip(values,counts)}
        rows.append(dict(observation=i,successor=j,shared_frame=int(e['frame_rows'][4]),shared_raw_frame_exact=True,source_group=c['group_index'],source_rays=len(source_ids),nearest_group_counts=histogram(np.ones(len(source_ids),dtype=bool)),mutual_nearest_rays=int(mutual.sum()),mutual_nearest_group_counts=histogram(mutual),distance_quantiles_m=np.quantile(dist,[0,.5,.9,1]).tolist(),mutual_distance_quantiles_m=np.quantile(dist[mutual],[0,.5,.9,1]).tolist() if mutual.any() else [],correspondence_is_ground_truth=False))
    write(OUT/'unknown_temporal_assignment.json',dict(cases=rows,code_sha256=sha(ROOT/'tools/v3/check_unknown_group_temporal_assignment.py'),labels_added=0,thresholds_used=False,interpretation='Current group indices are observation-local, not identities. Nearest and reciprocal-nearest returns are geometric diagnostics, not verified physical correspondence or branch correctness.'))
    print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__':main()

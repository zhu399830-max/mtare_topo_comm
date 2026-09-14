"""Saved schedules/targets/optimizer state, no forward or parameter update."""
from _bootstrap import PROJECT_ROOT as ROOT
import json
from collections import Counter
import numpy as np
import torch
from ai_junction_pilot import sha,write
from mtare_topo.representation.gse_structural_representation import GeometryStructureEncoder
from mtare_topo.representation.gse_conditional_geometry_loss import compile_conditional_loss_targets

BASE='results/gate3_semantics/'
ORIGINAL=BASE+'gate3_20260911_gse_new12_conditional_fit_v1_seed0'
TARGET=BASE+'gate3_20260911_gse_new12_conditional_geometry_v1_seed0'
AB=BASE+'gate3_20260911_gse_new12_corrected_ab_v1_seed0'
C=BASE+'gate3_20260911_gse_new12_axis_logit_correction_v1_seed0'
OUT='docs/figures/gse_conditional_geometry_fit_v1/training_wiring_audit.json'


def main():
    pins={}
    for run in (ORIGINAL,TARGET,AB,C):
        pins.update({p:h for h,p in (l.split('  ',1) for l in (ROOT/run/'artifacts/evidence_sha256.txt').read_text().splitlines())})
    used={}
    def checked(p):
        if sha(ROOT/p)!=pins[p]:raise ValueError('sealed drift '+p)
        used[p]=pins[p];return ROOT/p
    schedule=np.load(checked(ORIGINAL+'/artifacts/schedule.npy'),allow_pickle=False)
    assert schedule.shape==(2000,4)
    count=np.bincount(schedule.ravel(),minlength=12)
    for run in (AB,C):
        assert np.array_equal(np.load(checked(run+'/artifacts/schedule.npy'),allow_pickle=False),schedule)
    windows=[];mass=Counter()
    for i in range(12):
        with np.load(checked(TARGET+f'/artifacts/window_{i:02d}.npz'),allow_pickle=False) as y:
            t=compile_conditional_loss_targets(y,'cpu');known=t.known.numpy()[0];same=t.same.numpy()[0]
            h=t.height.numpy()[0];w=t.weights.numpy()[0].astype(float);n=y['neighbor_index'];c=y['patch_reference_component']
            pairs={tuple(sorted((int(c[p]),int(c[n[p,k]])))) for p,k in zip(*np.nonzero(known&~same))}
            values=dict(same_component=float(w[known&same].sum()),cross_component=float(w[known&~same].sum()),
                exactly_zero_height=float(w[known&(h==0)].sum()),abs_height_ge_01m=float(w[known&(abs(h)>=.1)].sum()),
                unknown=float(w[~known].sum()),known_weight=float(w[known].sum()))
            for k,v in values.items():mass[k]+=v*int(count[i])/schedule.size
            windows.append(dict(case=i,exposures=int(count[i]),cross_component_pairs=len(pairs),loss_mass=values))
    moments={}
    for variant,run in [('A',AB),('B',AB),('C',C)]:
        checkpoint=torch.load(checked(run+f'/artifacts/{variant}_final.pt'),map_location='cpu',weights_only=True)
        assert checkpoint['step']==2000 and checkpoint['variant']==variant
        model=GeometryStructureEncoder(variant);model.load_state_dict(checkpoint['model'],strict=True)
        names=list(model.named_parameters());opt=checkpoint['optimizer'];ids=[p for g in opt['param_groups'] for p in g['params']]
        assert len(ids)==len(names)
        rows=[]
        for key,(name,param) in zip(ids,names):
            state=opt['state'].get(key)
            if state is None:
                rows.append(dict(parameter=name,optimizer_state_present=False));continue
            avg=state['exp_avg'];sq=state['exp_avg_sq'];assert avg.shape==param.shape==sq.shape
            row=dict(parameter=name,optimizer_state_present=True,step=int(state['step']),
                exp_avg_abs_sum=float(avg.abs().sum()),exp_avg_sq_sum=float(sq.sum()))
            if name=='relation_head.2.weight':
                row.update(height_row_exp_avg_abs_sum=float(avg[1].abs().sum()),height_row_exp_avg_sq_sum=float(sq[1].sum()))
            if name=='relation_head.2.bias':row['height_bias_exp_avg_sq']=float(sq[1])
            rows.append(row)
        moments[variant]=rows
    sources=['src/mtare_topo/representation/gse_structural_representation.py',
        'src/mtare_topo/representation/gse_local_surface_affinity.py',
        'src/mtare_topo/representation/gse_surface_relation_model_v1.py',
        'src/mtare_topo/representation/gse_conditional_geometry_loss.py','tools/v3/train_conditional_geometry.py']
    # Verify interpretation uses exactly the source frozen for corrected C.
    spec=json.loads((ROOT/C/'config/run_spec.json').read_text())
    for p in sources:
        if sha(ROOT/p)!=spec['source_sha256'][p]:raise ValueError('current inspected source differs from run '+p)
        used[p]=sha(ROOT/p)
    result=dict(model_forwards=0,training_steps=0,shared_schedule_shape=list(schedule.shape),
        observations=12,exposures_per_model=int(schedule.size),windows=windows,exposure_weighted_loss_mass=dict(mass),optimizer_final_moments=moments,
        source_findings=dict(height_output='(head(sender,neighbor,neighbor-sender)[1] - head(neighbor,sender,sender-neighbor)[1])/2; no bounded height activation',
            position_input='All variants concatenate observed feature with patch XYZ/10 before learned adapter',
            geometry_input='B/C add unary geometry; C uses three sparse neighbor message layers with relativeXYZ/10',
            detached_inputs='Frozen observations/unary/relation only; no detach between new tokens and height loss',
            height_loss='Known weighted absolute error divided by10m, then averaged over four observations',
            pooled_region='Pooled region vector returned but not consumed by the current relation loss/readout',
            height_bias='Final height bias cancels under antisymmetric subtraction; zero bias gradient expected, not broken weight path'),
        limitations='Optimizer moments establish stored update evidence, not all per-step gradients or unique generalization cause. Only12 observations repeated; no independent cross-view consistency loss or topology consumer tested.',
        input_sha256=used)
    write(ROOT/OUT,result)
    print(json.dumps(dict(counts=count.tolist(),loss_mass=dict(mass),height_weights={v:next(r for r in rows if r['parameter']=='relation_head.2.weight') for v,rows in moments.items()}),indent=2))


if __name__=='__main__':main()

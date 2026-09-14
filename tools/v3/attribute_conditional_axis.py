"""Read-only exact loss/output attribution; no model forward or optimization."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
from summarize_conditional_geometry_fit import RUN,TARGET,OUT


def main():
    import numpy as np
    import torch
    from mtare_topo.representation.gse_conditional_geometry_loss import compile_conditional_loss_targets
    spec=json.loads((RUN/'config/run_spec.json').read_text())
    path='src/mtare_topo/representation/gse_conditional_geometry_loss.py'
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==spec['source_sha256'][path]
    for base in (RUN,TARGET):
        for line in (base/'artifacts/evidence_sha256.txt').read_text().splitlines():
            h,p=line.split('  ',1);assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    schedule=np.load(RUN/'artifacts/schedule.npy',allow_pickle=False)
    exposure=np.bincount(schedule.ravel(),minlength=12)/schedule.size
    all_y=[];all_w=[];rows=[];models={v:{'0':[],'2000':[]} for v in 'ABC'}
    for i in range(12):
        with np.load(TARGET/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as f:
            t=compile_conditional_loss_targets(f);mask=t.known[0].numpy();same=t.same[0].numpy()[mask]
            y=t.axis[0].numpy()[mask].astype(float);w=t.weights[0].numpy()[mask].astype(float)
            all_y.extend(y);all_w.extend(w*exposure[i])
            row=dict(observation=i,exposure=int((schedule==i).sum()),target_exact_one_weight=float(w[y==1].sum()),
                     same_weight=float(w[same].sum()),target_range=[float(y.min()),float(y.max())])
            rows.append(row)
            for v in 'ABC':
                for step in (0,2000):
                    with np.load(RUN/f'artifacts/{v}_step{step}_{i:02d}.npz',allow_pickle=False) as p:
                        values=p['axis'][0][mask].astype(float)
                    jac=values*(1-values);grad=w*np.sign(values-y)*jac
                    # Verify dL/dp independently; multiply sigmoid Jacobian only
                    # after autograd. This is the derivative at the saved output,
                    # not the full parameter gradient or past optimizer state.
                    pt=torch.tensor(values,dtype=torch.float64,requires_grad=True)
                    loss=(torch.tensor(w)*(pt-torch.tensor(y)).abs()).sum();loss.backward()
                    assert np.allclose(pt.grad.numpy()*jac,grad,rtol=1e-12,atol=1e-15)
                    models[v][str(step)].append(dict(observation=i,exposure_fraction=float(exposure[i]),
                        prediction_range=[float(values.min()),float(values.max())],
                        weighted_sigmoid_jacobian=float((w*jac).sum()),
                        output_mae=float((w*abs(values-y)).sum()),
                        logit_gradient_absolute_sum=float(abs(grad).sum()),
                        shared_logit_shift_gradient=float(grad.sum()),
                        cross_upward_weight=float(w[(~same)&(values<y)].sum()),
                        cross_downward_weight=float(w[(~same)&(values>y)].sum())))
    y=np.asarray(all_y);w=np.asarray(all_w);w/=w.sum();order=np.argsort(y)
    median=float(y[order][np.searchsorted(np.cumsum(w[order]),.5)])
    aggregate={}
    for v,steps in models.items():
        aggregate[v]={}
        for step,values in steps.items():
            aggregate[v][step]={k:sum(r['exposure_fraction']*r[k] for r in values) for k in
                ('weighted_sigmoid_jacobian','output_mae','logit_gradient_absolute_sum','shared_logit_shift_gradient')}
        aggregate[v]['jacobian_initial_over_final']=aggregate[v]['0']['weighted_sigmoid_jacobian']/aggregate[v]['2000']['weighted_sigmoid_jacobian']
    result=dict(scope='same12 saved predictions/labels and original loss; zero training/forward/weight change',
        exact_one_target_weight=float(w[y==1].sum()),weighted_target_median=median,
        constant_one_objective=float((w*abs(1-y)).sum()),constant_median_objective=float((w*abs(median-y)).sum()),
        parents=rows,aggregate=aggregate,per_parent_predictions=models,
        interpretation_limits=['Weighted median establishes optimal constant L1 prediction, not optimal input-conditioned network',
            'Derivative is dL/d symmetric axis logit at saved outputs, not full parameter gradients or unique causal history',
            'Sigmoid saturation and population imbalance do not prove missing input information or geometric learning impossible'])
    with (OUT/'axis_attribution.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('parents','per_parent_predictions')},indent=2))


if __name__=='__main__':main()

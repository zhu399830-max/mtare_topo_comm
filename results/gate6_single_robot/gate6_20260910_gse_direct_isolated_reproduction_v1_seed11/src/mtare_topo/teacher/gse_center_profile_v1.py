"""Bounded source-initialized nuisance refit at fixed +/-1m local centers.

Diagnostic only. The computational domain (axes .01..100m, exponent2..16,
angle +/-pi) is NOT an admissible-world prior or an acceptance gate. Boundary
solutions and exhausted evaluations are reported, never selected as valid
labels. No claim of global optimization or scan-equivalent alternative world.
"""
import numpy as np
from scipy.optimize import least_squares
from .gse_superellipse_constraints_v1 import local_constraints, finite_center_probe


def profile_center(points_uv_m, *, half_axes_m, exponent):
    points=np.asarray(points_uv_m,dtype=np.float64)
    base=local_constraints(points,center_uv_m=[0.,0.],half_axes_m=half_axes_m,exponent=exponent)
    probe=finite_center_probe(points,center_uv_m=[0.,0.],half_axes_m=half_axes_m,exponent=exponent)
    lower=np.array([np.log(.01),np.log(.01),np.log(2.),-np.pi])
    upper=np.array([np.log(100.),np.log(100.),np.log(16.),np.pi])
    initial=np.array([*np.log(half_axes_m),np.log(exponent),0.])
    if np.any(initial<lower) or np.any(initial>upper):
        raise ValueError('source outside declared diagnostic computational domain')
    rows=[]
    # The unshifted center receives the SAME nuisance refit. Comparing only
    # against raw source parameters would confound section-approximation error
    # with center ambiguity and unfairly favor the shifted fit.
    for step in [dict(sign=0,center_uv_m=[0.,0.]),*probe['probes']]:
        # If the first-order step leaves the domain, its center is still useful:
        # recover the same weak direction directly, not by clipping shape.
        if 'center_uv_m' not in step:
            j=base['jacobian'];n=j[:,2:]
            _,_,vh=np.linalg.svd(j[:,:2]-n@np.linalg.lstsq(n,j[:,:2],rcond=None)[0],full_matrices=True)
            direction=vh[-1]
            if direction[np.argmax(np.abs(direction))]<0:direction=-direction
            center=step['sign']*direction
        else:center=np.array(step['center_uv_m'])
        def evaluate(parameters):
            # Raw algebraic residual, same as previous local diagnostic.
            # Domain bounds keep powers finite for the fixed local observations.
            return local_constraints(points,center_uv_m=center,
                half_axes_m=np.exp(parameters[:2]),exponent=float(np.exp(parameters[2])),
                angle_rad=float(parameters[3]))
        def fun(parameters):return evaluate(parameters)['residual']
        def jac(parameters):return evaluate(parameters)['jacobian'][:,2:]
        result=least_squares(fun,initial,jac=jac,bounds=(lower,upper),max_nfev=100,
                             ftol=1e-8,xtol=1e-8,gtol=1e-8)
        rows.append(dict(sign=step['sign'],center_uv_m=center.tolist(),
            center_shift_m=float(np.linalg.norm(center)),nfev=result.nfev,
            solver_success=bool(result.success),solver_status=int(result.status),
            active_bound_indices=np.flatnonzero(result.active_mask).tolist(),
            residual_rms=float(np.sqrt(np.mean(result.fun**2))),
            half_axes_m=np.exp(result.x[:2]).tolist(),exponent=float(np.exp(result.x[2])),
            angle_rad=float(result.x[3])))
    return dict(original_residual_rms=float(np.sqrt(np.mean(base['residual']**2))),
        unshifted_profile=rows[0],profiles=rows[1:],qualified_label=False,scan_equivalence_verified=False)

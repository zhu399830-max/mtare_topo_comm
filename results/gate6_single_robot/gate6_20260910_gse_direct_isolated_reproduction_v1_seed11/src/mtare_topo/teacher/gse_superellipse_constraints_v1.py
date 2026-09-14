"""Local parameter sensitivity, not a fit, covariance or label qualifier.

Parameters: center/10m, log half-width, log half-height, log exponent, angle.
The nuisance shape/orientation subspace is removed before reporting center
sensitivity: a circle's undefined angle must not imply an undefined center.
"""
import numpy as np


def local_constraints(points_uv_m, *, center_uv_m, half_axes_m, exponent, angle_rad=0.):
    points=np.asarray(points_uv_m,dtype=np.float64)
    center=np.asarray(center_uv_m,dtype=np.float64)
    axes=np.asarray(half_axes_m,dtype=np.float64)
    if (points.ndim!=2 or points.shape[1:]!=(2,) or not 1<=len(points)<=57600
            or center.shape!=(2,) or axes.shape!=(2,) or np.any(axes<=0)
            or not all(np.isfinite(x).all() for x in (points,center,axes))
            or not np.isfinite(exponent) or exponent<2 or not np.isfinite(angle_rad)):
        raise ValueError('bounded finite points and valid source section parameters required')
    c,s=np.cos(angle_rad),np.sin(angle_rad)
    delta=points-center
    x=c*delta[:,0]+s*delta[:,1];y=-s*delta[:,0]+c*delta[:,1]
    normalized=np.column_stack((x,y))/axes
    absolute=np.abs(normalized)
    with np.errstate(over='raise',invalid='raise',divide='ignore'):
        power=absolute**exponent
        gradient=exponent*np.sign(normalized)*absolute**(exponent-1)/axes
        logs=np.zeros_like(absolute)
        np.log(absolute,out=logs,where=absolute>0)
        gx,gy=gradient.T
        jacobian=np.column_stack((10*(-c*gx+s*gy),10*(-s*gx-c*gy),
            -exponent*power[:,0],-exponent*power[:,1],
            exponent*np.sum(power*logs,axis=1),gx*y-gy*x))
    if not np.isfinite(jacobian).all():raise ValueError('nonfinite sensitivity')
    normalized_jacobian=jacobian/np.sqrt(len(points))
    nuisance=normalized_jacobian[:,2:]
    left,singular,_=np.linalg.svd(nuisance,full_matrices=False)
    tolerance=64*np.finfo(float).eps*max(nuisance.shape)*max(1.,float(singular[0]))
    basis=left[:,singular>tolerance]
    center_jacobian=normalized_jacobian[:,:2]
    profiled=center_jacobian-basis@(basis.T@center_jacobian)
    center_singular=np.linalg.svd(profiled,compute_uv=False)
    center_singular=np.pad(center_singular,(0,2-len(center_singular)))
    return dict(residual=power.sum(axis=1)-1.,jacobian=jacobian,
        nuisance_rank=int((singular>tolerance).sum()),
        center_profile_singular_values=center_singular,
        point_count=len(points),reference_parameters_used=True,
        qualified_label=False,
        limitation='First-order local sensitivity only; no noise model, global '
                   'uniqueness, confidence bound or deployed estimator claim.')


def finite_center_probe(points_uv_m, *, center_uv_m, half_axes_m, exponent, angle_rad=0.):
    """Fixed +/-1m local-section probes with first-order nuisance compensation.

    This is NOT reoptimization, a valid alternative world or a confidence
    interval. It need not retain the source centerline/sphere constraint. Exact
    nonlinear residuals show whether first-order predictions survive the step.
    """
    initial=local_constraints(points_uv_m,center_uv_m=center_uv_m,
        half_axes_m=half_axes_m,exponent=exponent,angle_rad=angle_rad)
    jac=initial['jacobian'];nuisance=jac[:,2:]
    coupling=np.linalg.lstsq(nuisance,jac[:,:2],rcond=None)[0]
    profiled=jac[:,:2]-nuisance@coupling
    _,_,right=np.linalg.svd(profiled,full_matrices=True)
    direction=right[-1]
    # Deterministic sign convention; both signs still evaluated.
    if direction[np.argmax(np.abs(direction))]<0:direction=-direction
    records=[]
    for sign in (-1,1):
        delta_center=sign*.1*direction  # center parameter unit is10m
        delta_nuisance=-coupling@delta_center
        center=np.asarray(center_uv_m)+10*delta_center
        try:
            with np.errstate(over='raise',under='raise',invalid='raise'):
                axes=np.asarray(half_axes_m)*np.exp(delta_nuisance[:2])
                power=float(exponent*np.exp(delta_nuisance[2]))
                angle=float(angle_rad+delta_nuisance[3])
            if power<2 or not np.isfinite(angle):
                records.append(dict(sign=sign,status='OUTSIDE_SOURCE_PARAMETER_DOMAIN'))
                continue
            changed=local_constraints(points_uv_m,center_uv_m=center,
                half_axes_m=axes,exponent=power,angle_rad=angle)
        except (FloatingPointError,ValueError,np.linalg.LinAlgError):
            records.append(dict(sign=sign,status='NUMERICAL_OR_PARAMETER_LIMIT'))
            continue
        linear=initial['residual']+jac@np.concatenate((delta_center,delta_nuisance))
        records.append(dict(sign=sign,status='FINITE_LOCAL_SECTION_PROBE',
            center_shift_m=float(np.linalg.norm(center-np.asarray(center_uv_m))),
            center_uv_m=center.tolist(),half_axes_m=axes.tolist(),exponent=power,angle_rad=angle,
            linear_residual_rms=float(np.sqrt(np.mean(linear**2))),
            nonlinear_residual_rms=float(np.sqrt(np.mean(changed['residual']**2)))))
    return dict(original_residual_rms=float(np.sqrt(np.mean(initial['residual']**2))),
        probes=records,qualified_label=False,
        limitation='Fixed1m local-section sensitivity check only; no physical '
                   'world feasibility, ray-equivalence or uncertainty interval.')

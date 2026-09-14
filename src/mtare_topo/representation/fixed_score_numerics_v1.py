"""Convex diagnostics of an unchanged affine head; no dataset or model search."""
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import expit


def equivalent_weights(masks, schedule):
    schedule=np.asarray(schedule,dtype=int)
    if schedule.ndim!=2 or schedule.shape[1]!=4 or np.any(schedule<0) or np.any(schedule>=len(masks)):
        raise ValueError('original four-observation batches required')
    counts=np.bincount(schedule.ravel(),minlength=len(masks));result=[]
    for (positive,negative),count in zip(masks,counts):
        p=np.asarray(positive,dtype=bool);n=np.asarray(negative,dtype=bool)
        if p.shape!=n.shape or np.any(p&n):raise ValueError('disjoint fixed labels required')
        groups=int(p.any())+int(n.any());w=np.zeros(len(p))
        for m in (p,n):
            if m.any():w[m]=count/(schedule.size*groups*int(m.sum()))
        result.append(w)
    return np.concatenate(result),counts


def objective(theta, X, y, weights):
    z=X@theta
    value=float(weights@np.logaddexp(0.,-y*z))
    grad=X.T@(-weights*y*expit(-y*z))
    return value,grad


def conditioning(X):
    # Full square invertible change of coordinates; no singular vector removed.
    _,s,v=np.linalg.svd(X,full_matrices=True)
    scales=np.full(X.shape[1],max(s[0]*1e-8,1e-12));scales[:len(s)]=np.maximum(s,scales[:len(s)])
    T=v.T/scales
    return T,dict(singular_values=s.tolist(),scale_floor=float(scales.min()),dimensions_retained=X.shape[1])


def exact_positive_margins(X,y,theta):
    t=[Fraction.from_float(float(v)) for v in theta]
    margins=[int(sign)*sum((Fraction.from_float(float(v))*a for v,a in zip(row,t)),Fraction(0)) for row,sign in zip(X,y)]
    return all(v>0 for v in margins),float(min(margins))


def separability(X,y):
    T,condition=conditioning(X);A=y[:,None]*(X@T)
    lp=linprog(np.zeros(X.shape[1]),A_ub=-A,b_ub=-np.ones(len(X)),bounds=[(None,None)]*X.shape[1],method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'time_limit':120.})
    out=dict(status='NUMERIC_UNDETERMINED',solver_status=int(lp.status),message=lp.message,conditioning=condition)
    if lp.success:
        theta=T@lp.x;valid,margin=exact_positive_margins(X,y,theta)
        out.update(witness_theta=theta.tolist(),exact_binary_rational_primal_verified=valid,min_signed_margin=margin)
        if valid:out['status']='STRICTLY_SEPARABLE_CERTIFIED'
    elif lp.status==2:
        # Do not equate floating point solver infeasibility with a proof.
        # Opposite labels at an exactly identical feature have a short exact certificate.
        for i in range(len(X)):
            for j in range(i):
                if y[i]!=y[j] and np.array_equal(X[i],X[j]):
                    out.update(status='NOT_STRICTLY_SEPARABLE_CERTIFIED',opposite_identical_rows=[j,i]);return out
        dual=linprog(np.zeros(len(X)),A_eq=np.vstack([A.T,np.ones(len(X))]),b_eq=np.r_[np.zeros(X.shape[1]),1.],
            bounds=(0,None),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'time_limit':120.})
        out['dual_solver_status']=int(dual.status)
        if dual.success:
            out['numerical_dual_weights']=dual.x.tolist()
            out['dual_residual_inf']=float(np.max(np.abs(A.T@dual.x)))
            # Recover and verify an EXACT convex dependence on binary-rational features.
            import sympy as sp
            support=np.flatnonzero(dual.x>1e-10)
            matrix=sp.Matrix([[sp.Rational(float(X[i,j]))*int(y[i]) for i in support] for j in range(X.shape[1])])
            basis=matrix.nullspace()
            if basis:
                N=sp.Matrix.hstack(*basis);nf=np.array(N,dtype=float)
                coeff=np.linalg.lstsq(nf,dual.x[support],rcond=None)[0]
                exact=N*sp.Matrix([sp.Rational(float(c)).limit_denominator(10**9) for c in coeff])
                if all(c>=0 for c in exact) and sum(exact)>0 and matrix*exact==sp.zeros(matrix.rows,1):
                    exact=exact/sum(exact)
                    out.update(status='NOT_STRICTLY_SEPARABLE_CERTIFIED',exact_dual_support=support.tolist(),exact_dual_weights=[str(c) for c in exact])
    return out


def solve_once(X,y,weights,theta0,callback=None):
    T,condition=conditioning(X);Z=X@T
    eta0=np.linalg.solve(T,theta0);history=[]
    def fun(eta):return objective(eta,Z,y,weights)
    def record(eta):
        value,g=fun(eta);row=dict(iteration=len(history)+1,loss=value,gradient_inf=float(np.max(np.abs(g))))
        history.append(row)
        if callback:callback(row)
    result=minimize(fun,eta0,jac=True,method='L-BFGS-B',callback=record,
        options={'maxiter':20000,'maxfun':50000,'ftol':1e-15,'gtol':1e-10,'maxls':50,'maxcor':30})
    theta=T@result.x;value,grad=objective(theta,X,y,weights);v2,g2=fun(result.x)
    sufficient=bool(result.success and np.max(np.abs(grad))<=1e-8 and np.max(np.abs(g2))<=1e-8 and abs(value-v2)<=1e-9)
    return theta,dict(solver_success=bool(result.success),message=str(result.message),iterations=int(result.nit),evaluations=int(result.nfev),
        loss=value,gradient_original_inf=float(np.max(np.abs(grad))),gradient_conditioned_inf=float(np.max(np.abs(g2))),
        objective_coordinate_difference=abs(value-v2),first_order_tolerance_met=sufficient,
        optimization_status='FIRST_ORDER_TOLERANCE_MET' if sufficient else 'OPTIMIZATION_NOT_SUFFICIENTLY_RESOLVED',
        finite_minimizer_claim=False,lower_bound_zero_gap=value,conditioning=condition,history=history)

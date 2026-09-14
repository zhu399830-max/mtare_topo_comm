"""Three prespecified ridge logistic heads, in original parameter coordinates."""
import numpy as np
from scipy.optimize import minimize
from .fixed_score_numerics_v1 import objective

LAMBDAS=(1e-2,1e-4,1e-6)

def regularized(theta,X,y,weights,lam):
    value,gradient=objective(theta,X,y,weights)
    penalty=theta.copy();penalty[-1]=0.
    return value+.5*lam*float(penalty@penalty),gradient+lam*penalty

def solve(X,y,weights,theta0,lam,callback=None):
    if lam not in LAMBDAS:raise ValueError('only three preregistered strengths')
    D=np.eye(X.shape[1])*lam;D[-1,-1]=0.
    # Full invertible curvature scaling; penalty remains in original weights.
    H=X.T@(weights[:,None]*X)*.25+D
    L=np.linalg.cholesky(H);T=np.linalg.solve(L.T,np.eye(len(theta0)))
    history=[]
    def fun(v):
        f,g=regularized(T@v,X,y,weights,lam);return f,T.T@g
    def record(v):
        f,g=fun(v);row=dict(iteration=len(history)+1,total=f,conditioned_gradient_inf=float(abs(g).max()));history.append(row)
        if callback:callback(row)
    result=minimize(fun,np.linalg.solve(T,theta0),jac=True,method='L-BFGS-B',callback=record,
        options={'maxiter':10000,'maxfun':30000,'ftol':1e-15,'gtol':1e-10,'maxls':50,'maxcor':30})
    theta=T@result.x;f,g=regularized(theta,X,y,weights,lam);_,cg=fun(result.x)
    converged=bool(result.success and abs(g).max()<=1e-8 and abs(cg).max()<=1e-8)
    return theta,dict(lambda_l2=lam,solver_success=bool(result.success),message=str(result.message),iterations=int(result.nit),
        function_evaluations=int(result.nfev),total=float(f),data_loss=objective(theta,X,y,weights)[0],
        gradient_original_inf=float(abs(g).max()),gradient_conditioned_inf=float(abs(cg).max()),converged=converged,
        weight_norm=float(np.linalg.norm(theta[:-1])),bias=float(theta[-1]),history=history)

def stable_scores(reference,executions,known):
    reference=np.asarray(reference);arrays=[np.asarray(v) for v in executions]
    error=max(float(np.max(abs(v-reference))) for v in arrays)
    minimum=min(float(np.min(abs(v[known]))) for v in [reference]+arrays)
    same=all(np.array_equal(v>=0,reference>=0) for v in arrays)
    return dict(max_logit_error=error,minimum_known_absolute_logit=minimum,all512_signs_agree=same,
        required_margin=max(1e-4,10*error),pass_stability=bool(same and minimum>max(1e-4,10*error)))

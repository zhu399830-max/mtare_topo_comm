"""No-update task-gradient Gram matrix on explicitly shared parameters."""
import torch


def task_gradients(terms, parameters):
    names=list(terms);vectors=[]
    for name in names:
        gradients=torch.autograd.grad(terms[name],parameters,retain_graph=True,allow_unused=True)
        vector=torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                          for p,g in zip(parameters,gradients)])
        if not torch.isfinite(vector).all():raise ValueError('nonfinite task gradient')
        vectors.append(vector)
    matrix=torch.stack(vectors)
    return names,matrix


def describe_gradients(names, matrix):
    gram=(matrix.double()@matrix.double().T).cpu()
    norms=gram.diag().clamp_min(0).sqrt()
    cos=gram/(norms[:,None]*norms[None,:]).clamp_min(1e-30)
    position=[i for i,n in enumerate(names) if n.endswith('_position')]
    other=[i for i in range(len(names)) if i not in position]
    cross=float(gram[position][:,other].sum())
    own=float(gram[position][:,position].sum())
    return dict(tasks=names,norms=norms.tolist(),gram=gram.tolist(),cosines=cos.tolist(),
        location_other_dot=cross,location_total_dot=own+cross,
        total_gradient_opposes_location=(own+cross)<0,
        interpretation='Euclidean first-order shared gradients at one checkpoint, not the Adam update or a causal root-cause proof')

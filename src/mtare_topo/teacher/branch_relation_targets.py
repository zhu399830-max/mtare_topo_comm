"""Conservative loss-side translation of archived weak section witnesses.

Different reference IDs alone NEVER produce a negative branch relation.
The archive has no observation-supported separation certificates; consequently
this translation does not qualify a population for fitting.
"""
import torch
from mtare_topo.representation.branch_relation_learning import BranchRelationTargets

def targets_from_section_witnesses(ray_ids,pair_indices,witness,*,device='cpu'):
    membership={int(i):set() for i in ray_ids}
    for key,indices in witness['supporting_ray_indices'].items():
        for i in indices:
            if i in membership:membership[i].add(key)
    values=[];known=[];refs=[];different=0;unsupported=0
    for a,b in pair_indices:
        left=membership[int(ray_ids[a])];right=membership[int(ray_ids[b])]
        unique=len(left)==len(right)==1
        positive=unique and left==right
        values.append(1. if positive else float('nan'));known.append(positive)
        refs.append('conditional_unique_section_ray_witness:'+next(iter(left)) if positive else '')
        different+=int(unique and left!=right);unsupported+=int(not unique)
    targets=BranchRelationTargets(torch.tensor(values,device=device),torch.tensor(known,device=device,dtype=torch.bool),tuple(refs),False)
    return targets,dict(weak_positive_pairs=sum(known),qualified_negative_pairs=0,
        different_section_pairs_kept_unknown=different,unsupported_or_competing_pairs=unsupported,
        training_qualified=False,old_reference_qualification_unchanged=True)

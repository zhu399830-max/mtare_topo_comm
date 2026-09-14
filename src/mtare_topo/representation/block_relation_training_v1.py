"""Paired builder using the original data/forward/loss/update interfaces.

No checkpoint selection or extra teacher. Both controls share the exact seeded
initial state and differ only in the explicit relation-attribute switch.
Existing point and query modules retain the original trainable policy; the
cached upstream encoder context remains frozen in the unchanged forward.
"""
from .development_paired_training import build_model,forward_observation,batch_schedule,state_sha256
from .development_corrective_training_v1 import objective,train_update
from .block_relation_increment_v1 import BlockRelationIncrementV1


def build_relation_model(*, relation_attributes, seed=0, device='cpu'):
    if type(relation_attributes) is not bool:raise ValueError('explicit C0/C1 switch required')
    model=build_model(seed=seed,device=device)
    model['head']=BlockRelationIncrementV1(model['head'],relation_attributes=relation_attributes)
    return model


__all__=['build_relation_model','forward_observation','batch_schedule','state_sha256','objective','train_update']

"""Shared forward for both groupings; loss has no branch terms or matching."""
import numpy as np
import torch
from .observed_anchor_training_v1 import build_model
from .observed_anchor_objective_v1 import observed_anchor_objective
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.teacher.gse_reference_exclusion_binding_v1 import bound_reference_exclusion


def forward(model,observation,grouping):
    if grouping not in ('SPATIAL','PRIMITIVE'):raise ValueError('only the registered two groupings')
    r=observation.student_representations[grouping]
    parameter=next(model.parameters())
    context=torch.as_tensor(np.array(r['context'],copy=True),device=parameter.device,dtype=parameter.dtype)
    return model['head'](r['blocks'],model['point'](r['blocks']),context)


def center_objective(output,observation):
    p=output.prediction;loss=observation.loss_only;f=loss['frozen_manifest'];produced=loss['produced_targets']
    if (canonical_sha(produced['record'])!=f['target_record_sha256']
            or produced['source_binding']!=f['source_binding']):raise ValueError('reference binding drift')
    target=loss['target'].position_m.to(p.position_m)
    positions=np.asarray([produced['record']['anchors'][i]['position_m'] for i in f['target_reference_indices']],float).reshape(-1,3)
    if not torch.equal(target,torch.as_tensor(positions,device=target.device,dtype=target.dtype)):raise ValueError('target changed')
    evidence=bound_reference_exclusion(loss['bundle'],loss['grid'],p.position_m.detach().cpu().double().numpy(),
        expected_binding=f['source_binding'],matching_radius_m=4.)
    result=observed_anchor_objective(p.position_m,p.presence_logits,target,
        confirmed_negative=torch.as_tensor(evidence['reference_negative_mask'],device=target.device,dtype=torch.bool))
    result['evidence']=evidence
    return result

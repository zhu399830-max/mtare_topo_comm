from types import SimpleNamespace
import numpy as np
import torch
from test_observed_anchor_detector_v1 import observation
from mtare_topo.representation.gse_block_points import bind_block_points
from mtare_topo.representation.grouping_pair_v1 import spatial_assignment
from mtare_topo.representation.grouping_center_training_v1 import forward
from mtare_topo.representation.grouping_zero_residual_init_v1 import build_model


def test_spatial_forward_uses_its_own_blocks_without_teacher_access():
    b=observation();s=bind_block_points(b.xyz_m,b.frame_index,spatial_assignment(b.xyz_m,len(b.block_ids)))
    class Student:
        student_representations={'SPATIAL':dict(blocks=s,context=np.zeros((len(s.block_ids),128),np.float32))}
        def __getattr__(self,k):raise AssertionError('no teacher access: '+k)
    model=build_model();model.eval()
    output=forward(model,Student(),'SPATIAL')
    assert torch.equal(output.prediction.position_m,output.query_positions_m)


def test_runner_routes_evaluation_and_training_to_spatial():
    import spatial_center_fit_v1 as wrapper
    import grouping_center_fit_v1 as executor
    from mtare_topo.representation.grouping_supervision_v2 import center_objective
    wrapper.configure()
    assert executor.GROUPING=='SPATIAL' and executor.center_objective is center_objective
    assert executor.POLICY['updates']==1000 and executor.POLICY['primitive_new_updates']==0

"""Synthetic training wiring, not authenticated real feature-cache evaluation."""
from dataclasses import replace
import pytest
import torch
from test_gse_surface_training_v1 import examples
from test_gse_surface_partial_losses_v2 import opening_setup
from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets
from mtare_topo.representation.gse_surface_training_context_v1 import SurfaceSourceLossContext
from mtare_topo.representation.gse_surface_training_v1 import train_paired_surface_models,SurfaceTrainingBudget


def setup():
    import numpy as np
    from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points
    from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches
    from mtare_topo.representation.gse_surface_relation_model_v1 import collate_surface_patches
    data,encoder=examples();_,kw=opening_setup()
    produced=kw['produced_targets'][0];manifest=kw['manifest_rows'][0]
    header=dict(data[0].header,target_binding_sha256=manifest['target_record_sha256'])
    s=kw['bundles'][0]['student']
    image=np.stack((s['ranges_m']/np.float32(50),s['valid_mask'].astype(np.float32)),axis=1)
    points,valid=register_causal_lidar_points(torch.from_numpy(image[None]),
        torch.from_numpy(s['relative_translation_current_sensor_m'][None]),
        torch.from_numpy(s['relative_yaw_current_sensor_deg'][None]))
    points=points.reshape(1,-1,3);valid=valid.reshape(1,-1)
    compact=replace(data[0].compact,points_xyz_m=points,valid=valid)
    patches=collate_surface_patches([extract_surface_patches(points[0].numpy(),valid[0].numpy(),np.repeat(np.arange(5),11520))])
    example=replace(data[0],header=header,compact=compact,patches=patches,targets=observed_targets([produced['record']]))
    context=SurfaceSourceLossContext(header['input_binding_sha256'],produced,manifest,
        kw['bundles'][0],kw['grids'][0],1.)
    return [example],encoder,[context]


def test_three_branches_share_source_loss_schedule_and_record_conditional_evidence():
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        data,encoder,contexts=setup()
        r=train_paired_surface_models(data,budget=SurfaceTrainingBudget(updates=1),
            frozen_encoder=encoder,loss_contexts=contexts)
        for branch in 'ABC':
            assert r.counts[branch]['optimizer_steps']==1
            assert r.history[branch][0]['example_indices']==r.schedule[0]
            for stage in ('initial','final'):
                e=r.evaluations[branch][stage][0]['source_loss_evidence']
                assert e['opening']['target_record_sha256']==contexts[0].manifest_row['target_record_sha256']
        assert len(set(r.initial_state_sha256.values()))==1
    finally:
        torch.set_num_threads(old)


def test_context_target_drift_rejected_before_training():
    data,encoder,contexts=setup()
    contexts[0].produced_targets['record']['openings'][0]['position_m'][0]=9.
    with pytest.raises(ValueError,match='fingerprint mismatch'):
        train_paired_surface_models(data,budget=SurfaceTrainingBudget(updates=1),
            frozen_encoder=encoder,loss_contexts=contexts)


def test_source_point_drift_rejected_before_optimizer():
    data,encoder,contexts=setup()
    data[0].compact.points_xyz_m[data[0].compact.valid]+=1.
    with pytest.raises(ValueError,match='point coordinates'):
        train_paired_surface_models(data,budget=SurfaceTrainingBudget(updates=1),
            frozen_encoder=encoder,loss_contexts=contexts)

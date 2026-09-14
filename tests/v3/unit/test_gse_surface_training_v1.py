"""Pure synthetic micro1x4 paired core; at most2 attempted updates per branch."""
from dataclasses import fields, replace

import numpy as np
import pytest
import torch
from torch import nn

from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures
from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches
from mtare_topo.representation.gse_surface_relation_model_v1 import SurfaceRelationModelV1, collate_surface_patches
from mtare_topo.representation.gse_surface_training_v1 import (
    SurfaceTrainingExample, SurfaceTrainingBudget, train_paired_surface_models,
    module_state_sha256, state_dict_sha256, FROZEN_READOUT_MODULES,
)
from tests.v3.unit.test_gse_surface_losses_v1 import targets


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def examples(unknown=False):
    torch.manual_seed(6); encoder=nn.Linear(2,3).eval().requires_grad_(False)
    encoder.register_buffer('fixed_buffer',torch.ones(1))
    digest=module_state_sha256(encoder)
    population=[]
    for sample in range(2):
        points=torch.zeros(1,57600,3);valid=torch.zeros(1,57600,dtype=torch.bool)
        for frame in range(5):
            for offset,location in enumerate(((.1,.1,1.1),(.2,.1,1.1),(.1,.2,1.1),(1.1,.1,2.1),(1.2,.1,2.1),(1.1,.2,2.1))):
                index=frame*16*720+offset;points[0,index]=torch.tensor(location)+sample*.02;valid[0,index]=True
        t=torch.arange(5)[:,None,None];az=torch.arange(720)[None,None]
        index=(t*180+az//4).expand(5,16,720).reshape(1,-1)
        compact=CompactFrozenDualPathFeatures(points,torch.zeros(1,900,128),valid,index)
        patch=collate_surface_patches([extract_surface_patches(points[0].numpy(),valid[0].numpy(),np.repeat(np.arange(5),16*720))])
        target=targets();target=replace(target,**{f.name:getattr(target,f.name).float() for f in fields(target) if getattr(target,f.name).dtype==torch.float64})
        if unknown:
            target=replace(target,**{name:torch.zeros_like(getattr(target,name)) for name in
                ('anchor_valid','opening_valid','direction_valid','dimension_valid','reachability_valid','physical_reference_valid','membership_valid')})
        header=dict(schema_version='gse_surface_cached_input_header_v1',observation_id=f'synthetic-{sample}',
            frame_indices=list(range(5*sample,5*sample+5)),coordinate_frame='current_sensor',input_binding_sha256='a'*64,
            target_binding_sha256='b'*64,frozen_encoder_state_sha256=digest)
        population.append(SurfaceTrainingExample(header,compact,patch,target))
    return population,encoder


def test_same_initial_schedule_fixed_final_actual_steps_and_frozen_readouts():
    data,encoder=examples();before=module_state_sha256(encoder)
    calls=[]
    encoder.register_forward_hook(lambda *args:calls.append(1))
    result=train_paired_surface_models(data,budget=SurfaceTrainingBudget(updates=2),frozen_encoder=encoder)
    assert not calls and module_state_sha256(encoder)==before
    assert len(set(result.initial_state_sha256.values()))==1
    assert len(result.schedule)==2 and all(len(group)==4 for group in result.schedule)
    for branch in 'ABC':
        c=result.counts[branch]
        assert c['optimizer_steps']==2 and c['effective_supervised_microbatches']==8
        assert c['training_microbatch_exposures']==8 and c['unique_population_observations']==2
        assert c['initial_eval_windows']==c['final_eval_windows']==2 and c['encoder_forward_windows']==0
        assert tuple(row['example_indices'] for row in result.history[branch])==result.schedule
        assert state_dict_sha256(result.final_states[branch])!=result.initial_state_sha256[branch]
        assert set(result.evaluations[branch])=={'initial','final'}
        for name,value in result.shared_initial_state.items():
            if name.split('.')[0] in FROZEN_READOUT_MODULES:
                assert torch.equal(value,result.final_states[branch][name])
    assert result.reliability_status=='UNTRAINED_UNCALIBRATED'


def test_all_unknown_groups_do_not_decay_any_parameter_or_fill_budget():
    data,encoder=examples(unknown=True)
    result=train_paired_surface_models(data,budget=SurfaceTrainingBudget(updates=2),frozen_encoder=encoder)
    for branch in 'ABC':
        count=result.counts[branch]
        assert count['optimizer_steps']==0 and count['skipped_all_unknown_groups']==2
        assert count['effective_supervised_microbatches']==0 and count['training_microbatch_exposures']==8
        assert state_dict_sha256(result.final_states[branch])==result.initial_state_sha256[branch]


def test_observed_partial_labels_train_detection_without_updating_unknown_task_heads():
    from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets
    data,encoder=examples()
    partial=[]
    for example in data:
        record=dict(schema='gse_surface_observed_targets_v1', coordinate_frame='current_sensor_m',
            source_frame_indices=example.header['frame_indices'],
            anchors=[dict(position_m=[.1,0.,0.],evidence='synthetic located anchor')],
            openings=[dict(position_m=[1.1,1.,0.],direction=None,width_m=None,height_m=None,
                           evidence='synthetic located opening; attributes unknown')],
            membership=[[None]], score_region=dict(center_m=[0.,0.,0.],radius_m=10.,
                anchors_complete=False,openings_complete=False,evidence='synthetic incomplete region'))
        partial.append(replace(example,targets=observed_targets([record])))
    result=train_paired_surface_models(partial,budget=SurfaceTrainingBudget(updates=1),frozen_encoder=encoder)
    for branch in 'ABC':
        assert result.counts[branch]['optimizer_steps']==1
        initial=result.shared_initial_state; final=result.final_states[branch]
        for name,value in initial.items():
            if name.split('.')[0] in ('opening_orientation','opening_dimensions','reachability',
                                      'member_anchor','member_opening'):
                assert torch.equal(value,final[name]), name
        assert any(not torch.equal(value,final[name]) for name,value in initial.items()
                   if name.startswith('opening_presence.'))


@pytest.mark.parametrize('kind',['header','encoder_parameter','encoder_buffer','encoder_mode'])
def test_actual_guard_drift_stops_before_optimizer(tmp_path,monkeypatch,kind):
    data,encoder=examples();original=SurfaceRelationModelV1.forward_compact
    def mutation(model,*args,**kwargs):
        result=original(model,*args,**kwargs)
        if kind=='header':data[0].header['frame_indices'][0]=999
        elif kind=='encoder_mode':encoder.train()
        else:
            with torch.no_grad():
                if kind=='encoder_parameter':encoder.weight.add_(1.)
                else:encoder.fixed_buffer.add_(1.)
        return result
    monkeypatch.setattr(SurfaceRelationModelV1,'forward_compact',mutation)
    def forbidden(*args,**kwargs):raise AssertionError('optimizer cannot run after drift')
    monkeypatch.setattr(torch.optim.AdamW,'step',forbidden)
    with pytest.raises(ValueError,match='changed|drift'):
        train_paired_surface_models(data,budget=SurfaceTrainingBudget(updates=2),frozen_encoder=encoder)


@pytest.mark.parametrize('budget',[SurfaceTrainingBudget(updates=True),SurfaceTrainingBudget(updates=2001),
    SurfaceTrainingBudget(updates=2,accumulation_steps=1),SurfaceTrainingBudget(updates=2,learning_rate=.01)])
def test_budget_cannot_silently_change(budget):
    data,encoder=examples()
    with pytest.raises(ValueError):train_paired_surface_models(data,budget=budget,frozen_encoder=encoder)


def test_trainable_encoder_or_forged_header_rejected_before_forward(monkeypatch):
    data,encoder=examples();encoder.weight.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'):
        train_paired_surface_models(data,budget=SurfaceTrainingBudget(updates=2),frozen_encoder=encoder)
    encoder.requires_grad_(False);data[0].header['frozen_encoder_state_sha256']='0'*64
    with pytest.raises(ValueError,match='fingerprint'):
        train_paired_surface_models(data,budget=SurfaceTrainingBudget(updates=2),frozen_encoder=encoder)


def test_deterministic_repetition_with_fixed_budget_and_seed():
    data,encoder=examples()
    first=train_paired_surface_models(data,budget=SurfaceTrainingBudget(updates=2),frozen_encoder=encoder)
    second=train_paired_surface_models(data,budget=SurfaceTrainingBudget(updates=2),frozen_encoder=encoder)
    assert first.schedule_sha256==second.schedule_sha256
    for branch in 'ABC':
        assert state_dict_sha256(first.final_states[branch])==state_dict_sha256(second.final_states[branch])
        assert first.history[branch]==second.history[branch]

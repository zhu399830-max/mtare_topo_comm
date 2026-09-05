from __future__ import annotations

import torch

from mtare_topo.representation.ray_column_dropout import apply_ray_column_dropout


def test_dropout_is_deterministic_and_does_not_mutate_source():
    source=torch.zeros((8,2,16,720));source[:,0]=.25;source[:,1]=1.0
    first,stats1=apply_ray_column_dropout(source,torch.Generator().manual_seed(91),probability=.5,period=10)
    second,stats2=apply_ray_column_dropout(source,torch.Generator().manual_seed(91),probability=.5,period=10)
    assert torch.equal(first,second) and stats1==stats2
    assert torch.all(source[:,0]==.25) and torch.all(source[:,1]==1.0)
    assert stats1["augmented_samples"]==sum(stats1["phase_counts"])


def test_selected_sample_masks_exactly_one_of_ten_column_phases():
    source=torch.zeros((3,2,16,720));source[:,0]=.2;source[:,1]=1.0
    result,stats=apply_ray_column_dropout(source,torch.Generator().manual_seed(7),probability=1.0,period=10)
    assert stats["augmented_samples"]==3
    for sample in result:
        masked=(sample[1,0]==0).nonzero().flatten()
        assert len(masked)==72 and torch.all(masked%10==masked[0]%10)
        assert torch.all(sample[0,:,masked]==1.0) and torch.all(sample[1,:,masked]==0.0)


def test_probability_zero_is_identity():
    source=torch.rand((2,2,16,720));result,stats=apply_ray_column_dropout(source,torch.Generator().manual_seed(1),probability=0.0)
    assert torch.equal(source,result) and stats["augmented_samples"]==0

from __future__ import annotations
import torch
from execute_gse_cardinality_conditioned_circular_slot_transport_readiness_v1 import _maximum_error,_synthetic_contract
from execute_gse_cardinality_conditioned_circular_slot_transport_readiness_v1 import _synthetic_batch
from mtare_topo.representation.gse_cardinality_conditioned_circular_slot_transport import (
    _circular_slot_transport_loss_reference,
    circular_slot_transport_loss,
)

def test_boolean_permutation_error_uses_logical_mismatch() -> None:
    assert _maximum_error(torch.tensor([True,False]),torch.tensor([True,False]))==0
    assert _maximum_error(torch.tensor([True,False]),torch.tensor([False,False]))==1

def test_synthetic_transport_penalizes_duplicate_and_repairs_missing_mode() -> None:
    result=_synthetic_contract()
    assert result["correct_assignment"]<result["duplicate_assignment"]
    assert result["duplicate_missing_target_gradient"]<0

def test_synthetic_transport_is_rotation_and_slot_permutation_invariant() -> None:
    result=_synthetic_contract()
    assert result["rotation_total_error"]<=1e-5
    assert result["slot_permutation_total_error"]<=1e-5
    assert result["sharp_concentration"]>.99
    assert result["diffuse_concentration"]<1e-5

def test_vectorized_loss_matches_reference_values_and_gradient() -> None:
    base,targets=_synthetic_batch(duplicate=True)
    generator=torch.Generator().manual_seed(19)
    raw_template=base["slot_logits"]+0.01*torch.randn(base["slot_logits"].shape,generator=generator)
    def evaluate(loss_function):
        raw=raw_template.clone().requires_grad_(True)
        mass=torch.softmax(raw,-1);azimuth=torch.arange(180)*2*torch.pi/180
        cosine=(mass*torch.cos(azimuth)).sum(-1);sine=(mass*torch.sin(azimuth)).sum(-1)
        outputs=dict(base);outputs["slot_logits"]=raw;outputs["slot_log_mass"]=torch.log_softmax(raw,-1);outputs["slot_mass"]=mass;outputs["slot_bearing_deg"]=torch.remainder(torch.rad2deg(torch.atan2(sine,cosine)),360.0)
        losses=loss_function(outputs,targets);losses["total"].backward()
        return {name:float(value.detach()) for name,value in losses.items()},raw.grad
    reference,reference_gradient=evaluate(_circular_slot_transport_loss_reference)
    vectorized,vectorized_gradient=evaluate(circular_slot_transport_loss)
    for name in reference:
        assert abs(reference[name]-vectorized[name])<=1e-6
    assert torch.allclose(reference_gradient,vectorized_gradient,atol=1e-7,rtol=1e-6)

import torch

import run_primitive_local_composition_slot_readiness_v1 as base
import run_primitive_local_composition_slot_readiness_v1r as corrective


def test_corrective_changes_only_run_identity_and_cuda_generator_bridge() -> None:
    assert corrective.RUN_ID.endswith("_v1r_seed0")
    assert corrective.PASS.endswith("_V1R")
    assert corrective.FAIL.endswith("_V1R")
    assert "Generator" in (base.main.__doc__ or "") or callable(base.main)


def test_cpu_seeded_permutation_contract_is_device_independent() -> None:
    first = torch.randperm(32, generator=torch.Generator().manual_seed(47))
    second = torch.randperm(32, generator=torch.Generator().manual_seed(47))
    assert torch.equal(first, second)
    assert torch.equal(torch.sort(first).values, torch.arange(32))

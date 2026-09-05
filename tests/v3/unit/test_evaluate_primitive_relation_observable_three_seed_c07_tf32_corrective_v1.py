import torch

from evaluate_primitive_relation_observable_three_seed_c07_tf32_corrective_v1 import (
    configure_numerical_contract,
)


def test_corrective_explicitly_matches_training_tf32_contract():
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    previous_precision = torch.get_float32_matmul_precision()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.use_deterministic_algorithms(False)
        assert configure_numerical_contract() == {
            "deterministic_algorithms": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn
        torch.set_float32_matmul_precision(previous_precision)
        torch.use_deterministic_algorithms(previous_deterministic)

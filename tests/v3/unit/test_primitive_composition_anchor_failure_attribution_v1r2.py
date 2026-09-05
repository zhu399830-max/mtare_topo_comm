import execute_primitive_composition_anchor_failure_attribution_v1r2 as executor
import run_primitive_composition_anchor_failure_attribution_v1r2 as runner


def test_v1r2_freezes_actual_source_evaluator_numerical_contract():
    assert executor.NUMERICAL_CONTRACT == {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "float32_matmul_precision": "highest",
        "deterministic_algorithms": False,
    }
    assert runner.ACTUAL_NUMERICAL_CONTRACT["cudnn_allow_tf32"] is True
    assert runner.ACTUAL_NUMERICAL_CONTRACT["deterministic_algorithms"] is False


def test_v1r2_binds_correct_sidecar_and_executor():
    assert runner._CorrectedAnchorRoot() / "artifacts/anchors" == runner.ANCHOR_TARGET_ROOT
    assert runner.ANCHOR_TARGET_ROOT.name == "anchor_targets"
    assert runner.EXECUTOR.name.endswith("failure_attribution_v1r2.py")
    assert runner.RUN_ID.endswith("_failure_attribution_v1r2_seed0")

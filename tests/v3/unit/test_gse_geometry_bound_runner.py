"""Run the proven full save/reload test against the corrected CPU tiny loop."""
import importlib
from pathlib import Path

import tests.v3.unit.test_gse_partial_structure_training_runner as previous


def test_actual_geometry_bound_training_saves_reproducible_complete_outputs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    runner = importlib.import_module("run_gse_geometry_bound_training_v1")
    monkeypatch.setattr(previous, "module", lambda _: runner)
    previous.test_real_cpu_three_branch_initial_final_scores_predictions_weights_and_previews(tmp_path, monkeypatch)


def test_unavailable_cuda_failure_sealed_and_retry_refused(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    runner = importlib.import_module("run_gse_geometry_bound_training_v1")
    monkeypatch.setattr(previous, "module", lambda _: runner)
    monkeypatch.setattr(runner, "validate_partial_structure_training_card", lambda _: None, raising=False)
    original_fixture = previous.lifecycle
    def adapted(root, patch):
        result = original_fixture(root, patch)
        patch.setattr(runner, "validate_geometry_bound_training_card", runner.validate_partial_structure_training_card)
        return result
    monkeypatch.setattr(previous, "lifecycle", adapted)
    previous.test_execute_cuda_absence_seals_failed_and_refuses_retry(tmp_path, monkeypatch)

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "gse_training_publisher_synthetic",
        ROOT / "tools/v3/publish_gse_training_figure_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validation(seed: int) -> dict:
    return {
        "loss": {"total": 2.0 + seed * 0.1},
        "event": {"macro_f1": 0.60 + seed * 0.01},
        "axis": {"mean_angular_error_deg": 6.0 + seed},
        "association_retrieval": {"top1_same_identity_precision": 0.98},
        "geometry_mae": {
            "width_m": 0.8,
            "height_m": 0.4,
            "slope_deg": 3.0,
            "curvature_per_m": 0.005,
        },
    }


def test_training_publisher_keeps_all_three_seed_curves_and_vector_sources(tmp_path: Path, monkeypatch) -> None:
    publisher = _load()
    fake_generator = tmp_path / "tools/v3/publish.py"
    fake_generator.parent.mkdir(parents=True)
    fake_generator.write_text("# synthetic frozen generator\n", encoding="utf-8")
    monkeypatch.setattr(publisher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "__file__", str(fake_generator))
    run = tmp_path / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
    (run / "metrics").mkdir(parents=True)
    state = {"state": "COMPLETED", "overall_status": publisher.EXPECTED_STATUS}
    seed_summaries = []
    sources = []
    for seed in (0, 1, 2):
        model_dir = run / f"artifacts/models/seed{seed}"
        model_dir.mkdir(parents=True)
        validation = _validation(seed)
        history = {
            "epoch": 1,
            "train_loss": {"total": 2.2 + seed * 0.1},
            "validation": validation,
        }
        history_path = model_dir / "epoch_metrics.jsonl"
        history_path.write_text(json.dumps(history) + "\n", encoding="utf-8")
        metrics_path = model_dir / "best_validation_metrics.json"
        metrics_path.write_text(json.dumps(validation), encoding="utf-8")
        child_path = model_dir / "summary.json"
        child_path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "epochs_completed": 1,
                    "strict_test_worlds_read": 0,
                    "mtare_worlds_read": 0,
                }
            ),
            encoding="utf-8",
        )
        sources.extend((history_path, metrics_path, child_path))
        seed_summaries.append({"seed": seed, "best_epoch": 1, "best_validation": validation})
    summary = {
        "overall_status": publisher.EXPECTED_STATUS,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "seeds": seed_summaries,
    }
    summary_path = run / "metrics/summary.json"
    state_path = run / "RUN_STATE.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    sources = [summary_path, state_path, *sources]
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
            for path in sources
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "docs/figures/gse_graph"
    result = publisher.publish(run, destination)
    assert result["published_files"] == 7
    for suffix in (".png", ".pdf", ".svg", ".csv", "_summary.json", "_provenance.json", "_sha256.txt"):
        assert (destination / f"gse_training_curves{suffix}").is_file()
    rows = (destination / "gse_training_curves.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4

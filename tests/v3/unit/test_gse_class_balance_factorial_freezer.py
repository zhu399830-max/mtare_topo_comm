import hashlib
import importlib
import json
from pathlib import Path

import pytest

from mtare_topo.governance import build_run_id
from mtare_topo.governance_class_balance_factorial import validate_class_balance_factorial_card
from tests.v3.unit.test_gse_geometry_bound_card import card as old_card


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    f = importlib.import_module("freeze_gse_class_balance_factorial_v1")
    monkeypatch.setattr(f, "PROJECT_ROOT", tmp_path)
    old = old_card()
    oldpath, rankpath = "configs/old_card.json", "configs/rank_card.json"
    write(tmp_path / oldpath, old)
    write(tmp_path / rankpath, {"synthetic": "rank metadata"})
    for relative in (*f.UNCHANGED_SOURCES, *f.TOOL_SOURCES):
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# synthetic source\n")
    src = {p: sha(tmp_path / p) for p in f.UNCHANGED_SOURCES}
    corrective = {"gate": 3, "seed": 0, "date": "20260905", "slug": "synthetic_corrective", "data_card": oldpath,
        "source_sha256": src, "command": ["env", "A=1", "B=1", "C=1", "D=1", "E=1", "/synthetic/bin/python"]}
    rank = {"gate": 3, "seed": 0, "date": "20260905", "slug": "synthetic_rank", "data_card": rankpath}
    write(tmp_path / f.CORRECTIVE_SPEC, corrective)
    write(tmp_path / f.RANK_SPEC, rank)
    lookup = {}
    for s, path, digest in ((corrective, f.CORRECTIVE_SPEC, f.CORRECTIVE_SEAL_SHA256), (rank, f.RANK_SPEC, f.RANK_SEAL_SHA256)):
        root = "results/gate3_semantics/" + build_run_id(s)
        entries = {root + "/config/run_spec.json": sha(tmp_path / path),
                   root + "/config/data_card.json": sha(tmp_path / s["data_card"]), root + "/metrics/summary.json": "e" * 64}
        if s is corrective:
            entries[root + "/artifacts/shared_initial_state.pt"] = "d" * 64
        lookup[digest] = entries
    lookup[f.EXPORT_SEAL_SHA256] = {v["path"]: v["sha256"] for v in old["sources"].values()}
    def selected(path, wanted, digest):
        entries = lookup[digest]
        assert wanted == set(entries)
        return dict(entries)
    monkeypatch.setattr(f, "selected_seal_entries", selected)
    p = tmp_path / f.METHOD_DOC
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("Synthetic method design\n")
    originalsha = f.sha
    def metadata_sha(path):
        assert path.suffix not in (".npz", ".pt", ".pth", ".ckpt")
        assert path.name not in ("summary.json", "manifest.json", "target_transport.json", "training_history.json")
        return originalsha(path)
    monkeypatch.setattr(f, "sha", metadata_sha)
    return f, lookup


def test_metadata_only_freezer_uses_six_payload_hashes_not_bytes(tmp_path, monkeypatch):
    f, _ = fixture(tmp_path, monkeypatch)
    c, s = f.documents()
    assert validate_class_balance_factorial_card(c).passed
    assert len(c["sources"]) == 6 and all(not (tmp_path / v["path"]).exists() for v in c["sources"].values())
    assert c["baseline_initial_state_sha256"] == "d" * 64
    assert not any(p.endswith(".pt") for p in c["sealed_sources"])
    assert s["training"]["total_steps"] == 2700
    assert s["evaluation"]["total_head_inference_windows"] == 3240
    assert s["expected_counts"]["xy_svg_figures"] + s["expected_counts"]["rank_svg_figures"] == 210
    assert s["command"][1] == "CUBLAS_WORKSPACE_CONFIG=:4096:8"
    assert s["command"][6] == "/synthetic/bin/python" and "CUDA_VISIBLE_DEVICES=" not in s["command"]
    assert set(f.TOOL_SOURCES) <= set(s["source_sha256"])
    assert not (tmp_path / f.CARD).exists() and not (tmp_path / f.SPEC).exists()


def test_old_metadata_snapshot_drift_is_rejected(tmp_path, monkeypatch):
    f, _ = fixture(tmp_path, monkeypatch)
    write(tmp_path / f.RANK_SPEC, {"gate": 3, "seed": 0, "date": "20260905", "slug": "synthetic_rank", "data_card": "configs/rank_card.json", "drift": True})
    with pytest.raises(ValueError, match="snapshot drift"):
        f.documents()


def test_sealed_original_loss_cannot_drift(tmp_path, monkeypatch):
    f, _ = fixture(tmp_path, monkeypatch)
    (tmp_path / f.UNCHANGED_SOURCES[1]).write_text("# changed old loss\n")
    with pytest.raises(ValueError, match="source drift"):
        f.documents()


def test_original_four_input_hash_drift_stops(tmp_path, monkeypatch):
    f, lookup = fixture(tmp_path, monkeypatch)
    first = next(iter(lookup[f.EXPORT_SEAL_SHA256]))
    lookup[f.EXPORT_SEAL_SHA256][first] = "0" * 64
    with pytest.raises(ValueError, match="hashes changed"):
        f.documents()


def test_no_overwrite(tmp_path, monkeypatch):
    f, _ = fixture(tmp_path, monkeypatch)
    write(tmp_path / f.CARD, {"preserve": True})
    with pytest.raises(RuntimeError, match="overwrite"):
        f.main()
    assert json.loads((tmp_path / f.CARD).read_text()) == {"preserve": True}

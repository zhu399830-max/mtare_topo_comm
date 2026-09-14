import copy
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys

import pytest

from mtare_topo.data.gse_surface_selection_v1 import select_parent, summarize_selection, VARIANTS
from mtare_topo import governance_surface_selection as scope_module


def fixture_report(parent="S01_flat_tree_small_C01", edges=16, decisions=3):
    intervals = []
    qbase, fbase = 0, 0
    for edge in range(edges):
        for direction in (0, 1):
            frames = [list(range(fbase + i, fbase + i + 5)) for i in range(decisions)]
            records = [{"variant": v, "task": parent + "__" + v,
                "sequence_rows": list(range(qbase, qbase + decisions)),
                "source_sequence_ids": list(range(qbase, qbase + decisions)), "frame_rows": frames,
                "decision_arc_m": [float(i + 4) for i in range(decisions)]} for v in VARIANTS]
            intervals.append({"traversal_id": f"{parent}:edge{edge:03}:d{direction}", "variants": records})
            qbase += decisions; fbase += decisions + 4
    return {"parent_id": parent, "partition": "c07" if parent.endswith("_C07") else "fit",
        "intervals": intervals, "short_intervals": [],
        "counts": {"variants": {v: {"frames": fbase, "sequences": qbase} for v in VARIANTS},
                   "traversals": len(intervals), "logical_sequences": qbase},
        "continuity_requires_source_audit": True, "metadata_truth_requires_source_reader": True, "duration_s": None}


def test_short_observations_valid_and_opposite_directions_are_one_edge():
    out = select_parent(fixture_report(), "fit")
    assert out["available_physical_edges"] == out["selected_physical_edges"] == 16
    rows = out["observations"]
    assert len(rows) == 48
    assert all(n == 3 for n in Counter(r["physical_edge_id"] for r in rows).values())
    assert len({r["traversal_id"] for r in rows}) == 16
    assert {r["decision_position_policy"] for r in rows} == {"first", "middle", "last"}
    for i in range(0, len(rows), 3):
        assert len({r["source_sequence_id"] for r in rows[i:i+3]}) == 1
    assert summarize_selection([out])["unique_variant_source_frames"] == 240
    assert summarize_selection([out])["independent_structure_count"] is None


def test_canonical_selection_independent_of_traversal_and_variant_order():
    source = fixture_report(edges=19)
    reverse = copy.deepcopy(source)
    reverse["intervals"].reverse()
    for t in reverse["intervals"]:
        t["variants"].reverse()
    assert select_parent(source, "fit") == select_parent(reverse, "fit")


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(parent_id="S01_flat_tree_small_C08"),
    lambda r: r.update(model_scores=[]),
    lambda r: r["intervals"][0]["variants"][0]["source_sequence_ids"].__setitem__(0, True),
    lambda r: r["intervals"][0]["variants"][0]["source_sequence_ids"].__setitem__(0, 208228),
    lambda r: r["intervals"][0]["variants"][0]["frame_rows"][0].__setitem__(4, 99),
    lambda r: r["intervals"][0]["variants"].pop(),
    lambda r: r["intervals"][0]["variants"][0]["decision_arc_m"].__setitem__(0, float("nan")),
    lambda r: r["intervals"].append(r["intervals"][0]),
    lambda r: r.update(continuity_requires_source_audit=False),
    lambda r: r["counts"].update(logical_sequences=1),
])
def test_input_drift_rejected(mutation):
    source = fixture_report()
    mutation(source)
    with pytest.raises(ValueError):
        select_parent(source, "fit")


def test_insufficient_distinct_edges_cannot_fill_using_both_directions():
    with pytest.raises(ValueError, match="INSUFFICIENT_DISTINCT_EDGES"):
        select_parent(fixture_report(edges=15), "fit")


def test_single_decision_and_empty_directions():
    source = fixture_report(decisions=1)
    assert len(select_parent(source, "fit")["observations"]) == 48
    empty = fixture_report(decisions=0)
    with pytest.raises(ValueError, match="INSUFFICIENT_DISTINCT_EDGES"):
        select_parent(empty, "fit")


def test_source_spacing_not_fabricated():
    source = fixture_report(decisions=3)
    for t in source["intervals"]:
        for v in t["variants"]:
            v["decision_arc_m"] = [1.25, 1.8, 2.625]
    selected = select_parent(source, "fit")["observations"]
    assert selected[0]["history_frame_spacing_m"] is None
    assert selected[0]["duration_s"] is None
    assert selected[0]["within_traversal_decision_spacing_m"] == [1.8 - 1.25, 2.625 - 1.8]


def make_card(file_hashes, seal_hash):
    scope = {"parent_ids": scope_module.PARENTS, "input_files_sha256": file_hashes,
        "source_seal": {"path": scope_module.INVENTORY + "/artifacts/evidence_sha256.txt", "sha256": seal_hash},
        "policy": scope_module.POLICY, "expected_observations": {"fit": 2880, "calibration": 240, "development": 240},
        "unknown_counts": {"unique_source_frames": None, "structure_entities": None, "labels": None, "duration_s": None},
        "time_basis": "sealed decision order and route arc; acquisition clock/history spacing unknown",
        "resources": {"wall_time_s": 120, "host_ram_bytes": 1073741824, "gpu_bytes": 0, "output_bytes": 33554432},
        "forbidden_payloads": ["scans", "poses", "mesh", "construction", "teacher", "models", "C08-C10", "benchmark"]}
    approval = {"status": "APPROVED", "approved_by": "synthetic-fixture-only", "approved_at": "2026-09-07",
        "authorized_operations": ["audit"], "authorized_gates": [3], "scope": "synthetic metadata fixture",
        "confirmation_reference": "unit test not a real user approval", "scope_sha256": scope_module.digest(scope)}
    return {"schema_version": scope_module.SCHEMA, "card_id": scope_module.CARD_ID, "operation": "audit",
        "purpose": "synthetic", "source_provenance": "synthetic", "limitations": "not real data",
        "scope": scope, "scope_sha256": scope_module.digest(scope), "approval": approval,
        "scientific_gate_pass": False, "training_eligibility": False}


def test_closed_card_cannot_grant_training_or_expand_worlds():
    good = make_card(dict.fromkeys(scope_module.INPUT_PATHS, "0" * 64), scope_module.SEAL_SHA256)
    assert scope_module.validate_surface_selection_card(good).passed
    bad = copy.deepcopy(good); bad["approval"]["authorized_operations"] = ["training"]
    assert not scope_module.validate_surface_selection_card(bad).passed
    bad = copy.deepcopy(good); bad["scope"]["parent_ids"] = ["S01_flat_tree_small_C08"]
    assert not scope_module.validate_surface_selection_card(bad).passed
    bad = copy.deepcopy(good); bad["training_eligibility"] = True
    assert not scope_module.validate_surface_selection_card(bad).passed


def load_runner():
    tools = Path(__file__).resolve().parents[3] / "tools/v3"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("surface_selection_runner_test", tools / "run_gse_surface_selection_v1.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("corrupt", [False, True])
def test_complete_runner_with_synthetic70parents_and_failure_seal(tmp_path, monkeypatch, corrupt):
    runner = load_runner()
    inventory = tmp_path / scope_module.INVENTORY / "artifacts"
    inventory.mkdir(parents=True)
    c07 = [p for p in scope_module.PARENTS if p.endswith("_C07")]
    split = {p: "fit" if not p.endswith("_C07") else ("calibration" if p in c07[:5] else "development") for p in scope_module.PARENTS}
    for p in scope_module.PARENTS:
        runner.write(inventory / (p + "_identity_intervals.json"), fixture_report(p))
    runner.write(inventory / "parent_split.json", split)
    runner.write(inventory / "parent_population.json", {"parents": scope_module.PARENTS, "complete_declared_parent_inventory": True})
    hashes = {p: runner.sha(tmp_path / p) for p in scope_module.INPUT_PATHS}
    seal = inventory / "evidence_sha256.txt"
    seal.write_text("".join(f"{h}  {p}\n" for p, h in hashes.items()))
    pin = runner.sha(seal)
    monkeypatch.setattr(scope_module, "SEAL_SHA256", pin)
    monkeypatch.setattr(runner, "SEAL_SHA256", pin)
    card = make_card(hashes, pin)
    card_path = tmp_path / "configs/card.json"; card_path.parent.mkdir()
    runner.write(card_path, card)
    spec = {"gate": 3, "date": "20260907", "slug": "synthetic_surface_fixture", "seed": 20260906,
            "operation": "audit", "data_card": "configs/card.json", "command": ["synthetic-only"],
            "source_sha256": {"configs/card.json": runner.sha(card_path)}}
    run = tmp_path / "results/gate3_semantics/gate3_20260907_synthetic_surface_fixture_seed20260906"
    for folder in ("config", "logs", "metrics", "artifacts"):
        (run / folder).mkdir(parents=True)
    runner.write(run / "RUN_STATE.json", {"state": "CREATED_NOT_EXECUTED"})
    runner.write(run / "config/run_spec.json", spec); runner.write(run / "config/data_card.json", card)
    (run / "config/command.txt").write_text(shlex.join(spec["command"]) + "\n")
    if corrupt:
        (inventory / (scope_module.PARENTS[0] + "_identity_intervals.json")).write_text("{}\n")
    assert runner.execute(spec, run, tmp_path) == int(corrupt)
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert summary["status"] == ("IDENTITY_SELECTION_FAIL" if corrupt else "IDENTITY_SELECTION_COMPLETE")
    if not corrupt:
        assert summary["result"]["observations"] == 3360
        assert summary["result"]["unique_variant_source_frames"] == 16800
    for line in (run / "artifacts/evidence_sha256.txt").read_text().splitlines():
        h, rel = line.split("  ", 1)
        assert runner.sha(tmp_path / rel) == h
    with pytest.raises(ValueError, match="fresh"):
        runner.execute(spec, run, tmp_path)

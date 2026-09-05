#!/usr/bin/env python3
"""One sealed, read-only inventory of the existing 180 composition observations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
import traceback

import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_scoped_inventory import ScopedCompositionInventory
from mtare_topo.governance import build_run_id, load_json, write_json
from mtare_topo.governance_inventory import validate_scoped_inventory_card


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def contained(relative):
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("input path outside project")
    return path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec, run = load_json(args.spec), args.run_dir.resolve()
    if (run != PROJECT_ROOT / "results/gate3_semantics" / build_run_id(spec)
            or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED"
            or load_json(run / "config/run_spec.json") != spec):
        raise RuntimeError("requires one fresh run snapshot; no overwrite or retry")
    started, error, summary, accessed = time.monotonic(), None, {}, {}
    write_json(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name})
    try:
        card = load_json(contained(spec["data_card"]))
        validation = validate_scoped_inventory_card(card)
        if not validation.passed or load_json(run / "config/data_card.json") != card:
            raise ValueError(f"inventory Data Card drift: {validation.errors}")
        if spec["operation"] != "audit" or card["observation_count"] != 180:
            raise ValueError("only the exact 180-row read-only inventory is authorized")
        for relative, digest in {**card["sealed_sources"], **spec["source_sha256"]}.items():
            if sha(contained(relative)) != digest:
                raise ValueError(f"frozen input/tool drift: {relative}")
        versions = {"python": platform.python_version(), "numpy": np.__version__, "zarr": zarr.__version__}
        if versions != spec["expected_versions"]:
            raise ValueError(f"environment drift: {versions}")
        write_json(run / "config/execution_environment.json", {**versions, "device": "cpu", "platform": platform.platform()})
        records = json.loads(contained(spec["selection_manifest"]).read_text())
        if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
            raise ValueError("selection manifest must be a list of row objects")
        selected = [{k: row[k] for k in ("task", "row_index", "source_global_sequence_index")} for row in records]
        if selected != card["selected_rows"] or len({r["task"] for r in records}) != 10:
            raise ValueError("selection differs from frozen 180-row source")
        # Seal indices are provenance metadata. No non-selected shard is opened.
        seals = {}
        for seal_path in spec["source_seals"]:
            with contained(seal_path).open() as stream:
                for line in stream:
                    digest, relative = line.strip().split(None, 1)
                    seals[str(contained(relative))] = digest
        reader = ScopedCompositionInventory(contained(spec["teacher_root"]), contained(spec["sidecar_root"]),
                                             selected, expected_sha256=seals)
        wanted_traces = {r["traversal"] for r in records}
        traces = {}
        with contained(spec["traversal_manifest"]).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["traversal_id"] in wanted_traces:
                    if row["traversal_id"] in traces:
                        raise ValueError("duplicate traversal metadata")
                    traces[row["traversal_id"]] = row
        if set(traces) != wanted_traces:
            raise ValueError("selected traversal metadata missing")
        observation_rows, task_rows = [], []
        with (run / "logs/inventory.log").open("x") as log:
            for task in reader.selection:
                data = reader.read_task(task)
                task_records = [r for r in records if r["task"] == task]
                construction_path = contained(spec["construction_root"]) / (task + ".json")
                if construction_path.resolve().parent != contained(spec["construction_root"]):
                    raise ValueError("construction path escapes selected root")
                digest = sha(construction_path)
                if seals.get(str(construction_path)) != digest:
                    raise ValueError("construction metadata not sealed")
                accessed[str(construction_path)] = digest
                construction = load_json(construction_path)
                if construction["parent_id"] != task.split("__")[0] or construction["geometry_realization"] != "c1_mixed":
                    raise ValueError("construction task identity drift")
                primitive_lookup = {str(p["primitive_id"]): i for i, p in enumerate(construction["realized_primitives"])}
                compositions = {str(c["node_id"]): c for c in construction["base_construction"]["composition_operations"]}
                for i, record in enumerate(task_records):
                    trace = traces[record["traversal"]]
                    node = str(record["target_node_scoring_only"])
                    if trace["to_node_id"] != node or trace["parent_id"] != task.split("__")[0]:
                        raise ValueError("selected node/traversal metadata drift")
                    composition = compositions[node]
                    supported_members = 0
                    present_members = 0
                    for member in composition["member_endpoints"]:
                        slots = np.flatnonzero(data["primitive_mask"][i] &
                                               (data["primitive_index"][i] == primitive_lookup[str(member["primitive_id"])]))
                        if len(slots) > 1:
                            raise ValueError("duplicate primitive slot")
                        if len(slots):
                            present_members += 1
                            supported_members += int(data["endpoint_observed"][i, slots[0], int(member["endpoint_index"])])
                    member_count = len(composition["member_endpoints"])
                    if int(composition["degree"]) != record["degree"] or member_count != record["degree"]:
                        raise ValueError("incident membership/degree metadata drift")
                    motion = data["relative_translation_current_sensor_m"][i]
                    window_chord_length = float(np.linalg.norm(np.diff(motion, axis=0), axis=1).sum())
                    observation_rows.append({"task": task, "row_index": record["row_index"], "degree": record["degree"],
                        "source_sequence": record["source_global_sequence_index"], "traversal_id_metadata_only": record["traversal"],
                        "node_id_metadata_only": node, "window_frame_rows": data["frame_row"][i].tolist(),
                        "window_measured_chord_length_m": window_chord_length, "whole_traversal_length_m": trace["length_m"],
                        "incident_members": member_count, "incident_primitives_present": present_members,
                        "incident_endpoints_supported": supported_members,
                        "complete_incident_endpoint_support": supported_members == member_count,
                        "all_observed_endpoints": int(data["endpoint_observed"][i].sum())})
                complete = sum(r["complete_incident_endpoint_support"] for r in observation_rows if r["task"] == task)
                task_rows.append({"task": task, "observations": len(task_records),
                                  "unique_source_frame_rows": len(np.unique(data["frame_row"])),
                                  "complete_incident_endpoint_support": complete})
                log.write(json.dumps(task_rows[-1]) + "\n"); log.flush()
        accessed.update(reader.opened)
        by_degree = {str(degree): {"observations": sum(r["degree"] == degree for r in observation_rows),
                                  "complete_support": sum(r["degree"] == degree and r["complete_incident_endpoint_support"] for r in observation_rows)}
                     for degree in (1, 2, 3, 4)}
        summary = {"observations": len(observation_rows), "parents": len(task_rows),
                   "unique_node_identities": len({(r["task"], r["node_id_metadata_only"]) for r in observation_rows}),
                   "referenced_window_frame_slots": len(observation_rows) * 5,
                   "unique_source_frames": sum(r["unique_source_frame_rows"] for r in task_rows),
                   "sensor_frames_decoded": 0, "model_inference": 0, "optimizer_steps": 0,
                   "duration_s": None, "time_basis": card["time_basis"], "by_degree": by_degree,
                   "complete_incident_support_observations": sum(r["complete_incident_endpoint_support"] for r in observation_rows),
                   "source_chunk_files_read": len(reader.opened), "training_ready": False,
                   "support_is_necessary_not_sufficient_for_event_label": True,
                   "note": "Existing endpoint support only; no event/port training labels generated. Degree alone is not an observable event label."}
        if summary["observations"] != 180 or summary["unique_node_identities"] != 100:
            raise ValueError("population drift")
        write_json(run / "artifacts/observation_inventory.json", observation_rows)
        write_json(run / "artifacts/task_inventory.json", task_rows)
        write_json(run / "artifacts/source_reads_sha256.json", {str(Path(p).relative_to(PROJECT_ROOT)): h for p, h in sorted(accessed.items())})
        for p, digest in accessed.items():
            if sha(Path(p)) != digest:
                raise ValueError("source changed during inventory")
        for p, digest in spec["source_sha256"].items():
            if sha(contained(p)) != digest:
                raise ValueError("tool changed during inventory")
        summary["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if summary["peak_rss_bytes"] > 4 * 1024**3 or time.monotonic() - started > 120:
            raise RuntimeError("inventory resource cap exceeded")
    except Exception:
        error = traceback.format_exc()
        (run / "logs/error.log").write_text(error)
    write_json(run / "metrics/summary.json", {"status": "INVENTORY_COMPLETE" if error is None else "INVENTORY_FAIL",
                "elapsed_s": time.monotonic() - started, "result": summary, "error": error, "scientific_gate_pass": False})
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "state": "COMPLETED" if error is None else "FAILED", "run_id": run.name, "error": error})
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text("".join(f"{sha(p)}  {p.relative_to(PROJECT_ROOT)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p != seal))
    print(json.dumps({"error": error, "result": summary, "seal_sha256": sha(seal)}))
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(main())

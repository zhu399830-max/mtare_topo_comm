"""Only synthetic metadata; no real source headers, arrays or checkpoints."""
import copy
import hashlib
import json

import pytest

from mtare_topo import governance_surface_input as governance
from mtare_topo.data import gse_surface_input_scope_v1 as scope_module
from mtare_topo.data.gse_surface_input_export_v1 import plan_array_access


def header(field, role):
    tail, dtype, first = {
        "range_m": ([16, 720], "<f4", 16), "valid_mask": ([16, 720], "|u1", 32),
        "frame_row": ([5], "<i4", 256), "source_global_sequence_index": ([], "<i8", 256),
        "relative_translation_current_sensor_m": ([5, 3], "<f4", 256),
        "relative_yaw_current_sensor_deg": ([5], "<f4", 256),
    }[field]
    n = 113 if role == "sensor" else 300
    return dict(zarr_format=2, shape=[n, *tail], chunks=[first, *tail], dtype=dtype,
                order="C", compressor=None, filters=None, fill_value=0)


def refreeze(card):
    card["scope_sha256"] = governance.digest(card["scope"])
    card["approval"]["scope_sha256"] = card["scope_sha256"]
    return card


def full_card():
    """Real210-task NAMES, entirely synthetic hashes and array populations."""
    tasks = governance.expected_tasks()
    files, plans = {}, {}
    for source in tasks.values():
        for role, names in governance.FIELDS.items():
            prefix = source[role] + "/"
            files.update({prefix + key: "a" * 64 for key in (".zgroup", ".zattrs")})
            for field in names:
                files[prefix + field + "/.zarray"] = "a" * 64
                plan = plan_array_access(header(field, role), field, role, list(range(80 if role == "sensor" else 16)))
                plans[prefix + field] = plan
                files.update({prefix + field + "/" + key: "a" * 64 for key in plan["chunk_keys"]})
    scope = dict(selection=dict(path=governance.SELECTION + "/artifacts/selection_manifest.json",
        sha256="b" * 64, seal_path=governance.SELECTION + "/artifacts/evidence_sha256.txt",
        seal_sha256=governance.SELECTION_SEAL), source_seals=copy.deepcopy(governance.SEALS),
        task_sources=tasks, fields=copy.deepcopy(governance.FIELDS), file_sha256=files, array_access=plans,
        population=dict(parents=70, physical_edge_units=1120, observations=3360, unique_variant_frames=16800,
            split_observations=dict(fit=2880, calibration=240, development=240), new_labels=0,
            effective_label_count=None, independent_structure_count=None, duration_s=None),
        spatial_temporal_basis="Synthetic five-frame indices; no acquisition clock.",
        teacher_source="Only stored relative motion and IDs, no teacher targets.",
        leakage_audit="Synthetic only; all actual210 authorized task names, no C08-C10 reads.",
        resources=copy.deepcopy(governance.RESOURCES), environment=copy.deepcopy(governance.ENVIRONMENT),
        forbidden_payloads=copy.deepcopy(governance.FORBIDDEN))
    return refreeze(dict(schema_version=governance.SCHEMA, card_id=governance.CARD_ID,
        operation="data_export", purpose="Synthetic validator fixture", limitations="No scientific labels.",
        scope=scope, scope_sha256="", approval=dict(status="APPROVED", authorized_operations=["data_export"],
            authorized_gates=[3], approved_by="synthetic", approved_at="synthetic", scope="synthetic only",
            confirmation_reference="Synthetic fixture, not an actual user conversation."),
        scientific_gate_pass=False, training_eligibility=False))


def test_full210task_scope_valid_and_six_fields_exact():
    card = full_card()
    assert len(card["scope"]["task_sources"]) == 210
    assert len(card["scope"]["array_access"]) == 1260
    assert governance.validate_surface_input_card(card).passed
    card["scope"]["fields"]["teacher"].append("primitive_mask")
    assert not governance.validate_surface_input_card(refreeze(card)).passed


@pytest.mark.parametrize("defect", ["none_plan", "collateral_lie", "extra_chunk", "missing_chunk", "role_swap", "float_count", "bool_zero", "float_gate", "C08", "selection_seal", "selection_hash_type"])
def test_resigned_forged_scope_fails_closed(defect):
    card = full_card(); scope = card["scope"]
    array = next(iter(scope["array_access"]))
    plan = scope["array_access"][array]
    if defect == "none_plan":
        scope["array_access"][array] = None
    elif defect == "collateral_lie":
        plan["decoded_actual_rows"] -= 1
    elif defect == "extra_chunk":
        scope["file_sha256"][array + "/999.0.0"] = "a" * 64
    elif defect == "missing_chunk":
        del scope["file_sha256"][array + "/" + plan["chunk_keys"][0]]
    elif defect == "role_swap":
        plan["role"] = "teacher"
    elif defect == "float_count":
        scope["population"]["observations"] = 3360.0
    elif defect == "bool_zero":
        scope["population"]["new_labels"] = False
    elif defect == "float_gate":
        card["approval"]["authorized_gates"] = [3.0]
    elif defect == "C08":
        source = next(iter(scope["task_sources"].values()))
        source["sensor"] = source["sensor"].replace("_C01", "_C08")
    elif defect == "selection_seal":
        scope["selection"]["seal_sha256"] = "c" * 64
    else:
        scope["selection"]["sha256"] = int("1" * 64)
    assert not governance.validate_surface_input_card(refreeze(card)).passed


def test_nonstring_file_key_returns_validation_report_not_exception():
    card = full_card()
    card["scope"]["file_sha256"][None] = "a" * 64
    assert not governance.validate_surface_input_card(card).passed


def small_compiler(monkeypatch, tmp_path, missing_chunk=False):
    task, source = next(iter(governance.expected_tasks().items()))
    tasks = {task: source}
    rows = [dict(task=task, parent_id=source["parent_id"], variant=source["variant"], split="fit",
        sequence_row=i, source_sequence_id=1000+i, frame_rows=list(range(i*5, i*5+5)),
        source_frame_count=113, source_sequence_count=300) for i in range(16)]
    raw = {}; seals = {}; selected_files = set()
    def add(path, value):
        raw[path] = json.dumps(value, sort_keys=True).encode()
        return hashlib.sha256(raw[path]).hexdigest()
    for role, names in governance.FIELDS.items():
        prefix = source[role] + "/"; entries = {}
        entries[prefix + ".zgroup"] = add(prefix + ".zgroup", {"zarr_format": 2})
        entries[prefix + ".zattrs"] = add(prefix + ".zattrs", dict(parent_id=source["parent_id"],
            partition=source["partition"], geometry_realization=source["variant"]))
        for field in names:
            entries[prefix + field + "/.zarray"] = add(prefix + field + "/.zarray", header(field, role))
            plan = plan_array_access(header(field, role), field, role, list(range(80 if role=="sensor" else 16)))
            for key in plan["chunk_keys"]:
                entries[prefix + field + "/" + key] = "a" * 64
        selected_files.update(entries)
        if missing_chunk and role == "sensor":
            del entries[prefix + "range_m/0.0.0"]
        # Excluded payload paths have NO bytes available, even to this fake store.
        entries[prefix.replace("_C01", "_C08") + "range_m/0.0.0"] = "a" * 64
        entries[prefix + "primitive_mask/0.0"] = "a" * 64
        path = role + "_synthetic_seal.txt"
        raw[path] = "".join(f"{h}  {p}\n" for p,h in entries.items()).encode()
        seals[role] = dict(path=path, sha256=hashlib.sha256(raw[path]).hexdigest())
    manifest_path = governance.SELECTION + "/artifacts/selection_manifest.json"
    manifest_hash = add(manifest_path, dict(schema_version="gse_surface_identity_selection_v1", observations=rows))
    seal_path = governance.SELECTION + "/artifacts/evidence_sha256.txt"
    raw[seal_path] = f"{manifest_hash}  {manifest_path}\n".encode()
    monkeypatch.setattr(scope_module, "SELECTION_SEAL", hashlib.sha256(raw[seal_path]).hexdigest())
    monkeypatch.setattr(scope_module, "SEALS", seals)
    monkeypatch.setattr(scope_module, "expected_tasks", lambda: tasks)
    # Only compile frontend is reduced: the210task validator above is not patched.
    monkeypatch.setattr(scope_module, "selected_rows", lambda manifest,tasks: {task:manifest["observations"]})
    opened = []
    def checked(root, path, expected, reads):
        opened.append(path)
        value = raw[path]  # Raises if compiler attempts any chunk/forbidden payload.
        actual = hashlib.sha256(value).hexdigest(); reads[path] = actual
        if actual != expected:
            raise ValueError("synthetic pinned drift")
        return value
    monkeypatch.setattr(scope_module, "checked_read", checked)
    return opened, selected_files


def test_compile_headers_only_filter_before_resolve_and_count_collateral(monkeypatch, tmp_path):
    opened, files = small_compiler(monkeypatch, tmp_path)
    scope, manifest, reads = scope_module.compile_input_scope(tmp_path)
    assert set(scope["file_sha256"]) == files
    assert set(opened) == set(reads) and len(opened) == len(set(opened)) == 14
    assert not any("_C08" in path or "primitive_mask" in path for path in opened)
    assert len(manifest["observations"]) == 16
    summaries = scope_module.access_summary(scope)
    assert summaries["sensor/range_m"]["decoded_actual_rows"] == 80
    assert summaries["sensor/valid_mask"]["decoded_actual_rows"] == 96
    assert summaries["teacher/frame_row"]["decoded_actual_rows"] == 256
    assert summaries["teacher/frame_row"]["selected_unique_rows"] == 16


def test_compile_missing_sealed_chunk_fails_before_any_payload(monkeypatch, tmp_path):
    small_compiler(monkeypatch, tmp_path, missing_chunk=True)
    with pytest.raises(ValueError, match="chunk absent"):
        scope_module.compile_input_scope(tmp_path)

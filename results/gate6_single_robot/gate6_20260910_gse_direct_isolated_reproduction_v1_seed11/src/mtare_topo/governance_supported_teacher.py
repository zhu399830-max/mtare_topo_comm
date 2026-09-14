"""Exact C01 construction-support teacher pilot, not model training authority."""
import re
from mtare_topo.governance_field_recovery import TASK, _digest, _relative_source, selection_sha256

SCHEMA = "v3_supported_construction_teacher_card_v1"
TEACHER_FIELDS = ("frame_row", "source_global_sequence_index", "primitive_index", "primitive_mask",
    "axis_control_current_sensor_m", "endpoint_half_axes_m", "endpoint_shape_exponent",
    "support_ray_count", "temporal_visibility", "endpoint_neighbor", "disconnected_overlap_packed",
    "relative_translation_current_sensor_m", "relative_yaw_current_sensor_deg")
SENSOR_FIELDS = ("range_m", "valid_mask", "primitive_membership_code", "sensor_xyz_m", "yaw_deg")
NATIVE_GEOMETRY = {"axis_spacing_m": .025, "mesh_axial_spacing_m": .05, "mesh_angular_segments": 64}
RESTRICTIONS = ("read_only_sources", "exact_C01_tasks_and_rows", "teacher_only_absolute_pose_and_identity",
    "no_model", "no_checkpoint", "no_old_predictions", "no_optimizer", "no_training", "no_calibration",
    "no_C02_C10", "no_scan_rerender", "no_new_worlds", "no_graph", "ignore_legacy_target_node",
    "unknown_unsupported_members", "unknown_terminal_without_cap", "no_partial_three_class_pass")


def validate_supported_teacher_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if not isinstance(card, dict): return ValidationReport(False, ("supported teacher card must be an object",))
    if card.get("schema_version") != SCHEMA: errors.append("invalid supported teacher schema")
    for key in ("card_id", "purpose", "teacher_source", "sampling_rule", "license_or_allowed_use", "support_proxy_limitations"):
        if not isinstance(card.get(key), str) or not card[key].strip(): errors.append(f"supported teacher {key} required")
    if card.get("partition") != "fit" or card.get("independent_sampling_unit") != "topology_parent":
        errors.append("original fit partition/topology_parent unit required")
    if "duration_s" not in card or card["duration_s"] is not None or card.get("time_basis") != "distance_sampled_no_acquisition_clock":
        errors.append("explicit untimed distance-sampled source required")
    for key, expected in {"observation_count":180,"parent_count":10,"legacy_node_count_inventory_only":100,
                          "unique_source_frame_count":900,"visible_fragment_count":1452,"frames_per_observation":5}.items():
        if type(card.get(key)) is not int or card[key] != expected: errors.append(f"exact {key}={expected} required")
    rows = card.get("selected_rows")
    if not isinstance(rows, list) or len(rows) != 180: errors.append("exact180 selected observations required"); rows=[]
    seen, source_seen, counts = set(), set(), {}
    for row in rows:
        if not isinstance(row, dict): errors.append("row must be object"); continue
        task,index,source = row.get("task"),row.get("row_index"),row.get("source_global_sequence_index")
        if (not isinstance(task,str) or not TASK.fullmatch(task) or type(index) is not int or index<0
                or type(source) is not int or source<0): errors.append("row outside exact C01 fit allowlist");continue
        if (task,index) in seen or (task,source) in source_seen: errors.append("duplicate row/source identity")
        seen.add((task,index));source_seen.add((task,source));counts[task]=counts.get(task,0)+1
    tasks=sorted(counts)
    if len(tasks)!=10 or any(n!=18 for n in counts.values()) or {t[:3] for t in tasks}!={f"S{i:02d}" for i in range(1,11)}:
        errors.append("exact ten S01--S10 C01 parents/eighteen rows each required")
    if card.get("tasks")!=tasks or card.get("worlds")!=[t.split("__")[0] for t in tasks]: errors.append("task/world selection mismatch")
    try: selection_digest=selection_sha256(rows)
    except (TypeError,ValueError): selection_digest=None;errors.append("finite JSON selection required")
    if not _digest(card.get("selection_sha256")) or card.get("selection_sha256")!=selection_digest:errors.append("selection hash mismatch")
    if card.get("teacher_fields")!=list(TEACHER_FIELDS) or card.get("sensor_fields")!=list(SENSOR_FIELDS):
        errors.append("teacher/sensor exact field allowlists required")
    native=card.get("native_geometry",{})
    if not isinstance(native,dict) or set(native)!=set(NATIVE_GEOMETRY): errors.append("exact native geometry parameters required");native={}
    for key,expected in NATIVE_GEOMETRY.items():
        if type(native.get(key)) is not type(expected) or native[key]!=expected:errors.append("native field/mesh parameter drift")
    if card.get("terminal_cap_policy")!="UNKNOWN_UNLESS_POSITIVE_SOURCE_CAP_EVIDENCE":errors.append("terminal cap uncertainty policy required")
    if card.get("terminal_cap_evidence_implemented") is not False:errors.append("this pilot has no implemented positive terminal cap evidence")
    for key in ("region_count","event_label_counts","member_label_count"):
        if key not in card or card[key] is not None:errors.append(f"new {key} must be unknown before pilot")
    if card.get("capacity_ready") is not False:errors.append("partial teacher cannot claim complete three-class capacity ready")
    restrictions=card.get("restrictions",{})
    for key in RESTRICTIONS:
        if not isinstance(restrictions,dict) or restrictions.get(key) is not True:errors.append(f"restriction missing: {key}")
    roots=card.get("source_roots",{})
    if not isinstance(roots,dict) or set(roots)!={"sensor","teacher","construction","codebook"}:
        errors.append("four exact source roots required");roots={}
    for root in roots.values():
        if (not _relative_source(root) or root.split("/")[-1]!="fit"
                or re.search(r"(?:^|[/_])C(?:0[2-9]|10)(?:[/_]|$)",root)):
            errors.append("source roots must be relative C01 fit roots")
    sources=card.get("sealed_sources",{})
    if not isinstance(sources,dict) or not sources:errors.append("source path/hash map required");sources={}
    for path,digest in sources.items():
        if not _relative_source(path) or not _digest(digest):errors.append("invalid source path/SHA256")
        elif path.endswith((".pt",".pth",".npz",".ckpt")):errors.append("checkpoint/prediction payload is not authorized")
    seals=card.get("source_seals",{})
    if not isinstance(seals,dict) or set(seals)!={"sensor","teacher"}:errors.append("sensor/teacher seals required");seals={}
    for path in seals.values():
        if not _relative_source(path) or not _digest(sources.get(path) if isinstance(path,str) else None):errors.append("source seals must be bound by hash")
    docs=card.get("task_json_sha256",{})
    expected_paths={roots[role]+"/"+task+".json" for role in ("construction","codebook") for task in tasks
                    if isinstance(roots.get(role),str)}
    if not isinstance(docs,dict) or set(docs)!=expected_paths or len(expected_paths)!=20:errors.append("exact20 construction/codebook JSON hashes required");docs={}
    for path,digest in docs.items():
        if not _relative_source(path) or not _digest(digest):errors.append("invalid teacher-only JSON source hash")
    for forbidden in ("checkpoint","checkpoints","inference","training","prediction_root","input_fields"):
        if forbidden in card:errors.append(f"forbidden model authority field: {forbidden}")
    approval=card.get("approval",{})
    if not isinstance(approval,dict):approval={}
    if (approval.get("status")!="APPROVED" or approval.get("authorized_operations")!=["teacher_generation"]
            or approval.get("authorized_gates")!=[3] or any(type(g) is not int for g in approval.get("authorized_gates",[]))):
        errors.append("teacher_generation-only Gate3 approval required")
    for key in ("approved_by","approved_at","scope","confirmation_reference"):
        if not isinstance(approval.get(key),str) or not approval[key].strip():errors.append(f"approval.{key} required")
    if not selection_digest or approval.get("selection_sha256")!=selection_digest:errors.append("approval must bind exact selection")
    return ValidationReport(not errors,tuple(errors))

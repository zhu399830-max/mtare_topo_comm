"""Narrow cached partial-target export authority; no training or new teacher."""
from pathlib import PurePosixPath
import re

from mtare_topo.governance_field_recovery import TASK, _digest, _relative_source, selection_sha256

SCHEMA = "v3_partial_structure_cache_export_card_v1"
SOURCE_NAMES = {"predictions":"all_predictions.npz","scoring_targets":"existing_scoring_targets.npz",
    "sample_manifest":"sample_manifest.json","identity_audit":"observation_audit.json",
    "partial_targets":"observation_targets.json"}
METHOD = "euclidean_control_point_mean_global_unique_orientation_unknown"
COORDINATE_COMMIT = "8ab2a09108e4f65e94ac606066474223b8403eb2"
TEACHER_COMMIT = "b15f01b48e27c6aa3e5aa4b1db7dd66a3e954fe7"
RAW_COUNTS = {"center_labels": 1206, "member_positive": 2155, "member_negative": 13510,
              "events": {"corridor": 872, "junction": 14, "terminal": 0}}
RESOURCES = {"wall_time_cap_s": 120, "host_ram_bytes": 4294967296, "output_bytes": 250000000, "gpu_bytes": 0}
RESTRICTIONS = ("read_only_sources", "exact_C01_tasks_and_rows", "only_five_cached_payloads",
    "no_model", "no_checkpoint", "no_optimizer", "no_training", "no_raw_sensor", "no_new_teacher",
    "no_calibration", "no_C02_C10", "no_graph", "no_checkpoint_selection", "all_32_prediction_slots",
    "gt_identity_and_masks_loss_only", "unknown_ambiguous_correspondence", "no_terminal_target",
    "no_complete_three_class_pass", "no_score_based_selection")
PRODUCER_FILES = {
    "coordinate_control": ("tools/v3/run_gse_coordinate_control_v1.py", "tools/v3/run_gse_point_axis_probe_v1.py"),
    "supported_teacher": ("src/mtare_topo/teacher/gse_supported_construction_teacher.py",
                          "tools/v3/run_gse_supported_construction_teacher_v1.py"),
}


def _exact_counts(value, expected):
    return (isinstance(value, dict) and set(value) == set(expected) and all(
        _exact_counts(value[k], n) if isinstance(n, dict) else type(value[k]) is int and value[k] == n
        for k, n in expected.items()))


def validate_partial_structure_export_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if not isinstance(card, dict): return ValidationReport(False, ("partial structure card must be an object",))
    if card.get("schema_version") != SCHEMA or card.get("operation") != "data_export":
        errors.append("exact partial cache data_export schema required")
    for key in ("card_id", "purpose", "teacher_source", "sampling_rule", "license_or_allowed_use", "identity_provenance"):
        if not isinstance(card.get(key), str) or not card[key].strip(): errors.append(f"{key} required")
    if card.get("partition") != "fit" or card.get("independent_sampling_unit") != "topology_parent":
        errors.append("C01 fit topology_parent units required")
    if card.get("time_basis") != "distance_sampled_no_acquisition_clock" or "duration_s" not in card or card["duration_s"] is not None:
        errors.append("distance sampling requires explicit null duration")
    for key, n in {"observation_count":180,"parent_count":10,"unique_source_frame_count":900,
                   "visible_fragment_count":1452,"frames_per_observation":5,"legacy_node_count_inventory_only":100}.items():
        if type(card.get(key)) is not int or card[key] != n: errors.append(f"exact {key}={n} required")
    rows = card.get("selected_rows")
    if not isinstance(rows, list) or len(rows) != 180: errors.append("exact180 selected rows required"); rows=[]
    seen, sources_seen, counts = set(), set(), {}
    for row in rows:
        if not isinstance(row, dict): errors.append("row must be object"); continue
        task,index,source = row.get("task"),row.get("row_index"),row.get("source_global_sequence_index")
        if (not isinstance(task,str) or not TASK.fullmatch(task) or type(index) is not int or index<0
                or type(source) is not int or source<0): errors.append("invalid C01 row/source"); continue
        if (task,index) in seen or (task,source) in sources_seen: errors.append("duplicate row/source")
        seen.add((task,index));sources_seen.add((task,source));counts[task]=counts.get(task,0)+1
    tasks=sorted(counts)
    if (len(tasks)!=10 or set(t[:3] for t in tasks)!={f"S{i:02d}" for i in range(1,11)}
            or any(n!=18 for n in counts.values())): errors.append("ten exact parents/eighteen rows each required")
    if card.get("tasks")!=tasks or card.get("worlds")!=[t.split("__")[0] for t in tasks]: errors.append("world/task drift")
    try: digest=selection_sha256(rows)
    except (TypeError,ValueError): digest=None
    if not _digest(card.get("selection_sha256")) or card.get("selection_sha256")!=digest: errors.append("selection hash drift")
    if card.get("prediction_branch")!="raw_coordinates" or card.get("native_method")!=METHOD: errors.append("frozen cache branch/method required")
    if card.get("cache_embeds_frame_identity") is not False: errors.append("cache lacks embedded frame IDs; explicit audit bridge required")
    if not _exact_counts(card.get("raw_teacher_counts"), RAW_COUNTS): errors.append("existing raw partial teacher counts drift")
    if card.get("effective_counts")!={"gt":None,"predicted":None}: errors.append("effective denominators must be unknown before export")
    if card.get("capacity_ready") is not False: errors.append("partial export cannot claim complete three-class capacity")
    if not _exact_counts(card.get("resources"), RESOURCES): errors.append("exact CPU export resource limits required")
    restrictions=card.get("restrictions")
    if not isinstance(restrictions,dict) or set(restrictions)!=set(RESTRICTIONS) or any(v is not True for v in restrictions.values()):
        errors.append("all exact export restrictions required")
    sealed=card.get("sealed_sources")
    if not isinstance(sealed,dict) or not sealed: errors.append("sealed source map required"); sealed={}
    for path,sha in sealed.items():
        if not _relative_source(path) or not _digest(sha): errors.append("invalid source path/hash")
        elif re.search(r"(?:^|[/_])C(?:0[2-9]|10)(?:[/_]|$)",path): errors.append("C02--C10 source paths forbidden")
        elif path.endswith((".pt",".pth",".ckpt")): errors.append("checkpoint payload forbidden")
    payloads=card.get("sources")
    if not isinstance(payloads,dict) or set(payloads)!=set(SOURCE_NAMES): errors.append("exact five cached sources required"); payloads={}
    paths=[]
    for key,value in payloads.items():
        if not isinstance(value,dict) or set(value)!={"path","sha256"}: errors.append("source must be path/SHA object"); continue
        path,sha=value["path"],value["sha256"]
        if (not _relative_source(path) or not _digest(sha) or PurePosixPath(path).name!=SOURCE_NAMES[key]
                or sealed.get(path)!=sha): errors.append("payload path/hash binding drift")
        paths.append(path)
    if len(paths)!=5 or len({p for p in paths if isinstance(p,str)})!=5: errors.append("five distinct payloads required")
    if any(p.endswith(".npz") and p not in paths for p in sealed if isinstance(p,str)): errors.append("extra numeric payload forbidden")
    seals=card.get("source_seals")
    if not isinstance(seals,dict) or set(seals)!={"coordinate_control","local_teacher","supported_teacher"}:
        errors.append("three original producer seals required"); seals={}
    for path in seals.values():
        if not isinstance(path,str) or not _relative_source(path) or path not in sealed: errors.append("unbound source seal")
    provenance=card.get("producer_provenance")
    if not isinstance(provenance,dict) or set(provenance)!=set(PRODUCER_FILES): errors.append("both frozen producers required");provenance={}
    for role,value in provenance.items():
        if not isinstance(value,dict) or set(value)!={"git_commit","spec","card","source_sha256"}: errors.append("invalid producer provenance");continue
        expected=COORDINATE_COMMIT if role=="coordinate_control" else TEACHER_COMMIT
        if value["git_commit"]!=expected: errors.append("frozen producer Git version drift")
        for key in ("spec","card"):
            path=value[key]
            if not isinstance(path,str) or not _relative_source(path) or path not in sealed: errors.append("unbound producer spec/card")
        hashes=value["source_sha256"]
        if not isinstance(hashes,dict) or set(hashes)!=set(PRODUCER_FILES[role]) or any(not _digest(v) for v in hashes.values()):
            errors.append("exact historical producer source hashes required")
    for key in ("training","optimizer","checkpoint","checkpoints","inference","sensor_root","teacher_root"):
        if key in card: errors.append(f"unauthorized field {key}")
    approval=card.get("approval",{})
    if not isinstance(approval,dict): approval={}
    if (approval.get("status")!="APPROVED" or approval.get("authorized_operations")!=["data_export"]
            or approval.get("authorized_gates")!=[3] or any(type(g) is not int for g in approval.get("authorized_gates",[]))):
        errors.append("data_export-only Gate3 approval required")
    for key in ("approved_by","approved_at","scope","confirmation_reference"):
        if not isinstance(approval.get(key),str) or not approval[key].strip(): errors.append(f"approval.{key} required")
    if not digest or approval.get("selection_sha256")!=digest: errors.append("approval must bind exact selection")
    if approval.get("sources")!=payloads: errors.append("approval must bind exact five source paths/hashes")
    return ValidationReport(not errors,tuple(errors))

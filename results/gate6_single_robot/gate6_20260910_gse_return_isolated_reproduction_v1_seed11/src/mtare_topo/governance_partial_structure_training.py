"""Independent fixed-budget cached-head training, never old export authority."""
from pathlib import PurePosixPath
import re

from mtare_topo.governance_field_recovery import _digest, _relative_source
from mtare_topo.governance_partial_structure import validate_partial_structure_export_card

SCHEMA="v3_partial_structure_head_training_card_v1"
SOURCE_NAMES={"inputs":"partial_training_inputs.npz","manifest":"manifest.json",
              "target_transport":"target_transport.json","summary":"summary.json"}
EXPORT_SEAL_SHA256="db28c9df03373dfa17a3222688f85d95679e6a0cbdabe339fdf74f15f35b514e"
IDENTITY_FIELDS=("partition","sampling_rule","license_or_allowed_use","time_basis","duration_s","independent_sampling_unit",
    "observation_count","parent_count","unique_source_frame_count","frames_per_observation","visible_fragment_count",
    "legacy_node_count_inventory_only","worlds","tasks","selected_rows","selection_sha256","producer_provenance","raw_teacher_counts")
TRAINING={"branches":["gt_axes","predicted_axes","predicted_no_relations"],"seed":0,"hidden":64,
    "steps_per_branch":300,"total_steps":900,"batch_size":18,"lr":.001,"optimizer":"Adam","weight_decay":0.,
    "device":"cuda:0","parameters_per_head":26826,"checkpoint_selection":"final_only",
    "sample_schedule":"shared_seed0_without_replacement_epochs","same_initial_state":True,
    "use_relations":{"gt_axes":True,"predicted_axes":True,"predicted_no_relations":False}}
EVALUATION={"stages":["initial","final"],"observations_per_stage_branch":180,"total_head_inference_windows":1080,
    "member_threshold":.5,"threshold_calibration":"none_explicit_uncalibrated",
    "region_matching":"center_geometry_only","read_training_matches":False,
    "known_event_classes":["corridor","junction"],"scientific_gate_pass":False}
RESOURCES={"wall_time_cap_s":1800,"host_ram_bytes":4294967296,"gpu_bytes":4294967296,"output_bytes":500000000}
EFFECTIVE_COUNTS={branch:{"member_positive":2152,"member_negative":13489,"unknown_member_positive":3,
    "unknown_member_negative":21,"center_labels":1206,"event_labels":886,"numeric_input_valid_queries":queries}
    for branch,queries in (("gt",2892),("predicted",11520))}
RESTRICTIONS=("read_only_sources","only_four_exported_payloads","exact_C01_tasks_and_rows","cached_axes_only",
    "all_32_prediction_slots","gt_masks_loss_only","no_backbone","no_input_checkpoint","no_new_sensor_frames",
    "no_new_teacher","no_calibration","no_C02_C10","no_graph","no_best_checkpoint_selection",
    "no_complete_three_class_claim","no_detection_or_generalization_claim","no_scientific_gate_pass",
    "same_initialization_order_budget","unknown_labels_unchanged","final_fixed_step300")


def _exact(value,expected):
    if type(value) is not type(expected): return False
    if isinstance(expected,dict): return set(value)==set(expected) and all(_exact(value[k],v) for k,v in expected.items())
    if isinstance(expected,list): return len(value)==len(expected) and all(_exact(a,b) for a,b in zip(value,expected))
    return value==expected


def validate_partial_structure_training_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    if not isinstance(card,dict): return ValidationReport(False,("partial head training card must be an object",))
    if card.get("schema_version")!=SCHEMA or card.get("operation")!="training": errors.append("independent partial-head training schema required")
    original=card.get("export_identity_reference")
    report=validate_partial_structure_export_card(original)
    if not report.passed: errors.append("valid original export identity evidence required: "+"; ".join(report.errors))
    if not isinstance(original,dict): original={}
    for key in IDENTITY_FIELDS:
        if key not in card or key not in original or not _exact(card[key],original[key]): errors.append(f"original export identity drift: {key}")
    for key in ("card_id","purpose","teacher_source","scope_limitations"):
        if not isinstance(card.get(key),str) or not card[key].strip(): errors.append(f"{key} required")
    for key,expected in (("training",TRAINING),("evaluation",EVALUATION),("resources",RESOURCES),("effective_counts",EFFECTIVE_COUNTS)):
        if not _exact(card.get(key),expected): errors.append(f"frozen {key} contract drift")
    if card.get("capacity_ready") is not False: errors.append("no complete three-class capacity")
    restrictions=card.get("restrictions")
    if not _exact(restrictions,dict.fromkeys(RESTRICTIONS,True)): errors.append("exact head-only restrictions required")
    sealed=card.get("sealed_sources")
    if not isinstance(sealed,dict) or not sealed: errors.append("sealed sources required");sealed={}
    for path,digest in sealed.items():
        if not _relative_source(path) or not _digest(digest): errors.append("invalid source path/hash")
        elif path.endswith((".pt",".pth",".ckpt")) or re.search(r"(?:^|[/_])C(?:0[2-9]|10)(?:[/_]|$)",path):
            errors.append("input checkpoint or non-C01 source forbidden")
    sources=card.get("sources")
    if not isinstance(sources,dict) or set(sources)!=set(SOURCE_NAMES): errors.append("exact four exported sources required");sources={}
    paths=[]
    for key,value in sources.items():
        if not isinstance(value,dict) or set(value)!={"path","sha256"}: errors.append("explicit source path/hash required");continue
        path,digest=value["path"],value["sha256"]
        if (not _relative_source(path) or not _digest(digest) or PurePosixPath(path).name!=SOURCE_NAMES[key]
                or sealed.get(path)!=digest): errors.append("source path/hash binding drift")
        paths.append(path)
    if len(paths)!=4 or len({p for p in paths if isinstance(p,str)})!=4: errors.append("four distinct payloads required")
    if any(p.endswith(".npz") and p not in paths for p in sealed if isinstance(p,str)): errors.append("old or extra numeric payload forbidden")
    reference=card.get("export_reference")
    if not isinstance(reference,dict) or set(reference)!={"spec","card","seal"}: errors.append("exact export spec/card/seal reference required");reference={}
    for path in reference.values():
        if not isinstance(path,str) or not _relative_source(path) or not _digest(sealed.get(path)): errors.append("export reference must be hash bound")
    if not isinstance(reference.get("seal"),str) or sealed.get(reference.get("seal"))!=EXPORT_SEAL_SHA256:
        errors.append("exact completed export seal required")
    for key in ("checkpoint","checkpoints","backbone","sensor_root","teacher_root","model_selection"):
        if key in card: errors.append(f"unauthorized field: {key}")
    approval=card.get("approval",{})
    if not isinstance(approval,dict): approval={}
    if (approval.get("status")!="APPROVED" or approval.get("authorized_operations")!=["training"]
            or not _exact(approval.get("authorized_gates"),[3])): errors.append("new training-only Gate3 approval required")
    for key in ("approved_by","approved_at","scope","confirmation_reference"):
        if not isinstance(approval.get(key),str) or not approval[key].strip(): errors.append(f"approval.{key} required")
    if approval.get("selection_sha256")!=card.get("selection_sha256") or not _digest(card.get("selection_sha256")):
        errors.append("approval selection hash drift")
    if not _exact(approval.get("sources"),sources) or not _exact(approval.get("training"),TRAINING):
        errors.append("approval must bind exact four sources and fixed training budget")
    return ValidationReport(not errors,tuple(errors))

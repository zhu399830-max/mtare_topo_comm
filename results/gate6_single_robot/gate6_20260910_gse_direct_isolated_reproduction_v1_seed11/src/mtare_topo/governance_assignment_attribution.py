"""Read-only cached assignment diagnosis, not training or model authority."""
from pathlib import PurePosixPath
import re

from mtare_topo.governance_field_recovery import _digest, _relative_source
from mtare_topo.governance_partial_structure_training import (
    IDENTITY_FIELDS, SOURCE_NAMES as EXPORT_SOURCES, EXPORT_SEAL_SHA256,
    validate_partial_structure_training_card, _exact,
)

SCHEMA="v3_gse_assignment_attribution_card_v1"
TRAINING_SEAL_SHA256="07f212c90a25785b55a173f0a1f66451b3bb47b9ac6b9b2ded71e78d949d5b1e"
SOURCE_NAMES={**EXPORT_SOURCES,"gt_final":"gt_axes__final__predictions.npz",
    "predicted_final":"predicted_axes__final__predictions.npz",
    "no_relations_final":"predicted_no_relations__final__predictions.npz",
    "history":"training_history.json","training_summary":"summary.json"}
ATTRIBUTION={"branches":["gt_axes","predicted_axes","predicted_no_relations"],"observations_per_branch":180,
    "cached_prediction_observations":540,"last_epoch_batches":10,"batch_size":18,"last_epoch_partition":"exact180_once",
    "joint_assignment":"original_region_set_losses","independent_evaluation":"original_evaluate_center_geometry_only",
    "member_threshold":.5,"model_windows":0,"checkpoint_reads":0,"optimizer_steps":0,"label_conditioned_diagnostic_only":True}
RESOURCES={"wall_time_cap_s":300,"host_ram_bytes":4294967296,"gpu_bytes":0,"output_bytes":500000000}
RESTRICTIONS=("read_only_sources","only_nine_cached_payloads","exact_C01_population","no_model","no_checkpoint",
    "no_optimizer","no_new_scan","no_new_teacher","no_threshold_search","no_calibration","no_C02_C10",
    "no_graph","no_old_result_modification","joint_matching_label_conditioned_only","no_scientific_gate_pass",
    "unknown_labels_unchanged","last_epoch_exact_partition")


def validate_assignment_attribution_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    if not isinstance(card,dict): return ValidationReport(False,("assignment attribution card must be an object",))
    if card.get("schema_version")!=SCHEMA or card.get("operation")!="data_export": errors.append("independent cached-attribution data_export schema required")
    old=card.get("training_card")
    report=validate_partial_structure_training_card(old)
    if not report.passed: errors.append("valid sealed training card identity required")
    if not isinstance(old,dict):old={}
    old_sources=old.get("sources",{})
    if not isinstance(old_sources,dict):old_sources={}
    for key in (*IDENTITY_FIELDS,"effective_counts"):
        if key not in card or key not in old or not _exact(card[key],old[key]):errors.append(f"training population/provenance drift: {key}")
    for key in ("card_id","purpose","teacher_source","scope_limitations"):
        if not isinstance(card.get(key),str) or not card[key].strip():errors.append(f"{key} required")
    if not _exact(card.get("attribution"),ATTRIBUTION) or not _exact(card.get("resources"),RESOURCES):errors.append("frozen attribution/budget drift")
    if card.get("scientific_gate_pass") is not False:errors.append("label-conditioned joint diagnosis is not scientific PASS")
    if not _exact(card.get("restrictions"),dict.fromkeys(RESTRICTIONS,True)):errors.append("all attribution restrictions required")
    sealed=card.get("sealed_sources")
    if not isinstance(sealed,dict) or not sealed:errors.append("source path/hash map required");sealed={}
    for path,digest in sealed.items():
        if not _relative_source(path) or not _digest(digest):errors.append("invalid source path/hash")
        elif path.endswith((".pt",".pth",".ckpt")) or re.search(r"(?:^|[/_])C(?:0[2-9]|10)(?:[/_]|$)",path):errors.append("checkpoint or non-C01 source forbidden")
    sources=card.get("sources")
    if not isinstance(sources,dict) or set(sources)!=set(SOURCE_NAMES):errors.append("exact nine cached sources required");sources={}
    paths=[]
    for name,value in sources.items():
        if not isinstance(value,dict) or set(value)!={"path","sha256"}:errors.append("explicit source path/hash object required");continue
        path,digest=value["path"],value["sha256"]
        if (not _relative_source(path) or not _digest(digest) or PurePosixPath(path).name!=SOURCE_NAMES[name]
                or sealed.get(path)!=digest):errors.append("cached source binding drift")
        paths.append(path)
        if name in EXPORT_SOURCES and not _exact(value,old_sources.get(name)):errors.append("original four exported sources must be unchanged")
    if len(paths)!=9 or len({p for p in paths if isinstance(p,str)})!=9:errors.append("nine distinct payloads required")
    if any(p.endswith(".npz") and p not in paths for p in sealed if isinstance(p,str)):errors.append("extra numeric payload forbidden")
    ref=card.get("training_reference")
    if not isinstance(ref,dict) or set(ref)!={"spec","card","seal"}:errors.append("training spec/card/seal required");ref={}
    for path in ref.values():
        if not isinstance(path,str) or not _relative_source(path) or not _digest(sealed.get(path)):errors.append("unbound training provenance")
    if not isinstance(ref.get("seal"),str) or sealed.get(ref.get("seal"))!=TRAINING_SEAL_SHA256:errors.append("completed training seal drift")
    export_ref=old.get("export_reference",{})
    export_seal=export_ref.get("seal") if isinstance(export_ref,dict) else None
    if not isinstance(export_seal,str) or sealed.get(export_seal)!=EXPORT_SEAL_SHA256:errors.append("completed export seal required")
    for key in ("training","optimizer","checkpoint","checkpoints","sensor_root","teacher_root","inference"):
        if key in card:errors.append(f"unauthorized field: {key}")
    approval=card.get("approval",{})
    if not isinstance(approval,dict):approval={}
    if (approval.get("status")!="APPROVED" or approval.get("authorized_operations")!=["data_export"]
            or not _exact(approval.get("authorized_gates"),[3])):errors.append("new data_export-only Gate3 approval required")
    for key in ("approved_by","approved_at","scope","confirmation_reference"):
        if not isinstance(approval.get(key),str) or not approval[key].strip():errors.append(f"approval.{key} required")
    if not _digest(card.get("selection_sha256")) or approval.get("selection_sha256")!=card.get("selection_sha256"):errors.append("approval selection drift")
    if not _exact(approval.get("sources"),sources):errors.append("approval must bind exact nine cached files")
    return ValidationReport(not errors,tuple(errors))

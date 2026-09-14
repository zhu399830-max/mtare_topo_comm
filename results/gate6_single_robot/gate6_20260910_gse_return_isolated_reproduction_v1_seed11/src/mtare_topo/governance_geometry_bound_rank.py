"""Read-only final-cache ranking diagnostic; no threshold/model selection."""
from pathlib import PurePosixPath
import re

from mtare_topo.governance_field_recovery import _digest,_relative_source
from mtare_topo.governance_geometry_bound import validate_geometry_bound_training_card
from mtare_topo.governance_partial_structure_training import IDENTITY_FIELDS,EXPORT_SEAL_SHA256,_exact

SCHEMA="v3_gse_geometry_bound_rank_card_v1"
CORRECTIVE_SEAL_SHA256="a8b1b111ad141fd4ea27584aaa2426f77090a830dabbae0273627a7d776c319a"
SOURCE_NAMES={"inputs":"partial_training_inputs.npz","manifest":"manifest.json","target_transport":"target_transport.json",
    "summary":"summary.json","gt_final":"gt_axes__final__predictions.npz","predicted_final":"predicted_axes__final__predictions.npz",
    "no_relations_final":"predicted_no_relations__final__predictions.npz","corrective_summary":"summary.json"}
RANK_POLICY={"member_population":"unique_center_query_and_scored_member_mask",
    "event_population":"unique_center_query_and_known_event_mask","event_score":"junction_probability",
    "metrics":["average_precision","roc_auc"],"tie_policy":"group_equal_scores_ap_half_credit_auc",
    "groups":["all","parent"],"member_threshold":.5,"event_decision":"original_argmax",
    "undefined_auc":"null_if_single_class","undefined_ap":"null_if_single_class",
    "probability_diagnostic":"final_logits_bce_vs_constant_prior",
    "threshold_search":False,"label":"cached_fit_rank_diagnostic_only"}
RESOURCES={"wall_time_cap_s":300,"host_ram_bytes":4294967296,"gpu_bytes":0,"output_bytes":500000000}
RESTRICTIONS=("read_only_sources","only_eight_cached_payloads","exact_C01_population","no_history","no_model",
    "no_checkpoint","no_optimizer","no_new_scan","no_new_teacher","no_threshold_search","no_calibration",
    "no_C02_C10","no_graph","no_old_result_modification","no_scientific_gate_pass","fixed_evaluation_population")


def validate_geometry_bound_rank_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    if not isinstance(card,dict):return ValidationReport(False,("rank diagnostic card must be object",))
    if card.get("schema_version")!=SCHEMA or card.get("operation")!="data_export":errors.append("independent rank diagnostic data_export schema required")
    original=card.get("corrective_card")
    if not validate_geometry_bound_training_card(original).passed:errors.append("valid corrective card identity required")
    if not isinstance(original,dict):original={}
    for key in (*IDENTITY_FIELDS,"effective_counts"):
        if key not in card or key not in original or not _exact(card[key],original[key]):errors.append(f"original population drift: {key}")
    for key in ("card_id","purpose","teacher_source","scope_limitations"):
        if not isinstance(card.get(key),str) or not card[key].strip():errors.append(f"{key} required")
    if not _exact(card.get("rank_policy"),RANK_POLICY) or not _exact(card.get("resources"),RESOURCES):errors.append("frozen rank policy/resources required")
    if card.get("scientific_gate_pass") is not False:errors.append("ranking diagnosis cannot claim sciencePASS")
    if not _exact(card.get("restrictions"),dict.fromkeys(RESTRICTIONS,True)):errors.append("exact read-only ranking restrictions required")
    sealed=card.get("sealed_sources")
    if not isinstance(sealed,dict) or not sealed:errors.append("sealed source hashes required");sealed={}
    for path,digest in sealed.items():
        if not _relative_source(path) or not _digest(digest):errors.append("invalid source path/hash")
        elif path.endswith((".pt",".pth",".ckpt","training_history.json")) or re.search(r"(?:^|[/_])C(?:0[2-9]|10)(?:[/_]|$)",path):errors.append("history/checkpoint/non-C01 reads forbidden")
    sources=card.get("sources")
    if not isinstance(sources,dict) or set(sources)!=set(SOURCE_NAMES):errors.append("exact eight cached payloads required");sources={}
    old_sources=original.get("sources",{})
    if not isinstance(old_sources,dict):old_sources={}
    paths=[]
    for key,value in sources.items():
        if not isinstance(value,dict) or set(value)!={"path","sha256"}:errors.append("explicit source path/SHA object required");continue
        path,digest=value["path"],value["sha256"]
        if not _relative_source(path) or not _digest(digest) or PurePosixPath(path).name!=SOURCE_NAMES[key] or sealed.get(path)!=digest:errors.append("payload hash/path binding drift")
        paths.append(path)
        if key in ("inputs","manifest","target_transport","summary") and not _exact(value,old_sources.get(key)):errors.append("original four exported inputs must be unchanged")
    if len(paths)!=8 or len({p for p in paths if isinstance(p,str)})!=8:errors.append("eight distinct cached sources required")
    if any(p.endswith(".npz") and p not in paths for p in sealed if isinstance(p,str)):errors.append("extra numeric cache forbidden")
    reference=card.get("corrective_reference")
    if not isinstance(reference,dict) or set(reference)!={"spec","card","seal"}:errors.append("corrective metadata reference required");reference={}
    for path in reference.values():
        if not isinstance(path,str) or not _relative_source(path) or not _digest(sealed.get(path)):errors.append("unbound corrective metadata")
    if not isinstance(reference.get("seal"),str) or sealed.get(reference.get("seal"))!=CORRECTIVE_SEAL_SHA256:errors.append("corrective final seal drift")
    export=original.get("export_reference",{})
    if not isinstance(export,dict) or not isinstance(export.get("seal"),str) or sealed.get(export["seal"])!=EXPORT_SEAL_SHA256:errors.append("original export seal required")
    for key in ("training","optimizer","checkpoint","checkpoints","sensor_root","teacher_root","inference","history"):
        if key in card:errors.append(f"unauthorized field: {key}")
    approval=card.get("approval",{})
    if not isinstance(approval,dict):approval={}
    if approval.get("status")!="APPROVED" or approval.get("authorized_operations")!=["data_export"] or not _exact(approval.get("authorized_gates"),[3]):errors.append("new data_export-only Gate3 approval required")
    for key in ("approved_by","approved_at","scope","confirmation_reference"):
        if not isinstance(approval.get(key),str) or not approval[key].strip():errors.append(f"approval.{key} required")
    if (not _digest(card.get("selection_sha256")) or approval.get("selection_sha256")!=card.get("selection_sha256")
            or not _exact(approval.get("sources"),sources) or not _exact(approval.get("rank_policy"),RANK_POLICY)):
        errors.append("approval must bind exact sources/selection and no-threshold-search policy")
    return ValidationReport(not errors,tuple(errors))

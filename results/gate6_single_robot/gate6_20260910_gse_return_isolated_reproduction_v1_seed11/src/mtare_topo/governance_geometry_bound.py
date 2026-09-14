"""Single-variable geometry-bound loss correction; original head/data budget."""
import re
from mtare_topo.governance_field_recovery import _digest, _relative_source
from mtare_topo.governance_partial_structure_training import (
    IDENTITY_FIELDS,RESTRICTIONS as BASE_RESTRICTIONS,EXPORT_SEAL_SHA256,
    validate_partial_structure_training_card,_exact,
)
from mtare_topo.governance_assignment_attribution import TRAINING_SEAL_SHA256

SCHEMA="v3_gse_geometry_bound_training_card_v1"
ATTRIBUTION_SEAL_SHA256="8c12d0188e53e54e48036c5602435192b2809c787fadb6e89d3293e10d634f30"
LOSS_CONTRACT={"loss_policy":"geometry_only_unique_center_binding_v1",
    "partial_known_targets_require_centers":True,"ambiguous_assignment":"unknown_all_losses_fixed_denominators",
    "supervised_empty_batch":"FAIL","original_loss_weights_scales_unchanged":True}
SHARED_FIELDS=(*IDENTITY_FIELDS,"sources","training","evaluation","resources","effective_counts","export_reference")
RESTRICTIONS=(*BASE_RESTRICTIONS,"no_old_checkpoint_reads","no_legacy_head_or_loss_modification","single_loss_binding_change_only")


def validate_geometry_bound_training_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    if not isinstance(card,dict):return ValidationReport(False,("geometry-bound training card must be object",))
    if card.get("schema_version")!=SCHEMA or card.get("operation")!="training":errors.append("independent geometry-bound training schema required")
    base=card.get("base_training_card")
    if not validate_partial_structure_training_card(base).passed:errors.append("valid original fixed-budget training card required")
    if not isinstance(base,dict):base={}
    for field in SHARED_FIELDS:
        if field not in base or field not in card or not _exact(card[field],base[field]):errors.append(f"single-variable contract drift: {field}")
    for key,expected in LOSS_CONTRACT.items():
        if not _exact(card.get(key),expected):errors.append(f"geometry binding contract drift: {key}")
    for key in ("card_id","purpose","teacher_source","scope_limitations"):
        if not isinstance(card.get(key),str) or not card[key].strip():errors.append(f"{key} required")
    if card.get("capacity_ready") is not False or card.get("scientific_gate_pass") is not False:errors.append("partial correction cannot claim scientific PASS")
    if not _exact(card.get("restrictions"),dict.fromkeys(RESTRICTIONS,True)):errors.append("exact single-variable correction restrictions required")
    sealed=card.get("sealed_sources")
    if not isinstance(sealed,dict) or not sealed:errors.append("sealed sources required");sealed={}
    sources=card.get("sources",{})
    if not isinstance(sources,dict):sources={}
    allowed=set()
    for value in sources.values():
        if isinstance(value,dict) and isinstance(value.get("path"),str):
            allowed.add(value["path"])
            if sealed.get(value["path"])!=value.get("sha256"):errors.append("four original payload hashes must be bound")
    for path,digest in sealed.items():
        if not _relative_source(path) or not _digest(digest):errors.append("invalid source path/hash")
        elif re.search(r"(?:^|[/_])C(?:0[2-9]|10)(?:[/_]|$)",path):errors.append("non-C01 metadata source forbidden")
        elif path.endswith((".pt",".pth",".ckpt")) or (path.endswith(".npz") and path not in allowed):errors.append("old checkpoint/prediction payload forbidden")
        elif path.rsplit("/",1)[-1] in ("summary.json","training_history.json","manifest.json","target_transport.json") and path not in allowed:
            errors.append("baseline/attribution payload is not metadata authority")
    for name,seal_digest in (("base_reference",TRAINING_SEAL_SHA256),("attribution_reference",ATTRIBUTION_SEAL_SHA256)):
        reference=card.get(name)
        if not isinstance(reference,dict) or set(reference)!={"spec","card","seal"}:errors.append(f"{name} spec/card/seal required");continue
        for path in reference.values():
            if not isinstance(path,str) or not _relative_source(path) or not _digest(sealed.get(path)):errors.append(f"unbound {name}")
        if not isinstance(reference["seal"],str) or sealed.get(reference["seal"])!=seal_digest:errors.append(f"{name} seal drift")
    export=card.get("export_reference",{})
    if not isinstance(export,dict) or not isinstance(export.get("seal"),str) or sealed.get(export["seal"])!=EXPORT_SEAL_SHA256:
        errors.append("original export seal required")
    for key in ("checkpoint","checkpoints","backbone","sensor_root","teacher_root","optimizer","loss_weights","loss_scales"):
        if key in card:errors.append(f"unauthorized correction field: {key}")
    approval=card.get("approval",{})
    if not isinstance(approval,dict):approval={}
    if (approval.get("status")!="APPROVED" or approval.get("authorized_operations")!=["training"]
            or not _exact(approval.get("authorized_gates"),[3])):errors.append("new correction training-only Gate3 approval required")
    for key in ("approved_by","approved_at","scope","confirmation_reference"):
        if not isinstance(approval.get(key),str) or not approval[key].strip():errors.append(f"approval.{key} required")
    if (not _digest(card.get("selection_sha256")) or approval.get("selection_sha256")!=card.get("selection_sha256")
            or not _exact(approval.get("sources"),sources) or not _exact(approval.get("loss_contract"),LOSS_CONTRACT)):
        errors.append("approval must newly bind original sources/selection and correction loss rule")
    return ValidationReport(not errors,tuple(errors))

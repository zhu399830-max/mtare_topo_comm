"""Narrow authority for source-interface diagnostics; no training labels."""
from mtare_topo.governance_surface_selection import digest

SCHEMA = "v3_joint_partial_targets_card_v3"
SLUG = "gse_joint_partial_targets_v3"
POLICY = {"radius_m":10.,"grid_voxel_m":.25,"surface_voxel_m":.5,"extra_range_error_m":0.,
          "wall_time_s":1800,"host_ram_bytes":4294967296,"output_bytes":536870912,
          "labels":0,"optimizer_steps":0,"no_retry":True}


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    if type(card) is not dict or set(card)!={"schema_version","card_id","operation","scope","scope_sha256","policy","approval"}:
        return ValidationReport(False,("closed interface diagnostic card required",))
    if (card["schema_version"],card["card_id"],card["operation"])!=(SCHEMA,SLUG,"data_export"):
        errors.append("interface-only export scope required")
    s=card["scope"];a=card["approval"]
    if type(s) is not dict or s.get("schema")!="gse_surface_teacher_source_plan_v1":
        return ValidationReport(False,("exact teacher source plan required",))
    if s.get("counts")!={"parents":10,"physical_edge_units":10,"observations":30,"selected_frames":150,
                        "construction_files":30,"codebook_files":30,"array_plans":90,"actual_payload_reads":0,"labels":0}:
        errors.append("fixed30 observation source count required")
    if digest(card["policy"])!=digest(POLICY):errors.append("fixed method and limits required")
    if (card["scope_sha256"]!=digest(s) or type(a) is not dict or a.get("scope_sha256")!=digest(s)
        or a.get("status")!="APPROVED" or a.get("authorized_operations")!=["data_export"] or a.get("authorized_gates")!=[3]
        or not a.get("confirmation_reference")):
        errors.append("exact standing authorization required")
    # Runtime recompiles the full source plan before payload, including all
    # selected tasks, source hashes and compressed chunk accounting.
    return ValidationReport(not errors,tuple(errors))



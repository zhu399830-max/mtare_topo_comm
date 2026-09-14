"""Fixed sealed30 material to blind-review files; no automatic labels."""
import json
from pathlib import Path
from mtare_topo.governance_surface_material import read_pinned, compile_scope as material_scope
from mtare_topo.governance_surface_selection import digest

SCHEMA="v3_surface_review_export_card_v1"
SLUG="gse_surface_review_export_v1"
MATERIAL="results/gate3_semantics/gate3_20260907_gse_surface_observed_material_v1_seed20260906"
ANCHOR="results/gate3_semantics/gate3_20260907_gse_surface_anchor_diagnostic_v3_seed20260906"
SEALS={MATERIAL:"9c0689d14b90d0a8f6e3c00ababd533c0f028440a1a7f09065dc909eb3df72a5",
       ANCHOR:"9e4cbc8b76b87bcc35fd150949b1d1f9db92cb7a39123df2ce1f77a87b55a18f"}
POLICY={"wall_time_s":600,"host_ram_bytes":4294967296,"output_bytes":536870912,"labels":0,"optimizer_steps":0,"no_retry":True}


def compile_scope(root):
    root=Path(root).resolve(strict=True)
    selected=material_scope(root)["selected"]
    paths={MATERIAL+"/artifacts/material_manifest.json"}
    for row in selected:
        paths.add(MATERIAL+"/artifacts/"+row["view_id"]+".npz")
        paths.add(ANCHOR+"/artifacts/"+row["view_id"]+".json")
    hashes={}
    for source,h in SEALS.items():
        raw=read_pinned(root,source+"/artifacts/evidence_sha256.txt",h)
        for line in raw.decode().splitlines():
            digest_value,p=line.split("  ",1)
            if p in paths:
                if p in hashes:raise ValueError("duplicate sealed source")
                hashes[p]=digest_value
    if set(hashes)!=paths:raise ValueError("fixed material/reference source missing")
    mp=MATERIAL+"/artifacts/material_manifest.json"
    manifest=json.loads(read_pinned(root,mp,hashes[mp]))
    if [r["selection"] for r in manifest["observations"]]!=selected:
        raise ValueError("material selection differs from fixed30")
    return dict(selected=selected,input_files_sha256=hashes,source_seals=SEALS,
        counts=dict(parents=10,physical_edge_units=10,observations=30,selected_frames=150),
        spacing="Exact source frame rows/decision arcs retained in selected; no invented time or history spacing.",
        teacher="Separate existing reference-conditioned diagnostics, NOT completed semantic labels; human review remains unperformed.",
        leakage="C01 fixed rank0 only; no new raw scan,world,checkpoint,C07-C10 or benchmark payload. Blind files contain only points/history/opaque view IDs.")


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    if type(card) is not dict or set(card)!={"schema_version","card_id","operation","scope","scope_sha256","policy","approval"}:
        return ValidationReport(False,("closed review export card required",))
    if (card["schema_version"],card["card_id"],card["operation"])!=(SCHEMA,SLUG,"data_export"):errors.append("review export only")
    s=card["scope"];a=card["approval"]
    if s.get("counts")!={"parents":10,"physical_edge_units":10,"observations":30,"selected_frames":150} or s.get("source_seals")!=SEALS:errors.append("fixed30 source scope required")
    if card["policy"]!=POLICY or card["scope_sha256"]!=digest(s):errors.append("policy/scope drift")
    if a.get("status")!="APPROVED" or a.get("scope_sha256")!=digest(s) or a.get("authorized_operations")!=["data_export"] or a.get("authorized_gates")!=[3]:errors.append("exact authorization required")
    return ValidationReport(not errors,tuple(errors))

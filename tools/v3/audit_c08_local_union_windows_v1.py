#!/usr/bin/env python3
"""Read-only C08 local-union window feasibility audit; no field/mesh/trajectory."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from _bootstrap import PROJECT_ROOT
from mtare_topo.data.local_implicit_union import edge_arc_incidence

BASE=PROJECT_ROOT/"results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
WORLDS=("S01_flat_tree_small_C08","S06_3d_branch_medium_C08","S10_3d_complex_C08")

def main() -> int:
    report=[]
    for world in WORLDS:
        root=BASE/world/"primary"
        graph=json.loads((root/"graph.json").read_text())
        splines=json.loads((root/"splines.json").read_text())
        records=edge_arc_incidence(graph,splines)
        incident=[item for item in records if int(next(node["degree"] for node in graph["nodes"] if str(node["id"])==item.node_id))>=3]
        max_error=max(item.connector_error_m for item in records)
        report.append({"world":world,"edge_count":len(graph["edges"]),"endpoint_records":len(records),"junction_endpoint_records":len(incident),"maximum_connector_error_m":max_error,"pass":bool(len(records)==2*len(graph["edges"]) and np.isfinite(max_error))})
    result={"schema_version":"cano_c08_edge_arc_incidence_audit_v1","worlds":report,"total_endpoint_records":sum(item["endpoint_records"] for item in report),"overall_pass":all(item["pass"] for item in report),"claim_boundary":"Read-only edge-to-spline projection/arc-incidence audit only; zero SDF, mesh, trajectory, rays, inference or graph."}
    print(json.dumps(result,sort_keys=True))
    return 0 if result["overall_pass"] else 2
if __name__=="__main__": raise SystemExit(main())

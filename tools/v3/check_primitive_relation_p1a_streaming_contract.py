#!/usr/bin/env python3
"""Real C01 no-asset pilot for the P1a streaming shard contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from numcodecs import Blosc
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_sensor_export import FramePoseCorrection, world_frame_poses_with_corrections
from mtare_topo.data.primitive_relation_dataset import PrimitiveMembershipCodebook
from mtare_topo.data.primitive_relation_sensor_export import render_primitive_sensor_frame
from mtare_topo.governance import load_json
from mtare_topo.teacher.csg_mesh_provenance import CSGMeshProvenanceRaycaster, mesh_swept_superellipse
from mtare_topo.teacher.geometry_variant_contract import GeometryRealization, realize_construction
from mtare_topo.teacher.primitive_construction_supervisor import build_primitive_construction_graph
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField
from run_primitive_relation_p1a_sensor_provenance_export_v1 import CORRECTION_PATH, MESH_ROOT, _load_traversals


WORLD = "S01_flat_tree_small_C01"


def digest(frame) -> str:
    value = hashlib.sha256(); value.update(frame.range_m.tobytes()); value.update(frame.valid_mask.tobytes()); value.update(frame.primitive_membership_code.tobytes()); return value.hexdigest()


def main() -> int:
    primary = MESH_ROOT / WORLD / "primary"; graph = load_json(primary / "graph.json"); splines = load_json(primary / "splines.json"); geometry = load_json(primary / "geometry_parameters.json")
    correction_rows = [row for row in load_json(CORRECTION_PATH)["corrections"] if row["world"] == WORLD]
    corrections = tuple(FramePoseCorrection(**{key: row[key] for key in ("global_frame_index", "traversal_id", "local_frame_index", "original_arc_m", "corrected_arc_m")}) for row in correction_rows)
    poses = world_frame_poses_with_corrections(parent_id=WORLD, traversal_manifest=_load_traversals()[WORLD], graph=graph, spline_document=splines, geometry_parameters=geometry, corrections=corrections)
    construction = build_primitive_construction_graph(graph, splines, geometry, endpoint_attachment_mode="free_space_overlap", node_degree_source="edge_incidence")
    rows = []
    with tempfile.TemporaryDirectory(prefix="primitive_p1a_pilot_") as directory:
        root = Path(directory)
        for realization in GeometryRealization:
            primitives = realize_construction(WORLD, construction, realization); field = SweptSuperellipseProvenanceField(primitives, spacing_m=.025)
            raycaster = CSGMeshProvenanceRaycaster([mesh_swept_superellipse(value, axial_spacing_m=.05, angular_segments=64) for value in primitives], operand_signed_distances=field.operand_signed_distances_sparse)
            codebook = PrimitiveMembershipCodebook(field.primitive_ids)
            first = render_primitive_sensor_frame(raycaster=raycaster, field=field, codebook=codebook, sensor_xyz_m=poses.sensor_xyz_m[0], yaw_deg=float(poses.yaw_deg[0]))
            second = render_primitive_sensor_frame(raycaster=raycaster, field=field, codebook=codebook, sensor_xyz_m=poses.sensor_xyz_m[0], yaw_deg=float(poses.yaw_deg[0]))
            path = root / f"{realization.value}.zarr"; group = zarr.open_group(str(path), mode="w"); codec = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
            group.create_dataset("range_m", data=first.range_m[None], chunks=(1,16,720), compressor=codec); group.create_dataset("valid_mask", data=first.valid_mask[None], chunks=(1,16,720), compressor=codec); group.create_dataset("primitive_membership_code", data=first.primitive_membership_code[None], chunks=(1,16,720), compressor=codec)
            reopened = zarr.open_group(str(path), mode="r")
            exact = np.array_equal(reopened["range_m"][0], first.range_m) and np.array_equal(reopened["valid_mask"][0], first.valid_mask) and np.array_equal(reopened["primitive_membership_code"][0], first.primitive_membership_code)
            rows.append({"realization": realization.value, "valid_returns": int(np.sum(first.valid_mask)), "ambiguous_rays": first.ambiguous_ray_count, "codebook_entries": len(codebook.source_sets), "maximum_membership": max(len(value) for value in codebook.source_sets), "repeat_exact": digest(first) == digest(second), "zarr_roundtrip_exact": exact})
    checks = {"all_three_realizations": len(rows) == 3, "all_repeat_exact": all(row["repeat_exact"] for row in rows), "all_zarr_roundtrip_exact": all(row["zarr_roundtrip_exact"] for row in rows), "all_valid_nonempty": all(row["valid_returns"] > 0 for row in rows), "all_codebooks_nonempty": all(row["codebook_entries"] > 1 for row in rows)}
    result = {"status": "PASS_PRIMITIVE_RELATION_P1A_STREAMING_CONTRACT" if all(checks.values()) else "FAIL_PRIMITIVE_RELATION_P1A_STREAMING_CONTRACT", "world": WORLD, "source_frames": len(poses.global_frame_indices), "pose_corrections": len(corrections), "rows": rows, "checks": checks, "assets_retained": 0, "c07_c08_c09_c10_read": 0}
    print(json.dumps(result, indent=2, sort_keys=True)); return 0 if all(checks.values()) else 1


if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Headless Isaac Sim validation for one generated tunnel USD stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result: dict[str, object] = {
        "schema_version": "isaac_tunnel_usd_import_v1",
        "status": "FAIL",
        "usd": str(args.usd),
        "updates_requested": args.updates,
    }
    simulation_app = None
    try:
        from isaacsim.simulation_app import SimulationApp

        simulation_app = SimulationApp(
            {
                "headless": True,
                "hide_ui": True,
                "create_new_stage": True,
                "disable_viewport_updates": True,
                "multi_gpu": False,
                "fast_shutdown": True,
            }
        )
        from pxr import UsdGeom, UsdPhysics

        if not simulation_app.context.open_stage(str(args.usd)):
            raise RuntimeError("Isaac USD context failed to open the stage")
        for _ in range(args.updates):
            simulation_app.update()
        stage = simulation_app.context.get_stage()
        if stage is None:
            raise RuntimeError("Isaac USD context returned no stage")
        mesh_prim = stage.GetPrimAtPath("/World/TunnelMesh")
        if not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
            raise RuntimeError("/World/TunnelMesh is missing or is not UsdGeom.Mesh")
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        extent = mesh.GetExtentAttr().Get()
        if points is None or face_counts is None or face_indices is None:
            raise RuntimeError("mesh arrays could not be read")
        if any(int(count) != 3 for count in face_counts):
            raise RuntimeError("mesh contains non-triangle faces")
        if len(face_indices) != 3 * len(face_counts):
            raise RuntimeError("face index count is inconsistent")
        if not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError("PhysicsCollisionAPI is missing")
        if not mesh_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            raise RuntimeError("PhysicsMeshCollisionAPI is missing")
        approximation = UsdPhysics.MeshCollisionAPI(mesh_prim).GetApproximationAttr().Get()
        navigation_prim = stage.GetPrimAtPath("/World/NavigationCenterlines")
        navigation_present = navigation_prim.IsValid()
        navigation_curve_count = 0
        navigation_point_count = 0
        if navigation_present:
            if not navigation_prim.IsA(UsdGeom.BasisCurves):
                raise RuntimeError(
                    "/World/NavigationCenterlines exists but is not UsdGeom.BasisCurves"
                )
            navigation_curves = UsdGeom.BasisCurves(navigation_prim)
            navigation_counts = navigation_curves.GetCurveVertexCountsAttr().Get()
            navigation_points = navigation_curves.GetPointsAttr().Get()
            if navigation_counts is None or navigation_points is None:
                raise RuntimeError("navigation centerline arrays could not be read")
            if sum(int(count) for count in navigation_counts) != len(navigation_points):
                raise RuntimeError("navigation curve counts do not match point count")
            navigation_curve_count = len(navigation_counts)
            navigation_point_count = len(navigation_points)
        up_axis = str(UsdGeom.GetStageUpAxis(stage))
        metres_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if up_axis != "Z":
            raise RuntimeError(f"stage up axis is {up_axis}, expected Z")
        if abs(metres_per_unit - 1.0) > 1e-12:
            raise RuntimeError(
                f"stage metresPerUnit is {metres_per_unit}, expected 1.0"
            )
        result.update(
            {
                "status": "PASS",
                "prim_path": str(mesh_prim.GetPath()),
                "vertex_count": len(points),
                "triangle_count": len(face_counts),
                "face_index_count": len(face_indices),
                "extent": [[float(value) for value in item] for item in extent],
                "up_axis": up_axis,
                "meters_per_unit": metres_per_unit,
                "collision_api": True,
                "mesh_collision_api": True,
                "collision_approximation": str(approximation),
                "navigation_centerlines_present": navigation_present,
                "navigation_curve_count": navigation_curve_count,
                "navigation_point_count": navigation_point_count,
                "updates_completed": args.updates,
                "simulation_app_running_before_close": simulation_app.is_running(),
            }
        )
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
    finally:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("MTARE_ISAAC_RESULT=" + json.dumps(result, sort_keys=True))
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception as close_error:
                print("MTARE_ISAAC_CLOSE_ERROR=" + repr(close_error))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

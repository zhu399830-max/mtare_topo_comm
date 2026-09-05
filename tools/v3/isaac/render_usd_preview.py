#!/usr/bin/env python3
"""Render whole-world and interior inspection previews in headless Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    topology = json.loads(args.topology.read_text(encoding="utf-8"))
    result: dict[str, object] = {
        "schema_version": "isaac_tunnel_preview_v1",
        "status": "FAIL",
        "usd": str(args.usd),
        "topology_parent_id": topology.get("topology_parent_id"),
    }
    simulation_app = None
    try:
        from isaacsim.simulation_app import SimulationApp

        simulation_app = SimulationApp(
            {
                "headless": True,
                "hide_ui": True,
                "create_new_stage": True,
                "disable_viewport_updates": False,
                "multi_gpu": False,
                "fast_shutdown": True,
                "renderer": "RealTimePathTracing",
                "width": 1000,
                "height": 1000,
            }
        )
        import carb.settings
        import omni.replicator.core as rep
        import omni.timeline
        from pxr import Gf, UsdGeom, UsdLux

        if not simulation_app.context.open_stage(str(args.usd)):
            raise RuntimeError("Isaac USD context failed to open the stage")
        for _ in range(4):
            simulation_app.update()
        stage = simulation_app.context.get_stage()
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/TunnelMesh"))
        extent = mesh.GetExtentAttr().Get()
        minimum = [float(value) for value in extent[0]]
        maximum = [float(value) for value in extent[1]]
        center = [(minimum[index] + maximum[index]) / 2 for index in range(3)]

        def define_camera(path: str, eye: list[float], target: list[float], up: list[float]):
            camera = UsdGeom.Camera.Define(stage, path)
            matrix = Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)
            ).GetInverse()
            UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(matrix)
            camera.CreateFocalLengthAttr(22.0)
            camera.CreateHorizontalApertureAttr(20.955)
            camera.CreateVerticalApertureAttr(20.955)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
            return camera

        overview_eye = [center[0], center[1], maximum[2] + 170.0]
        overview_camera = define_camera(
            "/World/InspectionOverviewCamera",
            overview_eye,
            center,
            [0.0, 1.0, 0.0],
        )

        node_positions = {
            node["id"]: [float(value) for value in node["position"]]
            for node in topology["nodes"]
        }
        first_edge = next(
            edge
            for edge in topology["edges"]
            if "n000" in {edge["u"], edge["v"]}
        )
        neighbour_id = first_edge["v"] if first_edge["u"] == "n000" else first_edge["u"]
        interior_eye = node_positions["n000"]
        neighbour = node_positions[neighbour_id]
        direction = [neighbour[index] - interior_eye[index] for index in range(3)]
        length = sum(value * value for value in direction) ** 0.5
        interior_target = [
            interior_eye[index] + 8.0 * direction[index] / length
            for index in range(3)
        ]
        interior_camera = define_camera(
            "/World/InspectionInteriorCamera",
            interior_eye,
            interior_target,
            [0.0, 0.0, 1.0],
        )

        dome = UsdLux.DomeLight.Define(stage, "/World/InspectionDomeLight")
        dome.CreateIntensityAttr(900.0)
        dome.CreateColorAttr(Gf.Vec3f(0.72, 0.80, 1.0))
        interior_light = UsdLux.SphereLight.Define(
            stage, "/World/InspectionInteriorLight"
        )
        interior_light.CreateIntensityAttr(35000.0)
        interior_light.CreateRadiusAttr(1.0)
        interior_light.CreateColorAttr(Gf.Vec3f(1.0, 0.78, 0.58))
        UsdGeom.Xformable(interior_light.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(*interior_eye)
        )

        rep.orchestrator.set_capture_on_play(False)
        carb.settings.get_settings().set("/rtx/post/dlss/execMode", 2)
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(3):
            simulation_app.update()
        timeline.pause()

        args.output_dir.mkdir(parents=True, exist_ok=True)

        def capture(camera, resolution: tuple[int, int], name: str) -> list[str]:
            capture_dir = args.output_dir / name
            capture_dir.mkdir(parents=True, exist_ok=True)
            render_product = rep.create.render_product(
                str(camera.GetPath()), resolution, name=f"{name}_render_product"
            )
            backend = rep.backends.get("DiskBackend")
            backend.initialize(output_dir=str(capture_dir))
            writer = rep.writers.get("BasicWriter")
            writer.initialize(backend=backend, rgb=True)
            writer.attach(render_product)
            render_product.hydra_texture.set_updates_enabled(True)
            rep.orchestrator.step(
                delta_time=0.0,
                rt_subframes=16,
                pause_timeline=True,
                wait_for_render=True,
            )
            rep.orchestrator.wait_until_complete()
            render_product.hydra_texture.set_updates_enabled(False)
            writer.detach()
            render_product.destroy()
            return sorted(str(path) for path in capture_dir.rglob("*.png"))

        overview_pngs = capture(overview_camera, (1000, 1000), "overview")
        interior_pngs = capture(interior_camera, (1200, 675), "interior")
        png_files = overview_pngs + interior_pngs
        if len(png_files) < 2:
            raise RuntimeError(f"expected two PNG previews, found {len(png_files)}")
        result.update(
            {
                "status": "PASS",
                "overview_camera_eye": overview_eye,
                "interior_camera_eye": interior_eye,
                "interior_camera_target": interior_target,
                "interior_edge": first_edge["id"],
                "png_files": png_files,
                "mesh_extent": [minimum, maximum],
                "renderer": "RealTimePathTracing",
            }
        )
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
    finally:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("MTARE_ISAAC_PREVIEW_RESULT=" + json.dumps(result, sort_keys=True))
        if simulation_app is not None:
            simulation_app.close()
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

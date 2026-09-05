#!/usr/bin/env python3
"""Capture the frozen 24-pose Cano diagnostic with Isaac Sim 6.0.1 RTX LiDAR."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--capture-stage-path", required=True, type=Path)
parser.add_argument("--capture-manifest-path", required=True, type=Path)
parser.add_argument("--capture-output-dir", required=True, type=Path)
parser.add_argument("--capture-metrics-path", required=True, type=Path)
parser.add_argument("--capture-config-name", default="MTARE_VLP16_720_50M_V1")
args, _unknown = parser.parse_known_args()

simulation_app = SimulationApp(
    {
        "headless": True,
        "enable_motion_bvh": True,
        "renderer": "RaytracedLighting",
    }
)

import numpy as np
import omni
import omni.timeline
import omni.usd
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, parse_generic_model_output_data
from pxr import Gf

sys.path.insert(0, "/workspace/src")
from mtare_topo.data.cano_sensor_smoke import rasterize_generic_model_output


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _array(value: object, dtype: object) -> np.ndarray:
    return np.asarray(value, dtype=dtype).copy()


def _field(gmo: object, name: str, count: int, dtype: object) -> np.ndarray:
    value = getattr(gmo, name, None)
    if value is None:
        return np.zeros(count, dtype=dtype)
    return _array(value, dtype)


def _capture_complete_scan(sensor: LidarSensor, discard_complete_scans: int = 1) -> tuple[dict, int]:
    complete_seen = 0
    previous_timestamp = None
    for frame in range(300):
        simulation_app.update()
        raw, _info = sensor.get_data("generic-model-output")
        if raw is None:
            continue
        gmo = parse_generic_model_output_data(raw)
        count = int(gmo.numElements)
        if count <= 0 or not bool(getattr(gmo, "scanComplete", False)):
            continue
        timestamp = int(getattr(gmo, "timestampNs", frame))
        if timestamp == previous_timestamp:
            continue
        previous_timestamp = timestamp
        complete_seen += 1
        if complete_seen <= discard_complete_scans:
            continue
        azimuth = _field(gmo, "x", count, np.float32)
        elevation = _field(gmo, "y", count, np.float32)
        distance = _field(gmo, "z", count, np.float32)
        raster, valid, raster_counts = rasterize_generic_model_output(
            azimuth, elevation, distance
        )
        return (
            {
                "timestamp_ns": timestamp,
                "scan_complete": True,
                "azimuth_deg": azimuth,
                "elevation_deg": elevation,
                "range_raw_m": distance,
                "channel_id": _field(gmo, "channelId", count, np.uint32),
                "emitter_id": _field(gmo, "emitterId", count, np.uint32),
                "time_offset_ns": _field(gmo, "timeOffsetNs", count, np.int64),
                "range_m": raster,
                "valid_mask": valid,
                "raster_counts": raster_counts,
            },
            frame + 1,
        )
    raise RuntimeError("no fresh complete RTX LiDAR scan within 300 rendered frames")


def main() -> int:
    print("CAPTURE_MAIN_ENTER", flush=True)
    stage_path = args.capture_stage_path.resolve()
    manifest_path = args.capture_manifest_path.resolve()
    output_dir = args.capture_output_dir.resolve()
    metrics_path = args.capture_metrics_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["poses"]) != 24:
        raise RuntimeError("Isaac capture accepts exactly 24 frozen poses")

    open_stage(str(stage_path))
    while is_stage_loading():
        simulation_app.update()
    for _ in range(10):
        simulation_app.update()

    success, lidar_prim = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path="/World/DiagnosticLidar",
        parent=None,
        config=args.capture_config_name,
        translation=Gf.Vec3d(0.0, 0.0, 0.0),
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        **{
            "omni:sensor:Core:auxOutputType": "BASIC",
            "omni:sensor:Core:outputFrameOfReference": "SENSOR",
        },
    )
    if not success or lidar_prim is None:
        raise RuntimeError(
            f"IsaacSensorCreateRtxLidar failed for {args.capture_config_name}"
        )
    lidar = Lidar(
        "/World/DiagnosticLidar",
        accumulate_outputs=True,
        aux_output_level="BASIC",
        tick_rate=10.0,
    )
    sensor = LidarSensor(lidar, annotators=["generic-model-output"])
    prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/DiagnosticLidar")
    frozen_attributes = {}
    for name in (
        "omni:sensor:Core:numberOfChannels",
        "omni:sensor:Core:numberOfEmitters",
        "omni:sensor:Core:nearRangeM",
        "omni:sensor:Core:farRangeM",
        "omni:sensor:Core:scanRateBaseHz",
        "omni:sensor:Core:patternFiringRateHz",
        "omni:sensor:tickRate",
        "omni:sensor:Core:accumulateOutputs",
        "omni:sensor:Core:auxOutputType",
    ):
        attribute = prim.GetAttribute(name)
        frozen_attributes[name] = attribute.Get() if attribute.IsValid() else None

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(30):
        simulation_app.update()

    samples = []
    for index, pose in enumerate(manifest["poses"]):
        position = np.asarray([pose["sensor_xyz_m"]], dtype=np.float64)
        half_yaw = math.radians(float(pose["yaw_deg"])) / 2.0
        orientation = np.asarray(
            [[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]],
            dtype=np.float64,
        )
        lidar.set_world_poses(positions=position, orientations=orientation)
        # One complete scan is always discarded after teleportation.  The next
        # complete revolution is stationary at the frozen pose.
        scan, rendered_frames = _capture_complete_scan(sensor, discard_complete_scans=1)
        output_path = output_dir / f"{pose['sample_id']}.npz"
        np.savez_compressed(
            output_path,
            azimuth_deg=scan["azimuth_deg"],
            elevation_deg=scan["elevation_deg"],
            range_raw_m=scan["range_raw_m"],
            channel_id=scan["channel_id"],
            emitter_id=scan["emitter_id"],
            time_offset_ns=scan["time_offset_ns"],
            range_m=scan["range_m"],
            valid_mask=scan["valid_mask"],
            sensor_xyz_m=position[0].astype(np.float32),
            yaw_deg=np.asarray(float(pose["yaw_deg"]), dtype=np.float32),
            timestamp_ns=np.asarray(scan["timestamp_ns"], dtype=np.int64),
        )
        samples.append(
            {
                "index": index,
                "sample_id": pose["sample_id"],
                "role": pose["role"],
                "raw_return_count": int(len(scan["range_raw_m"])),
                "raster_valid_count": int(np.sum(scan["valid_mask"])),
                "raster_valid_ratio": float(np.mean(scan["valid_mask"])),
                "rendered_frames_after_teleport": rendered_frames,
                "timestamp_ns": scan["timestamp_ns"],
                "raster_counts": scan["raster_counts"],
                "file": str(output_path),
            }
        )
        print(
            f"[{index + 1:02d}/24] {pose['sample_id']} "
            f"returns={len(scan['range_raw_m'])} valid={np.mean(scan['valid_mask']):.4f}",
            flush=True,
        )

    timeline.stop()
    _write_json(
        metrics_path,
        {
            "schema_version": "cano_isaac_rtx_capture_v1",
            "overall_status": "PASS_CAPTURED_24" if len(samples) == 24 else "FAIL",
            "isaac_sim_version": "6.0.1",
            "config_name": args.capture_config_name,
            "frozen_prim_attributes": frozen_attributes,
            "samples": samples,
            "counts": {"worlds": 1, "poses": len(samples), "formal_dataset_samples": 0},
        },
    )
    return 0


exit_code = 1
try:
    exit_code = main()
except Exception as error:
    args.capture_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.capture_metrics_path,
        {
            "schema_version": "cano_isaac_rtx_capture_v1",
            "overall_status": "FAIL_EXCEPTION",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "counts": {"formal_dataset_samples": 0},
        },
    )
    traceback.print_exc()
finally:
    simulation_app.close()

raise SystemExit(exit_code)

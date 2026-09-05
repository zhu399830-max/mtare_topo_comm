#!/usr/bin/env python3
"""Validate and capture one complete scan from the frozen local OmniLidar USDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--probe-sensor-usda", required=True, type=Path)
parser.add_argument("--probe-source-json", required=True, type=Path)
parser.add_argument("--probe-output-dir", required=True, type=Path)
parser.add_argument("--probe-metrics-dir", required=True, type=Path)
args, _unknown = parser.parse_known_args()

simulation_app = SimulationApp(
    {
        "headless": True,
        "enable_motion_bvh": True,
        "renderer": "RaytracedLighting",
    }
)

import numpy as np
import omni.timeline
import omni.usd
import omni.replicator.core as rep
from isaacsim.core.utils.stage import create_new_stage
from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, parse_generic_model_output_data
from isaacsim.sensors.experimental.rtx.sensor_checker import ModelInfo, SensorCheckerUtil
from omni.replicator.core import Writer
from pxr import Gf, UsdGeom

sys.path.insert(0, "/workspace/src")
from mtare_topo.data.cano_sensor_smoke import rasterize_generic_model_output


EXPECTED_ATTRIBUTES: dict[str, Any] = {
    "omni:sensor:Core:accumulateOutputs": True,
    "omni:sensor:Core:azimuthErrorMean": 0.0,
    "omni:sensor:Core:azimuthErrorStd": 0.0,
    "omni:sensor:Core:elevationErrorMean": 0.0,
    "omni:sensor:Core:elevationErrorStd": 0.0,
    "omni:sensor:Core:emitterState:s001:azimuthDeg": [0.0] * 16,
    "omni:sensor:Core:emitterState:s001:channelId": list(range(1, 17)),
    "omni:sensor:Core:emitterState:s001:elevationDeg": list(
        np.arange(-15.0, 16.0, 2.0)
    ),
    "omni:sensor:Core:emitterState:s001:fireTimeNs": [0] * 16,
    "omni:sensor:Core:elementsCoordsType": "SPHERICAL",
    "omni:sensor:Core:farRangeM": 50.0,
    "omni:sensor:Core:intensityMappingType": "LINEAR",
    "omni:sensor:Core:intensityProcessing": "NORMALIZATION",
    "omni:sensor:Core:maxReturns": 1,
    "omni:sensor:Core:minReflectance": 0.0,
    "omni:sensor:Core:minReflectionRangeM": 50.0,
    "omni:sensor:Core:nearRangeM": 0.3,
    "omni:sensor:Core:numberOfChannels": 16,
    "omni:sensor:Core:numberOfEmitters": 16,
    "omni:sensor:Core:outputFrameOfReference": "SENSOR",
    "omni:sensor:Core:outputMotionCompensationState": "NONCOMPENSATED",
    "omni:sensor:Core:patternFiringRateHz": 7200,
    "omni:sensor:Core:peakPowerW": 10.0,
    "omni:sensor:Core:pulseTimeNs": 6,
    "omni:sensor:Core:rangeAccuracyM": 0.0,
    "omni:sensor:Core:rangeResolutionM": 0.001,
    "omni:sensor:Core:rayType": "IDEALIZED",
    "omni:sensor:Core:rotationDirection": "CCW",
    "omni:sensor:Core:scanRateBaseHz": 10,
    "omni:sensor:Core:scanType": "ROTARY",
    "omni:sensor:Core:skipDroppingInvalidPoints": False,
    "omni:sensor:Core:startAzimuthOffsetDeg": 0.0,
    "omni:sensor:Core:waveLengthNm": 903.0,
    "omni:sensor:tickRate": 10.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return [_json_value(item) for item in value]
    except TypeError:
        return str(value)


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        try:
            actual_array = np.asarray(actual)
            expected_array = np.asarray(expected)
        except Exception:
            return False
        if actual_array.shape != expected_array.shape:
            return False
        if np.issubdtype(expected_array.dtype, np.number):
            return bool(np.allclose(actual_array, expected_array, rtol=0.0, atol=1e-6))
        return actual_array.tolist() == expected_array.tolist()
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-6
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str):
        return str(actual) == expected
    return actual == expected


def _create_closed_box() -> None:
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World")
    walls = {
        "WallPosX": ((8.0, 0.0, 0.5), (0.2, 16.0, 5.0)),
        "WallNegX": ((-8.0, 0.0, 0.5), (0.2, 16.0, 5.0)),
        "WallPosY": ((0.0, 8.0, 0.5), (16.0, 0.2, 5.0)),
        "WallNegY": ((0.0, -8.0, 0.5), (16.0, 0.2, 5.0)),
        "Floor": ((0.0, 0.0, -2.0), (16.0, 16.0, 0.2)),
        "Ceiling": ((0.0, 0.0, 3.0), (16.0, 16.0, 0.2)),
    }
    for name, (translation, scale) in walls.items():
        cube = UsdGeom.Cube.Define(stage, f"/World/ProbeBox/{name}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.45, 0.48)])
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))


def _read_and_validate_attributes(prim: Any) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    mismatches: list[dict[str, Any]] = []
    for name, expected in EXPECTED_ATTRIBUTES.items():
        attribute = prim.GetAttribute(name)
        actual = attribute.Get() if attribute.IsValid() else None
        observed[name] = _json_value(actual)
        if not _values_equal(actual, expected):
            mismatches.append(
                {"attribute": name, "expected": expected, "actual": _json_value(actual)}
            )
    report = {
        "schema_version": "exact_rtx_profile_attribute_readback_v1",
        "status": "PASS" if not mismatches else "FAIL",
        "prim_path": str(prim.GetPath()),
        "prim_type": prim.GetTypeName(),
        "applied_schemas": list(prim.GetAppliedSchemas()),
        "expected": EXPECTED_ATTRIBUTES,
        "observed": observed,
        "mismatches": mismatches,
        "derived": {
            "ticks_per_scan": int(
                observed["omni:sensor:Core:patternFiringRateHz"]
                / observed["omni:sensor:Core:scanRateBaseHz"]
            ),
            "nominal_rays_per_scan": int(
                observed["omni:sensor:Core:numberOfEmitters"]
                * observed["omni:sensor:Core:patternFiringRateHz"]
                / observed["omni:sensor:Core:scanRateBaseHz"]
            ),
        },
    }
    return report


def _field(gmo: Any, name: str, count: int, dtype: Any) -> np.ndarray:
    value = getattr(gmo, name, None)
    if value is None:
        return np.zeros(count, dtype=dtype)
    return np.asarray(value, dtype=dtype).copy()


class _ExactProfileGmoCaptureWriter(Writer):
    """Capture GMO through the Writer callback used by Isaac Sim 6.0.1 tests."""

    def __init__(self) -> None:
        self.data_structure = "renderProduct"
        self.annotators = [rep.annotators.get("GenericModelOutput")]
        self.scan: dict[str, Any] | None = None
        self.callback_count = 0
        self.zero_element_count = 0
        self.parse_failures: list[str] = []

    def write(self, data: Any) -> None:
        self.callback_count += 1
        if self.scan is not None or "renderProducts" not in data:
            return
        for _render_product, render_product_data in data["renderProducts"].items():
            raw = render_product_data.get("GenericModelOutput")
            if isinstance(raw, dict):
                raw = raw.get("data")
            if raw is None:
                continue
            try:
                gmo = parse_generic_model_output_data(raw)
            except Exception as error:
                self.parse_failures.append(
                    f"callback={self.callback_count}: {type(error).__name__}: {error}"
                )
                continue
            count = int(gmo.numElements)
            if count <= 0:
                self.zero_element_count += 1
                continue
            if not bool(getattr(gmo, "scanComplete", False)):
                continue
            azimuth = _field(gmo, "x", count, np.float32)
            elevation = _field(gmo, "y", count, np.float32)
            distance = _field(gmo, "z", count, np.float32)
            raster, valid, raster_counts = rasterize_generic_model_output(
                azimuth, elevation, distance
            )
            self.scan = {
                "timestamp_ns": int(getattr(gmo, "timestampNs", self.callback_count)),
                "azimuth_deg": azimuth,
                "elevation_deg": elevation,
                "range_raw_m": distance,
                "channel_id": _field(gmo, "channelId", count, np.uint32),
                "emitter_id": _field(gmo, "emitterId", count, np.uint32),
                "time_offset_ns": _field(gmo, "timeOffsetNs", count, np.int64),
                "range_m": raster,
                "valid_mask": valid,
                "raster_counts": raster_counts,
                "parse_failures_before_complete_scan": list(self.parse_failures),
            }
            return


def _capture_first_complete_scan(
    writer: _ExactProfileGmoCaptureWriter,
) -> tuple[dict[str, Any], int]:
    for frame in range(300):
        simulation_app.update()
        if writer.scan is not None:
            return writer.scan, frame + 1
    raise RuntimeError(
        "no valid complete GMO scan from Writer callback within 300 render frames; "
        f"callbacks={writer.callback_count}, zero_elements={writer.zero_element_count}, "
        f"parse_failures={writer.parse_failures[-5:]}"
    )


def main() -> int:
    sensor_usda = args.probe_sensor_usda.resolve()
    source_json = args.probe_source_json.resolve()
    output_dir = args.probe_output_dir.resolve()
    metrics_dir = args.probe_metrics_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    source_profile = json.loads(source_json.read_text(encoding="utf-8"))
    if source_profile["profile"]["patternFiringRateHz"] != 7200:
        raise RuntimeError("source JSON no longer freezes 7200 Hz pattern firing")
    if source_profile["profile"]["numberOfEmitters"] != 16:
        raise RuntimeError("source JSON no longer freezes 16 emitters")

    create_new_stage()
    _create_closed_box()
    for _ in range(5):
        simulation_app.update()

    lidar = Lidar.create(
        path="/World/ProbeLidar",
        usd_path=str(sensor_usda),
        accumulate_outputs=True,
        aux_output_level="BASIC",
        tick_rate=10.0,
        attributes={"omni:sensor:Core:outputFrameOfReference": "SENSOR"},
        translations=[[0.0, 0.0, 0.0]],
        orientations=[[1.0, 0.0, 0.0, 0.0]],
    )
    if len(lidar) != 1:
        raise RuntimeError(f"expected one lidar prim, received {len(lidar)}")
    prim = lidar.prims[0]
    if prim.GetTypeName() != "OmniLidar":
        raise RuntimeError(f"expected OmniLidar, received {prim.GetTypeName()!r}")

    attribute_report = _read_and_validate_attributes(prim)
    attribute_report["runtime_aux_output_level"] = lidar.aux_output_level
    _write_json(metrics_dir / "attribute_readback.json", attribute_report)
    if attribute_report["status"] != "PASS":
        raise RuntimeError(f"frozen attribute mismatch: {attribute_report['mismatches']}")
    if attribute_report["derived"]["ticks_per_scan"] != 720:
        raise RuntimeError("derived ticks_per_scan is not 720")
    if attribute_report["derived"]["nominal_rays_per_scan"] != 11520:
        raise RuntimeError("derived nominal_rays_per_scan is not 11520")

    model_info = ModelInfo()
    model_info.modelName = "lidar.core"
    model_info.modelVersion = "1.0"
    model_info.schemaVersion = "1.0"
    model_info.modelVendor = "nv"
    model_info.marketName = "GenericLidar"
    checker = SensorCheckerUtil()
    init_result = checker.init(model_info)
    checker_error = checker.validateParams(prim)
    validated_params = checker.getValidatedParams()
    checker_report = {
        "schema_version": "exact_rtx_profile_sensor_checker_v1",
        "status": "PASS" if checker_error in (None, "") else "FAIL",
        "init_result": init_result,
        "validation_error": checker_error,
        "validated_parameter_count": int(validated_params.numParams),
    }
    _write_json(metrics_dir / "sensor_checker.json", checker_report)
    if checker_report["status"] != "PASS" or checker_report["validated_parameter_count"] <= 0:
        raise RuntimeError(f"SensorChecker rejected exact profile: {checker_report}")

    stage_path = output_dir / "probe_stage.usda"
    omni.usd.get_context().get_stage().GetRootLayer().Export(str(stage_path))

    rep.WriterRegistry.register(_ExactProfileGmoCaptureWriter)
    sensor = LidarSensor(lidar)
    writer = sensor.attach_writer("_ExactProfileGmoCaptureWriter")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(30):
        simulation_app.update()
    scan, rendered_frames = _capture_first_complete_scan(writer)
    timeline.stop()

    scan_path = output_dir / "complete_scan.npz"
    np.savez_compressed(
        scan_path,
        azimuth_deg=scan["azimuth_deg"],
        elevation_deg=scan["elevation_deg"],
        range_raw_m=scan["range_raw_m"],
        channel_id=scan["channel_id"],
        emitter_id=scan["emitter_id"],
        time_offset_ns=scan["time_offset_ns"],
        range_m=scan["range_m"],
        valid_mask=scan["valid_mask"],
        timestamp_ns=np.asarray(scan["timestamp_ns"], dtype=np.int64),
    )
    valid_count = int(np.sum(scan["valid_mask"]))
    gmo_report = {
        "schema_version": "exact_rtx_profile_gmo_summary_v1",
        "status": "PASS" if valid_count > 0 else "FAIL",
        "render_frames_to_first_complete_scan": rendered_frames,
        "writer_callback_count": writer.callback_count,
        "writer_zero_element_count": writer.zero_element_count,
        "scan_complete": True,
        "timestamp_ns": scan["timestamp_ns"],
        "raw_return_count": int(len(scan["range_raw_m"])),
        "raster_shape": list(scan["range_m"].shape),
        "raster_valid_count": valid_count,
        "raster_valid_ratio": float(np.mean(scan["valid_mask"])),
        "raster_counts": scan["raster_counts"],
        "parse_failures_before_complete_scan": scan["parse_failures_before_complete_scan"],
        "scan_file": str(scan_path),
    }
    _write_json(metrics_dir / "gmo_summary.json", gmo_report)
    if gmo_report["status"] != "PASS" or gmo_report["raster_shape"] != [16, 720]:
        raise RuntimeError(f"invalid complete scan: {gmo_report}")

    _write_json(
        metrics_dir / "capture_summary.json",
        {
            "schema_version": "exact_rtx_profile_capture_summary_v1",
            "overall_status": "PASS_EXACT_PROFILE_COMPLETE_SCAN",
            "isaac_sim_version": "6.0.1",
            "sensor_creation": "Lidar.create(usd_path=...)",
            "sensor_usda": str(sensor_usda),
            "sensor_usda_sha256": _sha256(sensor_usda),
            "source_json": str(source_json),
            "source_json_sha256": _sha256(source_json),
            "counts": {
                "analytic_test_worlds": 1,
                "cano_worlds": 0,
                "sensors": 1,
                "poses": 1,
                "complete_diagnostic_scans": 1,
                "formal_dataset_samples": 0,
                "training_samples": 0,
                "models": 0,
            },
        },
    )
    return 0


exit_code = 1
try:
    exit_code = main()
except Exception as error:
    _write_json(
        args.probe_metrics_dir / "capture_failure.json",
        {
            "schema_version": "exact_rtx_profile_capture_failure_v1",
            "overall_status": "FAIL_STOPPED_NO_FALLBACK",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "counts": {
                "cano_worlds": 0,
                "formal_dataset_samples": 0,
                "training_samples": 0,
                "models": 0,
            },
        },
    )
    traceback.print_exc()
finally:
    simulation_app.close()

raise SystemExit(exit_code)

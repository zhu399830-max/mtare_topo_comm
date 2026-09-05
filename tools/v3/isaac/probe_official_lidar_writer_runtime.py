#!/usr/bin/env python3
"""Probe the pinned Isaac built-in lidar Writer runtime without project data."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--control-output-dir", required=True, type=Path)
parser.add_argument("--control-metrics-dir", required=True, type=Path)
args, _unknown = parser.parse_known_args()

simulation_app = SimulationApp({"headless": True, "enable_motion_bvh": True})

import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.experimental.objects import Cube
from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, parse_generic_model_output_data
from isaacsim.sensors.experimental.rtx import generic_model_output
from omni.replicator.core import Writer


MAX_RENDER_FRAMES = 300
EXPECTED_GMO_MAGIC = int(generic_model_output.getMagicNumberGMO())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class OfficialGmoRuntimeControlWriter(Writer):
    """Count official GenericModelOutput callbacks without saving point samples."""

    def __init__(self) -> None:
        self.data_structure = "renderProduct"
        self.annotators = [rep.annotators.get("GenericModelOutput")]
        self.callback_count = 0
        self.payload_count = 0
        self.valid_header_count = 0
        self.positive_element_count = 0
        self.zero_element_count = 0
        self.complete_scan_count = 0
        self.first_valid_header: dict[str, Any] | None = None
        self.parse_failures: list[str] = []

    def write(self, data: Any) -> None:
        self.callback_count += 1
        if "renderProducts" not in data:
            return
        for render_product, render_product_data in data["renderProducts"].items():
            raw = render_product_data.get("GenericModelOutput")
            if isinstance(raw, dict):
                raw = raw.get("data")
            if raw is None:
                continue
            self.payload_count += 1
            try:
                gmo = parse_generic_model_output_data(raw)
            except Exception as error:
                self.parse_failures.append(
                    f"callback={self.callback_count}: {type(error).__name__}: {error}"
                )
                continue
            magic = int(getattr(gmo, "magicNumber", 0))
            count = int(getattr(gmo, "numElements", 0))
            scan_complete = bool(getattr(gmo, "scanComplete", False))
            if magic == EXPECTED_GMO_MAGIC:
                self.valid_header_count += 1
                if self.first_valid_header is None:
                    self.first_valid_header = {
                        "callback": self.callback_count,
                        "render_product": str(render_product),
                        "magic_number": magic,
                        "num_elements": count,
                        "scan_complete": scan_complete,
                        "timestamp_ns": int(getattr(gmo, "timestampNs", 0)),
                    }
            if count > 0:
                self.positive_element_count += 1
            else:
                self.zero_element_count += 1
            if count > 0 and scan_complete:
                self.complete_scan_count += 1


def main() -> int:
    output_dir = args.control_output_dir.resolve()
    metrics_dir = args.control_metrics_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cube_positions = {
        "/World/cube_front": np.array([5.0, 0.0, 0.0]),
        "/World/cube_left": np.array([0.0, 5.0, 0.0]),
        "/World/cube_right": np.array([0.0, -5.0, 0.0]),
        "/World/cube_above": np.array([0.0, 0.0, 5.0]),
    }
    for path, position in cube_positions.items():
        Cube(path, positions=position, scales=np.array([2.0, 2.0, 2.0]))

    lidar = Lidar.create(
        "/World/lidar",
        config="Example_Rotary",
        translations=np.array([0.0, 0.0, 1.0]),
        aux_output_level="BASIC",
    )
    sensor = LidarSensor(lidar, annotators=[])
    rep.WriterRegistry.register(OfficialGmoRuntimeControlWriter)
    writer = sensor.attach_writer("OfficialGmoRuntimeControlWriter")

    stage_path = output_dir / "official_control_stage.usda"
    omni.usd.get_context().get_stage().Export(str(stage_path))

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _frame in range(MAX_RENDER_FRAMES):
        simulation_app.update()
    timeline.stop()

    if writer.callback_count == 0:
        status = "FAIL_RUNTIME_WRITER_CALLBACK_ZERO"
        classification = "HOST_OR_HEADLESS_REPLICATOR_RUNTIME_BLOCKED"
    elif writer.valid_header_count == 0:
        status = "FAIL_RUNTIME_WRITER_GMO_INVALID"
        classification = "OFFICIAL_WRITER_CALLBACK_PRESENT_BUT_GMO_INVALID"
    elif writer.positive_element_count == 0:
        status = "FAIL_RUNTIME_WRITER_ZERO_ELEMENTS"
        classification = "OFFICIAL_WRITER_RUNTIME_PRESENT_BUT_NO_RETURNS"
    else:
        status = "PASS_OFFICIAL_WRITER_RUNTIME_HEALTH"
        classification = "CUSTOM_SENSOR_OR_SCENE_COUPLING_REMAINS"

    report = {
        "schema_version": "official_lidar_writer_runtime_control_v1",
        "overall_status": status,
        "classification": classification,
        "isaac_sim_version": "6.0.1",
        "control_sensor": "Example_Rotary",
        "control_scene": "official_four_cube_layout",
        "max_render_frames": MAX_RENDER_FRAMES,
        "expected_gmo_magic": EXPECTED_GMO_MAGIC,
        "writer": {
            "callback_count": writer.callback_count,
            "payload_count": writer.payload_count,
            "valid_header_count": writer.valid_header_count,
            "positive_element_count": writer.positive_element_count,
            "zero_element_count": writer.zero_element_count,
            "complete_scan_count": writer.complete_scan_count,
            "first_valid_header": writer.first_valid_header,
            "parse_failures": writer.parse_failures,
        },
        "counts": {
            "official_control_worlds": 1,
            "cano_worlds": 0,
            "built_in_control_sensors": 1,
            "custom_project_sensors": 0,
            "poses": 1,
            "saved_point_samples": 0,
            "formal_dataset_samples": 0,
            "labels": 0,
            "training_samples": 0,
            "models": 0,
        },
        "stage_file": str(stage_path),
        "claim_boundary": (
            "This control tests only whether the pinned headless Isaac runtime can trigger "
            "an official built-in lidar GMO Writer. It is not the project sensor, a dataset "
            "sample, a Cano test, or evidence for learning or planning."
        ),
    }
    _write_json(metrics_dir / "runtime_control.json", report)
    return 0 if status.startswith("PASS") else 2


exit_code = 1
try:
    exit_code = main()
except Exception as error:
    _write_json(
        args.control_metrics_dir / "runtime_control_failure.json",
        {
            "schema_version": "official_lidar_writer_runtime_control_failure_v1",
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
finally:
    simulation_app.close()

raise SystemExit(exit_code)

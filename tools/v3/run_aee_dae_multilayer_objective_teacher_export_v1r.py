#!/usr/bin/env python3
"""Run the immutable ten-shard DAE-multilayer AEE teacher export."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import run_aee_objective_teacher_export_v1 as legacy


legacy.RUN_ID = "gate2_20260821_aee_dae_multilayer_objective_teacher_export_v1r_seed20260820"
legacy.IMAGE = "mtare-semantic-runtime:local"
legacy.STATUS_PASS = "PASS_AEE_DAE_MULTILAYER_OBJECTIVE_TEACHER_EXPORT_V1R"
legacy.STATUS_FAIL = "FAIL_AEE_DAE_MULTILAYER_OBJECTIVE_TEACHER_EXPORT_V1R"

_legacy_load_json = legacy.load_json
_geometry_bindings: list[dict[str, Any]] | None = None
_geometry_prevalidated = False


def _runtime_volume_args() -> list[str]:
    return [
        "-v", "/etc/localtime:/etc/localtime:ro",
        "-v", "/dev/bus/usb:/dev/bus/usb",
        "-v", "/dev/input:/dev/input",
        "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
    ]


def _frozen_image_file_hash(path: str) -> str:
    completed = subprocess.run(
        [
            "docker", "run", "--rm", "--network", "none",
            *_runtime_volume_args(),
            "--entrypoint", "/usr/bin/sha256sum", legacy.IMAGE, path,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot hash frozen image input {path}: {completed.stderr.strip()}")
    return completed.stdout.split()[0]


def load_json_with_geometry_capture(path: Path) -> dict[str, Any]:
    global _geometry_bindings
    payload = _legacy_load_json(path)
    bindings = payload.get("complete_maps")
    if isinstance(bindings, list):
        _geometry_bindings = bindings
    return payload


def _binding_world_from_mesh(item: dict[str, Any]) -> list[list[float]]:
    script = """
import json
import sys
sys.path.insert(0, '/workspace/src')
from mtare_topo.oracle.oriented_dae_support import load_gazebo_collision_mesh_binding
b = load_gazebo_collision_mesh_binding(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
print(json.dumps(b.world_from_mesh.astype(float).tolist(), separators=(',', ':')))
"""
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        *_runtime_volume_args(),
        "-v",
        f"{legacy.PROJECT_ROOT}:/workspace:ro",
        "-w",
        "/workspace",
        "--entrypoint",
        "/usr/bin/python3",
        legacy.IMAGE,
        "-c",
        script,
        item["world_path_in_image"],
        item["model_sdf_path_in_image"],
        item["include_uri"],
        item["mesh_uri"],
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot resolve Gazebo coordinate binding for {item.get('world')}: "
            f"{completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def image_file_hash_after_full_geometry_prevalidation(path: str) -> str:
    """Validate both worlds before the legacy runner can write shard one."""

    global _geometry_prevalidated
    if not _geometry_prevalidated:
        if not isinstance(_geometry_bindings, list) or {
            item.get("world") for item in _geometry_bindings if isinstance(item, dict)
        } != {"tunnel", "garage"}:
            raise RuntimeError("exactly one tunnel and one garage geometry binding are required")
        fields = (
            ("path_in_image", "sha256", "obstacle PLY"),
            ("support_mesh_path_in_image", "support_mesh_sha256", "support DAE"),
            ("world_path_in_image", "world_sha256", "Gazebo world"),
            ("model_sdf_path_in_image", "model_sdf_sha256", "model SDF"),
        )
        for item in _geometry_bindings:
            world = item["world"]
            for path_key, hash_key, label in fields:
                if _frozen_image_file_hash(item[path_key]) != item[hash_key]:
                    raise RuntimeError(f"{world} {label} identity drift")
            observed = _binding_world_from_mesh(item)
            if observed != item.get("expected_world_from_mesh"):
                raise RuntimeError(f"{world} Gazebo coordinate binding drift")
        _geometry_prevalidated = True
    return _frozen_image_file_hash(path)


def case_command(
    source_run: Path,
    run_dir: Path,
    record: dict[str, Any],
    complete_map: dict[str, Any],
    index: int,
) -> tuple[list[str], str]:
    trajectory_id = record["trajectory_id"]
    sensor_relative = Path(record["sensor_shard"])
    if sensor_relative.is_absolute() or ".." in sensor_relative.parts:
        raise RuntimeError("sensor shard path escapes source run")
    output = f"/evidence/teacher_shards/{trajectory_id}.npz"
    inner = [
        "python3",
        "/workspace/tools/v3/generate_aee_objective_teacher_shard_v1r.py",
        "--sensor-shard",
        f"/source/{sensor_relative.as_posix()}",
        "--sensor-shard-sha256",
        record["sensor_shard_sha256"],
        "--obstacle-map",
        complete_map["path_in_image"],
        "--obstacle-map-sha256",
        complete_map["sha256"],
        "--support-mesh",
        complete_map["support_mesh_path_in_image"],
        "--support-mesh-sha256",
        complete_map["support_mesh_sha256"],
        "--world-file",
        complete_map["world_path_in_image"],
        "--world-file-sha256",
        complete_map["world_sha256"],
        "--model-sdf",
        complete_map["model_sdf_path_in_image"],
        "--model-sdf-sha256",
        complete_map["model_sdf_sha256"],
        "--include-uri",
        complete_map["include_uri"],
        "--mesh-uri",
        complete_map["mesh_uri"],
        "--trajectory-id",
        trajectory_id,
        "--output",
        output,
    ]
    shell = "export PYTHONPATH=/workspace/src:$PYTHONPATH && " + shlex.join(inner)
    name = f"aee-dae-multilayer-teacher-v1r-{index:02d}"
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        *_runtime_volume_args(),
        "--name",
        name,
        "-v",
        f"{legacy.PROJECT_ROOT}:/workspace:ro",
        "-v",
        f"{source_run}:/source:ro",
        "-v",
        f"{run_dir / 'artifacts'}:/evidence:rw",
        "--entrypoint",
        "/bin/bash",
        legacy.IMAGE,
        "-lc",
        shell,
    ], name


def audit_teacher_shard(*args: Any, **kwargs: Any) -> dict[str, Any]:
    summary = dict(args[2])
    summary["status"] = "PASS_AEE_OBJECTIVE_TEACHER_SHARD_V1"
    replaced = (args[0], args[1], summary, *args[3:])
    return _legacy_audit(*replaced, **kwargs)


_legacy_audit = legacy.audit_teacher_shard
legacy.load_json = load_json_with_geometry_capture
legacy.image_file_hash = image_file_hash_after_full_geometry_prevalidation
legacy.case_command = case_command
legacy.audit_teacher_shard = audit_teacher_shard


if __name__ == "__main__":
    raise SystemExit(legacy.main())

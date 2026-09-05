#!/home/zeng-workstation/anaconda3/bin/python
"""Execute and seal the approved 64-frame AEE sensor-operator audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate2_20260821_aee_sensor_operator_parity_v1_seed20260820"
HOST_PYTHON = Path("/home/zeng-workstation/anaconda3/bin/python")
RAYCAST_PYTHON = Path("/tmp/mtare_gate4_meshing_sidecar_v1/bin/python")
RAYCASTER = PROJECT_ROOT / "tools/v3/raycast_aee_dae_same_pose_v1.py"
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_aee_sensor_operator_parity_v1.py"
CONTAINER_ID = "f6d9ada42365"
IMAGE_ID = "sha256:9819eea672b2001ed18f12ee09c67f0da53a5ba6ce69192e2f21c128e5e78f9a"
DAE = {
    "tunnel": ("/home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/mesh/tunnel/meshes/tunnel.dae", "10b4e640ca2db08aecfc4fe15c8baf3a7763eac5b7c33c596ff0d68fed6abc13"),
    "garage": ("/home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/mesh/garage/meshes/garage.dae", "06c6686758fbf31cffdf952ae34b4757ba0230bdca1cfbbfa252a9f3b6ab2b4e"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(run_dir: Path) -> tuple[int, str]:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    text = "".join(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files)
    destination.write_text(text)
    return len(files), hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED":
        raise RuntimeError("immutable run identity/state mismatch")
    if spec.get("gate") != 2 or spec.get("sample_contract") != {"tunnel": 32, "garage": 32, "total": 64}:
        raise RuntimeError("audit scope drift")
    if spec.get("data_card"):
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("supplied Data Card is not approved")
    for name, item in spec["frozen_tools"].items():
        if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen tool drift: {name}")
    for name, relative in (("sensor", spec["source_runs"]["sensor"]), ("teacher", spec["source_runs"]["teacher"])):
        source_seal = PROJECT_ROOT / relative / "artifacts/evidence_sha256.txt"
        if sha256(source_seal) != spec["source_runs"][f"{name}_seal_sha256"]:
            raise RuntimeError(f"{name} source seal drift")
    inspected = json.loads(subprocess.check_output(["docker", "inspect", CONTAINER_ID], text=True))[0]
    if inspected["Image"] != IMAGE_ID or inspected["State"]["Status"] != "exited":
        raise RuntimeError("frozen stopped-container identity/state drift")
    geometry = run_dir / "artifacts/geometry"; geometry.mkdir(parents=True, exist_ok=False)
    for world, (source, expected) in DAE.items():
        destination = geometry / f"{world}.dae"
        subprocess.run(["docker", "cp", f"{CONTAINER_ID}:{source}", str(destination)], check=True)
        if sha256(destination) != expected:
            raise RuntimeError(f"{world} DAE hash drift")
    environment = {
        "host": platform.platform(), "host_python": str(HOST_PYTHON),
        "raycast_python": str(RAYCAST_PYTHON), "docker_container": CONTAINER_ID, "docker_image_id": IMAGE_ID,
        "versions": {
            "host": subprocess.check_output([str(HOST_PYTHON), "-c", "import numpy,torch,matplotlib;print(numpy.__version__,torch.__version__,matplotlib.__version__)"], text=True).strip(),
            "raycast": subprocess.check_output([str(RAYCAST_PYTHON), "-c", "import numpy,open3d;print(numpy.__version__,open3d.__version__)"], text=True).strip(),
        },
    }
    write_json(run_dir / "config/environment.json", environment)
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
    ideal = run_dir / "artifacts/ideal_scans.npz"; log = run_dir / "logs/raw.log"; started = time.monotonic()
    commands = [
        [str(RAYCAST_PYTHON), str(RAYCASTER), "--assets-dir", str(geometry), "--output", str(ideal)],
        [str(HOST_PYTHON), str(EXECUTOR), "--run-dir", str(run_dir), "--ideal-scans", str(ideal)],
    ]
    with log.open("w") as stream:
        for command in commands:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stream.write("COMMAND " + json.dumps(command) + "\n" + completed.stdout)
            stream.flush()
            if completed.returncode != 0:
                write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "FAILED", "return_code": completed.returncode})
                seal(run_dir); return completed.returncode
    summary = load_json(run_dir / "metrics/summary.json")
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED",
        "overall_status": summary["overall_status"], "scientific_conclusion": summary["conclusion"],
        "duration_seconds": time.monotonic() - started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    count, seal_hash = seal(run_dir)
    print(json.dumps({"overall_status": summary["overall_status"], "conclusion": summary["conclusion"], "sealed_files": count, "seal_sha256": seal_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

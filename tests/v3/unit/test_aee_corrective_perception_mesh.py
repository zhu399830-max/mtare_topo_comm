"""Zero-material contracts for the corrective perception-mesh executor."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
E1 = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_topology_v1/bin/python")


def _probe() -> dict:
    code = """
import json, sys
sys.path.insert(0, 'tools/v3')
import execute_aee_corrective_perception_mesh_v1 as executor
print(json.dumps({'parents': executor._parents(), 'scope': executor.EXPECTED_SCOPE}))
"""
    done = subprocess.run(
        [str(E1), "-c", code], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return json.loads(done.stdout)


def test_formal_runner_imports_in_frozen_mesh_environment() -> None:
    code = """
import sys
sys.path.insert(0, 'tools/v3')
import run_aee_corrective_perception_mesh_v1 as runner
assert runner.E1 == runner.Path('/home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_topology_v1/bin/python')
"""
    subprocess.run([str(E1), "-c", code], cwd=ROOT, check=True)


def test_v1r_runner_propagates_fixed_checkout_environment() -> None:
    code = """
import sys
from pathlib import Path
sys.path.insert(0, 'tools/v3')
import run_aee_corrective_perception_mesh_v1r as runner
environment = runner._environment()
paths = environment['PYTHONPATH'].split(':')
expected = str(Path('external/procedural-subt-gen/src').resolve())
assert runner.RUN_ID.endswith('_v1r_seed20260821')
assert expected in paths
assert 'env=runner._environment()' not in Path(runner.__file__).read_text()
assert 'env=_environment()' in Path(runner.__file__).read_text()
"""
    subprocess.run([str(E1), "-c", code], cwd=ROOT, check=True)


def test_v1r2_process_tree_rss_monitor_passes_and_fails_closed() -> None:
    code = r'''
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, 'tools/v3')
import run_aee_corrective_perception_mesh_v1r2 as runner
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    ok = runner._run_monitored(
        [str(runner.E1), '-c', "print('ok')"],
        cwd=runner.PROJECT_ROOT,
        environment=runner._environment(),
        log_path=root / 'ok.log',
        trace_path=root / 'ok.jsonl',
        time_limit_seconds=10.0,
        rss_limit_bytes=2**63,
        sample_interval_seconds=0.01,
    )
    assert ok[0] == 0
    assert ok[2]['within_rss_limit'] is True
    assert ok[2]['stop_reason'] is None
    assert len((root / 'ok.jsonl').read_text().splitlines()) >= 1
    stopped = runner._run_monitored(
        [str(runner.E1), '-c', 'import time; time.sleep(10)'],
        cwd=runner.PROJECT_ROOT,
        environment=runner._environment(),
        log_path=root / 'stopped.log',
        trace_path=root / 'stopped.jsonl',
        time_limit_seconds=10.0,
        rss_limit_bytes=1,
        sample_interval_seconds=0.01,
    )
    assert stopped[0] != 0
    assert stopped[2]['within_rss_limit'] is False
    assert stopped[2]['stop_reason'] == 'RSS_LIMIT_EXCEEDED'
    assert stopped[2]['forced_kill'] is False
print(json.dumps({'pass_case': True, 'rss_fail_closed': True}))
'''
    subprocess.run([str(E1), "-c", code], cwd=ROOT, check=True)


def test_v1r3_batch_runner_is_lightweight_and_scope_exact() -> None:
    code = r'''
import json, sys
sys.path.insert(0, 'tools/v3')
sys.path.insert(0, 'src')
import run_aee_corrective_perception_mesh_v1r3 as runner
assert 'open3d' not in sys.modules
parents = runner._parents()
assert len(parents) == 20
assert len({item['parent_id'] for item in parents}) == 20
assert sum(item['split'] == 'corrective_train' for item in parents) == 10
assert sum(item['split'] == 'corrective_validation' for item in parents) == 10
assert runner.EXPECTED_SCOPE['native_materializations'] == 20
assert runner.EXPECTED_SCOPE['lidar_observations'] == 0
assert runner.RSS_LIMIT_BYTES == 2 * 1024**3
print(json.dumps({'lightweight': True, 'parents': len(parents)}))
'''
    subprocess.run([str(E1), "-c", code], cwd=ROOT, check=True)


def test_v1r3_worker_rejects_parent_identity_before_materialization() -> None:
    code = r'''
import sys, tempfile
from pathlib import Path
sys.path.insert(0, 'tools/v3')
sys.path.insert(0, 'src')
import execute_aee_corrective_perception_mesh_parent_v1r3 as worker
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / 'artifacts/meshes').mkdir(parents=True)
    (root / 'previews/train_complete_maps').mkdir(parents=True)
    (root / 'metrics/worker_receipts').mkdir(parents=True)
    try:
        worker.execute_parent(root, 0, 'WRONG_PARENT')
    except RuntimeError as error:
        assert 'index/id mismatch' in str(error)
    else:
        raise AssertionError('worker accepted a mismatched parent ID')
    assert list((root / 'artifacts/meshes').iterdir()) == []
'''
    subprocess.run([str(E1), "-c", code], cwd=ROOT, check=True)


def test_v1r4_worker_installs_bounded_geometry_and_rejects_wrong_parent() -> None:
    code = r'''
import sys, tempfile
from pathlib import Path
sys.path.insert(0, 'tools/v3')
sys.path.insert(0, 'src')
import execute_aee_corrective_perception_mesh_parent_v1r4 as worker
from mtare_topo.data.cano_memory_bounded_geometry import DEFAULT_SCRATCH_LIMIT_BYTES
assert DEFAULT_SCRATCH_LIMIT_BYTES == 32 * 1024**2
source = Path(worker.__file__).read_text()
assert 'with patched_cano_points_inside()' in source
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / 'artifacts/meshes').mkdir(parents=True)
    (root / 'previews/train_complete_maps').mkdir(parents=True)
    (root / 'metrics/worker_receipts').mkdir(parents=True)
    try:
        worker.execute_parent(root, 0, 'WRONG_PARENT')
    except RuntimeError as error:
        assert 'index/id mismatch' in str(error)
    else:
        raise AssertionError('worker accepted a mismatched parent ID')
    assert list((root / 'artifacts/meshes').iterdir()) == []
'''
    subprocess.run([str(E1), "-c", code], cwd=ROOT, check=True)


def test_v1r4_runner_is_lightweight_and_contract_exact() -> None:
    code = r'''
import json, sys
sys.path.insert(0, 'tools/v3')
sys.path.insert(0, 'src')
import run_aee_corrective_perception_mesh_v1r4 as runner
assert 'open3d' not in sys.modules
assert runner.RUN_ID.endswith('_v1r4_seed20260821')
assert runner.MEMORY_IMPLEMENTATION == 'exact_query_row_chunking_v1'
assert runner.RSS_LIMIT_BYTES == 2 * 1024**3
assert len(runner._parents()) == 20
assert runner.EXPECTED_SCOPE['native_materializations'] == 20
assert runner.EXPECTED_SCOPE['c09_reads'] == 0
assert runner.EXPECTED_SCOPE['c10_reads'] == 0
print(json.dumps({'lightweight': True, 'parents': 20}))
'''
    subprocess.run([str(E1), "-c", code], cwd=ROOT, check=True)


def test_v1r4_independent_verifier_rejects_sealed_v1r3_failure() -> None:
    failed = (
        ROOT
        / "results/gate2_representation/gate2_20260822_aee_corrective_perception_mesh_v1r3_seed20260821"
    )
    done = subprocess.run(
        [
            str(E1),
            "tools/v3/verify_aee_corrective_perception_mesh_v1r4.py",
            "--run-dir",
            str(failed),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    result = json.loads(done.stdout)
    assert done.returncode == 2
    assert result["passed"] is False
    assert result["seal_entries"] == 114
    assert result["maximum_parent_peak_process_tree_rss_bytes"] == 2199797760


def test_selected_parent_adapter_is_exact_and_split_disjoint() -> None:
    parents = _probe()["parents"]
    assert len(parents) == 20
    assert sum(item["split"] == "corrective_train" for item in parents) == 10
    assert sum(item["split"] == "corrective_validation" for item in parents) == 10
    assert len({item["parent_id"] for item in parents}) == 20
    assert len({item["canonical_parent_identity"] for item in parents}) == 20
    assert all((ROOT / item["source_graph"]).is_file() for item in parents)
    assert all((ROOT / item["source_splines"]).is_file() for item in parents)


def test_mesh_scope_has_zero_downstream_operations() -> None:
    scope = _probe()["scope"]
    assert scope["native_materializations"] == scope["sanitized_primary_assets"] == 20
    assert scope["train_complete_previews"] == 10
    for key in (
        "validation_complete_previews",
        "lidar_observations",
        "teacher_labels",
        "formal_dataset_samples",
        "training_samples",
        "models",
        "c09_reads",
        "c10_reads",
        "mtare_changes",
    ):
        assert scope[key] == 0

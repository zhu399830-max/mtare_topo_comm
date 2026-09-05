from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[3] / "tools/v3"
sys.path.insert(0, str(TOOLS))
import publish_gse_teacher_distribution_figure_v1 as module


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublishGSETeacherDistributionFigureTest(unittest.TestCase):
    def _source(self, root: Path) -> Path:
        run = root / "results" / module.EXPECTED_RUN_ID
        status = {"state": "COMPLETED", "overall_status": module.EXPECTED_RUN_STATUS}
        summary = {
            "overall_status": module.EXPECTED_EXECUTOR_STATUS,
            "train_worlds_read": 80,
            "validation_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
        }
        _write_json(run / "RUN_STATE.json", status)
        _write_json(run / "metrics/summary.json", summary)
        _write_json(run / "metrics/runner_summary.json", {"overall_status": module.EXPECTED_RUN_STATUS})
        for relative in module.SOURCE_FILES:
            path = run / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode())
        lines = []
        for relative in module.SOURCE_FILES:
            path = run / relative
            lines.append(f"{_sha256(path)}  {path.relative_to(root)}\n")
        seal = run / "artifacts/evidence_sha256.txt"
        seal.write_text("".join(lines), encoding="utf-8")
        return run

    def test_publishes_complete_provenance_bundle_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._source(root)
            destination = root / "docs/figures/gse_graph"
            result = module.publish(run, destination, project_root=root)
            self.assertEqual(result["published_files"], 6)
            self.assertTrue((destination / "gse_teacher_distribution.pdf").is_file())
            self.assertTrue((destination / "gse_teacher_distribution_sha256.txt").is_file())
            with self.assertRaisesRegex(RuntimeError, "refusing overwrite"):
                module.publish(run, destination, project_root=root)

    def test_rejects_unsealed_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._source(root)
            (run / "artifacts/event_distribution.csv").write_text("drift", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "drifted"):
                module.publish(
                    run,
                    root / "docs/figures/gse_graph",
                    project_root=root,
                )


if __name__ == "__main__":
    unittest.main()

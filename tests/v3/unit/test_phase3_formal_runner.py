import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_cano_phase3_multitask_structural_semantics_v1 as runner


class _Result:
    def __init__(self,return_code,stdout="",stderr=""):
        self.returncode=return_code
        self.stdout=stdout
        self.stderr=stderr


class Phase3FormalRunnerTest(unittest.TestCase):
    def test_training_environment_freezes_cublas_before_child_launch(self):
        env=runner.training_environment()
        self.assertEqual(env["CUBLAS_WORKSPACE_CONFIG"],":4096:8")
        self.assertIn("src",env["PYTHONPATH"])
        self.assertIn("tools/v3",env["PYTHONPATH"])

    def test_probe_persists_failure_then_success(self):
        identity={"torch":"x","cuda_available":True}
        with tempfile.TemporaryDirectory() as raw:
            run=Path(raw)
            with patch.object(runner.subprocess,"run",side_effect=[_Result(1,"","transient"),_Result(0,json.dumps(identity),"")]) as run_mock,patch.object(runner.time,"sleep") as sleep:
                self.assertEqual(runner.probe_environment_identity(run),identity)
            for call in run_mock.call_args_list:
                self.assertEqual(call.kwargs["env"],runner.os.environ.copy())
            self.assertEqual((run/"logs/environment_probe_attempt_1.stderr.log").read_text(),"transient")
            evidence=json.loads((run/"logs/environment_probe_attempts.json").read_text())
            self.assertEqual([x["return_code"] for x in evidence["attempts"]],[1,0])
            sleep.assert_called_once_with(runner.ENVIRONMENT_PROBE_RETRY_DELAY_SECONDS)

    def test_probe_stops_after_two_failures(self):
        with tempfile.TemporaryDirectory() as raw:
            run=Path(raw)
            with patch.object(runner.subprocess,"run",side_effect=[_Result(1,stderr="first"),_Result(2,stderr="second")]),patch.object(runner.time,"sleep"):
                with self.assertRaisesRegex(RuntimeError,"failed after 2 attempts"):
                    runner.probe_environment_identity(run)
            evidence=json.loads((run/"logs/environment_probe_attempts.json").read_text())
            self.assertEqual([x["return_code"] for x in evidence["attempts"]],[1,2])


if __name__=="__main__":
    unittest.main()

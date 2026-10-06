"""Non-evaluation queue checks; native integration requires the real installed skill."""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from common import write_json, digest
from run_candidates import inventory, claim_batch, absolute_budget, BudgetEnded, reflect_attempts
from window_budget import ensure_window, remaining_seconds

class QueueContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/"configs/candidates.json").read_text())

    def test_exact_inventory_excludes_parked_ideas_and_has_no_retries(self):
        names = inventory(self.cfg)
        self.assertEqual(len(names), 18)
        self.assertEqual(len(set(names)), 18)
        self.assertFalse(set(names) & set(self.cfg["parked"]))
        with self.assertRaises(ValueError):
            inventory({**self.cfg, "max_retries_per_trial": 1})

    def test_candidate_scope_keeps_original_clock_and_cannot_launch_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); window = ensure_window(root, epoch=time.time()-100)
            claim_batch(root, window, "engineering")
            self.assertLess(remaining_seconds(ensure_window(root)), 28800-99)
            with self.assertRaises(FileExistsError):
                claim_batch(root, window, "another")
            self.assertEqual(window["started_epoch"], ensure_window(root)["started_epoch"])

    def test_unknown_previous_job_is_not_automatically_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); window = ensure_window(root)
            write_json(root/"runs/windows"/window["window_id"]/"attempt.json", {"run_id": "unknown"})
            with self.assertRaisesRegex(RuntimeError, "active/unknown"):
                claim_batch(root, window, "engineering")

    def test_absolute_budget_interrupts_python_preprocessing(self):
        with self.assertRaises(BudgetEnded):
            with absolute_budget(.05):
                time.sleep(.2)

    def test_timeout_overrides_stale_child_success_and_keeps_later_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); records = []
            for name, native_status, code in (("first", "timeout", -9), ("second", "completed", 0)):
                workspace = root/"runs"/name/"workspace"; result = workspace/"out/result.json"
                write_json(result, {"method": name, "status": "DEVELOPMENTAL_SCORED"})
                records.append({"trial_id": name, "status": native_status, "exit_code": code,
                    "cwd": str(workspace), "attempt_path": "runs/"+name, "seconds": .01,
                    "output_refs": [{"path": result.relative_to(root).as_posix(), "sha256": digest(result)}]})
            values = reflect_attempts(root, ["first", "second", "third"], {"attempts": records})
            self.assertEqual([r["status"] for r in values], ["TIMEOUT", "DEVELOPMENTAL_SCORED", "CARRYOVER_NOT_STARTED"])

    def test_missing_output_cannot_count_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = {"attempts": [{"trial_id": "one", "status": "completed", "exit_code": 0,
                       "attempt_path": "runs/one", "cwd": str(root/"runs/one/workspace"),
                       "seconds": .1, "output_refs": []}]}
            self.assertEqual(reflect_attempts(root, ["one"], receipt)[0]["status"], "FAILED")

NATIVE_ROOT = os.environ.get("RESEARCH_AUTOPILOT_ROOT")
AVAILABLE = bool(NATIVE_ROOT and (Path(NATIVE_ROOT)/"scripts/run_experiments.py").is_file())

@unittest.skipUnless(AVAILABLE, "Complete real Research Autopilot not installed; no test host double")
class NativeContinuation(unittest.TestCase):
    def run_jobs(self, commands, timeout=1):
        sys.path.insert(0, str(Path(NATIVE_ROOT)/"scripts"))
        import run_experiments as native
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root/"program.py"
            script.write_text("import sys,time\nif sys.argv[1]=='bad': raise RuntimeError('engineering failure')\nif sys.argv[1]=='slow': time.sleep(5)\nprint('engineering done')\n")
            ref = {"path": "program.py", "sha256": digest(script)}
            jobs = [{"trial_id": name, "command": [sys.executable, str(script), action], "cwd": ".",
                     "input_refs": [], "code_refs": [ref], "output_paths": [], "seed": 0,
                     "group": "engineering-process-control", "arm_role": "baseline"} for name, action in commands]
            plan = native.make_plan(root, run_id="engineering", jobs=jobs,
                provenance={"git_revision": "engineering", "model_revision": "NONE", "data_revision": "NONE",
                            "environment_digest": hashlib.sha256(b"engineering").hexdigest()},
                limits={"max_attempts": len(jobs), "max_development_trials": len(jobs),
                        "max_confirmation_trials": 0, "max_retries_per_trial": 0,
                        "wall_time_seconds": 30, "attempt_timeout_seconds": timeout},
                purpose="engineering", evidence_mode="developmental")
            return native.run_plan(root, plan, authorizer=lambda scope: True)

    def test_real_process_error_continues_to_next_real_process(self):
        receipt = self.run_jobs([("broken", "bad"), ("later", "good")])
        self.assertEqual([r["status"] for r in receipt["attempts"]], ["failed", "completed"])

    def test_real_individual_timeout_continues_to_next_real_process(self):
        receipt = self.run_jobs([("slow", "slow"), ("later", "good")])
        self.assertEqual([r["status"] for r in receipt["attempts"]], ["timeout", "completed"])

if __name__ == "__main__":
    unittest.main()

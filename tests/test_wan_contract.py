"""CPU algebra/parser engineering fixtures; never model/benchmark evaluation."""
import hashlib
import json
import math
import shutil
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/wan_cache"))
from probe_stats import corr, summarize
from make_result_packet import write_packet
from verify_checkpoint import content_hash


class WanContract(unittest.TestCase):
    def test_spearman_uses_midranks_for_ties(self):
        self.assertAlmostEqual(corr([1, 1, 2], [1, 2, 3])["spearman"], math.sqrt(3) / 2)

    def test_correlation_rejects_nonfinite_or_misaligned_pairs(self):
        for xs, ys in (([1], [1, 2]), ([1, math.nan], [1, 2])):
            with self.assertRaises(ValueError):
                corr(xs, ys)

    def test_rms_gain_is_amplitude_invariant_in_linear_regime(self):
        # Scalar algebra identity only, not generated model results.
        rows = [{"terminal_mse": .04, "reference_guided_local_output_rms": .1},
                {"terminal_mse": .16, "reference_guided_local_output_rms": .2}]
        gain = summarize(rows)["empirical_rms_gain"]
        self.assertAlmostEqual(gain["mean"], 2)
        self.assertAlmostEqual(gain["std"], 0)

    def packet_fixture(self, root):
        # Parser inventory contains identifiers only; it is never a benchmark input.
        write = lambda name, data: (root / name).write_text(json.dumps(data))
        write("manifest.json", {"prompts": [{"benchmark_index": 0}], "force_steps": [1, 2]})
        write("preflight.json", {"status": "PASS", "scientific_evidence": False})
        write("summary.json", {"coverage_complete": True})
        write("native-receipt.json", {"status": "completed", "provenance": {"git_revision": "fixture"}})
        ref = root / "prompt_00000"
        ref.mkdir()
        (ref / "reference_probe.json").write_text(json.dumps({"native_parity_checked_branches": ["cond", "uncond"]}))
        rows = [{"benchmark_index": 0, "force_step": step, "terminal_mse": .04,
                 "reference_guided_local_output_rms": .1,
                 "forced_calls": [{"branch": branch, "step_idx": step, "forced_cache": True}
                                  for branch in ("cond", "uncond")]} for step in (1, 2)]
        (root / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return rows

    def test_complete_requires_raw_inventory_and_native_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.packet_fixture(root)
            self.assertEqual(write_packet(root, "fixture", 0, "completed"), "COMPLETE_ENGINEERING")
            (root / "native-receipt.json").unlink()
            self.assertEqual(write_packet(root, "fixture", 0, "completed"), "FAILED_OR_INCOMPLETE")

    def test_summary_success_cannot_hide_partial_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.packet_fixture(root)
            (root / "metrics.jsonl").write_text(json.dumps(rows[0]) + "\n")
            self.assertEqual(write_packet(root, "fixture", 0, "completed"), "FAILED_OR_INCOMPLETE")
            self.assertEqual(write_packet(root, "fixture", -1, "budget_exhausted"), "CARRYOVER_BUDGET_BOUNDARY")

    def test_duplicate_rows_cannot_hide_missing_intervention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.packet_fixture(root)
            (root / "metrics.jsonl").write_text((json.dumps(rows[0]) + "\n") * 2)
            self.assertEqual(write_packet(root, "fixture", 0, "completed"), "FAILED_OR_INCOMPLETE")

    def test_missing_real_reference_parity_blocks_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.packet_fixture(root)
            (root / "prompt_00000/reference_probe.json").unlink()
            self.assertEqual(write_packet(root, "fixture", 0, "completed"), "FAILED_OR_INCOMPLETE")

    def test_git_blob_hash_matches_actual_pinned_source(self):
        fixture_root = Path(os.environ["QWEN_SOURCE_FIXTURES"])
        path = fixture_root / "src/evaluation/mmeb_v2/eval_embedding.py"
        self.assertEqual(content_hash(path, "sha1", True), "fe25a1563d0adbf57b25b707dc0b4ead4ec0eff1")

    def test_nonfinite_cli_budget_fails_before_execution(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/run_wan_window.py"), "--hours", "nan"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bounded by at most", result.stderr)

    def test_real_entrypoint_imports_after_native_staging(self):
        skill = Path(os.environ["RESEARCH_AUTOPILOT_ROOT"])
        sys.path.insert(0, str(skill / "scripts"))
        import run_experiments as native
        # Run the real committed-source candidates' help interfaces. No torch,
        # model, data download or fabricated benchmark is executed.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = list((ROOT / "tools").glob("*.py")) + list((ROOT / "experiments/wan_cache").glob("*.py"))
            refs = []
            for source in sources:
                target = root / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                refs.append({"path": target.relative_to(root).as_posix(), "sha256": content_hash(target)})
            for index, relative in enumerate(("experiments/wan_cache/preflight.py",
                                              "experiments/wan_cache/run_job.py", "tools/run_wan_window.py")):
                plan = native.make_plan(root, run_id=f"entry-help-{index}",
                    jobs=[{"trial_id": "help", "command": [sys.executable, str(root / relative), "--help"],
                           "cwd": ".", "input_refs": [], "code_refs": refs, "output_paths": [],
                           "seed": 0, "group": "engineering", "arm_role": "fixture"}],
                    provenance={"git_revision": "CPU-entrypoint-check-only", "model_revision": "none",
                                "data_revision": "none", "environment_digest": "1" * 64},
                    limits={"max_attempts": 1, "max_development_trials": 1, "max_confirmation_trials": 0,
                            "max_retries_per_trial": 0, "wall_time_seconds": 5, "attempt_timeout_seconds": 5})
                result = native.run_plan(root, plan,
                    authorizer=lambda scope: scope["plan_digest"] == plan["plan_digest"])
                self.assertEqual(result["status"], "completed")
                self.assertFalse(result["gate_advanced"])


if __name__ == "__main__":
    unittest.main()

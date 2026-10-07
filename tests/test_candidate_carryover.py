"""Retained-file engineering fixtures; no native host, model, labels or scorer.

These records exercise conservative inventory/budget decisions. They are not
proof that the private runner or a benchmark executed.
"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from common import digest, read_json, write_json
from plan_candidate_carryover import audit, canonical_digest, check_bundle_protocol


class CarryoverFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run_id = "r003-engineering-files"
        self.window = {"window_id": "a" * 32, "started_epoch": 1000, "status": "started"}
        self.cfg = read_json(ROOT / "configs/candidates.json")
        write_json(self.root / "configs/candidates.json", self.cfg)
        write_json(self.root / "setup-receipt.json", self.window)
        self.summary_path = self.root / "runs/candidate-batches" / self.run_id / "summary.json"
        self.summary = {"run_id": self.run_id, "window": self.window,
                        "status": "INTERRUPTED", "inventory": self.cfg["controls"] + self.cfg["selected"],
                        "native_run_ids": []}
        write_json(self.summary_path, self.summary)
        self.marker = self.root / "runs/windows" / self.window["window_id"]
        write_json(self.marker / "candidate-batch.json", {
            "run_id": self.run_id, "window_id": self.window["window_id"], "started_epoch": 1000,
            "max_preparation_attempts": 1, "max_method_and_control_attempts": 18, "retry_count": 0})
        write_json(self.root / "runs/latest-candidates.json", {"run_id": self.run_id})
        self.program = self.root / "engineering-marker.txt"
        self.program.write_text("engineering file only; never executed\n")

    def history(self, records, suffix="heads"):
        native_id = self.run_id + "-" + suffix
        native_path = Path("runs/attempts") / native_id
        config_ref = {"path": "configs/candidates.json", "sha256": digest(self.root / "configs/candidates.json")}
        code_ref = {"path": self.program.name, "sha256": digest(self.program)}
        jobs = [{"trial_id": name, "command": ["engineering-never-executed"], "cwd": ".",
                 "input_refs": [config_ref], "code_refs": [code_ref], "output_paths": [],
                 "seed": self.cfg["seed"], "group": "engineering-file-check", "arm_role": "baseline"}
                for name, status, code in records]
        plan = {"schema_id": "experiment-run-plan", "schema_version": "1.0.0", "run_id": native_id,
                "purpose": "engineering", "evidence_mode": "developmental", "protocol_ref": None,
                "protocol_digest": None, "provenance": {"git_revision": "engineering-files-only"},
                "jobs": jobs, "limits": {"max_attempts": len(jobs), "max_development_trials": len(jobs),
                "max_confirmation_trials": 0, "max_retries_per_trial": 0,
                "wall_time_seconds": 100, "attempt_timeout_seconds": 50}}
        plan["plan_digest"] = canonical_digest(plan)
        attempts = []
        for job, (_, status, code) in zip(jobs, records):
            path = native_path / (job["trial_id"] + "-a1-engineering")
            record = {"attempt_path": path.as_posix(), "trial_id": job["trial_id"], "status": status,
                      "exit_code": code, "retry_index": 0, "seconds": 1,
                      "provenance": plan["provenance"], "output_refs": [],
                      **{key: copy.deepcopy(job[key]) for key in ("input_refs", "code_refs", "seed", "group", "arm_role")}}
            write_json(self.root / path / "attempt.json", record)
            attempts.append(record)
        receipt = {"schema_id": "experiment-run-receipt", "schema_version": "1.0.0", "run_id": native_id,
                   "purpose": "engineering", "evidence_mode": "developmental", "status": "failed",
                   "plan_digest": plan["plan_digest"], "attempts": attempts,
                   "resources": {"seconds": len(attempts)}}
        write_json(self.root / native_path / "plan.json", plan)
        write_json(self.root / native_path / "receipt.json", receipt)
        self.summary["native_run_ids"].append(native_id)
        write_json(self.summary_path, self.summary)
        return native_path, plan, receipt

    def inspect(self, **kwargs):
        return audit(self.root, epoch=1100, **kwargs)

    def codes(self, report):
        return {record["code"] for record in report["blockers"]}

    def test_missing_state_does_not_create_clock_or_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = audit(tmp, epoch=1100)
            self.assertTrue(report["blockers"])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_repeated_read_only_audit_preserves_every_byte(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        first, second = self.inspect(), self.inspect()
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(first["unstarted_inventory"], second["unstarted_inventory"])
        self.assertEqual(first["budget"]["remaining_original_seconds"], 28700)
        self.assertFalse(first["dispatch_supported"])
        self.assertIsNone(first["resume_command"])

    def test_unstarted_batch_keeps_all_18_and_requires_original_preparation(self):
        report = self.inspect()
        self.assertEqual(report["status"], "ORIGINAL_PREPARATION_REQUIRED")
        self.assertEqual(len(report["unstarted_inventory"]), 18)
        self.assertEqual(report["eligible_head_inventory"], [])
        self.assertEqual(report["budget"]["remaining_preparation_attempts"], 1)
        self.assertEqual(report["scientific_verdict"], "NONE")

    def test_expired_original_clock_cannot_become_a_new_window(self):
        report = audit(self.root, epoch=1000 + 28801)
        self.assertIn("ORIGINAL_BUDGET_EXHAUSTED", self.codes(report))
        self.assertEqual(report["eligible_head_inventory"], [])
        self.assertEqual(read_json(self.root / "setup-receipt.json"), self.window)

    def test_tail_is_preserved_and_lower_hours_do_not_add_time(self):
        report = self.inspect(hours=2)
        self.assertEqual(report["budget"]["remaining_dispatch_seconds"], 7200 - 100 - 120)
        near_tail = audit(self.root, epoch=1000 + 28800 - 119)
        self.assertIn("ORIGINAL_BUDGET_EXHAUSTED", self.codes(near_tail))

    def test_all_attempted_statuses_consume_one_try_without_scientific_success(self):
        self.history([("I02", "failed", 1), ("I09", "timeout", -9), ("I14", "completed", 0)])
        report = self.inspect()
        self.assertEqual(report["budget"]["head_attempts_used"], 3)
        self.assertFalse({"I02", "I09", "I14"} & set(report["unstarted_inventory"]))
        self.assertEqual([a["native_status"] for a in report["attempts"]], ["failed", "timeout", "completed"])
        self.assertTrue(all(not a["retry_allowed"] for a in report["attempts"]))
        self.assertEqual(report["eligible_head_inventory"], [])

    def test_failed_preparation_cannot_be_retried(self):
        self.history([("prepare", "failed", 1)], suffix="prepare")
        report = self.inspect()
        self.assertIn("PREPARATION_ALREADY_FAILED_NO_RETRY", self.codes(report))
        self.assertEqual(report["budget"]["remaining_preparation_attempts"], 0)
        self.assertEqual(report["retained_attempt_inventory"][0]["native_status"], "failed")

    def test_changed_protocol_preserves_failure_and_blocks_reuse(self):
        self.history([("I02", "failed", 1)])
        self.cfg["seed"] += 1
        write_json(self.root / "configs/candidates.json", self.cfg)
        report = self.inspect()
        self.assertIn("INVALID_OR_UNKNOWN_NATIVE_HISTORY", self.codes(report))
        self.assertEqual(report["retained_attempt_inventory"][0]["native_status"], "failed")
        self.assertNotIn("I02", report["unstarted_inventory"])

    def test_changed_producer_code_blocks_reuse(self):
        self.history([("I02", "failed", 1)])
        self.program.write_text("changed engineering file\n")
        self.assertIn("INVALID_OR_UNKNOWN_NATIVE_HISTORY", self.codes(self.inspect()))

    def test_unknown_attempt_directory_blocks_all_dispatch(self):
        path, _, _ = self.history([("I02", "failed", 1)])
        (self.root / path / "I09-a1-unknown/workspace").mkdir(parents=True)
        report = self.inspect()
        self.assertIn("INVALID_OR_UNKNOWN_NATIVE_HISTORY", self.codes(report))
        self.assertEqual(report["eligible_head_inventory"], [])

    def test_missing_terminal_receipt_blocks_all_dispatch(self):
        path, _, _ = self.history([("I02", "failed", 1)])
        (self.root / path / "receipt.json").unlink()
        report = self.inspect()
        self.assertIn("INVALID_OR_UNKNOWN_NATIVE_HISTORY", self.codes(report))
        self.assertEqual(len(report["retained_attempt_inventory"]), 1)

    def test_hidden_native_run_is_not_silently_ignored(self):
        (self.root / "runs/attempts" / (self.run_id + "-hidden")).mkdir(parents=True)
        self.assertIn("UNRECONCILED_NATIVE_RUNS", self.codes(self.inspect()))

    def test_active_shared_r001_run_is_not_assumed_finished(self):
        write_json(self.marker / "attempt.json", {"run_id": "r001-active"})
        self.assertTrue(self.inspect()["blockers"])

    def test_replaced_window_identity_is_blocked(self):
        write_json(self.root / "setup-receipt.json", {**self.window, "started_epoch": 1050})
        self.assertIn("MISSING_OR_INVALID_RETAINED_STATE", self.codes(self.inspect()))

    def test_duplicate_lifetime_trial_is_retained_and_blocked(self):
        self.history([("I02", "failed", 1)], suffix="heads")
        self.history([("I02", "completed", 0)], suffix="extra")
        report = self.inspect()
        self.assertIn("LIFETIME_TRIAL_CEILING_EXCEEDED", self.codes(report))
        self.assertEqual(len(report["retained_attempt_inventory"]), 2)

    def test_modified_raw_output_is_detected(self):
        path, _, receipt = self.history([("I02", "completed", 0)])
        record = receipt["attempts"][0]
        out = self.root / record["attempt_path"] / "workspace/out/engineering.txt"
        out.parent.mkdir(parents=True)
        out.write_text("original engineering bytes\n")
        record["output_refs"] = [{"path": out.relative_to(self.root).as_posix(), "sha256": digest(out)}]
        write_json(self.root / record["attempt_path"] / "attempt.json", record)
        write_json(self.root / path / "receipt.json", receipt)
        out.write_text("changed engineering bytes\n")
        self.assertIn("INVALID_OR_UNKNOWN_NATIVE_HISTORY", self.codes(self.inspect()))

    def test_receipt_and_attempt_disagreement_is_blocked(self):
        path, _, receipt = self.history([("I02", "failed", 1)])
        record = copy.deepcopy(receipt["attempts"][0])
        record["status"] = "completed"
        write_json(self.root / record["attempt_path"] / "attempt.json", record)
        self.assertIn("INVALID_OR_UNKNOWN_NATIVE_HISTORY", self.codes(self.inspect()))

    def test_unsupported_native_schema_is_blocked(self):
        path, plan, receipt = self.history([("I02", "failed", 1)])
        plan["schema_version"] = receipt["schema_version"] = "99.0.0"
        plan["plan_digest"] = receipt["plan_digest"] = canonical_digest(plan)
        write_json(self.root / path / "plan.json", plan)
        write_json(self.root / path / "receipt.json", receipt)
        self.assertIn("INVALID_OR_UNKNOWN_NATIVE_HISTORY", self.codes(self.inspect()))

    def test_reused_bundle_requires_exact_producer_configuration(self):
        # Protocol metadata only; no benchmark examples, features, labels or scores.
        with self.assertRaisesRegex(ValueError, "configuration"):
            check_bundle_protocol({}, {"input_refs": [{"path": "configs/candidates.json", "sha256": "0" * 64}]},
                                  self.cfg, "1" * 64)

    def test_old_single_task_bundle_cannot_masquerade_as_current_scope(self):
        config_sha = digest(self.root / "configs/candidates.json")
        producer = {"input_refs": [{"path": "configs/candidates.json", "sha256": config_sha}]}
        train = {"dataset_id": self.cfg["training_dataset"], "revision": self.cfg["training_revision"],
                 "split": self.cfg["training_split"], "tasks": self.cfg["training_tasks"]}
        manifest = {"train": train, "evaluation": [{"task": "ScienceQA"}],
                    "teacher": {"temperature": self.cfg["head"]["temperature"]}}
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            check_bundle_protocol(manifest, producer, self.cfg, config_sha)


if __name__ == "__main__":
    unittest.main()

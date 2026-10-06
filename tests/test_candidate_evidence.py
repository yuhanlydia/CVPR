"""File-integrity regression checks, not native execution or benchmark fixtures."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from common import write_json, digest
from collect_candidates import arm_predictions
from experiments.embedding_heads.bundle import preparation_receipt, UPSTREAM


class EvidenceContracts(unittest.TestCase):
    def producer(self, root):
        relative = "runs/attempts/engineering-prepare/prepare-a1-example"
        attempt = root/relative
        manifest = attempt/"workspace/out/manifest.json"
        write_json(manifest, {"engineering_file_marker": True})
        producer = {"attempt_path": relative, "trial_id": "prepare", "status": "completed", "exit_code": 0,
                    "command": [sys.executable, str(ROOT/"tools/prepare_candidate_bundle.py")],
                    "provenance": {"upstream_revision": UPSTREAM},
                    "code_refs": [{"path": "tools/prepare_candidate_bundle.py",
                                   "sha256": digest(ROOT/"tools/prepare_candidate_bundle.py")}],
                    "output_refs": [{"path": manifest.relative_to(root).as_posix(), "sha256": digest(manifest)}]}
        write_json(attempt/"attempt.json", producer)
        return manifest, producer

    def arm(self, root):
        relative = "runs/attempts/engineering-heads/I02-a1-example"
        attempt = root/relative
        pred = attempt/"workspace/out/ScienceQA_pred.jsonl"
        pred.parent.mkdir(parents=True)
        # No scientific query/label or evaluator is constructed in this check.
        pred.write_text('{"engineering_file_marker":true}\n')
        child = {"method": "I02", "status": "DEVELOPMENTAL_SCORED",
                 "tasks": [{"task": "ScienceQA", "denominator": 1, "prediction_sha256": digest(pred)}]}
        result_path = pred.with_name("result.json")
        write_json(result_path, child)
        native = {"attempt_path": relative, "trial_id": "I02", "status": "completed", "exit_code": 0,
                  "output_refs": [{"path": p.relative_to(root).as_posix(), "sha256": digest(p)}
                                  for p in (pred, result_path)]}
        write_json(attempt/"attempt.json", native)
        return {"method": "I02", "attempt_path": relative, "result": copy.deepcopy(child)}, pred, native

    def test_manifest_receipt_works_after_contained_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"another/workspace"
            manifest, _ = self.producer(root)
            self.assertEqual(preparation_receipt(manifest), manifest.parent.parent.parent/"attempt.json")

    def test_manifest_without_producer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = self.producer(Path(tmp))
            (manifest.parent.parent.parent/"attempt.json").unlink()
            with self.assertRaises(OSError):
                preparation_receipt(manifest)

    def test_changed_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _ = self.producer(Path(tmp))
            write_json(manifest, {"engineering_file_marker": "changed"})
            with self.assertRaises(ValueError):
                preparation_receipt(manifest)

    def test_wrong_preparation_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, producer = self.producer(Path(tmp))
            producer["code_refs"][0]["sha256"] = "0"*64
            write_json(manifest.parent.parent.parent/"attempt.json", producer)
            with self.assertRaises(ValueError):
                preparation_receipt(manifest)

    def test_unchanged_bound_prediction_can_be_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); record, pred, _ = self.arm(root)
            path, rows = arm_predictions(root, record, {"task": "ScienceQA", "denominator": 1})
            self.assertEqual(path, pred)
            self.assertEqual(rows, [{"engineering_file_marker": True}])

    def test_changed_prediction_cannot_bypass_child_replay_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); record, pred, native = self.arm(root)
            pred.write_text('{"engineering_file_marker":"changed"}\n')
            native["output_refs"][0]["sha256"] = digest(pred)
            write_json(pred.parent.parent.parent/"attempt.json", native)
            with self.assertRaisesRegex(ValueError, "scorer replay receipt"):
                arm_predictions(root, record, {"task": "ScienceQA", "denominator": 1})

    def test_changed_child_summary_cannot_bypass_native_output_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); record, pred, native = self.arm(root)
            result = record["result"].copy(); result["extra"] = "changed"
            write_json(pred.with_name("result.json"), result)
            native["output_refs"][1]["sha256"] = digest(pred.with_name("result.json"))
            write_json(pred.parent.parent.parent/"attempt.json", native)
            with self.assertRaisesRegex(ValueError, "child summary"):
                arm_predictions(root, record, {"task": "ScienceQA", "denominator": 1})


if __name__ == "__main__":
    unittest.main()

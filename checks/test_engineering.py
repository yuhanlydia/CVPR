"""Engineering/math checks only. No fabricated task samples, labels or scores.

Optional source fixtures are actual fetched author files outside the public repo.
The runner fixture exercises process lifecycle, not a scientific evaluator.
"""
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from common import digest, write_json
from prepare_assets import extract_referenced
from qualify import admission
from source_adapter import EXPECTED_BLOBS, blob_sha, transform
from window_budget import ensure_window, remaining_seconds, claim_attempt


class EngineeringChecks(unittest.TestCase):
    def test_setup_reuses_original_clock_and_exhausts_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            first = ensure_window(directory, epoch=1000)
            second = ensure_window(directory, epoch=1100)
            self.assertEqual(first["started_epoch"], second["started_epoch"])
            self.assertEqual(first["window_id"], second["window_id"])
            self.assertEqual(remaining_seconds(second, epoch=1100), 28700)
            self.assertEqual(remaining_seconds(second, epoch=40000), 0)

    def test_retained_window_prevents_second_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            window = ensure_window(directory, epoch=1000)
            claim_attempt(directory, window, "engineering-first")
            with self.assertRaises(FileExistsError):
                claim_attempt(directory, window, "engineering-second")

    def test_insufficient_time_is_not_admitted(self):
        self.assertFalse(admission(100, 2, 1, 100, 50, 2, 10)["admit"])
        self.assertTrue(admission(1000, 2, 1, 100, 50, 2, 10)["admit"])

    def test_archive_only_extracts_declared_paths_and_rejects_missing(self):
        # Ordinary archive fixture, not evaluation data.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("container/engineering/a.bin", b"engineering fixture")
                z.writestr("unrelated/b.bin", b"not selected")
            extract_referenced(archive, {"engineering/a.bin"}, root / "out")
            self.assertEqual((root / "out/engineering/a.bin").read_bytes(), b"engineering fixture")
            self.assertFalse((root / "out/unrelated/b.bin").exists())
            with self.assertRaises(ValueError):
                extract_referenced(archive, {"missing.bin"}, root / "missing")

    def test_archive_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../escape.bin", b"fixture")
            with self.assertRaises(ValueError):
                extract_referenced(archive, {"escape.bin"}, Path(directory) / "out")

    def test_actual_pinned_source_transform(self):
        fixture_root = os.environ.get("QWEN_SOURCE_FIXTURES")
        if not fixture_root:
            self.skipTest("Original pinned source fixtures were not provided")
        for relative, expected in EXPECTED_BLOBS.items():
            original = (Path(fixture_root) / relative).read_bytes()
            self.assertEqual(blob_sha(original), expected)
            edited = transform(relative, original.decode())
            if relative.endswith("eval_embedding.py"):
                self.assertNotIn("torch.bfloat16", edited)
                self.assertNotIn("'flash_attention_2'", edited)
                self.assertIn("reps.cpu().float()", edited)
            if relative.endswith("qwen3_vl_embedding.py"):
                self.assertIn("no NULL substitution allowed", edited)
        with self.assertRaises(ValueError):
            transform("eval_embedding.py", "wrong upstream source")

    def test_native_runner_success_and_deadline(self):
        skill = os.environ.get("RESEARCH_AUTOPILOT_ROOT")
        if not skill:
            self.skipTest("Installed native runner not supplied")
        sys.path.insert(0, str(Path(skill) / "scripts"))
        import run_experiments as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "lifecycle.py"
            script.write_text('import sys,time,pathlib\n'
                              'if sys.argv[1]=="delay": time.sleep(10)\n'
                              'pathlib.Path("receipt.txt").write_text("engineering completion")\n')
            reference = {"path": "lifecycle.py", "sha256": digest(script)}
            limits = dict(max_attempts=1, max_development_trials=1, max_confirmation_trials=0,
                          max_retries_per_trial=0, wall_time_seconds=5, attempt_timeout_seconds=0.3)
            for run_id, mode in (("success", "done"), ("deadline", "delay")):
                plan = runner.make_plan(root, run_id=run_id,
                    jobs=[dict(trial_id="lifecycle", command=[sys.executable, str(script), mode],
                               cwd=".", input_refs=[], code_refs=[reference], output_paths=["receipt.txt"],
                               seed=0, group="engineering", arm_role="fixture")],
                    provenance=dict(git_revision="engineering-fixture", model_revision="none-engineering",
                                    data_revision="none-engineering", environment_digest="1" * 64), limits=limits)
                result = runner.run_plan(root, plan, authorizer=lambda scope: scope["plan_digest"] == plan["plan_digest"])
                self.assertFalse(result["gate_advanced"])
                self.assertEqual(len(result["attempts"]), 1)
                if mode == "done":
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(len(result["attempts"][0]["output_refs"]), 1)
                else:
                    self.assertEqual(result["attempts"][0]["status"], "timeout")
                    self.assertLess(result["resources"]["seconds"], 3)
                    self.assertEqual(result["attempts"][0]["output_refs"], [])


class AlgebraChecks(unittest.TestCase):
    def test_projection_kkt_and_curvature_remainder(self):
        import numpy as np
        g, anchor, b = np.array([1.0, -3.0]), np.array([0.0, 2.0]), 0.25
        multiplier = max(0.0, -b - anchor @ g) / (anchor @ anchor)
        v = g + multiplier * anchor
        self.assertGreaterEqual(anchor @ v, -b - 1e-12)
        self.assertLess(abs(multiplier * (anchor @ v + b)), 1e-12)
        self.assertTrue(np.allclose(v - g, multiplier * anchor))
        eta, curvature = 0.1, 3.0
        # Quadratic identity, not a task evaluation.
        exact_change = -eta * (anchor @ v) + curvature * eta ** 2 * (v @ v) / 2
        bound = eta * b + curvature * eta ** 2 * (v @ v) / 2
        self.assertLessEqual(exact_change, bound + 1e-12)

    def test_robust_margin_at_interval_extreme(self):
        estimate_i, estimate_j, ui, uj = 0.8, 0.3, 0.1, 0.15
        margin = estimate_i - estimate_j - ui - uj
        self.assertAlmostEqual((estimate_i - ui) - (estimate_j + uj), margin)
        self.assertGreater(margin, 0)

    def test_generalized_eigen_equation(self):
        import numpy as np
        from scipy.linalg import eigh
        signal, noise = np.diag([4.0, 1.0]), np.diag([2.0, 3.0])
        values, vectors = eigh(signal, noise)
        for value, v in zip(values, vectors.T):
            self.assertTrue(np.allclose(signal @ v, value * noise @ v))
            self.assertAlmostEqual(v @ signal @ v / (v @ noise @ v), value)

    def test_routing_lagrangian_choice(self):
        gain, cost, multiplier = 0.25, 0.5, 0.2
        chosen = int(gain > multiplier * cost)
        objectives = [0.0, -gain + multiplier * cost]
        self.assertLessEqual(objectives[chosen], objectives[1 - chosen])

    def test_negative_softmax_derivative_and_mixture_identity(self):
        import numpy as np
        scores, tau, step = np.array([0.5, 0.2, -0.3]), 0.2, 1e-6
        def loss(x):
            y = x / tau
            return np.log(np.exp(y - y.max()).sum()) + y.max() - y[0]
        plus, minus = scores.copy(), scores.copy()
        plus[1] += step
        minus[1] -= step
        probability = np.exp(scores[1] / tau) / np.exp(scores / tau).sum()
        self.assertAlmostEqual((loss(plus) - loss(minus)) / (2 * step), probability / tau, places=8)
        pi, positive, negative = 0.2, 3.0, 1.5
        mix = pi * positive + (1 - pi) * negative
        self.assertAlmostEqual((mix - pi * positive) / (1 - pi), negative)

    def test_partial_transport_mass_constraint_is_essential(self):
        import numpy as np
        from scipy.optimize import linprog
        costs = np.array([1.0, 2.0, 3.0, 4.0])
        free = linprog(costs, bounds=(0, None), method="highs")
        fixed = linprog(costs, A_eq=np.ones((1, 4)), b_eq=[1.0], bounds=(0, None), method="highs")
        self.assertTrue(free.success and fixed.success)
        self.assertAlmostEqual(free.x.sum(), 0.0)
        self.assertAlmostEqual(fixed.x.sum(), 1.0)

    def test_reward_group_probability_and_control_variable(self):
        import itertools
        p, group = 0.2, 4
        exact = 0.0
        for outcomes in itertools.product([0, 1], repeat=group):
            if len(set(outcomes)) == 1:
                exact += p ** sum(outcomes) * (1 - p) ** (group - sum(outcomes))
        self.assertAlmostEqual(exact, p ** group + (1 - p) ** group)
        # Score-function expectation for a Bernoulli distribution in its probability parameter.
        score_yes, score_no = 1 / p, -1 / (1 - p)
        self.assertAlmostEqual(p * score_yes + (1 - p) * score_no, 0)

    def test_error_propagation_and_smoothing_attenuation(self):
        import numpy as np
        jacobians, errors, initial = [1.1, 0.9, 1.2], [0.01, -0.02, 0.04], 0.1
        delta = initial
        for j, e in zip(jacobians, errors):
            delta = j * delta + e
        expanded = np.prod(jacobians) * initial
        expanded += sum(np.prod(jacobians[k + 1:]) * e for k, e in enumerate(errors))
        self.assertAlmostEqual(delta, expanded)
        omega, regularizer = np.pi / 2, 2.0
        response = 1 / (1 + regularizer * abs(1 - np.exp(-1j * omega)) ** 2)
        self.assertGreater(response, 0)
        self.assertLess(response, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

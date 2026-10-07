"""Algebra/software fixtures only. No benchmark cases, scoring or model quality."""
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.embedding_heads.heads import Training, Blocked, fit, features, pair_data, ridge_delta, unit, rbf

class MathematicalOperators(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/"configs/candidates.json").read_text())["head"]
        self.cfg = {**self.cfg, "protected_group": "engineering-a", "protected_rank": 2,
                    "components": 3, "landmarks": 8, "steps": 10, "irls_steps": 4}
        rng = np.random.default_rng(4)
        q, c = unit(rng.normal(size=(12, 4))), unit(rng.normal(size=(18, 4)))
        positive = np.zeros((12, 18), bool); positive[np.arange(12), np.arange(12)] = True
        allowed = positive.copy(); allowed[np.arange(12), np.arange(12)+6] = True
        teacher = q@np.diag([1.1, 0.7, 1.2, 0.9])@c.T/self.cfg["temperature"]
        self.t = Training(q, c, positive, allowed, teacher, np.array(["engineering-a"]*6+["engineering-b"]*6))

    def test_weighted_dual_equals_regularized_primal(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(5, 11)); r = rng.normal(size=5); weights = rng.uniform(.1, 1, 5)
        actual = ridge_delta(x, r, .3, weights)
        expected = np.linalg.solve(x.T@(weights[:, None]*x)+len(x)*.3*np.eye(11), x.T@(weights*r))
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_fisher_step_matches_dense_hessian(self):
        x, target, _, _ = pair_data(self.t, self.cfg["temperature"])
        base = x@np.eye(4).reshape(-1)
        p0 = 1/(1+np.exp(-np.clip(base, -60, 60))); pt = 1/(1+np.exp(-np.clip(target, -60, 60)))
        weights = np.maximum(p0*(1-p0), self.cfg["fisher_floor"])
        h = x.T@(weights[:, None]*x)/len(x)+self.cfg["ridge"]*np.eye(16)
        gradient = x.T@(p0-pt)/len(x)
        delta = -np.linalg.solve(h, gradient)
        actual = fit("I09", self.t, self.cfg)["matrix"]-np.eye(4)
        np.testing.assert_allclose(actual.reshape(-1), delta, atol=1e-11)

    def test_protected_bilinear_scores_preserved(self):
        head = fit("I02", self.t, self.cfg)
        _, _, vt = np.linalg.svd(self.t.q[:6], full_matrices=False)
        protected = vt[:2]
        np.testing.assert_allclose(protected@(head["matrix"]-np.eye(4)), 0, atol=1e-12)
        np.testing.assert_allclose(features(head, protected, "q")@self.t.c.T, protected@self.t.c.T, atol=1e-12)

    def test_nystrom_feature_product_identity(self):
        head = fit("I16", self.t, self.cfg); x = self.t.q[:3]; y = self.t.c[:4]
        k = rbf(head["anchors"], head["anchors"], head["bandwidth"])+self.cfg["ridge"]*np.eye(len(head["anchors"]))
        expected = rbf(x, head["anchors"], head["bandwidth"])@np.linalg.solve(
            k, rbf(head["anchors"], y, head["bandwidth"]))
        actual = features(head, x, "q")@features(head, y, "c").T
        np.testing.assert_allclose(actual, expected, atol=1e-11)

    def test_diffusion_extension_retains_time_power(self):
        head = fit("I12", self.t, self.cfg); anchors = head["anchors"]
        k = rbf(anchors, anchors, head["bandwidth"]); d = k.sum(1)
        values, vectors = np.linalg.eigh(k/np.sqrt(d[:, None]*d[None, :]))
        chosen = [i for i in np.argsort(values)[::-1][1:] if values[i] > 1e-8][:self.cfg["components"]]
        expected = vectors[:, chosen]/np.sqrt(d[:, None])*values[chosen]**self.cfg["diffusion_time"]
        np.testing.assert_allclose(features(head, anchors, "q"), expected, atol=1e-11)

    def test_cca_regularized_training_covariance_constraints(self):
        head = fit("I10", self.t, self.cfg)
        for side, values in (("q", self.t.q), ("c", self.t.c[np.argmax(self.t.positive, axis=1)])):
            values = values-values.mean(0); cov = values.T@values/len(values)+self.cfg["ridge"]*np.eye(4)
            transform = head[side+"_map"]
            np.testing.assert_allclose(transform.T@cov@transform, np.eye(transform.shape[1]), atol=1e-11)

    def test_two_view_features_are_unit_normalized(self):
        for method in ("I10", "I13", "C_WHITEN"):
            with self.subTest(method=method):
                head = fit(method, self.t, self.cfg)
                for side, values in (("q", self.t.q), ("c", self.t.c)):
                    mapped = features(head, values, side)
                    np.testing.assert_allclose(np.linalg.norm(mapped, axis=1), 1, atol=1e-12)

    def test_i03_returns_psd_distance_metric(self):
        head = fit("I03", self.t, self.cfg)
        self.assertEqual(head["kind"], "distance")
        metric = head["matrix"]
        np.testing.assert_allclose(metric, metric.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(metric).min(), -1e-10)
        self.assertEqual(metric.shape, (self.t.q.shape[1], self.t.q.shape[1]))

    def test_every_unblocked_registered_head_has_finite_features(self):
        cfg = json.loads((ROOT/"configs/candidates.json").read_text())
        for method in cfg["selected"]+cfg["controls"]:
            if method in {"I01", "I05"}:
                continue
            with self.subTest(method=method):
                head = fit(method, self.t, self.cfg)
                if head["kind"] not in {"rbf", "distance"}:
                    self.assertTrue(np.isfinite(features(head, self.t.q, "q")).all())
                    self.assertTrue(np.isfinite(features(head, self.t.c, "c")).all())
                elif head["kind"] == "distance":
                    self.assertTrue(np.isfinite(head["matrix"]).all())

    def test_missing_interval_and_single_positive_do_not_fake_eligibility(self):
        with self.assertRaisesRegex(Blocked, "CALIBRATION"):
            fit("I01", self.t, self.cfg)
        with self.assertRaisesRegex(Blocked, "MULTI_POSITIVE"):
            fit("I05", self.t, self.cfg)

    def test_group_or_supervision_conditions_are_enforced(self):
        with self.assertRaisesRegex(Blocked, "AT_LEAST_TWO"):
            fit("I11", replace(self.t, groups=np.array(["engineering-a"]*12)), self.cfg)
        with self.assertRaisesRegex(Blocked, "NEGATIVE"):
            fit("C_KD", replace(self.t, allowed=self.t.positive.copy()), self.cfg)
        with self.assertRaisesRegex(Blocked, "BOOLEAN"):
            replace(self.t, positive=self.t.positive.astype(float)).validate()

    def test_multi_positive_operator_accepts_only_existing_known_positive_mask(self):
        positive = self.t.positive.copy(); allowed = self.t.allowed.copy()
        positive[0, 1] = True; allowed[0, 1] = True
        head = fit("I05", replace(self.t, positive=positive, allowed=allowed), self.cfg)
        self.assertTrue(np.isfinite(head["matrix"]).all())

    def test_zero_feature_is_not_silently_normalized(self):
        with self.assertRaisesRegex(Blocked, "ZERO"):
            unit(np.zeros((2, 4)))

if __name__ == "__main__":
    unittest.main()

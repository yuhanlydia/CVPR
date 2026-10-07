"""Local semantic checks using the actual released bundle, not handmade evaluation.

This qualifies only named implementation properties. No scientific gate/score here.
Run as a bounded zero-GPU remote harness acceptance job.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.special import expit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import read_json, write_json
from experiments.embedding_heads.heads import fit, pair_data, unit, rbf
from experiments.embedding_heads.audit_controls import (
    registry, pair_objective, positive_objective, fit_audit, audit_features)
from experiments.embedding_heads.bundle import (
    validate_manifest, load_training, load_projection, referenced, project, check_original_source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "upstream", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    arms = registry(config)
    check_original_source(args.upstream)
    manifest = validate_manifest(args.bundle)
    training = load_training(args.bundle, manifest, config["head"]["temperature"])
    projection = load_projection(args.bundle, manifest)
    checks, blocked = [], []
    def check(name, predicate, details):
        checks.append({"property": name, "passed": bool(predicate), "details": details})
    objectives = [(mode, pair_objective(training, config["head"], mode))
                  for mode in ("ce", "erm", "dro")]
    if (training.positive.sum(1) > 1).any():
        objectives += [(mode, positive_objective(training, config["head"], mode))
                       for mode in ("mp_inside", "mp_outside", "single_clean")]
    else:
        blocked += [{"arm": name, "reason": "RELEASED_REAL_MULTI_POSITIVE_ROWS_REQUIRED"}
                    for name in ("A_MP_IN_LBFGS", "A_MP_OUT_LBFGS", "A_SINGLE_CLEAN_LBFGS")]
    for name, (objective, initial) in objectives:
        _, gradient = objective(initial)
        coordinates = np.unique(np.linspace(0, len(initial) - 1, 7, dtype=int))
        errors = []
        epsilon = 1e-5
        for coordinate in coordinates:
            direction = np.zeros(len(initial)); direction[coordinate] = epsilon
            finite_difference = (objective(initial + direction)[0] - objective(initial - direction)[0]) / (2 * epsilon)
            errors.append(float(abs(finite_difference - gradient[coordinate])))
        check("analytic_gradient_" + name, max(errors) < 1e-6,
              {"absolute_errors": errors, "actual_training_rows": len(training.q), "epsilon": epsilon})
    x, teacher, _, _ = pair_data(training, config["head"]["temperature"])
    initial = np.eye(training.q.shape[1]).reshape(-1)
    base = x @ initial
    curvature = np.maximum(expit(base) * (1 - expit(base)), config["head"]["fisher_floor"])
    hessian = x.T @ (curvature[:, None] * x) / len(x) + config["head"]["ridge"] * np.eye(x.shape[1])
    gradient = x.T @ (expit(base) - expit(teacher)) / len(x)
    newton = fit("I09", training, config["head"])
    direction = newton["matrix"].reshape(-1) - initial
    residual = float(np.linalg.norm(hessian @ direction + gradient))
    check("sample_space_newton_dense_equation", residual < 1e-8,
          {"residual_l2": residual, "hessian_dimension": x.shape[1], "training_rows": len(x)})
    damped = fit_audit(arms["A_I09_ARMIJO"], training, config)
    diagnostic = damped["diagnostics"]
    check("training_only_armijo_bound",
          diagnostic["regularized_objective_after"] <= diagnostic["regularized_objective_before"] +
          config["armijo"]["c1"] * diagnostic["step"] * diagnostic["direction_slope"],
          diagnostic)
    metric = fit("I03", training, config["head"])["matrix"]
    for task in manifest["evaluation"]:
        with np.load(referenced(Path(args.bundle).parent, task["features_ref"]), allow_pickle=False) as data:
            fullq, fullc = data["q"].copy(), data["c"].copy()
        q, c = project(fullq, projection), project(fullc, projection)
        # Native feature rows for semantics only, never reported as a benchmark score.
        qrows, crows = q[:min(7, len(q))], c[:min(97, len(c))]
        direct = np.array([-np.einsum("nd,de,ne->n", crows - row, metric, crows - row) for row in qrows])
        expanded = (2 * qrows @ metric @ crows.T -
                    np.einsum("nd,de,ne->n", crows, metric, crows)[None, :] -
                    np.einsum("nd,de,ne->n", qrows, metric, qrows)[:, None])
        check("distance_expansion_" + task["task"], np.allclose(direct, expanded, atol=1e-10, rtol=1e-10),
              {"max_error": float(np.max(np.abs(direct - expanded)))})
        exact_rbf = rbf(qrows, crows, config["head"]["bandwidth"])
        monotone = np.exp((qrows @ crows.T - 1) / config["head"]["bandwidth"] ** 2)
        check("unit_dot_rbf_identity_" + task["task"], np.allclose(exact_rbf, monotone, atol=1e-12, rtol=1e-12),
              {"max_error": float(np.max(np.abs(exact_rbf - monotone)))})
        for dimension in (64, 128, 256, 512, 1024, 2048):
            head = fit_audit(arms["A_MRL" + str(dimension)], training, config)
            actual = audit_features(head, projection, fullq, "q")
            expected = unit(unit(fullq)[:, :dimension])
            check("native_prefix_algebra_" + task["task"] + "_" + str(dimension),
                  np.allclose(actual, expected, atol=1e-12, rtol=1e-12),
                  {"dimension": dimension, "max_error": float(np.max(np.abs(actual - expected)))})
        for method in ("I10", "I13", "C_WHITEN"):
            head = fit(method, training, config["head"])
            actual = audit_features(head, projection, fullc, "c")
            check("unit_two_view_" + task["task"] + "_" + method,
                  np.allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-10),
                  {"norm_min": float(np.linalg.norm(actual, axis=1).min()),
                   "norm_max": float(np.linalg.norm(actual, axis=1).max())})
    if training.radius is None:
        blocked.append({"arm": "I01", "reason": "INDEPENDENT_INTERVAL_CALIBRATION_NOT_AVAILABLE"})
    report = {"schema": "cvpr.audit-local-semantics.v1",
              "status": "PROPERTY_CHECKS_PASS" if all(c["passed"] for c in checks) else "PROPERTY_CHECKS_FAIL",
              "checks": checks, "blocked": blocked, "scientific_verdict": "NONE", "gate_advanced": False,
              "full_native_scoring_qualification": "SEPARATE_REQUIRED",
              "scope": "derivatives/linear algebra/feature semantics on original native inputs"}
    write_json(args.out, report)
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

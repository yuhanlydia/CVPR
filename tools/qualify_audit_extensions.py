"""Local semantic acceptance on real native training inputs; not benchmark scoring."""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import write_json
from experiments.embedding_heads.audit_inventory import load_inventory
from experiments.embedding_heads.audit_controls import pair_objective
from experiments.embedding_heads.audit_extensions import (
    training_identity, selected_rows, subset_training, extension_objective,
    whitening_head, protected_head, preflight_extension, pair_selection)
from experiments.embedding_heads.bundle import (
    validate_manifest, load_training, referenced, jsonl, check_original_source, sha256)
from experiments.embedding_heads.heads import Blocked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "extension-config", "upstream", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    config, arms = load_inventory(args.config, args.extension_config)
    check_original_source(args.upstream)
    manifest = validate_manifest(args.bundle)
    if (manifest["train"]["revision"] != config["training_revision"] or
            manifest["train"]["tasks"] != config["training_tasks"] or
            [t["task"] for t in manifest["evaluation"]] != config["evaluation_tasks"] or
            manifest["train"]["projection_rank"] != config["projection_rank"]):
        raise ValueError("Native train/three-task/projection identity changed")
    training = load_training(args.bundle, manifest, config["head"]["temperature"])
    rows = jsonl(referenced(Path(args.bundle).parent, manifest["train"]["rows_ref"]))
    candidates = manifest["train"]["candidate_ids"]
    training_identity(training, rows, candidates)
    checks, blocked = [], []
    def check(name, passed, detail):
        checks.append({"property": name, "passed": bool(passed), "detail": detail})
    for arm in config["_extension"]["arms"]:
        try:
            preflight_extension(arm, training, rows, candidates)
        except Blocked as error:
            blocked.append({"arm": arm["id"], "reason": str(error)})
    blocked_ids = {a["arm"] for a in blocked}
    pair_kinds = {"query_balanced_all_pairs", "teacher_hard_negative",
                  "uniform_native_negative", "teacher_label_mixture"}
    for arm in config["_extension"]["arms"]:
        if arm["id"] in blocked_ids or arm["kind"] not in pair_kinds:
            continue
        objective, initial, diagnostic = extension_objective(arm, training, config["head"], rows, candidates)
        # Deterministic non-identity probe parameters; input rows are real, never synthetic eval.
        probe = initial + .01 * np.sin(np.arange(len(initial)) + 1)
        value, gradient = objective(probe)
        errors = []
        coordinates = np.unique(np.linspace(0, len(initial)-1, 7, dtype=int))
        epsilon = 1e-5
        for j in coordinates:
            direction = np.zeros(len(initial)); direction[j] = epsilon
            estimate = (objective(probe + direction)[0] - objective(probe - direction)[0]) / (2*epsilon)
            errors.append(float(abs(estimate-gradient[j])))
        check("gradient_" + arm["id"], np.isfinite(value) and max(errors) < 1e-6,
              {"absolute_errors": errors, "diagnostics": diagnostic, "epsilon": epsilon})
    if not (training.allowed & ~training.positive).any(1).all():
        blocked.append({"property": "mix_endpoints", "reason": "NATIVE_NEGATIVE_REQUIRED"})
    else:
        for alpha, mode in ((0.0, "erm"), (1.0, "ce")):
            objective, initial, _ = extension_objective(
                {"kind": "teacher_label_mixture", "alpha": alpha}, training,
                config["head"], rows, candidates)
            parent, _ = pair_objective(training, config["head"], mode)
            value, gradient = objective(initial)
            pv, pg = parent(initial)
            check("mix_endpoint_" + str(alpha),
                  np.isclose(value, pv, atol=1e-12) and np.allclose(gradient, pg, atol=1e-12),
                  {"objective_error": float(abs(value-pv)), "gradient_max_error": float(np.max(abs(gradient-pg)))})
        for arm in config["_extension"]["arms"]:
            if arm["kind"] != "uniform_native_negative":
                continue
            p, neg = pair_selection(arm, training, rows, candidates)
            reverse = np.arange(len(candidates))[::-1]
            from experiments.embedding_heads.heads import Training
            permuted = Training(training.q, training.c[reverse], training.positive[:, reverse],
                                training.allowed[:, reverse], training.teacher[:, reverse],
                                training.groups, None if training.radius is None else training.radius[:, reverse])
            _, other = pair_selection(arm, permuted, rows, [candidates[j] for j in reverse])
            check("random_negative_id_order_" + arm["id"],
                  [candidates[j] for j in neg] == [candidates[reverse[j]] for j in other],
                  {"scope": "negative IDs only; first-positive parent convention intentionally unchanged"})
    for arm in config["_extension"]["arms"]:
        if arm["id"] in blocked_ids:
            continue
        if arm["kind"] in {"source_restricted_head", "nested_head_subset"}:
            chosen = selected_rows(arm, training, rows)
            sub, sr, sc, diagnostic = subset_training(training, rows, candidates, chosen)
            columns = [candidates.index(c) for c in sc]
            check("native_subset_remap_" + arm["id"],
                  np.array_equal(sub.q, training.q[chosen]) and
                  np.array_equal(sub.c, training.c[columns]) and
                  np.array_equal(sub.teacher, training.teacher[np.ix_(chosen, columns)]),
                  diagnostic)
        elif arm["kind"] in {"balanced_query_candidate_second_moment", "positive_weighted_second_moment"}:
            head = whitening_head(arm, training, config["head"])
            mean, transform = head["q_mean"], head["q_map"]
            q, c = training.q-mean, training.c-mean
            mass = (np.full(len(c), 1/len(c)) if arm["kind"] == "balanced_query_candidate_second_moment"
                    else (training.positive/training.positive.sum(1)[:, None]).mean(0))
            # Independent weighted-stack construction of the anchored second moment.
            values = np.vstack((q, c))
            weights = np.concatenate((np.full(len(q), .5/len(q)), .5*mass))
            moment = values.T @ (weights[:, None]*values)
            d, alpha, ridge = q.shape[1], config["head"]["shrinkage"], config["head"]["ridge"]
            covariance = (1-alpha)*moment + alpha*np.trace(moment)/d*np.eye(d) + ridge*np.eye(d)
            error = float(np.linalg.norm(transform @ covariance @ transform - np.eye(d)))
            parent_mean = (training.q.sum(0)+training.c.sum(0))/(len(q)+len(c))
            check("weighted_whitening_" + arm["id"],
                  error < 1e-8 and np.allclose(mean, parent_mean, atol=1e-12),
                  {"whitening_residual": error, **head["diagnostics"]})
        elif arm["kind"] in {"source_svd_protection", "random_orthogonal_protection"}:
            head = protected_head(arm, training, config["head"])
            delta = head["matrix"]-np.eye(training.q.shape[1])
            r = head["protected_basis"]
            error = float(np.linalg.norm(r.T @ delta))
            check("protected_subspace_" + arm["id"],
                  error < 1e-8 and head["diagnostics"]["orthogonality_residual"] < 1e-10,
                  head["diagnostics"])
    for parent in ("I09", "I13", "A_CE_LBFGS"):
        previous = set()
        for count in (64, 128, 256):
            arm = arms["B_N_" + parent + "_" + str(count)]
            if arm["id"] in blocked_ids:
                continue
            current = set(selected_rows(arm, training, rows))
            check("nested_selection_" + arm["id"], previous <= current,
                  {"previous_rows": len(previous), "rows": len(current)})
            previous = current
    success = bool(checks) and all(c["passed"] for c in checks)
    report = {"schema": "cvpr.extension-local-semantics.v1",
              "status": "PROPERTY_CHECKS_PASS" if success else "PROPERTY_CHECKS_FAIL",
              "checks": checks, "blocked": blocked,
              "config_sha256": sha256(args.config),
              "extension_config_sha256": sha256(args.extension_config),
              "bundle_manifest_sha256": sha256(args.bundle),
              "scientific_verdict": "NONE", "gate_advanced": False,
              "scope": "native training identity/gradients/remap/weights/protection only",
              "native_scoring_qualification": "SEPARATE_REQUIRED",
              "blocked_conditions_do_not_pass": True}
    write_json(args.out, report)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

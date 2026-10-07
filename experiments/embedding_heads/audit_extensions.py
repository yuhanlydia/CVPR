"""Existing-method extension controls. Generated source; Local acceptance pending.

Only native training inputs enter fit; no scorer, queue, verdict or test selection.
"""
import hashlib
import json
import numpy as np
from scipy.special import expit
from .heads import Blocked, Training, pair_data, linear, ridge_delta, invsqrt
from .audit_controls import fit_audit, optimize


def identity_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def training_identity(training, rows, candidate_ids):
    training.validate()
    if len(rows) != len(training.q) or len(candidate_ids) != len(training.c):
        raise Blocked("NATIVE_TRAINING_IDENTITY_DIMENSIONS")
    if len(set(candidate_ids)) != len(candidate_ids) or any(not isinstance(c, str) for c in candidate_ids):
        raise Blocked("NATIVE_CANDIDATE_IDS_INVALID")
    keys = [(r["group"], r["query_id"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise Blocked("DUPLICATE_NATIVE_QUERY_GROUP")
    index = {c: j for j, c in enumerate(candidate_ids)}
    for i, row in enumerate(rows):
        if (row["split"] != "original" or row["group"] != training.groups[i]
                or not isinstance(row["query_id"], str)):
            raise Blocked("NATIVE_QUERY_IDENTITY_CHANGED")
        p, n = row["positive_candidate_ids"], row["negative_candidate_ids"]
        if not p or set(p) & set(n) or len(p) != len(set(p)) or len(n) != len(set(n)):
            raise Blocked("INVALID_NATIVE_POSITIVE_NEGATIVE_IDS")
        if not set(p + n) <= set(index):
            raise Blocked("NATIVE_LABEL_ID_NOT_IN_CACHE")
        if (set(np.flatnonzero(training.positive[i])) != {index[c] for c in p}
                or set(np.flatnonzero(training.allowed[i])) != {index[c] for c in p + n}):
            raise Blocked("NATIVE_LABEL_MASK_IDENTITY_CHANGED")
    return keys


def selected_rows(arm, training, rows):
    if arm["kind"] == "source_restricted_head":
        chosen = np.flatnonzero(training.groups == arm["group"]).tolist()
        if len(chosen) < 2:
            raise Blocked("SOURCE_GROUP_HAS_INSUFFICIENT_NATIVE_QUERIES")
        return chosen
    count, seed = arm["unique_queries_per_group"], arm["seed"]
    chosen = []
    for group in ("ScienceQA", "A-OKVQA"):
        indices = np.flatnonzero(training.groups == group).tolist()
        if len(indices) < count:
            raise Blocked("INSUFFICIENT_NATIVE_UNIQUE_QUERIES:" + group)
        indices.sort(key=lambda i: (identity_digest([seed, group, rows[i]["query_id"]]),
                                    rows[i]["query_id"]))
        chosen.extend(indices[:count])
    # Selection by hash, original order for deterministic first-pair parent behavior.
    return sorted(chosen)


def subset_training(training, rows, candidate_ids, chosen):
    chosen = np.asarray(chosen, dtype=int)
    columns = np.flatnonzero(training.allowed[chosen].any(axis=0))
    if len(chosen) < 2 or not len(columns):
        raise Blocked("EMPTY_NATIVE_TRAINING_SUBSET")
    ix = np.ix_(chosen, columns)
    subset = Training(training.q[chosen], training.c[columns],
                      training.positive[ix], training.allowed[ix], training.teacher[ix],
                      training.groups[chosen],
                      None if training.radius is None else training.radius[ix]).validate()
    sr, sc = [rows[i] for i in chosen], [candidate_ids[j] for j in columns]
    training_identity(subset, sr, sc)
    metadata = {"selected_query_group_ids": [[r["group"], r["query_id"]] for r in sr],
                "selected_candidate_ids": sc}
    return subset, sr, sc, {
        "training_selection_sha256": identity_digest(metadata),
        "selected_training_rows": len(chosen), "selected_training_candidates": len(columns),
        "selected_group_counts": {str(g): int((subset.groups == g).sum())
                                  for g in np.unique(subset.groups)},
        "pca_fitted_on_full_parent_training": True,
        "scope": "head supervision only; not end-to-end source or sample isolation"}


def pair_selection(arm, training, rows, candidate_ids):
    positives = np.argmax(training.positive, axis=1)
    native_negatives = training.allowed & ~training.positive
    if not native_negatives.any(1).all():
        raise Blocked("NATIVE_NEGATIVE_REQUIRED")
    selected = np.argmax(native_negatives, axis=1)
    for i, row in enumerate(rows):
        choices = np.flatnonzero(training.allowed[i] & ~training.positive[i])
        if arm["kind"] == "teacher_hard_negative":
            selected[i] = choices[np.argmax(training.teacher[i, choices])]
        elif arm["kind"] == "uniform_native_negative":
            ordered = sorted(choices.tolist(), key=lambda j: candidate_ids[j])
            digest = identity_digest([arm["seed"], row["group"], row["query_id"]])
            rng = np.random.default_rng(int(digest[:16], 16))
            selected[i] = ordered[int(rng.integers(len(ordered)))]
    return positives, selected


def extension_objective(arm, training, config, rows, candidate_ids):
    """Chunked exact query-balanced all-pairs or one pair per native query."""
    temp, ridge, chunk = config["temperature"], config["ridge"], arm.get("pair_chunk_size", 256)
    d, n = training.q.shape[1], len(training.q)
    initial = np.eye(d).reshape(-1)
    all_pairs = arm["kind"] == "query_balanced_all_pairs"
    positive_sets = [np.flatnonzero(p) for p in training.positive]
    negative_sets = [np.flatnonzero(a & ~p) for a, p in zip(training.allowed, training.positive)]
    if any(not len(x) for x in negative_sets):
        raise Blocked("NATIVE_NEGATIVE_REQUIRED")
    if not all_pairs:
        p, neg = pair_selection(arm, training, rows, candidate_ids)
        x = np.einsum("ni,nj->nij", training.q, training.c[p] - training.c[neg]).reshape(n, -1) / temp
        logits = training.teacher[np.arange(n), p] - training.teacher[np.arange(n), neg]
        alpha = arm.get("alpha", 1.0)
        target = (1 - alpha) + alpha * expit(logits)
        selection = [[rows[i]["group"], rows[i]["query_id"], candidate_ids[p[i]], candidate_ids[neg[i]]]
                     for i in range(n)]
        diagnostic = {"teacher_by_group": {
                          str(g): {"rows": int((training.groups == g).sum()),
                                   "negative_margin_fraction": float((logits[training.groups == g] < 0).mean()),
                                   "target_mean": float(target[training.groups == g].mean())}
                          for g in np.unique(training.groups)},
                      "pair_selection_sha256": identity_digest(selection),
                      "pairs": n, "teacher_negative_margin_fraction": float((logits < 0).mean()),
                      "target_mean": float(target.mean()), "teacher_mix_alpha": alpha,
                      "negative_selector": arm["kind"]}
        def objective(w):
            z = x @ w
            delta = w - initial
            return (float(np.mean(np.logaddexp(0, z) - target * z) + ridge * (delta @ delta) / 2),
                    x.T @ (expit(z) - target) / n + ridge * delta)
        return objective, initial, diagnostic
    # No giant pairs x d² allocation: enumerate each real query's Cartesian product in chunks.
    pair_count = sum(len(p) * len(neg) for p, neg in zip(positive_sets, negative_sets))
    redundant = all(len(p) == len(neg) == 1 for p, neg in zip(positive_sets, negative_sets))
    def blocks(with_row=False):
        for i, (p, neg) in enumerate(zip(positive_sets, negative_sets)):
            count = len(p) * len(neg)
            for start in range(0, count, chunk):
                flat = np.arange(start, min(start + chunk, count))
                pp, nn = p[flat // len(neg)], neg[flat % len(neg)]
                diff = training.c[pp] - training.c[nn]
                x = (training.q[i][None, :, None] * diff[:, None, :]).reshape(len(flat), -1) / temp
                logits = training.teacher[i, pp] - training.teacher[i, nn]
                if with_row:
                    yield i, logits, 1 / (n * count)
                else:
                    yield x, logits, 1 / (n * count)
    def objective(w):
        value, gradient = 0.0, np.zeros_like(w)
        for x, logits, mass in blocks():
            z, target = x @ w, expit(logits)
            value += mass * np.sum(np.logaddexp(0, z) - target * z)
            gradient += mass * (x.T @ (expit(z) - target))
        delta = w - initial
        return float(value + ridge * (delta @ delta) / 2), gradient + ridge * delta
    conflict, mean_target = 0.0, 0.0
    group_stats = {str(g): {"rows": int((training.groups == g).sum()), "negative_margin_fraction": 0.0,
                           "target_mean": 0.0} for g in np.unique(training.groups)}
    for i, logits, mass in blocks(with_row=True):
        conflict += mass * np.sum(logits < 0)
        mean_target += mass * np.sum(expit(logits))
        stat = group_stats[str(training.groups[i])]
        stat["negative_margin_fraction"] += float(mass * n / stat["rows"] * np.sum(logits < 0))
        stat["target_mean"] += float(mass * n / stat["rows"] * np.sum(expit(logits)))
    diagnostic = {"teacher_by_group": group_stats, "pairs": pair_count, "pair_chunk_size": chunk, "query_equal_weight": True,
                  "teacher_negative_margin_fraction": float(conflict), "target_mean": float(mean_target),
                  "redundant_with_first_pair": redundant,
                  "pair_selection_sha256": identity_digest({
                      "rule": "all_native_positive_x_negative_query_balanced",
                      "query_ids": [[r["group"], r["query_id"]] for r in rows],
                      "candidate_ids": candidate_ids,
                      "positive_indices": [p.tolist() for p in positive_sets],
                      "negative_indices": [p.tolist() for p in negative_sets]})}
    return objective, initial, diagnostic


def whitening_head(arm, training, config):
    d, n, m = training.q.shape[1], len(training.q), len(training.c)
    mean = np.vstack((training.q, training.c)).mean(0)  # same parent centering rule
    q, c = training.q - mean, training.c - mean
    if arm["kind"] == "balanced_query_candidate_second_moment":
        mass = np.full(m, 1 / m)
    else:
        mass = (training.positive / training.positive.sum(1)[:, None]).mean(0)
    if not np.isclose(mass.sum(), 1):
        raise Blocked("NATIVE_WHITENING_MASS_NOT_NORMALIZED")
    moment = .5 * (q.T @ q / n) + .5 * (c.T @ (mass[:, None] * c))
    alpha, ridge = config["shrinkage"], config["ridge"]
    covariance = (1-alpha) * moment + alpha * np.trace(moment) / d * np.eye(d) + ridge * np.eye(d)
    transform = invsqrt(covariance, ridge)
    values = np.linalg.eigvalsh(covariance)
    return {"kind": "two_view", "q_mean": mean, "c_mean": mean,
            "q_map": transform, "c_map": transform, "diagnostics": {
                "statistic": "anchored_second_moment_fixed_parent_mean",
                "query_mass": .5, "candidate_mass": .5,
                "parent_query_mass": n / (n + m), "parent_candidate_mass": m / (n + m),
                "candidate_weight_sha256": identity_digest(mass.tolist()),
                "uses_training_positive_labels": arm["kind"] == "positive_weighted_second_moment",
                "covariance_min_eigenvalue": float(values.min()),
                "covariance_max_eigenvalue": float(values.max()),
                "covariance_condition": float(values.max() / values.min()),
                "shrinkage": alpha}}


def protected_head(arm, training, config):
    d = training.q.shape[1]
    x, target, positive, negative = pair_data(training, config["temperature"])
    selected = training.q[training.groups == config["protected_group"]]
    k = arm["protected_rank"]
    if len(selected) < 2 or not 0 < k < d:
        raise Blocked("INVALID_NATIVE_PROTECTED_SUBSPACE")
    if arm["kind"] == "random_orthogonal_protection":
        r, _ = np.linalg.qr(np.random.default_rng(arm["seed"]).standard_normal((d, k)), mode="reduced")
    else:
        _, singular, vt = np.linalg.svd(selected, full_matrices=False)
        if int((singular > 1e-10).sum()) < k:
            raise Blocked("REQUESTED_PROTECTED_RANK_NOT_AVAILABLE")
        r = vt[:k].T
    identity = np.eye(d)
    projection = identity - r @ r.T
    diff = training.c[positive] - training.c[negative]
    xp = np.einsum("ni,nj->nij", training.q @ projection, diff).reshape(len(x), -1) / config["temperature"]
    delta = projection @ ridge_delta(xp, target - x @ identity.reshape(-1),
                                     config["ridge"]).reshape(d, d)
    energies = np.sum((selected @ r)**2, axis=1) / np.sum(selected**2, axis=1)
    head = linear(identity + delta, protected_rank=k, protected_group=config["protected_group"],
                  constraint_residual=float(np.linalg.norm(r.T @ delta)),
                  orthogonality_residual=float(np.linalg.norm(r.T @ r - np.eye(k))),
                  protected_query_energy_mean=float(energies.mean()),
                  protected_query_energy_min=float(energies.min()),
                  protected_query_energy_max=float(energies.max()),
                  subspace_source=arm["kind"], random_seed=arm.get("seed"),
                  numpy_version=np.__version__, whole_group_guarantee=False)
    head["protected_basis"] = r  # persisted/digested; features uses matrix only
    return head


def preflight_extension(arm, training, rows, candidate_ids):
    training_identity(training, rows, candidate_ids)
    if arm["kind"] in {"source_restricted_head", "nested_head_subset"}:
        selected_rows(arm, training, rows)
    if arm["kind"] in {"source_svd_protection", "random_orthogonal_protection"}:
        if not 0 < arm["protected_rank"] < training.q.shape[1]:
            raise Blocked("PROTECTED_RANK_OUT_OF_RANGE")
        selected = training.q[training.groups == "ScienceQA"]
        if len(selected) < 2:
            raise Blocked("DECLARED_PROTECTED_TRAINING_GROUP_MISSING")
        if arm["kind"] == "source_svd_protection" and int((np.linalg.svd(selected, compute_uv=False) > 1e-10).sum()) < arm["protected_rank"]:
            raise Blocked("REQUESTED_PROTECTED_RANK_NOT_AVAILABLE")
    if arm["kind"] in {"query_balanced_all_pairs", "teacher_hard_negative",
                       "uniform_native_negative", "teacher_label_mixture"}:
        if not (training.allowed & ~training.positive).any(1).all():
            raise Blocked("NATIVE_NEGATIVE_REQUIRED")


def fit_extension(arm, training, config, rows, candidate_ids):
    preflight_extension(arm, training, rows, candidate_ids)
    original_counts = {"parent_rows": len(training.q), "parent_candidates": len(training.c)}
    selected = {}
    if arm["kind"] in {"source_restricted_head", "nested_head_subset"}:
        chosen = selected_rows(arm, training, rows)
        training, rows, candidate_ids, selected = subset_training(training, rows, candidate_ids, chosen)
        parent = next(a for a in config["arms"] if a["id"] == arm["parent"])
        head = fit_audit(parent, training, config)
    elif arm["kind"] in {"query_balanced_all_pairs", "teacher_hard_negative",
                         "uniform_native_negative", "teacher_label_mixture"}:
        objective, initial, diagnostic = extension_objective(arm, training, config["head"], rows, candidate_ids)
        vector, opt = optimize(objective, initial, config["optimizer"], strong_convexity=config["head"]["ridge"])
        head = linear(vector.reshape(training.q.shape[1], training.q.shape[1]), **diagnostic, **opt)
    elif arm["kind"] in {"balanced_query_candidate_second_moment", "positive_weighted_second_moment"}:
        head = whitening_head(arm, training, config["head"])
    elif arm["kind"] in {"source_svd_protection", "random_orthogonal_protection"}:
        head = protected_head(arm, training, config["head"])
    else:
        raise ValueError("Unknown extension construction")
    head["diagnostics"].update(original_counts, **selected, constructed_arm=arm["id"],
                              training_rows=len(training.q), training_candidates=len(training.c),
                              training_dim=training.q.shape[1], fit_uses_test_labels=False,
                              numpy_version=np.__version__,
                              training_identity_sha256=identity_digest({
                                  "queries": [[r["group"], r["query_id"]] for r in rows],
                                  "candidate_ids": candidate_ids}))
    return head

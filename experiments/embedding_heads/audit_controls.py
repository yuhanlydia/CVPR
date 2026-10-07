"""Known controls for the existing r003 methods; Web-generated, unexecuted.

No benchmark, scientific verdict, scheduler or gate is implemented here.
The head fit only receives native training features and released supervision.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from .heads import Blocked, fit, features, linear, pair_data, unit
from .bundle import project


class EvaluationLimit(RuntimeError):
    pass


def registry(config):
    arms = config["arms"]
    ids = [a["id"] for a in arms]
    if len(ids) != len(set(ids)) or len(ids) != 43:
        raise ValueError("The reviewed 43-record audit inventory changed")
    if config["schema"] != "cvpr.existing-method-audit.v1":
        raise ValueError("Unreviewed audit config")
    if config["evaluation_tasks"] != ["ScienceQA", "ChartQA", "MSCOCO_i2t"]:
        raise ValueError("Full native three-task coverage required")
    if config["head"]["temperature"] <= 0 or config["head"]["ridge"] <= 0:
        raise ValueError("Positive temperature and ridge required")
    return {a["id"]: a for a in arms}


def pair_objective(training, config, mode):
    x, teacher, _, _ = pair_data(training, config["temperature"])
    w0 = np.eye(training.q.shape[1]).reshape(-1)
    target = expit(teacher) if mode == "ce" else np.ones(len(x))
    ridge = config["ridge"]
    groups = np.unique(training.groups)
    if mode == "dro" and len(groups) < 2:
        raise Blocked("NATIVE_TRAINING_GROUPS_REQUIRED")
    def objective(w):
        margins = x @ w
        losses = np.logaddexp(0, margins) - target * margins
        residual = expit(margins) - target
        if mode == "dro":
            group_losses = np.array([losses[training.groups == g].mean() for g in groups])
            tau = config["dro_temperature"]
            masses = np.exp(group_losses / tau - logsumexp(group_losses / tau))
            row_mass = np.zeros(len(x))
            for g, mass in zip(groups, masses):
                rows = training.groups == g
                row_mass[rows] = mass / rows.sum()
            value = tau * logsumexp(group_losses / tau)
        else:
            row_mass = np.full(len(x), 1 / len(x))
            value = losses.mean()
        delta = w - w0
        value += ridge * (delta @ delta) / 2
        gradient = x.T @ (row_mass * residual) + ridge * delta
        return float(value), gradient
    return objective, w0


def positive_objective(training, config, mode):
    if not (training.positive.sum(1) > 1).any():
        raise Blocked("RELEASED_REAL_MULTI_POSITIVE_ROWS_REQUIRED")
    positive = training.positive.copy()
    allowed = training.allowed.copy()
    if mode == "single_clean":
        first = np.argmax(positive, axis=1)
        keep = np.zeros_like(positive)
        keep[np.arange(len(keep)), first] = True
        allowed &= ~(positive & ~keep)
        positive = keep
    counts = positive.sum(1)
    target = positive / counts[:, None]
    d = training.q.shape[1]
    w0 = np.eye(d).reshape(-1)
    ridge, temp = config["ridge"], config["temperature"]
    def objective(w):
        scores = training.q @ w.reshape(d, d) @ training.c.T / temp
        masked = np.where(allowed, scores, -np.inf)
        normalizer = logsumexp(masked, axis=1, keepdims=True)
        mass = np.exp(masked - normalizer)
        if mode == "mp_inside":
            pmasked = np.where(positive, scores, -np.inf)
            posnorm = logsumexp(pmasked, axis=1, keepdims=True)
            posmass = np.exp(pmasked - posnorm)
            losses = normalizer[:, 0] - posnorm[:, 0] + np.log(counts)
        else:
            posmass = target
            losses = normalizer[:, 0] - (target * scores).sum(1)
        delta = w - w0
        value = losses.mean() + ridge * (delta @ delta) / 2
        gradient = (training.q.T @ (mass - posmass) @ training.c /
                    (len(training.q) * temp)).reshape(-1) + ridge * delta
        return float(value), gradient
    return objective, w0


def optimize(objective, initial, settings, *, strong_convexity=None):
    """Strict analytic-gradient call ceiling; keep capped results explicitly unqualified."""
    calls = 0
    best = None
    history = []
    last = None
    def measured(w):
        nonlocal calls, best, last
        if calls >= settings["maxfun"]:
            raise EvaluationLimit("analytic objective call ceiling reached")
        calls += 1
        value, gradient = objective(w)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise Blocked("NONFINITE_NATIVE_TRAINING_OBJECTIVE")
        if best is None or value < best[0]:
            best = (value, w.copy(), gradient.copy())
        last = (w.copy(), value, gradient.copy())
        return value, gradient
    before, _ = measured(initial)
    def checkpoint(w):
        if last is not None and np.array_equal(w, last[0]):
            value, gradient = last[1:]
        else:
            value, gradient = measured(w)
        history.append({"objective": value, "gradient_linf": float(np.max(np.abs(gradient)))})
    try:
        result = minimize(measured, initial.copy(), method="L-BFGS-B", jac=True,
                          callback=checkpoint, options=settings)
        vector = result.x
        scipy_success, message, iterations = bool(result.success), str(result.message), int(result.nit)
        # Retain a lower observed objective if the optimizer returns a worse vector.
        if best is not None and float(result.fun) > best[0]:
            vector = best[1]
            scipy_success = False
            message += "; returning lowest observed objective"
    except EvaluationLimit as error:
        vector = best[1]
        scipy_success, message, iterations = False, str(error), len(history)
    after, gradient = objective(vector)
    norm = float(np.max(np.abs(gradient)))
    diagnostic = {"optimizer": "scipy.optimize.minimize/L-BFGS-B",
                  "analytic_optimizer_objective_calls": calls, "maxfun": settings["maxfun"],
                  "final_receipt_objective_calls": 1,
                  "iterations": iterations, "scipy_success": scipy_success,
                  "termination": message, "objective_before": before, "objective_after": after,
                  "gradient_linf": norm, "gradient_stationary": norm <= settings["gtol"],
                  "convergence_qualified": scipy_success and norm <= settings["gtol"],
                  "iteration_history": history, "global_optimum_claim": False}
    if strong_convexity is not None:
        diagnostic["conditional_suboptimality_upper_bound"] = float(
            (gradient @ gradient) / (2 * strong_convexity))
        diagnostic["strong_convexity"] = strong_convexity
    return vector, diagnostic


def fit_audit(arm, training, config):
    training.validate()
    head_config = dict(config["head"])
    overrides = arm.get("parameter_overrides", {})
    if set(overrides) - {"shrinkage"}:
        raise ValueError("Unreviewed result-affecting parameter override")
    head_config.update(overrides)
    kind = arm["kind"]
    if kind == "prefix":
        dimension = arm["dimension"]
        if dimension not in {64, 128, 256, 512, 1024, 2048}:
            raise ValueError("Unsupported native MRL prefix")
        head = {"kind": "prefix", "dimension": dimension, "input_policy": "native_prefix",
                "diagnostics": {"no_head_training": True}}
    elif kind == "pca_rank":
        head = linear(np.eye(arm["dimension"]), no_head_training=True)
        head.update(input_policy="pca_truncated", dimension=arm["dimension"])
    elif kind in {"reuse", "legacy_dot", "raw_two_view", "normalized_map", "distance_map"}:
        head = fit(arm["source_method"], training, head_config)
        if kind == "legacy_dot":
            if head["kind"] != "distance":
                raise ValueError("Expected trained I03 metric")
            head["kind"] = "linear"
            head["diagnostics"]["historical_bilinear_scorer_reconstruction"] = True
        elif kind == "raw_two_view":
            if head["kind"] != "two_view":
                raise ValueError("Expected two-view feature head")
            head["output_policy"] = "raw_two_view"
        elif kind == "normalized_map":
            head["output_policy"] = "normalized_map"
        elif kind == "distance_map":
            head["map_kind"] = head["kind"]
            head["kind"] = "distance"
            head["matrix"] = np.eye(head["weights"].shape[1])
            head["output_policy"] = "map_distance"
    elif kind in {"ce_lbfgs", "erm_lbfgs", "dro_lbfgs", "mp_inside", "mp_outside", "single_clean"}:
        if kind in {"mp_inside", "mp_outside", "single_clean"}:
            objective, initial = positive_objective(training, head_config, kind)
        else:
            mode = {"ce_lbfgs": "ce", "erm_lbfgs": "erm", "dro_lbfgs": "dro"}[kind]
            objective, initial = pair_objective(training, head_config, mode)
        strong = None if kind == "mp_inside" else head_config["ridge"]
        vector, diagnostic = optimize(objective, initial, config["optimizer"], strong_convexity=strong)
        diagnostic["objective_id"] = kind
        diagnostic["nonconvex_local_search"] = kind == "mp_inside"
        head = linear(vector.reshape(training.q.shape[1], training.q.shape[1]), **diagnostic)
    elif kind in {"newton_scaled", "newton_armijo"}:
        head = fit("I09", training, head_config)
        objective, initial = pair_objective(training, head_config, "ce")
        direction = head["matrix"].reshape(-1) - initial
        before, gradient = objective(initial)
        slope = float(gradient @ direction)
        history = []
        if kind == "newton_scaled":
            step = arm["step"]
        else:
            step = None
            if slope > 1e-12:
                raise Blocked("NEWTON_DIRECTION_NOT_DESCENT")
            for k in range(config["armijo"]["max_backtracks"]):
                alpha = 2.0 ** -k
                value, _ = objective(initial + alpha * direction)
                bound = before + config["armijo"]["c1"] * alpha * slope
                history.append({"alpha": alpha, "objective": value, "armijo_bound": bound})
                if value <= bound:
                    step = alpha
                    break
            if step is None:
                raise Blocked("BOUNDED_ARMIJO_NOT_ACCEPTED: " + repr(history))
        vector = initial + step * direction
        after, residual = objective(vector)
        head["matrix"] = vector.reshape(training.q.shape[1], training.q.shape[1])
        x, _, _, _ = pair_data(training, head_config["temperature"])
        diagnostics = head["diagnostics"]
        diagnostics["full_step_max_logit_step"] = diagnostics["max_logit_step"]
        diagnostics["full_step_kd_after"] = diagnostics["kd_after"]
        diagnostics["max_logit_step"] = float(np.max(np.abs(x @ (step * direction))))
        diagnostics["kd_after"] = float(after - head_config["ridge"] *
                                       ((vector - initial) @ (vector - initial)) / 2)
        head["diagnostics"].update(step=step, regularized_objective_before=before,
                                  regularized_objective_after=after, direction_slope=slope,
                                  gradient_linf=float(np.max(np.abs(residual))),
                                  training_only_step_selection=True, backtracking_history=history,
                                  convergence_qualified=False, global_optimum_claim=False)
    else:
        raise ValueError("Unknown reviewed audit construction: " + kind)
    positives = training.positive.sum(1)
    head["diagnostics"].update(training_rows=len(training.q), training_dim=training.q.shape[1],
        real_multi_positive_rows=int((positives > 1).sum()), real_multi_positive_fraction=float((positives > 1).mean()),
        positives_per_group={str(g): {"rows": int((training.groups == g).sum()),
            "multi_positive_rows": int(((training.groups == g) & (positives > 1)).sum()),
            "max_positives": int(positives[training.groups == g].max())} for g in np.unique(training.groups)},
        fit_uses_test_labels=False, constructed_arm=arm["id"])
    if arm.get("source_method") in {"I13", "C_WHITEN"}:
        values = np.vstack((training.q, training.c))
        values -= values.mean(0)
        covariance = values.T @ values / len(values)
        alpha = head_config["shrinkage"] if arm["source_method"] == "I13" else 0.0
        covariance = ((1 - alpha) * covariance + alpha * np.trace(covariance) /
                      covariance.shape[0] * np.eye(covariance.shape[0]) +
                      head_config["ridge"] * np.eye(covariance.shape[0]))
        eigenvalues = np.linalg.eigvalsh(covariance)
        head["diagnostics"].update(covariance_min_eigenvalue=float(eigenvalues.min()),
                                  covariance_max_eigenvalue=float(eigenvalues.max()),
                                  covariance_condition=float(eigenvalues.max() / eigenvalues.min()))
    return head


def audit_features(head, projection, full, side):
    """Called only after fit has returned; no labels enter this transform."""
    policy = head.get("input_policy", "pca")
    if policy == "native_prefix":
        if full.ndim != 2 or full.shape[1] != 2048:
            raise ValueError("Complete 2048-dimensional native cache required")
        return unit(np.asarray(full[:, :head["dimension"]], dtype=np.float64))
    if policy == "pca_truncated":
        mean, basis = projection
        dimension = head["dimension"]
        if basis.shape[1] < dimension:
            raise Blocked("TRAINING_PCA_RANK_TOO_SMALL")
        return unit((np.asarray(full, dtype=np.float64) - mean) @ basis[:, :dimension])
    values = project(full, projection)
    output = head.get("output_policy")
    if output == "raw_two_view":
        return (values - head[side + "_mean"]) @ head[side + "_map"]
    if output == "map_distance":
        mapped_head = dict(head, kind=head["map_kind"])
        return features(mapped_head, values, side)
    if output == "normalized_map":
        return unit(features(head, values, side))
    if head["kind"] in {"rbf", "distance"}:
        return values
    return features(head, values, side)

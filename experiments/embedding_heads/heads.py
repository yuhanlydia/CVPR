"""Mathematical prototypes fitted on training-only frozen features (CPU float64).
No evaluator or scientific gate is implemented. Loops have finite ceilings.
"""
from dataclasses import dataclass
import numpy as np

class Blocked(ValueError):
    """Missing data/math condition, not scientific falsification."""

@dataclass(frozen=True)
class Training:
    q: np.ndarray
    c: np.ndarray
    positive: np.ndarray
    allowed: np.ndarray
    teacher: np.ndarray
    groups: np.ndarray
    radius: np.ndarray | None = None

    def validate(self):
        if self.q.ndim != 2 or self.c.ndim != 2:
            raise Blocked("INVALID_TRAINING_DIMENSIONS")
        n, d = self.q.shape
        shape = (n, len(self.c))
        if n < 2 or d < 2 or self.c.shape[1] != d:
            raise Blocked("INVALID_TRAINING_DIMENSIONS")
        if any(x.shape != shape for x in (self.positive, self.allowed, self.teacher)):
            raise Blocked("TRAINING_SUPERVISION_SHAPE_MISMATCH")
        if self.positive.dtype != bool or self.allowed.dtype != bool:
            raise Blocked("SUPERVISION_MASKS_MUST_BE_BOOLEAN")
        if self.groups.shape != (n,) or not self.positive.any(axis=1).all():
            raise Blocked("MISSING_ORIGINAL_POSITIVES_OR_GROUPS")
        if (self.positive & ~self.allowed).any():
            raise Blocked("POSITIVE_OUTSIDE_ORIGINAL_SUPERVISION")
        if any(not np.isfinite(x).all() for x in (self.q, self.c, self.teacher)):
            raise Blocked("NONFINITE_TRAINING")
        if self.radius is not None:
            if self.radius.shape != shape or not np.isfinite(self.radius).all() or (self.radius < 0).any():
                raise Blocked("INVALID_CALIBRATED_RADIUS")
        return self

def unit(x):
    lengths = np.linalg.norm(x, axis=1, keepdims=True)
    if (lengths < 1e-12).any() or not np.isfinite(x).all():
        raise Blocked("ZERO_OR_NONFINITE_FEATURE")
    return x / lengths

def invsqrt(a, epsilon):
    values, vectors = np.linalg.eigh((a + a.T) / 2)
    if not np.isfinite(values).all() or values.min() < -epsilon:
        raise Blocked("NON_PSD_COVARIANCE")
    return (vectors * np.maximum(values, epsilon)**-0.5) @ vectors.T

def rbf(x, y, bandwidth):
    distances = np.maximum(0, (x*x).sum(1)[:, None] + (y*y).sum(1)[None, :] - 2*x@y.T)
    return np.exp(-distances / (2 * bandwidth**2))

def pair_data(t, temperature):
    positive = np.argmax(t.positive, axis=1)
    negatives = t.allowed & ~t.positive
    if not negatives.any(axis=1).all():
        raise Blocked("ORIGINAL_NEGATIVE_REQUIRED_FOR_EVERY_TRAINING_ROW")
    negative = np.argmax(negatives, axis=1)
    difference = t.c[positive] - t.c[negative]
    x = np.einsum("ni,nj->nij", t.q, difference).reshape(len(t.q), -1) / temperature
    teacher = t.teacher[np.arange(len(t.q)), positive] - t.teacher[np.arange(len(t.q)), negative]
    return x, teacher, positive, negative

def ridge_delta(x, residual, ridge, weights=None):
    """Weighted ridge through an n x n system, never a d^4 Hessian."""
    weights = np.ones(len(x)) if weights is None else weights
    a = x * np.sqrt(weights[:, None])
    b = residual * np.sqrt(weights)
    dual = np.linalg.solve(a @ a.T + len(x) * ridge * np.eye(len(x)), b)
    return a.T @ dual

def linear(matrix, **diagnostics):
    return {"kind": "linear", "matrix": matrix, "diagnostics": diagnostics}

def fit(method, t, config):
    t.validate()
    d = t.q.shape[1]
    ridge, temp = config["ridge"], config["temperature"]
    identity = np.eye(d)
    if method == "C_PROJECTED":
        return linear(identity)
    if method in {"I01", "I02", "I09", "I14", "I11", "C_KD", "C_ERM"}:
        x, target, positive, negative = pair_data(t, temp)
        w0 = identity.reshape(-1)
        base = x @ w0
        if method == "C_KD":
            return linear(identity + ridge_delta(x, target-base, ridge).reshape(d, d))
        if method == "I02":
            group = config["protected_group"]
            selected = t.q[t.groups == group]
            if len(selected) < 2:
                raise Blocked("DECLARED_PROTECTED_TRAINING_GROUP_MISSING")
            _, singular, vt = np.linalg.svd(selected, full_matrices=False)
            k = min(config["protected_rank"], int((singular > 1e-10).sum()), d-1)
            if k < 1:
                raise Blocked("EMPTY_PROTECTED_SUBSPACE")
            r = vt[:k].T
            p = identity-r@r.T
            diff = t.c[positive]-t.c[negative]
            xp = np.einsum("ni,nj->nij", t.q@p, diff).reshape(len(t.q), -1)/temp
            delta = p @ ridge_delta(xp, target-base, ridge).reshape(d, d)
            return linear(identity+delta, protected_rank=k,
                          constraint_residual=float(np.linalg.norm(r.T@delta)), protected_group=group)
        if method == "I09":
            p0 = 1/(1+np.exp(-np.clip(base, -60, 60)))
            pt = 1/(1+np.exp(-np.clip(target, -60, 60)))
            curvature = np.maximum(p0*(1-p0), config["fisher_floor"])
            delta = ridge_delta(x, (pt-p0)/curvature, ridge, curvature)
            after = base+x@delta
            bce = lambda z: float(np.mean(np.logaddexp(0, z)-pt*z))
            return linear(identity+delta.reshape(d, d), kd_before=bce(base), kd_after=bce(after),
                          max_logit_step=float(np.max(np.abs(after-base))), quadratic_steps=1)
        if method == "I14":
            delta = np.zeros(d*d)
            history = []
            for _ in range(config["irls_steps"]):
                residual = base+x@delta-target
                weights = np.minimum(1, config["huber_delta"]/np.maximum(np.abs(residual), 1e-12))
                updated = ridge_delta(x, target-base, ridge, weights)
                history.append(float(np.linalg.norm(updated-delta)))
                delta = updated
            return linear(identity+delta.reshape(d, d), irls_update_norms=history, exact_optimum_claim=False)
        if method == "I01":
            if t.radius is None:
                raise Blocked("INDEPENDENT_INTERVAL_CALIBRATION_NOT_AVAILABLE")
            lower = target-t.radius[np.arange(len(t.q)), positive]-t.radius[np.arange(len(t.q)), negative]
            eligible = lower > 0
            if not eligible.any():
                raise Blocked("NO_CERTIFIED_TEACHER_PAIRS_UNDER_DECLARED_INTERVALS")
            x, lower = x[eligible], lower[eligible]
            w = w0.copy()
            for _ in range(config["steps"]):
                active = (x@w < lower).astype(float)
                w -= config["learning_rate"]*(-x.T@active/len(x)+ridge*(w-w0))
            return linear(w.reshape(d, d), interval_pair_coverage=float(eligible.mean()),
                          guarantee_scope="conditional on declared relevance intervals")
        groups = np.unique(t.groups)
        if method == "I11" and len(groups) < 2:
            raise Blocked("GROUP_DRO_REQUIRES_AT_LEAST_TWO_ORIGINAL_TRAINING_GROUPS")
        w = w0.copy()
        for _ in range(config["steps"]):
            margin = x@w
            losses = np.logaddexp(0, -margin)
            residual = -1/(1+np.exp(np.clip(margin, -60, 60)))
            row_weights = np.full(len(x), 1/len(x))
            if method == "I11":
                group_loss = np.array([losses[t.groups == g].mean() for g in groups])
                mass = np.exp((group_loss-group_loss.max())/config["dro_temperature"]); mass /= mass.sum()
                row_weights = np.zeros(len(x))
                for g, a in zip(groups, mass):
                    select = t.groups == g
                    row_weights[select] = a/select.sum()
            w -= config["learning_rate"]*(x.T@(row_weights*residual)+ridge*(w-w0))
        return linear(w.reshape(d, d), objective="smooth_group_max" if method == "I11" else "pair_erm")
    if method in {"I05", "C_SINGLE"}:
        if method == "I05" and t.positive.sum(axis=1).max() < 2:
            raise Blocked("ORIGINAL_MULTI_POSITIVE_SUPERVISION_NOT_AVAILABLE")
        positive_mask = t.positive.copy()
        if method == "C_SINGLE":
            positive_mask[:] = False
            positive_mask[np.arange(len(t.q)), np.argmax(t.positive, axis=1)] = True
        matrix = identity.copy()
        def distribution(s):
            e = np.exp(s-np.max(s, axis=1, keepdims=True))
            return e/e.sum(axis=1, keepdims=True)
        for _ in range(config["steps"]):
            scores = t.q@matrix@t.c.T/temp
            masked = np.where(t.allowed, scores, -np.inf)
            positive = np.where(positive_mask, scores, -np.inf)
            gradient = t.q.T@(distribution(masked)-distribution(positive))@t.c/(len(t.q)*temp)
            matrix -= config["learning_rate"]*(gradient+ridge*(matrix-identity))
        return linear(matrix)
    if method == "I03":
        _, _, positive, negative = pair_data(t, temp)
        dp, dn = t.q-t.c[positive], t.q-t.c[negative]
        a = dp.T@dp/len(dp)+ridge*identity
        b = dn.T@dn/len(dn)
        whitening = invsqrt(a, ridge)
        eigenvalues, vectors = np.linalg.eigh(whitening@b@whitening)
        k = min(config["components"], d)
        transform = whitening@vectors[:, -k:]
        return linear(transform@transform.T, generalized_eigenvalues=eigenvalues[-k:].tolist())
    if method == "I10":
        q, c = t.q, t.c[np.argmax(t.positive, axis=1)]
        mq, mc = q.mean(0), c.mean(0)
        q, c = q-mq, c-mc
        aq = invsqrt(q.T@q/len(q)+ridge*identity, ridge)
        ac = invsqrt(c.T@c/len(c)+ridge*identity, ridge)
        u, singular, vt = np.linalg.svd(aq@(q.T@c/len(q))@ac, full_matrices=False)
        k = min(config["components"], d)
        return {"kind": "two_view", "q_mean": mq, "c_mean": mc,
                "q_map": aq@u[:, :k], "c_map": ac@vt[:k].T,
                "diagnostics": {"canonical_correlations": singular[:k].tolist()}}
    if method in {"I13", "C_WHITEN"}:
        values = np.vstack((t.q, t.c)); mean = values.mean(0); values = values-mean
        cov = values.T@values/len(values)
        alpha = config["shrinkage"] if method == "I13" else 0.0
        cov = (1-alpha)*cov+alpha*np.trace(cov)/d*identity+ridge*identity
        transform = invsqrt(cov, ridge)
        return {"kind": "two_view", "q_mean": mean, "c_mean": mean,
                "q_map": transform, "c_map": transform, "diagnostics": {"shrinkage": alpha}}
    if method == "I17":
        c = t.c[np.argmax(t.positive, axis=1)]
        signal = np.maximum((t.q*c).mean(0), 0)
        noise = ((t.q-c)**2).mean(0)/2
        reliability = signal/np.maximum(signal+noise, 1e-12)
        if reliability.max() <= 1e-12:
            raise Blocked("DECLARED_SHARED_SIGNAL_MODEL_HAS_NO_POSITIVE_DIRECTION")
        return linear(np.diag(reliability), reliability=reliability.tolist(), noise=noise.tolist())
    if method in {"I12", "I16", "C_RBF"}:
        if method == "C_RBF":
            return {"kind": "rbf", "bandwidth": config["bandwidth"], "diagnostics": {}}
        values = np.vstack((t.q, t.c)); count = min(config["landmarks"], len(values))
        anchors = values[np.linspace(0, len(values)-1, count, dtype=int)]
        kernel = rbf(anchors, anchors, config["bandwidth"])
        if method == "I16":
            weights = invsqrt(kernel+ridge*np.eye(count), ridge)
        else:
            degrees = kernel.sum(1)
            symmetric = kernel/np.sqrt(degrees[:, None]*degrees[None, :])
            eigenvalues, vectors = np.linalg.eigh(symmetric)
            indices = [i for i in np.argsort(eigenvalues)[::-1][1:] if eigenvalues[i] > 1e-8]
            indices = indices[:config["components"]]
            if not indices:
                raise Blocked("NO_NONSTATIONARY_POSITIVE_DIFFUSION_DIRECTION")
            phi = vectors[:, indices]/np.sqrt(degrees[:, None])
            weights = phi*eigenvalues[indices]**(config["diffusion_time"]-1)
        return {"kind": "diffusion" if method == "I12" else "nystrom", "anchors": anchors,
                "weights": weights, "bandwidth": config["bandwidth"], "diagnostics": {"landmark_count": count}}
    raise ValueError(f"Unknown registered method: {method}")

def features(head, x, side):
    kind = head["kind"]
    if kind == "linear":
        return x@head["matrix"] if side == "q" else x
    if kind == "two_view":
        return (x-head[side+"_mean"])@head[side+"_map"]
    if kind in {"diffusion", "nystrom"}:
        kernel = rbf(x, head["anchors"], head["bandwidth"])
        if kind == "diffusion":
            kernel /= np.maximum(kernel.sum(1, keepdims=True), 1e-12)
        return kernel@head["weights"]
    raise ValueError(f"No explicit feature map for {kind}")

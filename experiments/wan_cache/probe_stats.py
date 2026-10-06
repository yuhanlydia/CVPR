"""Descriptive diagnostics only; these quantities do not score VBench quality."""
import math
import statistics


def midranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + end - 1) / 2.0
        for original, _ in indexed[start:end]:
            ranks[original] = rank
        start = end
    return ranks


def corr(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("Correlation pairs must align")
    if any(not isinstance(x, (float, int)) or isinstance(x, bool) or not math.isfinite(x)
           for x in list(xs) + list(ys)):
        raise ValueError("Correlation inputs must be finite numbers")
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"pearson": None, "spearman": None}
    return {"pearson": float(statistics.correlation(xs, ys)),
            "spearman": float(statistics.correlation(midranks(xs), midranks(ys)))}


def summarize(rows):
    usable = [r for r in rows if r.get("reference_local_denoiser_output_rel_l1")
              not in (None, 0)]
    pairs = lambda key: [(r[key], r["terminal_mse"]) for r in usable if r.get(key) is not None]
    correlations = {}
    for name, key in (
        ("local_denoiser_error", "reference_local_denoiser_output_rel_l1"),
        ("raw_timestep_proxy", "reference_raw_modulated_rel_l1"),
        ("published_teacache_rescaled_proxy", "reference_teacache_rescaled_proxy")):
        xy = pairs(key)
        correlations[f"corr_{name}_to_terminal_mse"] = corr(
            [x for x, y in xy], [y for x, y in xy])
    # Combine CFG branches before the local norm to retain cancellation.
    gains = [math.sqrt(r["terminal_mse"]) / r["reference_guided_local_output_rms"]
             for r in rows if r.get("reference_guided_local_output_rms", 0) > 0]
    return {"num_forced_runs": len(rows), "num_usable_pairs": len(usable),
            **correlations,
            "empirical_rms_gain": {
                "definition": "terminal_video_RMS / absolute_CFG_denoiser_error_RMS",
                "num_pairs": len(gains),
                "mean": statistics.fmean(gains) if gains else None,
                "std": statistics.pstdev(gains) if gains else None,
                "min": min(gains) if gains else None,
                "max": max(gains) if gains else None,
                "interpretation": "direction-specific descriptive gain, not an operator norm or perceptual score"},
            "scientific_verdict": "NONE; native scoring and stronger controls pending"}

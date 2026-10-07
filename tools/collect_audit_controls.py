"""Collect the audit's native receipts, all arms and paired descriptive analyses.

Uses the official scorer live. Missing/failed arms remain visible; never tunes.
Run as a bounded zero-GPU remote harness collection job at the execution revision.
"""
import argparse
import collections
import hashlib
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import read_json, write_json, digest
from experiments.embedding_heads.audit_controls import registry
from experiments.embedding_heads.bundle import (
    validate_manifest, referenced, jsonl, check_original_source)


def image_units(base, manifest, task):
    """Bind repeated-image clusters to retained actual native samples/assets."""
    refs = {r["path"]: r for r in manifest["provenance_refs"]}
    sample_ref = refs["provenance/" + task["task"] + "_samples.jsonl"]
    asset_ref = refs["provenance/baseline_assets.json"]
    samples = jsonl(referenced(base, sample_ref))
    assets = read_json(referenced(base, asset_ref))
    original = jsonl(referenced(base, task["original_predictions_ref"]))
    if len(samples) != task["denominator"]:
        raise ValueError("Native sample unit coverage changed")
    units = []
    for index, (row, prediction) in enumerate(zip(samples, original)):
        labels = row["label_name"] if isinstance(row["label_name"], list) else [row["label_name"]]
        if row["released_row_index"] != index or labels != prediction["label"]:
            raise ValueError("Native sample row order/labels changed")
        image = row["query_input"].get("image")
        if image:
            relative = Path(image).relative_to(Path(assets["image_root"])).as_posix()
            units.append("image:" + assets["image_hashes"][relative])
        else:
            encoded = repr(sorted(row["query_input"].items())).encode()
            units.append("query:" + hashlib.sha256(encoded).hexdigest())
    return units, [sample_ref, asset_ref]


def bootstrap(difference, unit_ids, replicates, seed):
    indices = {}
    for index, unit in enumerate(unit_ids):
        indices.setdefault(unit, []).append(index)
    if len(indices) < 2:
        raise ValueError("At least two actual sampling units required")
    sums = np.array([difference[rows].sum() for rows in indices.values()])
    counts = np.array([len(rows) for rows in indices.values()])
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for index in range(replicates):
        selected = rng.integers(0, len(sums), len(sums))
        draws[index] = sums[selected].sum() / counts[selected].sum()
    return {"cluster_bootstrap_percentile_95": np.quantile(draws, [.025, .975]).tolist(),
            "sampling_units": len(indices), "replicates": replicates, "seed": seed,
            "unit_rule": "same native query-image SHA256 clusters; text-only exact-input clusters",
            "assumption": "clusters exchangeable/independent conditional on this fixed candidate corpus",
            "multiplicity": "descriptive only; no familywise winner significance or scientific PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "upstream", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--receipt", action="append", required=True,
                        help="Actual native run receipt JSON; repeat across retained windows")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    config = read_json(args.config)
    arms = registry(config)
    manifest = validate_manifest(args.bundle)
    check_original_source(args.upstream)
    sys.path.insert(0, str(Path(args.upstream).resolve()))
    from src.evaluation.mmeb_v2.utils.eval_utils.metrics import RankingMetrics
    baseline_metric = RankingMetrics(["hit"])
    config_hash = digest(args.config)
    tasks = {t["task"]: t for t in manifest["evaluation"]}
    expected = {(arm, task): {"method": arm, "task": task, "status": "PENDING"}
                for arm in arms for task in tasks}
    hits, predictions, parameter_hashes, retained, errors = {}, {}, {}, [], []
    required_code = ["tools/run_audit_method.py", "tools/run_method.py",
                     "experiments/embedding_heads/audit_controls.py",
                     "experiments/embedding_heads/heads.py", "experiments/embedding_heads/bundle.py"]
    seen = set()
    for receipt_path in args.receipt:
        receipt_path = Path(receipt_path).resolve()
        receipt_path.relative_to(root)
        receipt = read_json(receipt_path)
        retained.append({"path": receipt_path.relative_to(root).as_posix(), "sha256": digest(receipt_path)})
        for attempt in receipt["attempts"]:
            key = tuple(attempt["trial_id"].rsplit("__", 1))
            if key not in expected:
                errors.append({"trial_id": attempt["trial_id"], "reason": "OUTSIDE_FROZEN_AUDIT_INVENTORY"})
                continue
            if key in seen:
                errors.append({"trial_id": attempt["trial_id"], "reason": "DUPLICATE_ATTEMPT_NEEDS_EXPLICIT_REVIEW"})
                expected[key]["status"] = "AMBIGUOUS_DUPLICATE"
                hits.pop(key, None)
                predictions.pop(key, None)
                parameter_hashes.pop(key, None)
                continue
            seen.add(key)
            record = expected[key]
            record.update(native_status=attempt["status"], exit_code=attempt["exit_code"],
                          attempt_path=attempt["attempt_path"], seconds=attempt["seconds"])
            actual_dir = (root / attempt["attempt_path"]).resolve()
            actual_dir.relative_to(root)
            actual = read_json(actual_dir / "attempt.json")
            if actual != attempt:
                raise ValueError("Native receipt/attempt record mismatch; retain and reconcile")
            def checked(filename):
                path = actual_dir / "workspace/out" / filename
                refs = [r for r in actual["output_refs"] if r["path"] == path.relative_to(root).as_posix()]
                if len(refs) != 1 or digest(path) != refs[0]["sha256"]:
                    raise ValueError("Changed/missing actual audit output: " + filename)
                retained.append(refs[0])
                return path
            try:
                child = read_json(checked("result.json"))
                if child["method"] != key[0] or child["effective_task"] != key[1]:
                    raise ValueError("Actual arm/group identity changed")
                record["result"] = child
                if actual["status"] != "completed" or actual["exit_code"] != 0:
                    record["status"] = ("BLOCKED" if actual["exit_code"] == 20 and child["status"] == "BLOCKED"
                                        else str(actual["status"]).upper())
                    continue
                if child["status"] != "DEVELOPMENTAL_SCORED" or child["audit_config_sha256"] != config_hash:
                    raise ValueError("Actual design/status differs from pinned audit")
                refs = {(r["path"], r["sha256"]) for r in actual["code_refs"]}
                if not all((name, digest(root / name)) in refs for name in required_code):
                    raise ValueError("Collector must use the exact execution source revision")
                if not any(r["sha256"] == config_hash for r in actual["input_refs"]):
                    raise ValueError("Native attempt lacks audit config input binding")
                checked("head.npz"); checked("head.json")
                task = tasks[key[1]]
                replay = child["tasks"]
                if len(replay) != 1 or replay[0]["task"] != key[1] or replay[0]["denominator"] != task["denominator"]:
                    raise ValueError("Native scorer coverage differs")
                pred_path = checked(key[1] + "_pred.jsonl")
                score_path = checked(key[1] + "_score.json")
                checked(key[1] + "_replay.json")
                if digest(pred_path) != replay[0]["prediction_sha256"] or digest(score_path) != replay[0]["score_sha256"]:
                    raise ValueError("Replay output digest changed")
                rows = jsonl(pred_path)
                original = jsonl(referenced(Path(args.bundle).parent, task["original_predictions_ref"]))
                if len(rows) != task["denominator"]:
                    raise ValueError("Native denominator changed")
                for own, native in zip(rows, original):
                    if (own["label"] != native["label"] or
                            collections.Counter(own["prediction"]) != collections.Counter(native["prediction"])):
                        raise ValueError("Native candidate/label contract changed")
                official = RankingMetrics(task["metrics"]).evaluate(rows)
                score = read_json(score_path)
                if any(abs(float(score[k]) - float(v)) > 1e-12 for k, v in official.items()):
                    raise ValueError("Actual live official scorer replay differs")
                hits[key] = np.array([baseline_metric.hit_at_k(r["prediction"], r["label"], 1) for r in rows])
                predictions[key] = rows
                parameter_hashes[key] = child["parameter_sha256"]
                record["status"] = "DEVELOPMENTAL_SCORED_LIVE_REPLAYED"
            except Exception as error:
                record.update(status="INVALID_EVIDENCE", error_type=type(error).__name__, error=str(error))
    comparisons = []
    all_contrasts = [dict(c, analysis_level="primary") for c in config["primary_contrasts"]]
    all_contrasts += [dict(c, analysis_level="secondary") for c in config["secondary_contrasts"]]
    for contrast in all_contrasts:
        for name, task in tasks.items():
            item = {"contrast": contrast["id"], "analysis_level": contrast["analysis_level"],
                    "task": name, "weights": contrast["weights"],
                    "scientific_verdict": "NONE", "denominator": task["denominator"]}
            missing = [arm for arm in contrast["weights"] if (arm, name) not in hits]
            if missing:
                item.update(status="PENDING_COMPARISON", missing=missing)
            else:
                difference = sum(weight * hits[(arm, name)] for arm, weight in contrast["weights"].items())
                item.update(status="EXPLORATORY_POINT_ESTIMATE", paired_delta_hit_at_1=float(difference.mean()))
                try:
                    units, refs = image_units(Path(args.bundle).parent, manifest, task)
                    item.update(bootstrap(difference, units, config["analysis"]["bootstrap_replicates"],
                                          config["analysis"]["seed"]), unit_provenance_refs=refs)
                    item["status"] = "EXPLORATORY_CONDITIONAL_INTERVAL"
                except Exception as error:
                    item.update(uncertainty_status="PENDING_UNIT_AUDIT", uncertainty_error=str(error))
                if len(contrast["weights"]) == 2:
                    positive = next(a for a, w in contrast["weights"].items() if w == 1)
                    negative = next(a for a, w in contrast["weights"].items() if w == -1)
                    a, b = predictions[(positive, name)], predictions[(negative, name)]
                    item["different_top1_rows"] = sum(x["prediction"][0] != y["prediction"][0] for x, y in zip(a, b))
                    item["different_full_rankings"] = sum(x["prediction"] != y["prediction"] for x, y in zip(a, b))
            comparisons.append(item)
    coverage = []
    for arm in arms:
        keys = [(arm, name) for name in tasks]
        values = [parameter_hashes[k] for k in keys if k in parameter_hashes]
        coverage.append({"method": arm, "completed_tasks": sum(k in hits for k in keys),
                         "required_tasks": list(tasks),
                         "same_fit_across_tasks": all(k in hits for k in keys) and
                                                 len(values) == len(tasks) and len(set(values)) == 1,
                         "optimizer_convergence": [expected[k].get("result", {}).get("diagnostics", {}).get(
                             "convergence_qualified") for k in keys]})
    report = {"schema": "cvpr.audit-return.v1", "status": "DEVELOPMENTAL_AUDIT_SNAPSHOT",
              "arms": list(expected.values()), "coverage": coverage, "comparisons": comparisons,
              "retained_output_refs": retained, "errors": errors, "config_sha256": config_hash,
              "scientific_verdict": "NONE", "gate_advanced": False,
              "all_failures_retained": True, "all_43_records_included": True,
              "primary_contrasts": len(config["primary_contrasts"]),
              "primary_task_endpoints": 3 * len(config["primary_contrasts"]),
              "secondary_contrasts": len(config["secondary_contrasts"]),
              "confirmation": "not performed; inspected development outputs cannot confirm chosen winners"}
    write_json(args.out, report)
    print("Audit snapshot:", args.out)
    print("Completed native group jobs:", len(hits), "; scientific_verdict=NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

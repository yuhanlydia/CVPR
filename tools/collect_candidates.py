"""Collect existing real outputs and exploratory paired uncertainty. Never fit/tune."""
import argparse
import collections
import json
import sys
import tarfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import read_json, write_json, digest
from experiments.embedding_heads.bundle import validate_manifest, referenced, jsonl, check_original_source

def collect(root, run_id):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("Invalid batch ID")
    batch = root/"runs/candidate-batches"/run_id
    report = read_json(batch/"summary.json")
    cfg = read_json(root/"configs/candidates.json")
    # Keep each failure/block/timeout and each missing comparison; no winner-only report.
    result = {"run_id": run_id, "batch_status": report["status"], "methods": report["methods"],
              "comparisons": [], "scientific_verdict": "NONE", "gate_advanced": False,
              "scope": "exploratory native per-task hit@1, no selected-winner confirmation"}
    completed = {r["method"]: r for r in report["methods"] if r["status"] == "DEVELOPMENTAL_SCORED"}
    if report.get("bundle"):
        manifest = validate_manifest(report["bundle"])
        upstream = root/"sources/Qwen3-VL-Embedding"; check_original_source(upstream)
        sys.path.insert(0, str(upstream))
        from src.evaluation.mmeb_v2.utils.eval_utils.metrics import RankingMetrics
        metric = RankingMetrics(["hit"])
        rng = np.random.default_rng(cfg["seed"])
        for method in cfg["selected"]:
            required = cfg["required_controls"][method]
            missing = [name for name in [method]+required if name not in completed]
            if missing:
                result["comparisons"].append({"method": method, "status": "PENDING_COMPARISON", "missing": missing})
                continue
            for task in manifest["evaluation"]:
                n = task["denominator"]
                original = jsonl(referenced(Path(report["bundle"]).parent, task["original_predictions_ref"]))
                arms = ["ORIGINAL_FULL"]+required
                own = root/completed[method]["attempt_path"]/"workspace/out"/(task["task"]+"_pred.jsonl")
                a = jsonl(own)
                for control in arms:
                    b = original if control == "ORIGINAL_FULL" else jsonl(
                        root/completed[control]["attempt_path"]/"workspace/out"/(task["task"]+"_pred.jsonl"))
                    if len(a) != n or len(b) != n:
                        raise ValueError("Paired native denominator mismatch")
                    for x, y in zip(a, b):
                        if x["label"] != y["label"] or collections.Counter(x["prediction"]) != collections.Counter(y["prediction"]):
                            raise ValueError("Comparison row/label/candidate identities differ")
                    # Native per-case endpoint, not a replacement scorer/benchmark.
                    difference = np.array([metric.hit_at_k(x["prediction"], x["label"], 1)-
                                           metric.hit_at_k(y["prediction"], y["label"], 1) for x, y in zip(a, b)])
                    samples = np.array([difference[rng.integers(0, n, n)].mean()
                                        for _ in range(cfg["bootstrap_replicates"])])
                    result["comparisons"].append({"method": method, "control": control, "task": task["task"],
                        "status": "EXPLORATORY_COMPLETE", "denominator": n, "paired_delta_hit_at_1": float(difference.mean()),
                        "bootstrap_percentile_95": np.quantile(samples, [0.025, 0.975]).tolist(),
                        "replicates": cfg["bootstrap_replicates"], "seed": cfg["seed"],
                        "paired_query_resampling_assumes_exchangeable_queries": True,
                        "multiple_candidate_selection_not_confirmed": True,
                        "method_prediction_sha256": digest(own), "scientific_verdict": "NONE"})
    small = root/"rounds/r003/returns"/(run_id+".json")
    write_json(small, result)
    files = [p for p in batch.rglob("*") if p.is_file() and not p.is_symlink()]
    for native_id in report["native_run_ids"]:
        run = root/"runs/attempts"/native_id
        files += [p for p in run.rglob("*") if p.is_file() and not p.is_symlink()
                  and "sources" not in p.relative_to(run).parts
                  and (p.suffix in {".json", ".jsonl", ".log"} or p.name == "head.npz")]
    archive = root/"runs"/(run_id+"-return.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(set(files)):
            tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    print(f"Report: {small}\nRaw evidence archive: {archive}")
    return result

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-id"); a = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    run_id = a.run_id or read_json(root/"runs/latest-candidates.json")["run_id"]
    collect(root, run_id)

if __name__ == "__main__":
    main()

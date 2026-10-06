"""Collect real outputs and paired uncertainty; never fit/tune or discard failures."""
import argparse
import collections
import sys
import tarfile
import traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import read_json, write_json, digest
from experiments.embedding_heads.bundle import validate_manifest, referenced, jsonl, check_original_source


def arm_predictions(root, record, task):
    """Reject modified predictions or result summaries before any comparison."""
    root = Path(root).resolve()
    relative = Path(record["attempt_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid method attempt path")
    attempt = (root/relative).resolve()
    attempt.relative_to(root)
    native = read_json(attempt/"attempt.json")
    if (native["status"] != "completed" or native["exit_code"] != 0
            or native["trial_id"] != record["method"]
            or native["attempt_path"] != relative.as_posix()):
        raise ValueError("Method has no matching completed native receipt")
    def checked(filename):
        path = (attempt/"workspace/out"/filename).resolve()
        path.relative_to(root)
        refs = [r for r in native["output_refs"] if r["path"] == path.relative_to(root).as_posix()]
        if len(refs) != 1 or not path.is_file() or digest(path) != refs[0]["sha256"]:
            raise ValueError("Missing/changed native method output: "+filename)
        return path
    result = read_json(checked("result.json"))
    if (result != record["result"] or result["method"] != record["method"]
            or result["status"] != "DEVELOPMENTAL_SCORED"):
        raise ValueError("Batch child summary differs from actual native result")
    name = task["task"]
    rows = [v for v in result["tasks"] if v["task"] == name]
    if len(rows) != 1 or rows[0]["denominator"] != task["denominator"]:
        raise ValueError("Method native task identity or denominator changed")
    path = checked(name+"_pred.jsonl")
    if digest(path) != rows[0]["prediction_sha256"]:
        raise ValueError("Method prediction differs from scorer replay receipt")
    return path, jsonl(path)


def paired_comparison(metric, a, b, n, cfg, rng):
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0 or len(a) != n or len(b) != n:
        raise ValueError("Paired native denominator mismatch")
    for x, y in zip(a, b):
        if (x["label"] != y["label"]
                or collections.Counter(x["prediction"]) != collections.Counter(y["prediction"])):
            raise ValueError("Comparison row/label/candidate identities differ")
    # Native per-case endpoint, not a replacement scorer or benchmark.
    difference = np.array([metric.hit_at_k(x["prediction"], x["label"], 1)-
                           metric.hit_at_k(y["prediction"], y["label"], 1) for x, y in zip(a, b)])
    samples = np.array([difference[rng.integers(0, n, n)].mean()
                        for _ in range(cfg["bootstrap_replicates"])])
    return {"status": "EXPLORATORY_COMPLETE", "denominator": n,
            "paired_delta_hit_at_1": float(difference.mean()),
            "bootstrap_percentile_95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "replicates": cfg["bootstrap_replicates"], "seed": cfg["seed"],
            "paired_query_resampling_assumes_exchangeable_queries": True,
            "multiple_candidate_selection_not_confirmed": True, "scientific_verdict": "NONE"}


def comparisons(root, report, cfg, manifest, completed):
    upstream = root/"sources/Qwen3-VL-Embedding"
    check_original_source(upstream)
    sys.path.insert(0, str(upstream))
    from src.evaluation.mmeb_v2.utils.eval_utils.metrics import RankingMetrics
    metric = RankingMetrics(["hit"])
    rng = np.random.default_rng(cfg["seed"])
    result = []
    for method in cfg["selected"]:
        required = cfg["required_controls"][method]
        missing = [name for name in [method]+required if name not in completed]
        if missing:
            result.append({"method": method, "status": "PENDING_COMPARISON", "missing": missing})
            continue
        for task in manifest["evaluation"]:
            for control in ["ORIGINAL_FULL"]+required:
                item = {"method": method, "control": control, "task": task["task"]}
                try:
                    own, a = arm_predictions(root, completed[method], task)
                    if control == "ORIGINAL_FULL":
                        other = referenced(Path(report["bundle"]).parent, task["original_predictions_ref"])
                        b = jsonl(other)
                    else:
                        other, b = arm_predictions(root, completed[control], task)
                    item.update(paired_comparison(metric, a, b, task["denominator"], cfg, rng),
                                method_prediction_sha256=digest(own), control_prediction_sha256=digest(other))
                except Exception as error:
                    item.update(status="INVALID_EVIDENCE", error_type=type(error).__name__, error=str(error),
                                scientific_verdict="NONE", error_traceback=traceback.format_exc())
                result.append(item)
    return result


def collect(root, run_id):
    root = Path(root).resolve()
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("Invalid batch ID")
    batch = root/"runs/candidate-batches"/run_id
    report = read_json(batch/"summary.json")
    cfg = read_json(root/"configs/candidates.json")
    result = {"run_id": run_id, "batch_status": report["status"], "methods": report["methods"],
              "comparisons": [], "scientific_verdict": "NONE", "gate_advanced": False,
              "scope": "exploratory native per-task hit@1, no selected-winner confirmation",
              "collection_status": "RETAINED"}
    completed = {r["method"]: r for r in report["methods"] if r["status"] == "DEVELOPMENTAL_SCORED"}
    if report.get("bundle"):
        try:
            manifest = validate_manifest(report["bundle"])
            result["comparisons"] = comparisons(root, report, cfg, manifest, completed)
            if any(v["status"] == "INVALID_EVIDENCE" for v in result["comparisons"]):
                result["collection_status"] = "RETAINED_WITH_INVALID_COMPARISONS"
        except Exception as error:
            # Still return failed logs/raw attempts when shared evidence is invalid.
            result.update(collection_status="INVALID_SHARED_EVIDENCE",
                          collection_error_type=type(error).__name__, collection_error=str(error),
                          collection_error_traceback=traceback.format_exc())
    small = root/"rounds/r003/returns"/(run_id+".json")
    write_json(small, result)
    files = [p for p in batch.rglob("*") if p.is_file() and not p.is_symlink()]
    for native_id in report["native_run_ids"]:
        if not native_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in native_id):
            raise ValueError("Invalid native run ID")
        run = root/"runs/attempts"/native_id
        files += [p for p in run.rglob("*") if p.is_file() and not p.is_symlink()
                  and "sources" not in p.relative_to(run).parts
                  and (p.suffix in {".json", ".jsonl", ".log"} or p.name == "head.npz")]
    archive = root/"runs"/(run_id+"-return.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(set(files)):
            tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    print("Report:", small)
    print("Raw evidence archive:", archive)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id")
    a = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    run_id = a.run_id or read_json(root/"runs/latest-candidates.json")["run_id"]
    collect(root, run_id)


if __name__ == "__main__":
    main()

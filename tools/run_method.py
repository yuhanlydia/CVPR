"""One registered prototype plus live original MMEB scoring on real complete caches."""
import argparse
import collections
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embedding_heads.heads import Blocked, fit, features, rbf
from experiments.embedding_heads.bundle import (validate_manifest, load_training, load_projection,
    project, referenced, jsonl, check_original_source, sha256, SCORER_PATH)
from common import now, read_json, write_json

def evaluate(task, head, bundle_path, manifest, projection, upstream, out, *, feature_builder=None):
    import torch
    sys.path.insert(0, str(Path(upstream).resolve()))
    from src.evaluation.mmeb_v2.models import MMEBEmbeddingModel
    from src.evaluation.mmeb_v2.utils.eval_utils.metrics import RankingMetrics
    base = Path(bundle_path).parent
    infos = jsonl(referenced(base, task["metadata_ref"]))
    original = jsonl(referenced(base, task["original_predictions_ref"]))
    original_score = read_json(referenced(base, task["original_scores_ref"]))
    n = task["denominator"]
    if len(infos) != n or len(original) != n or original_score["num_data"] != n or original_score["num_pred"] != n:
        raise ValueError("Native task denominator changed")
    scorer = RankingMetrics(task["metrics"])
    original_replay = scorer.evaluate(original)
    if any(abs(float(original_score[k])-float(v)) > 1e-12 for k, v in original_replay.items()):
        raise ValueError("Actual official baseline scorer replay failed")
    with np.load(referenced(base, task["features_ref"]), allow_pickle=False) as data:
        fullq, fullc = data["q"].copy(), data["c"].copy()
    if len(fullq) != n or len(fullc) != len(task["candidate_ids"]):
        raise ValueError("Native feature coverage changed")
    if not np.isfinite(fullq).all() or not np.isfinite(fullc).all():
        raise ValueError("Nonfinite actual evaluation features")
    if feature_builder is None:
        q, c = project(fullq, projection), project(fullc, projection)
        if head["kind"] not in {"rbf", "distance"}:
            q, c = features(head, q, "q"), features(head, c, "c")
    else:
        q = feature_builder(fullq, "q")
        c = feature_builder(fullc, "c")
    if head["kind"] not in {"rbf", "distance"}:
        if q.ndim != 2 or c.ndim != 2 or q.shape[1] != c.shape[1]:
            raise ValueError("Transformed query/candidate dimensions disagree")
        if not np.isfinite(q).all() or not np.isfinite(c).all():
            raise ValueError("Nonfinite transformed features")
        if (np.linalg.norm(q, axis=1) < 1e-12).any() or (np.linalg.norm(c, axis=1) < 1e-12).any():
            raise ValueError("Zero transformed feature")
    if head["kind"] == "distance":
        matrix = np.asarray(head["matrix"], dtype=np.float64)
        if matrix.shape != (q.shape[1], q.shape[1]) or not np.isfinite(matrix).all():
            raise ValueError("Invalid distance metric dimensions")
        if not np.allclose(matrix, matrix.T, atol=1e-10):
            raise ValueError("Distance metric is not symmetric")
        if np.linalg.eigvalsh(matrix).min() < -1e-8:
            raise ValueError("Distance metric is not positive semidefinite")
    q_norm, c_norm = np.linalg.norm(q, axis=1), np.linalg.norm(c, axis=1)
    feature_diagnostics = {"query_dim": int(q.shape[1]), "candidate_dim": int(c.shape[1]),
                           "query_norm_min": float(q_norm.min()), "query_norm_max": float(q_norm.max()),
                           "candidate_norm_min": float(c_norm.min()), "candidate_norm_max": float(c_norm.max())}
    candidate_ids = task["candidate_ids"]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Duplicate native candidate IDs")
    index = {name: i for i, name in enumerate(candidate_ids)}
    predictions, tie_rows = [], 0
    pred_path = out/(task["task"]+"_pred.jsonl")
    with pred_path.open("x", encoding="utf-8") as stream:
        for i, info in enumerate(infos):
            names = candidate_ids if task["eval_type"] == "global" else info["cand_names"]
            labels = info["label_name"] if isinstance(info["label_name"], list) else [info["label_name"]]
            if (collections.Counter(original[i]["prediction"]) != collections.Counter(names)
                    or original[i]["label"] != labels):
                raise ValueError("Original candidate/label contract mismatch")
            candidates = c[[index[name] for name in names]]
            if head["kind"] == "rbf":
                tensor = torch.as_tensor(rbf(q[i:i+1], candidates, head["bandwidth"])[0], dtype=torch.float64)
            elif head["kind"] == "distance":
                delta = candidates-q[i:i+1]
                scores = -np.einsum("md,de,me->m", delta, matrix, delta)
                tensor = torch.as_tensor(scores, dtype=torch.float64)
            else:
                tensor = MMEBEmbeddingModel.compute_similarity(None,
                    torch.as_tensor(q[i:i+1], dtype=torch.float64),
                    torch.as_tensor(candidates, dtype=torch.float64)).squeeze(0)
            if not torch.isfinite(tensor).all():
                raise ValueError("Nonfinite method scores")
            # Original torch.sort, with CPU/float64 explicitly recorded for these heads.
            sorted_scores, ranked = torch.sort(tensor, descending=True)
            tie_rows += int(bool(torch.any(sorted_scores[1:] == sorted_scores[:-1])))
            pred = {"prediction": [names[j] for j in ranked.tolist()], "label": labels,
                    "rel_scores": info.get("rel_scores")}
            predictions.append(pred)
            stream.write(json.dumps(pred, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
    score = scorer.evaluate(predictions)
    score.update(num_pred=n, num_data=n)
    score_path = out/(task["task"]+"_score.json")
    write_json(score_path, score)
    replay = scorer.evaluate(jsonl(pred_path))
    if any(abs(float(score[k])-float(v)) > 1e-12 for k, v in replay.items()):
        raise ValueError("Persisted native prediction replay mismatch")
    receipt = {"task": task["task"], "denominator": n, "unique_candidates": len(candidate_ids),
               "native_scorer_sha256": sha256(Path(upstream)/SCORER_PATH),
               "prediction_sha256": sha256(pred_path), "score_sha256": sha256(score_path),
               "replay": "matched", "baseline_replay": "matched", "scores": score,
               "score_device": "cpu", "score_dtype": "float64", "rows_with_exact_ties": tie_rows,
               "full_candidate_rankings": True, "score_kind": head["kind"],
               "feature_diagnostics": feature_diagnostics}
    write_json(out/(task["task"]+"_replay.json"), receipt)
    return receipt

def main():
    p = argparse.ArgumentParser()
    for name in ("bundle", "config", "method", "upstream", "out"):
        p.add_argument("--"+name, required=True)
    a = p.parse_args()
    config = read_json(a.config)
    if a.method not in config["controls"]+config["selected"]:
        raise ValueError("Method outside the frozen inventory")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = {"method": a.method, "status": "RUNNING", "started_at": now(), "tasks": [],
              "scientific_verdict": "NONE", "gate_advanced": False, "fit_uses_test_labels": False}
    write_json(out/"result.json", result)
    exit_code = 0
    try:
        check_original_source(a.upstream)
        manifest = validate_manifest(a.bundle)
        if (manifest["train"]["revision"] != config["training_revision"]
                or [t["task"] for t in manifest["evaluation"]] != config["evaluation_tasks"]):
            raise ValueError("Bundle/config identity changed")
        training = load_training(a.bundle, manifest, config["head"]["temperature"])
        fit_start = time.monotonic()
        head = fit(a.method, training, config["head"])
        result.update(setup_seconds=fit_start-start, fit_seconds=time.monotonic()-fit_start,
                      diagnostics=head["diagnostics"], head_kind=head["kind"])
        np.savez_compressed(out/"head.npz", **{k: v for k, v in head.items() if isinstance(v, np.ndarray)})
        write_json(out/"head.json", {k: v for k, v in head.items() if not isinstance(v, np.ndarray)})
        write_json(out/"result.json", result)
        projection = load_projection(a.bundle, manifest)
        # Evaluation rows are first loaded after fit() returned.
        for task in manifest["evaluation"]:
            result["tasks"].append(evaluate(task, head, a.bundle, manifest, projection, a.upstream, out))
            write_json(out/"result.json", result)
        result["status"] = "DEVELOPMENTAL_SCORED"
    except Blocked as error:
        result.update(status="BLOCKED", error=str(error), error_type=type(error).__name__)
        exit_code = 20
        print(str(error), file=sys.stderr, flush=True)
    except Exception as error:
        result.update(status="FAILED", error=str(error), error_type=type(error).__name__)
        exit_code = 1
        traceback.print_exc()
    finally:
        result.update(finished_at=now(), elapsed_seconds=time.monotonic()-start)
        write_json(out/"result.json", result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded real training-cache preparation; original training images must exist.
Evaluation caches are exported only from our own completed native qualification.
"""
import argparse
import collections
import hashlib
import json
import pickle
import shutil
import sys
import time
from pathlib import Path, PurePosixPath
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import digest, read_json, write_json, now
from source_adapter import adapt
from experiments.embedding_heads.bundle import UPSTREAM, check_original_source
from experiments.embedding_heads.heads import unit

def identity(text, image_hash):
    value = [" ".join(text.replace("<|image_1|>", "").split()), image_hash]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()

def image_file(root, relative):
    if not relative:
        return None
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Unsafe original image path")
    path = (Path(root)/p).resolve()
    path.relative_to(Path(root).resolve())
    if not path.is_file():
        raise FileNotFoundError(f"Original training image missing: {relative}")
    return path

def original_pairs(text, images):
    text = "" if text is None else text
    images = "" if images is None else images
    text = [text] if isinstance(text, str) else text
    images = [images] if isinstance(images, str) else images
    if not isinstance(text, list) or not isinstance(images, list) or len(text) != len(images):
        raise ValueError("Unsupported original MMEB training fields")
    if not all(isinstance(x, str) for x in text+images):
        raise ValueError("Original training fields must contain strings")
    return [(t, im) for t, im in zip(text, images) if t or im]

def baseline_evidence(out, cfg):
    out = Path(out).resolve()
    attempt_path = out.parent.parent/"attempt.json"
    attempt = read_json(attempt_path)
    original_cwd = Path(attempt["cwd"])
    # The unique attempt directory survives staging; absolute roots may change.
    if (attempt["status"] != "completed" or attempt["exit_code"] != 0
            or original_cwd.name != out.parent.name or original_cwd.parent.name != out.parent.parent.name
            or not any(Path(arg).name == "qualify.py" for arg in attempt["command"])
            or attempt["provenance"]["upstream_revision"] != UPSTREAM):
        raise ValueError("Only completed pinned native qualification output may be exported")
    expected = digest(Path(__file__).with_name("qualify.py"))
    if not any(r["path"] == "tools/qualify.py" and r["sha256"] == expected for r in attempt["code_refs"]):
        raise ValueError("Actual cache producer differs from reviewed qualification code")
    summary = read_json(out/"summary.json")
    completed = {t["task"]: t for t in summary["completed_tasks"]}
    if any(n not in completed or completed[n]["scorer_replay"] != "matched" for n in cfg["evaluation_tasks"]):
        raise ValueError("Required full native baseline qualification missing")
    return out, attempt_path, completed, read_json(out/"assets/assets.json")

def prepare(a):
    cfg = read_json(a.config)
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    base, attempt, completed, assets = baseline_evidence(a.baseline_out, cfg)
    check_original_source(a.upstream)
    for name, expected in assets["model_weight_hashes"].items():
        if digest(Path(assets["model_path"])/name) != expected:
            raise ValueError("Frozen model weights changed after baseline qualification")
    manifest = {"schema": "r003.frozen-embedding-bundle.v1", "prepared_at": now(),
        "upstream_commit": UPSTREAM, "model_id": assets["model_id"], "model_revision": assets["model_revision"],
        "evaluation": [], "provenance_refs": [], "interval_calibration": None,
        "teacher": {"kind": "frozen_full_dimension_same_2B", "temperature": cfg["head"]["temperature"],
                    "second_model_loaded": False}}
    def ref(path):
        return {"path": Path(path).relative_to(out).as_posix(), "sha256": digest(path)}
    def copy(source, name):
        target = out/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target); return ref(target)
    for source, name in ((attempt, "provenance/baseline_attempt.json"),
                         (base/"summary.json", "provenance/baseline_summary.json"),
                         (base/"assets/assets.json", "provenance/baseline_assets.json")):
        manifest["provenance_refs"].append(copy(source, name))
    eval_images = set(assets["image_hashes"].values()); eval_ids = set()
    import yaml
    for name in cfg["evaluation_tasks"]:
        taskroot = base/"baselines"/name
        replay = read_json(taskroot/"replay.json")
        if replay["status"] != "matched" or replay["prediction_sha256"] != digest(taskroot/(name+"_pred.jsonl")):
            raise ValueError("Original native prediction replay identity changed")
        # Only our independently identified local producer's pickles are read.
        with (taskroot/(name+"_qry")).open("rb") as stream:
            q = np.asarray(pickle.load(stream), dtype=np.float64)
        with (taskroot/(name+"_tgt")).open("rb") as stream:
            corpus = pickle.load(stream)
        ids = list(corpus); c = np.asarray([corpus[k] for k in ids], dtype=np.float64)
        if (len(q) != completed[name]["native_queries"] or len(c) != completed[name]["unique_candidates"]
                or not np.isfinite(q).all() or not np.isfinite(c).all()):
            raise ValueError("Incomplete/nonfinite actual native cache")
        np.savez_compressed(out/(name+".npz"), q=q, c=c)
        native = yaml.safe_load((taskroot/"native-config.yaml").read_text())[name]
        manifest["evaluation"].append({"task": name, "denominator": len(q), "candidate_ids": ids,
            "eval_type": native.get("eval_type", "global"),
            "metrics": native.get("metrics", ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]),
            "features_ref": ref(out/(name+".npz")),
            "metadata_ref": copy(taskroot/(name+"_info.jsonl"), name+"_info.jsonl"),
            "original_predictions_ref": copy(taskroot/(name+"_pred.jsonl"), name+"_original_pred.jsonl"),
            "original_scores_ref": copy(taskroot/(name+"_score.json"), name+"_original_score.json")})
        sample = base/"native-samples"/(name+".jsonl")
        manifest["provenance_refs"].append(copy(sample, "provenance/"+name+"_samples.jsonl"))
        for line in sample.read_text().splitlines():
            inp = json.loads(line)["query_input"]; img = inp.get("image")
            eval_ids.add(identity(inp.get("text") or "", digest(img) if img else ""))
    from datasets import load_dataset
    inputs, candidates, images = [], [], {}
    rows, index = collections.OrderedDict(), {}
    def parsed(text, image):
        path = image_file(a.train_image_root, image); h = digest(path) if path else ""
        if h: images[image] = h
        key = identity(text, h)
        if h in eval_images or key in eval_ids:
            raise ValueError("Actual train/test overlap; no automatic exclusions")
        return key, {"text": text.replace("<|image_1|>", "").strip(), "image": str(path) if path else None}
    for group in cfg["training_tasks"]:
        ds = load_dataset(cfg["training_dataset"], group, split=cfg["training_split"], revision=cfg["training_revision"])
        for row_index, raw in enumerate(ds.select(range(min(len(ds), cfg["training_rows_per_task"])))):
            if not isinstance(raw["qry"], str) or not isinstance(raw["qry_image_path"], str):
                raise ValueError("Unsupported released training query layout")
            qid, inp = parsed(raw["qry"], raw["qry_image_path"]); key = (group, qid)
            if key not in rows:
                rows[key] = {"query_id": qid, "group": group, "split": "original", "released_row_indices": [],
                             "positive_candidate_ids": [], "negative_candidate_ids": []}
                inputs.append(inp)
            row = rows[key]; row["released_row_indices"].append(row_index)
            for role, tk, ik in (("positive", "pos_text", "pos_image_path"), ("negative", "neg_text", "neg_image_path")):
                for text, image in original_pairs(raw.get(tk, ""), raw.get(ik, "")):
                    cid, cand = parsed(text, image)
                    if cid not in index:
                        index[cid] = len(candidates); candidates.append(cand)
                    if cid not in row[role+"_candidate_ids"]:
                        row[role+"_candidate_ids"].append(cid)
    trainrows = list(rows.values())
    if len(inputs) < 2 or not candidates or any(not r["positive_candidate_ids"] for r in trainrows):
        raise ValueError("Missing original training supervision")
    if any(set(r["positive_candidate_ids"]) & set(r["negative_candidate_ids"]) for r in trainrows):
        raise ValueError("Conflicting released positive/negative labels")
    manifest["train"] = {"dataset_id": cfg["training_dataset"], "revision": cfg["training_revision"],
        "split": cfg["training_split"], "tasks": cfg["training_tasks"], "rows": len(trainrows),
        "selection": "first declared training rows; never performance-selected"}
    adapt(a.upstream)  # Isolated staged checkout, not the clean source checkout.
    sys.path.insert(0, str(Path(a.upstream).resolve()))
    import torch
    from src.evaluation.mmeb_v2.models import MMEBEmbeddingModel
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Actual single CUDA GPU required for feature preparation")
    torch.manual_seed(cfg["seed"]); torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model = MMEBEmbeddingModel.load(assets["model_path"], normalize=True,
        torch_dtype=torch.float16, attn_implementation="sdpa")
    def encode(items):
        result = []
        for i, inp in enumerate(items):
            with torch.inference_mode():
                v = model.encode_input(inp).detach().cpu().float().numpy()[0]
            if not np.isfinite(v).all(): raise ValueError("Nonfinite training encoding")
            result.append(v)
            print(json.dumps({"stage": "train_encode", "completed": i+1, "total": len(items)}), flush=True)
        return np.asarray(result, dtype=np.float64)
    fullq, fullc = encode(inputs), encode(candidates)
    if any(np.max(np.abs(np.linalg.norm(v, axis=1)-1)) > 0.005 for v in (fullq, fullc)):
        raise ValueError("Original normalized encoder contract changed")
    values = np.vstack((fullq, fullc)); mean = values.mean(0)
    _, singular, vt = np.linalg.svd(values-mean, full_matrices=False)
    rank = min(cfg["projection_rank"], int((singular > 1e-10).sum()))
    if rank < 2: raise ValueError("Insufficient original training rank")
    basis = vt[:rank].T
    np.savez_compressed(out/"projection.npz", mean=mean, basis=basis)
    positive = np.zeros((len(fullq), len(fullc)), dtype=bool); allowed = positive.copy()
    for i, row in enumerate(trainrows):
        for cid in row["positive_candidate_ids"]: positive[i, index[cid]] = True
        for cid in row["positive_candidate_ids"]+row["negative_candidate_ids"]: allowed[i, index[cid]] = True
    np.savez_compressed(out/"train.npz", q=unit((fullq-mean)@basis), c=unit((fullc-mean)@basis),
        positive=positive, allowed=allowed, teacher=fullq@fullc.T/cfg["head"]["temperature"])
    with (out/"train_rows.jsonl").open("x", encoding="utf-8") as stream:
        for row in trainrows: stream.write(json.dumps(row, ensure_ascii=False)+"\n")
    write_json(out/"leakage_audit.json", {"train_image_sha256": sorted(set(images.values())),
        "eval_image_sha256": sorted(eval_images), "train_query_ids": [r["query_id"] for r in trainrows],
        "eval_query_ids": sorted(eval_ids)})
    write_json(out/"provenance/train_images.json", images)
    manifest["provenance_refs"].append(ref(out/"provenance/train_images.json"))
    manifest["projection_ref"] = ref(out/"projection.npz")
    manifest["leakage_audit_ref"] = ref(out/"leakage_audit.json")
    manifest["train"].update(features_ref=ref(out/"train.npz"), rows_ref=ref(out/"train_rows.jsonl"),
                             candidate_ids=list(index), projection_rank=rank)
    manifest["resources"] = {"gpu_name": torch.cuda.get_device_name(0),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    write_json(out/"manifest.json", manifest)
    return manifest

def main():
    p = argparse.ArgumentParser()
    for name in ("config", "baseline-out", "train-image-root", "upstream", "out"):
        p.add_argument("--"+name, required=True)
    a = p.parse_args(); started = time.monotonic(); manifest = prepare(a)
    print(json.dumps({"status": "PREPARED_DEVELOPMENTAL_ASSETS", "elapsed_seconds": time.monotonic()-started,
                      "training_rows": manifest["train"]["rows"], "scientific_verdict": "NONE"}), flush=True)

if __name__ == "__main__":
    main()

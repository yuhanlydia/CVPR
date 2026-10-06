"""One native-runner-owned acquisition/calibration/baseline job.

Native rows/candidates/labels and scorer are retained. Calibration produces costs,
not benchmark performance. Full native task results are developmental qualification.
No research method, training sweep or confirmation is dispatched here.
"""
import argparse
import collections
import contextlib
import hashlib
import json
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path
from common import digest, now, read_json, write_json
from source_adapter import adapt


def admission(remaining, query_seconds, cand_seconds, n_queries, n_candidates,
              factor, tail):
    estimate = factor * (query_seconds * n_queries + cand_seconds * n_candidates)
    return {"admit": estimate + tail < remaining, "estimated_encode_seconds": estimate,
            "basis": "extrapolated from native calibration, not a measured task runtime"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--upstream", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--image-root")
    a = p.parse_args()
    cfg, out = read_json(a.config), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started, deadline = time.monotonic(), time.monotonic() + a.seconds
    summary = {"started_at": now(), "purpose": cfg["purpose"], "scientific_verdict": "NONE",
               "method_training_started": False, "gate_advanced": False,
               "completed_tasks": [], "pending_tasks": cfg["core_tasks"] + cfg["reserve_tasks"]}

    def event(kind, **fields):
        record = {"at": now(), "elapsed_seconds": time.monotonic() - started, "kind": kind, **fields}
        with (out / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        write_json(out / "summary.json", summary)

    failed = False
    try:
        if cfg["method_training_enabled"]:
            raise RuntimeError("Method execution requires a separate scientifically qualified protocol")
        event("acquisition_started")
        cmd = [sys.executable, str(Path(__file__).with_name("prepare_assets.py")),
               "--config", a.config, "--out", str(out / "assets")]
        if a.image_root:
            cmd += ["--image-root", a.image_root]
        subprocess.run(cmd, check=True, timeout=min(cfg["acquisition_timeout_seconds"],
                                                   max(1, deadline - time.monotonic() - cfg["reserved_tail_seconds"])))
        assets = read_json(out / "assets/assets.json")
        summary["asset_revisions"] = {key: assets[key] for key in
                                      ("model_revision", "metadata_revision", "image_revision")}
        event("acquisition_completed", native_task_sizes=assets["native_task_sizes"])
        upstream = Path(a.upstream).resolve()
        summary["source_adapter"] = adapt(upstream)
        sys.path.insert(0, str(upstream))
        import numpy as np
        import torch
        import yaml
        import datasets
        from PIL import Image
        # Every native parser load is bound to the already-resolved benchmark revision.
        original_dataset_load = datasets.load_dataset
        def pinned_dataset(path, *args, **kwargs):
            if path != assets["metadata_id"]:
                raise ValueError(f"Undeclared benchmark repository: {path}")
            kwargs["revision"] = assets["metadata_revision"]
            return original_dataset_load(path, *args, **kwargs)
        datasets.load_dataset = pinned_dataset
        from src.evaluation.mmeb_v2 import eval_embedding as ev
        from src.evaluation.mmeb_v2.arguments import ModelArguments, DataArguments
        from src.evaluation.mmeb_v2.data.collator import MultimodalEvalDataCollator
        from src.evaluation.mmeb_v2.data.datasets.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
        torch.manual_seed(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one GPU is required")
        if any(key in __import__("os").environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
            raise RuntimeError("Launch directly with Python; inherited distributed settings are unsupported")
        image_configs = yaml.safe_load((upstream / "scripts/evaluation/mmeb_v2/image.yaml").read_text())
        selected = {name: image_configs[name] for name in cfg["core_tasks"] + cfg["reserve_tasks"]}
        ma = ModelArguments(model_name_or_path=assets["model_path"], normalize=True)
        da = DataArguments()
        original_instantiate = AutoEvalPairDataset.instantiate
        inventory = {}
        def instantiate(*args, **kwargs):
            kwargs["image_root"] = assets["image_root"]
            # No num_sample_per_subset override: full native task and candidate pool.
            if "num_sample_per_subset" in kwargs:
                raise RuntimeError("Native task subsampling is not enabled in this round")
            queries, corpus = original_instantiate(*args, **kwargs)
            name = kwargs["dataset_name"]
            if len(queries) != assets["native_task_sizes"][name]:
                raise ValueError("Native parser changed the released denominator")
            candidates = generate_cand_dataset(queries, corpus)
            inventory[name] = (len(queries), len(candidates))
            manifest = out / "native-samples" / (name + ".jsonl")
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with manifest.open("w", encoding="utf-8") as f:
                for row_id, row in enumerate(queries):
                    info = row["dataset_infos"]
                    data = {"released_row_index": row_id, "query_input": row["query_input"],
                            "label_name": info["label_name"],
                            "candidate_list_sha256": hashlib.sha256(
                                json.dumps(info["cand_names"], ensure_ascii=False).encode()).hexdigest()}
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            event("native_task_loaded", task=name, queries=len(queries), candidates=len(candidates),
                  sample_manifest_sha256=digest(manifest))
            return queries, corpus
        AutoEvalPairDataset.instantiate = staticmethod(instantiate)
        first = cfg["core_tasks"][0]
        queries, corpus = instantiate(model_args=ma, data_args=da, **selected[first])
        candidates = generate_cand_dataset(queries, corpus)
        event("model_load_started")
        model = ev.MMEBEmbeddingModel.load(assets["model_path"], normalize=True,
                                          torch_dtype=torch.float16, attn_implementation="sdpa")
        model.eval()
        event("model_load_completed")
        original_encode = ev.encode_embeddings
        times = []
        def encode(*args, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.monotonic()
            result = original_encode(*args, **kwargs)
            torch.cuda.synchronize()
            seconds = time.monotonic() - t
            embeds, metadata = result
            if not np.isfinite(embeds).all() or embeds.shape[0] != kwargs["full_dataset_len"]:
                raise ValueError("Non-finite or incomplete native embeddings")
            observation = {"side": kwargs["encode_side"], "count": len(metadata), "seconds": seconds,
                           "seconds_per_item": seconds / len(metadata),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            times.append(observation)
            event("encoding_completed", **observation)
            return result
        ev.encode_embeddings = encode
        # Calibration uses original benchmark cases. Selection is cost-based, never label-based.
        areas = []
        for i, row in enumerate(queries):
            path = row["query_input"].get("image")
            if path:
                with Image.open(path) as im:
                    areas.append((im.width * im.height, i))
        n = cfg["calibration_queries"]
        query_indices = sorted(set(range(min(n, len(queries)))) |
                               {i for _, i in sorted(areas, reverse=True)[:n]})
        m = cfg["calibration_candidates"]
        lengths = [(len(row["cand_input"].get("text", "")), i) for i, row in enumerate(candidates)]
        cand_indices = sorted(set(range(min(m, len(candidates)))) |
                              {i for _, i in sorted(lengths, reverse=True)[:m]})
        event("calibration_started", query_row_indices=query_indices, candidate_row_indices=cand_indices,
              metrics_reported=False, task=first)
        qds, cds = queries.select(query_indices), candidates.select(cand_indices)
        for ds, side in ((qds, "qry"), (cds, "cand")):
            loader = ev.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                                   collate_fn=MultimodalEvalDataCollator(side))
            encode(model=model, loader=loader, encode_side=side, full_dataset_len=len(ds),
                   description="Native resource calibration")
        qcost, ccost = times[-2]["seconds_per_item"], times[-1]["seconds_per_item"]
        summary["calibration"] = {"task": first, "observations": times[-2:],
                                  "query_row_indices": query_indices, "candidate_row_indices": cand_indices,
                                  "timing_is_extrapolation": True, "benchmark_metrics_reported": False}
        # Subsequent native main calls reuse the exact same frozen model, not a second copy.
        def cached_load(*args, **kwargs):
            if kwargs.get("model_name_or_path", args[0] if args else None) != assets["model_path"]:
                raise ValueError("Model identity changed")
            return model
        ev.MMEBEmbeddingModel.load = staticmethod(cached_load)
        for name in cfg["core_tasks"] + cfg["reserve_tasks"]:
            if name not in inventory:
                instantiate(model_args=ma, data_args=da, **selected[name])
            nq, nc = inventory[name]
            decision = admission(deadline - time.monotonic(), qcost, ccost, nq, nc,
                                 cfg["admission_safety_factor"], cfg["reserved_tail_seconds"])
            # Different tasks/images can cost more; this is a conservative heuristic, not a guarantee.
            event("task_admission", task=name, **decision)
            if not decision["admit"]:
                continue
            task_out = out / "baselines" / name
            task_out.mkdir(parents=True, exist_ok=True)
            task_config = dict(selected[name], image_root=assets["image_root"])
            config_path = task_out / "native-config.yaml"
            config_path.write_text(yaml.safe_dump({name: task_config}))
            argv = ["native_eval", "--model_name_or_path", assets["model_path"],
                    "--normalize", "true", "--per_device_eval_batch_size", "1",
                    "--dataloader_num_workers", "0", "--dataset_config", str(config_path),
                    "--encode_output_path", str(task_out)]
            old_argv = sys.argv
            try:
                sys.argv = argv
                ev.main()
            finally:
                sys.argv = old_argv
            pred_path, score_path = task_out / (name + "_pred.jsonl"), task_out / (name + "_score.json")
            infos = [json.loads(line) for line in (task_out / (name + "_info.jsonl")).read_text().splitlines()]
            preds = [json.loads(line) for line in pred_path.read_text().splitlines()]
            if len(infos) != nq or len(preds) != nq:
                raise ValueError("Incomplete full-task native prediction coverage")
            for info, pred in zip(infos, preds):
                label = info["label_name"] if isinstance(info["label_name"], list) else [info["label_name"]]
                if pred["label"] != label or collections.Counter(pred["prediction"]) != collections.Counter(info["cand_names"]):
                    raise ValueError("Native candidate list or label contract changed")
            with (task_out / (name + "_tgt")).open("rb") as f:
                own_candidate_cache = pickle.load(f)  # Only our just-generated in-process artifact.
            if len(own_candidate_cache) != nc:
                raise ValueError("Incomplete native candidate encoding")
            subprocess.run([sys.executable, str(Path(__file__).with_name("replay_native.py")),
                            "--upstream", str(upstream), "--predictions", str(pred_path),
                            "--scores", str(score_path), "--out", str(task_out / "replay.json")],
                           check=True, timeout=max(1, min(300, deadline - time.monotonic())))
            scores = read_json(score_path)
            summary["completed_tasks"].append({"task": name, "scores": scores, "native_queries": nq,
                                               "unique_candidates": nc, "scorer_replay": "matched",
                                               "status": "developmental_baseline_qualification"})
            summary["pending_tasks"].remove(name)
            event("task_completed", task=name, hit_at_1=scores["hit@1"], denominator=nq)
        summary["status"] = "qualification_finished" if not summary["pending_tasks"] else "partial_qualification"
    except BaseException as error:
        failed = True
        summary["status"] = "failed_or_interrupted"
        summary["error_type"] = type(error).__name__
        summary["error"] = str(error)
        traceback.print_exc()
    finally:
        summary["finished_at"] = now()
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(out / "summary.json", summary)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

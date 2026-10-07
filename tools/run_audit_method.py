"""One existing-method audit arm on released native caches; generated_unexecuted.

Run ONLY as an admitted remote run_harness/native-runner job. No dispatch here.
An optional task selects one official group; the design requires all three groups.
"""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embedding_heads.heads import Blocked
from experiments.embedding_heads.audit_controls import registry, fit_audit, audit_features
from experiments.embedding_heads.bundle import (
    validate_manifest, load_training, load_projection, check_original_source, sha256)
from common import now, read_json, write_json
from run_method import evaluate


def parameter_digest(head):
    digest = hashlib.sha256()
    for key in sorted(k for k, v in head.items() if isinstance(v, np.ndarray)):
        value = np.ascontiguousarray(head[key], dtype="<f8")
        digest.update(key.encode() + b"\0" + str(value.shape).encode() + b"\0" + value.tobytes())
    policy = {k: v for k, v in head.items() if not isinstance(v, np.ndarray) and k != "diagnostics"}
    digest.update(json.dumps(policy, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "method", "upstream", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--task", choices=("ScienceQA", "ChartQA", "MSCOCO_i2t"))
    args = parser.parse_args()
    config = read_json(args.config)
    arms = registry(config)
    if args.method not in arms:
        raise ValueError("Arm outside the frozen audit inventory")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"method": args.method, "effective_task": args.task, "status": "RUNNING",
              "started_at": now(), "tasks": [], "scientific_verdict": "NONE", "gate_advanced": False,
              "fit_uses_test_labels": False, "audit_config_sha256": sha256(args.config),
              "source_status": config["status"], "scope": "existing-method developmental audit"}
    write_json(out / "result.json", result)
    code = 0
    try:
        check_original_source(args.upstream)
        manifest = validate_manifest(args.bundle)
        if (manifest["train"]["revision"] != config["training_revision"] or
                manifest["train"]["tasks"] != config["training_tasks"] or
                [t["task"] for t in manifest["evaluation"]] != config["evaluation_tasks"] or
                manifest["train"]["projection_rank"] != config["projection_rank"]):
            raise ValueError("Original train/task/projection identity changed")
        training = load_training(args.bundle, manifest, config["head"]["temperature"])
        fit_started = time.monotonic()
        head = fit_audit(arms[args.method], training, config)
        result.update(setup_seconds=fit_started - started, fit_seconds=time.monotonic() - fit_started,
                      diagnostics=head["diagnostics"], head_kind=head["kind"],
                      parameter_sha256=parameter_digest(head), model_revision=manifest["model_revision"])
        np.savez_compressed(out / "head.npz",
                            **{k: v for k, v in head.items() if isinstance(v, np.ndarray)})
        write_json(out / "head.json",
                   {k: v for k, v in head.items() if not isinstance(v, np.ndarray)})
        write_json(out / "result.json", result)
        projection = load_projection(args.bundle, manifest)
        # Evaluation features and labels are first opened after fit_audit returned.
        selected = [t for t in manifest["evaluation"] if args.task is None or t["task"] == args.task]
        if not selected:
            raise ValueError("Missing required native task")
        transform = lambda full, side: audit_features(head, projection, full, side)
        for task in selected:
            result["tasks"].append(evaluate(task, head, args.bundle, manifest, projection,
                                           args.upstream, out, feature_builder=transform))
            write_json(out / "result.json", result)
        result["status"] = "DEVELOPMENTAL_SCORED"
    except Blocked as error:
        result.update(status="BLOCKED", error=str(error), error_type=type(error).__name__)
        code = 20
        print(str(error), file=sys.stderr, flush=True)
    except Exception as error:
        result.update(status="FAILED", error=str(error), error_type=type(error).__name__)
        code = 1
        traceback.print_exc()
    finally:
        result.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
        write_json(out / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

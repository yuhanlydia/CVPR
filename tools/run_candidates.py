"""Finite r003 batch: optional one preparation, then 12 methods + 6 controls.
Use the complete installed Research Autopilot native runner. Never vendor it.
Ordinary method errors/individual timeouts continue; cancellation/global budget stop.
"""
import argparse
import contextlib
import fcntl
import hashlib
import math
import os
import signal
import sys
import time
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import digest, git, now, read_json, run_plan_compat, write_json
from window_budget import ensure_window, remaining_seconds
from run_window import skill_root
from experiments.embedding_heads.bundle import validate_manifest, all_refs, referenced, check_original_source, UPSTREAM

def inventory(config):
    selected, parked, controls = config["selected"], config["parked"], config["controls"]
    if (len(selected) != 12 or len(parked) != 8 or len(set(selected+parked)) != 20
            or set(selected+parked) != {f"I{i:02d}" for i in range(1, 21)}
            or len(set(controls)) != len(controls) or set(selected+parked) & set(controls)
            or config["max_method_and_control_attempts"] != len(selected+controls)
            or config["max_retries_per_trial"] != 0):
        raise ValueError("Invalid frozen 20 -> 12 inventory or attempt ceiling")
    numeric = ["prepare_timeout_seconds", "attempt_timeout_seconds", "reserved_tail_seconds"]
    for key in numeric:
        if isinstance(config[key], bool) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Invalid finite queue budget")
    for key in ("temperature", "ridge", "learning_rate", "bandwidth", "fisher_floor", "huber_delta", "dro_temperature"):
        value = config["head"][key]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid head parameter")
    return controls+selected

def claim_batch(root, window, run_id):
    """An explicitly added finite scope, within the ORIGINAL retained clock."""
    directory = Path(root)/"runs/windows"/window["window_id"]
    legacy = directory/"attempt.json"
    if legacy.exists():
        prior = read_json(legacy)
        receipt = Path(root)/"runs/attempts"/prior["run_id"]/"receipt.json"
        if not receipt.is_file() or read_json(receipt)["status"] not in {"completed", "failed", "budget_exhausted", "interrupted"}:
            raise RuntimeError("Earlier attempt is active/unknown; reconcile it before launching")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"candidate-batch.json").open("x", encoding="utf-8") as stream:
        import json
        json.dump({"run_id": run_id, "window_id": window["window_id"], "started_epoch": window["started_epoch"],
                   "max_preparation_attempts": 1, "max_method_and_control_attempts": 18, "retry_count": 0}, stream)
        stream.flush(); os.fsync(stream.fileno())

class BudgetEnded(KeyboardInterrupt):
    pass

@contextlib.contextmanager
def absolute_budget(seconds):
    """Also bound hashing/staging between native subprocess watchdogs (Linux)."""
    previous = signal.getsignal(signal.SIGALRM)
    def expire(_signal, _frame):
        raise BudgetEnded("Original cumulative window exhausted")
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

def reflect_attempts(root, expected, receipt):
    """Execution exit/timeout overrides a child report; never turn a crash into success."""
    statuses = {name: {"method": name, "status": "CARRYOVER_NOT_STARTED"} for name in expected}
    for attempt in receipt.get("attempts", []):
        name = attempt["trial_id"]
        if name not in statuses:
            continue
        record = {"method": name, "native_status": attempt["status"], "exit_code": attempt["exit_code"],
                  "attempt_path": attempt["attempt_path"], "seconds": attempt["seconds"],
                  "status": {"timeout": "TIMEOUT", "interrupted": "INTERRUPTED"}.get(attempt["status"], "FAILED")}
        output = Path(attempt["cwd"])/"out/result.json"
        try:
            output.resolve().relative_to(Path(root).resolve())
            ref = next((r for r in attempt["output_refs"] if (Path(root)/r["path"]).resolve() == output.resolve()), None)
            if ref and output.is_file() and digest(output) == ref["sha256"]:
                child = read_json(output)
                if child["method"] != name:
                    raise ValueError("Child method identity changed")
                record["result"] = child
                if attempt["status"] == "completed" and attempt["exit_code"] == 0 and child["status"] == "DEVELOPMENTAL_SCORED":
                    record["status"] = "DEVELOPMENTAL_SCORED"
                elif attempt["status"] == "failed" and attempt["exit_code"] == 20 and child["status"] == "BLOCKED":
                    record["status"] = "BLOCKED"
        except (OSError, ValueError, KeyError, TypeError) as error:
            record["artifact_error"] = str(error)
        statuses[name] = record
    return list(statuses.values())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", help="Existing verified manifest within this checkout; skips preparation")
    p.add_argument("--baseline-out", help="Our native r001 attempt's workspace/out directory")
    p.add_argument("--train-image-root", help="Existing original MMEB training images, containing images/...")
    p.add_argument("--skill-root")
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if not math.isfinite(a.hours) or not 0 < a.hours <= 8:
        raise ValueError("One retained cumulative window, at most eight hours")
    root = Path(__file__).resolve().parent.parent
    cfg_path = root/"configs/candidates.json"; cfg = read_json(cfg_path); names = inventory(cfg)
    window = ensure_window(root)
    run_id = "r003-"+time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())+"-"+uuid.uuid4().hex[:8]
    batch_root = root/"runs/candidate-batches"/run_id
    summary = {"run_id": run_id, "status": "PLANNED", "window": window, "started_at": now(),
               "inventory": names, "selected_candidates": cfg["selected"], "parked_candidates": cfg["parked"],
               "methods": [{"method": m, "status": "PENDING"} for m in names], "native_run_ids": [],
               "scientific_verdict": "NONE", "gate_advanced": False, "gpu_model_execution_claim": "READ_ACTUAL_RECEIPTS"}
    if not a.execute:
        print(f"Inventory: 20 candidates -> 12 prototypes + {len(cfg['controls'])} controls; retries=0.")
        print(f"Remaining ORIGINAL clock: {remaining_seconds(window, a.hours):.0f}s.")
        print("Execution requires --execute and an existing bundle, or --baseline-out plus --train-image-root.")
        return
    batch_root.mkdir(parents=True, exist_ok=False)
    def save():
        write_json(batch_root/"summary.json", summary)
        write_json(root/"runs/latest-candidates.json", {"run_id": run_id, "summary": str(batch_root/"summary.json")})
    save()
    print("Batch ledger:", batch_root / "summary.json", flush=True)
    try:
        installed = skill_root(a.skill_root); sys.path.insert(0, str(installed/"scripts"))
        import run_experiments as native
        summary["installed_native_runner_sha256"] = digest(installed/"scripts/run_experiments.py")
        if git(root, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("Commit tracked edits before freezing code")
        upstream = root/"sources/Qwen3-VL-Embedding"
        if git(upstream, "rev-parse", "HEAD") != UPSTREAM or git(upstream, "status", "--porcelain"):
            raise RuntimeError("Pinned clean upstream checkout required")
        check_original_source(upstream)
        seconds = remaining_seconds(window, a.hours)-cfg["reserved_tail_seconds"]
        if seconds <= 0:
            raise BudgetEnded("No remaining original window budget")
        end = time.monotonic()+seconds
        device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        lock_path = Path("/tmp")/("cvpr-gpu-"+hashlib.sha256(device.encode()).hexdigest()[:16]+".lock")
        with lock_path.open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            claim_batch(root, window, run_id)
            summary["status"] = "RUNNING"; save()
            with absolute_budget(seconds):
                from host_check import inspect
                host = inspect(root); host_path = batch_root/"host.json"; write_json(host_path, host)
                def ref(path):
                    path = Path(path).resolve(); relative = path.relative_to(root).as_posix()
                    return {"path": relative, "sha256": digest(path)}
                code = sorted((root/"tools").glob("*.py"))
                code += sorted((root/"experiments/embedding_heads").glob("*.py"))
                code += [v for v in upstream.rglob("*") if v.is_file() and ".git" not in v.parts
                         and (v.suffix in {".py", ".yaml", ".yml"} or v.name == "LICENSE")]
                code_refs = [ref(v) for v in code]
                def run_stage(suffix, jobs, per_attempt):
                    remaining = min(end-time.monotonic(), remaining_seconds(window, a.hours)-cfg["reserved_tail_seconds"])
                    if remaining <= 0: raise BudgetEnded("Cumulative budget exhausted")
                    plan = native.make_plan(root, run_id=run_id+"-"+suffix, jobs=jobs,
                        provenance={"git_revision": git(root, "rev-parse", "HEAD"), "upstream_revision": UPSTREAM,
                                    "environment_digest": digest(host_path), "environment_refs": [ref(host_path)],
                                    "model_revision": "ACTUAL_PIN_IN_BUNDLE_OR_PREPARATION_RECEIPT",
                                    "data_revision": cfg["training_revision"],
                                    "scientific_claim": "NONE; user-directed developmental prototypes"},
                        limits={"max_attempts": len(jobs), "max_development_trials": len(jobs),
                                "max_confirmation_trials": 0, "max_retries_per_trial": 0,
                                "wall_time_seconds": remaining, "attempt_timeout_seconds": min(per_attempt, remaining)},
                        purpose="engineering", evidence_mode="developmental")
                    write_json(batch_root/(suffix+"-plan.json"), plan)
                    summary["native_run_ids"].append(plan["run_id"]); save()
                    result = run_plan_compat(native, root, plan,
                        authorizer=lambda scope: scope.get("run_id") == plan["run_id"] and scope.get("plan_digest") == plan["plan_digest"],
                        process_fds=(lease.fileno(),))
                    # The native runner handles KeyboardInterrupt while killing its
                    # process group. Distinguish our expired alarm from cancellation.
                    if time.monotonic() >= end:
                        raise BudgetEnded("Original cumulative deadline reached during native execution")
                    return result
                common_inputs = [ref(cfg_path), ref(host_path)]
                if a.bundle:
                    bundle = Path(a.bundle).resolve(); bundle.relative_to(root)
                else:
                    if not a.baseline_out or not a.train_image_root:
                        raise RuntimeError("Provide real native baseline output and existing original training images")
                    baseline = Path(a.baseline_out).resolve(); relative = baseline.relative_to(root)
                    # Stage the exact actual cache/receipt; no unknown future-input hashes.
                    cache_files = [baseline/"summary.json", baseline/"assets/assets.json", baseline.parent.parent/"attempt.json"]
                    for name in cfg["evaluation_tasks"]:
                        taskroot = baseline/"baselines"/name
                        cache_files += [taskroot/(name+tail) for tail in ("_qry", "_tgt", "_info.jsonl", "_pred.jsonl", "_score.json")]
                        cache_files += [taskroot/"replay.json", taskroot/"native-config.yaml", baseline/"native-samples"/(name+".jsonl")]
                    metadata_refs = []
                    metadata_arg = []
                    metadata_setting = cfg.get("training_metadata_root")
                    if metadata_setting:
                        metadata_root = (root/metadata_setting).resolve()
                        metadata_root.relative_to(root)
                        metadata_files = sorted(metadata_root.glob("**/*.parquet"))
                        if not metadata_files:
                            raise RuntimeError(f"No local training parquet shards under {metadata_root}")
                        metadata_refs = [ref(path) for path in metadata_files]
                        metadata_arg = ["--train-metadata-root", metadata_setting]
                    job = {"trial_id": "prepare", "command": [sys.executable, str(root/"tools/prepare_candidate_bundle.py"),
                        "--config", str(cfg_path), "--baseline-out", relative.as_posix(),
                        "--train-image-root", str(Path(a.train_image_root).resolve()), "--upstream", "sources/Qwen3-VL-Embedding", "--out", "out"] + metadata_arg,
                        "cwd": ".", "input_refs": common_inputs+metadata_refs+[ref(v) for v in cache_files], "code_refs": code_refs,
                        "output_paths": ["out/manifest.json"], "seed": cfg["seed"], "group": "prepare-native-feature-assets", "arm_role": "baseline"}
                    receipt = run_stage("prepare", [job], cfg["prepare_timeout_seconds"])
                    attempt = receipt["attempts"][0] if receipt["attempts"] else None
                    if not attempt or attempt["status"] != "completed" or attempt["exit_code"] != 0:
                        summary["status"] = "BLOCKED_SHARED_INPUT"
                        summary["methods"] = [{"method": m, "status": "BLOCKED_SHARED_INPUT", "preparation_receipt": receipt["run_id"]} for m in names]
                        save(); return 1
                    bundle = Path(attempt["cwd"])/"out/manifest.json"
                manifest = validate_manifest(bundle)
                producer = bundle.parent.parent.parent/"attempt.json"
                bundle_inputs = [ref(bundle), ref(producer)]+[ref(referenced(bundle.parent, v)) for v in all_refs(manifest)]
                jobs = [{"trial_id": m, "command": [sys.executable, str(root/"tools/run_method.py"),
                        "--bundle", str(bundle), "--config", str(cfg_path), "--method", m,
                        "--upstream", "sources/Qwen3-VL-Embedding", "--out", "out"],
                         "cwd": ".", "input_refs": common_inputs+bundle_inputs, "code_refs": code_refs,
                         "output_paths": ["out/result.json", "out/head.json", "out/head.npz"]+
                            ["out/"+t["task"]+suffix for t in manifest["evaluation"]
                             for suffix in ("_pred.jsonl", "_score.json", "_replay.json")],
                         "seed": cfg["seed"], "group": "native-development-heads",
                         "arm_role": "baseline" if m in cfg["controls"] else "method"} for m in names]
                receipt = run_stage("heads", jobs, cfg["attempt_timeout_seconds"])
                summary["methods"] = reflect_attempts(root, names, receipt)
                summary["status"] = receipt["status"].upper()
                summary["bundle"] = str(bundle)
    except BudgetEnded as error:
        summary.update(status="CARRYOVER_BUDGET", error=str(error))
    except KeyboardInterrupt:
        summary.update(status="INTERRUPTED", error="User cancellation; do not automatically restart")
    except Exception as error:
        summary.update(status="SETUP_OR_SHARED_INPUT_FAILED", error_type=type(error).__name__, error=str(error))
        summary["methods"] = [{"method": m, "status": "BLOCKED_SHARED_INPUT", "reason": str(error)} for m in names]
    finally:
        # Recover native attempt records even if interrupted during staging/receipt writing.
        for native_id in summary["native_run_ids"]:
            runroot = root/"runs/attempts"/native_id
            records = []
            for path in sorted(runroot.glob("*/attempt.json")):
                try:
                    records.append(read_json(path))
                except (OSError, ValueError) as error:
                    summary.setdefault("receipt_read_errors", []).append({"path": str(path), "error": str(error)})
            if records and any(r["trial_id"] in names for r in records):
                summary["methods"] = reflect_attempts(root, names, {"attempts": records})
        summary["methods"] = [{**r, "status": "CARRYOVER_NOT_STARTED" if r["status"] == "PENDING" else r["status"]}
                              for r in summary["methods"]]
        summary.update(finished_at=now(), remaining_original_seconds=remaining_seconds(window, a.hours))
        save()
    print(f"Batch: {batch_root/'summary.json'}; {summary['status']}; scientific gates unchanged.")
    return 0 if all(r["status"] == "DEVELOPMENTAL_SCORED" for r in summary["methods"]) else 1

if __name__ == "__main__":
    raise SystemExit(main())


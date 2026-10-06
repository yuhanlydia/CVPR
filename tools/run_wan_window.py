"""Run the Wan engineering qualification in one native, finite attempt."""
import argparse
import fcntl
import hashlib
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from common import digest, git, run_plan_compat, write_json
from window_budget import ensure_window, remaining_seconds, claim_attempt
from run_window import skill_root

SOURCE_PINS = {"WAN_ROOT": "9737cba9c1c3c4d04b33fcad41c111989865d315",
               "VBENCH_ROOT": "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not math.isfinite(args.hours) or not 0 < args.hours <= 7.5:
        raise ValueError("Wan model time is bounded by at most 7.5 hours")
    root = Path(__file__).resolve().parent.parent
    window = ensure_window(root)
    skill = skill_root()
    sys.path.insert(0, str(skill / "scripts"))
    import run_experiments as native
    sources = {}
    for name, pin in SOURCE_PINS.items():
        path = Path(os.environ[name]).resolve()
        relative = path.relative_to(root)
        if git(path, "rev-parse", "HEAD") != pin or git(path, "status", "--porcelain"):
            raise RuntimeError(f"{name} must be a clean checkout at its pinned commit inside this repository")
        sources[name] = (path, relative)
    checkpoint = Path(os.environ["WAN_CKPT"]).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Commit tracked edits before running a frozen batch")
    commit = git(root, "rev-parse", "HEAD")
    # Host inspection is finite and the retained clock already includes it.
    import subprocess
    host = root / "runs/wan-host.json"
    host.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(root / "scripts/inspect_host.py"),
                    "--output", str(host)], check=True, timeout=90)
    wan, wan_rel = sources["WAN_ROOT"]
    vbench, vbench_rel = sources["VBENCH_ROOT"]
    prompt_json = vbench / "vbench/VBench_full_info.json"
    def ref(path):
        return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}
    code = list((root / "experiments/wan_cache").glob("*.py"))
    code += list((root / "tools").glob("*.py"))
    code += [path for path in wan.rglob("*") if path.is_file() and ".git" not in path.parts
             and path.suffix in {".py", ".json", ".yaml", ".yml"}]
    code_refs = [ref(path) for path in sorted(set(code))]
    run_id = "r002-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    command = [sys.executable, str(root / "experiments/wan_cache/run_job.py"),
               "--source-wan-root", str(wan), "--source-vbench-root", str(vbench),
               "--wan-root", wan_rel.as_posix(), "--ckpt-dir", str(checkpoint),
               "--vbench-json", prompt_json.relative_to(root).as_posix(), "--output-dir", "out"]
    for option, env, default in (
        ("num-prompts", "NUM_PROMPTS", "1"), ("sampling-steps", "SAMPLING_STEPS", "50"),
        ("frame-num", "FRAME_NUM", "81"), ("width", "WIDTH", "832"),
        ("height", "HEIGHT", "480"), ("shift", "SHIFT", "8"),
        ("guide-scale", "GUIDE_SCALE", "6"), ("base-seed", "BASE_SEED", "20261006")):
        command += ["--" + option, os.environ.get(env, default)]
    command += ["--force-steps", os.environ.get("FORCE_STEPS", "5")]
    # Reserve the end of the original eight-hour clock for receipts/handoff.
    seconds = min(args.hours * 3600, remaining_seconds(window) - 120)
    if seconds < 120:
        raise RuntimeError("Retained window exhausted; preserve receipts before another reviewed window")
    command += ["--max-wall-hours", str(seconds / 3600)]
    plan = native.make_plan(root, run_id=run_id, purpose="engineering", evidence_mode="developmental",
        provenance={"git_revision": commit, "upstream_revision": SOURCE_PINS["WAN_ROOT"],
                    "model_revision": "37ec512624d61f7aa208f7ea8140a131f93afc9a; content checked in bounded preflight",
                    "data_revision": SOURCE_PINS["VBENCH_ROOT"], "environment_digest": digest(host),
                    "environment_refs": [ref(host)], "scientific_claim": "NONE; native VBench scoring pending"},
        jobs=[{"trial_id": "wan-qualification", "command": command, "cwd": ".",
               "input_refs": [ref(prompt_json), ref(host), ref(root / "research/ROUND_002_WAN_CACHE_PROPAGATION.md"),
                              ref(root / "research/SOURCES_ROUND_002.md")],
               "code_refs": code_refs, "output_paths": ["out/preflight.json", "out/manifest.json", "out/summary.json", "out/metrics.jsonl"],
               "seed": int(os.environ.get("BASE_SEED", "20261006")), "group": "wan-runtime-qualification", "arm_role": "baseline"}],
        limits={"max_attempts": 1, "max_development_trials": 1, "max_confirmation_trials": 0,
                "max_retries_per_trial": 0, "wall_time_seconds": seconds, "attempt_timeout_seconds": seconds})
    plan_path = root / "runs/plans" / (run_id + ".json")
    write_json(plan_path, plan)
    print(f"Engineering plan {run_id}; {seconds:.0f}s including preflight and model loading; digest {plan['plan_digest']}", flush=True)
    if not args.execute:
        print("Plan validation only; no model process started.")
        return
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    lock_path = Path("/tmp") / ("cvpr-gpu-" + hashlib.sha256(device.encode()).hexdigest()[:16] + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        claim_attempt(root, window, run_id)
        result = run_plan_compat(native, root, plan,
            authorizer=lambda scope: scope.get("run_id") == run_id and scope.get("plan_digest") == plan["plan_digest"],
            process_fds=(lock.fileno(),))
    # Copy only small raw evidence out of the native attempt. Do not re-score.
    destination = root / "artifacts/round_002" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(host, destination / "host.json")
    shutil.copy2(plan_path, destination / "plan.json")
    write_json(destination / "native-receipt.json", result)
    exit_code = -1
    for attempt in result["attempts"]:
        attempt_root = root / attempt["attempt_path"]
        shutil.copy2(attempt_root / "stdout.log", destination / "run.log")
        shutil.copy2(attempt_root / "stderr.log", destination / "stderr.log")
        shutil.copy2(attempt_root / "attempt.json", destination / "attempt.json")
        exit_code = attempt["exit_code"] if attempt["exit_code"] is not None else -1
        output = attempt_root / "workspace/out"
        if output.exists():
            for path in output.rglob("*"):
                relative = path.relative_to(output)
                if path.is_file() and not path.is_symlink() and "large" not in relative.parts and path.suffix in {".json", ".jsonl"}:
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
    sys.path.insert(0, str(root / "experiments/wan_cache"))
    write_json(destination / "evidence-manifest.json", {
        "run_id": run_id, "source_commit": commit,
        "files": [{"path": p.relative_to(destination).as_posix(), "sha256": digest(p)}
                  for p in sorted(destination.rglob("*")) if p.is_file()]})
    from make_result_packet import write_packet
    status = write_packet(destination, commit, exit_code, result["status"])
    write_json(root / "runs/latest-wan.json", {"run_id": run_id, "native_status": result["status"],
                                               "packet_status": status, "evidence": destination.relative_to(root).as_posix()})
    print(f"{status}: {destination.relative_to(root)}/RESULT.md; scientific verdict NONE", flush=True)
    if status != "COMPLETE_ENGINEERING":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

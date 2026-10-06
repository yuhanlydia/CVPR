"""Use the installed Research Autopilot native bounded runner; qualify baselines.

No private skill code is republished. --execute authorizes this finite scope.
The registered plan is engineering/developmental, with no scientific gate claim.
"""
import argparse
import fcntl
import hashlib
import os
import sys
import time
import uuid
from pathlib import Path
from common import digest, git, now, read_json, run_plan_compat, write_json
from window_budget import ensure_window, remaining_seconds, claim_attempt


def skill_root(explicit=None):
    candidates = [explicit, os.environ.get("RESEARCH_AUTOPILOT_ROOT"),
                  Path.home() / ".agents/skills/research-autopilot",
                  Path.home() / ".codex/skills/research-autopilot"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "scripts/run_experiments.py").is_file():
            return Path(candidate).resolve()
    raise RuntimeError("Installed Research Autopilot required; set RESEARCH_AUTOPILOT_ROOT to its complete folder")


def build_plan(root, runner, run_id, seconds, image_root=None):
    cfg = read_json(root / "configs/batch.json")
    upstream = root / "sources/Qwen3-VL-Embedding"
    if git(upstream, "rev-parse", "HEAD") != cfg["upstream_commit"] or git(upstream, "status", "--porcelain"):
        raise RuntimeError("Pinned upstream checkout is missing, dirty or at a different commit")
    code_files = sorted((root / "tools").glob("*.py"))
    code_files += [p for p in upstream.rglob("*") if p.is_file() and ".git" not in p.parts
                   and (p.suffix in {".py", ".yaml", ".yml"} or p.name == "LICENSE")]
    def ref(path):
        return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}
    command = [sys.executable, str(root / "tools/qualify.py"),
               "--config", str(root / "configs/batch.json"),
               "--upstream", "sources/Qwen3-VL-Embedding", "--out", "out",
               "--seconds", str(seconds)]
    if image_root:
        command += ["--image-root", str(Path(image_root).resolve())]
    host = root / "runs/host.json"
    return runner.make_plan(root, run_id=run_id,
        provenance={"git_revision": git(root, "rev-parse", "HEAD"),
                    "model_revision": "RESOLVED_AND_PINNED_DURING_BOUNDED_ACQUISITION",
                    "data_revision": "RESOLVED_AND_PINNED_DURING_BOUNDED_ACQUISITION",
                    "environment_digest": digest(host), "environment_refs": [ref(host)],
                    "upstream_revision": cfg["upstream_commit"],
                    "scientific_claim": "NONE; baseline/resource qualification only"},
        jobs=[{"trial_id": "qualify", "command": command, "cwd": ".",
               "input_refs": [ref(root / "configs/batch.json"), ref(host)],
               "code_refs": [ref(p) for p in code_files],
               "output_paths": ["out/summary.json", "out/events.jsonl", "out/assets/assets.json"],
               "seed": cfg["seed"], "group": "native-baseline-qualification", "arm_role": "baseline"}],
        limits={"max_attempts": 1, "max_development_trials": 1, "max_confirmation_trials": 0,
                "max_retries_per_trial": 0, "wall_time_seconds": seconds,
                "attempt_timeout_seconds": seconds}, purpose="engineering", evidence_mode="developmental")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skill-root")
    p.add_argument("--image-root", help="Existing original MMEB image directory; content hashes are recorded")
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if not 0 < a.hours <= 8:
        raise ValueError("This round is authorized for at most one eight-hour window")
    root = Path(__file__).resolve().parent.parent
    # Native harness code is imported from the actual complete installation.
    skill = skill_root(a.skill_root)
    sys.path.insert(0, str(skill / "scripts"))
    import run_experiments as native
    from host_check import inspect
    host = inspect(root)
    write_json(root / "runs/host.json", host)
    window = ensure_window(root)
    seconds = remaining_seconds(window, a.hours)
    if seconds < 600:
        raise RuntimeError("Current setup/window budget exhausted; preserve receipts and review before another window")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Commit tracked local edits before execution; no silent dirty-source run")
    run_id = "r001-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    plan = build_plan(root, native, run_id, seconds, a.image_root)
    plan_path = root / "runs/plans" / (run_id + ".json")
    write_json(plan_path, plan)
    print(f"Pinned plan: {plan_path}; digest: {plan['plan_digest']}; seconds remaining: {seconds:.0f}", flush=True)
    if not a.execute:
        print("Validated only. Add --execute to run this finite qualification scope.")
        return
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    # The native watchdog inherits this whole-device lock; it remains held if the parent dies.
    lock_path = Path("/tmp") / ("cvpr-gpu-" + hashlib.sha256(device.encode()).hexdigest()[:16] + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            claim_attempt(root, window, run_id)
        except FileExistsError as error:
            raise RuntimeError("This retained window already launched its one attempt; preserve it and review the result") from error
        result = run_plan_compat(native, root, plan,
            authorizer=lambda scope: scope.get("run_id") == run_id and scope.get("plan_digest") == plan["plan_digest"],
            process_fds=(lock.fileno(),))
    write_json(root / "runs/latest.json", {"run_id": run_id, "status": result["status"],
                                          "completed_at": now(), "receipt": f"runs/attempts/{run_id}/receipt.json"})
    print(f"Window status: {result['status']}; run: {run_id}; scientific gates unchanged.")
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

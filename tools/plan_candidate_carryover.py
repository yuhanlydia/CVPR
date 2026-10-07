"""Read-only audit of r003's retained budget and never-started inventory.

This is a planning report, not a native runner plan or a resume executor. It
never creates a setup receipt, changes a batch marker, imports a model/scorer,
retries a trial, or launches a process. Native plans/receipts remain authoritative.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

from common import digest, now, read_json
from window_budget import remaining_seconds

IDENTITY = re.compile(r"[A-Za-z0-9_-]+\Z")
TERMINAL = {"completed", "failed", "timeout", "interrupted", "budget_exhausted"}
LIMITS = {"window_seconds": 28800, "prepare_timeout_seconds": 7200,
          "attempt_timeout_seconds": 600, "reserved_tail_seconds": 120,
          "max_retries_per_trial": 0, "max_preparation_attempts": 1,
          "max_method_and_control_attempts": 18}


def contained(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Path must be project-relative: " + str(relative))
    path = root / relative
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Symlinked evidence path: " + str(relative))
    path.resolve().relative_to(root)
    return path


def canonical_digest(plan):
    value = {key: val for key, val in plan.items() if key != "plan_digest"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def check_bundle_protocol(manifest, producer, cfg, config_sha):
    """A prior manifest must have been produced for this exact frozen config."""
    refs = [ref for ref in producer["input_refs"] if ref["path"] == "configs/candidates.json"]
    if len(refs) != 1 or refs[0]["sha256"] != config_sha:
        raise ValueError("Bundle producer used a different or unbound configuration")
    train = manifest["train"]
    if (train["dataset_id"] != cfg["training_dataset"] or train["revision"] != cfg["training_revision"]
            or train["split"] != cfg["training_split"] or train["tasks"] != cfg["training_tasks"]
            or [task["task"] for task in manifest["evaluation"]] != cfg["evaluation_tasks"]
            or manifest["teacher"]["temperature"] != cfg["head"]["temperature"]):
        raise ValueError("Bundle training/evaluation protocol changed")
    selected = train["selected_released_row_indices"]
    if set(selected) != set(cfg["training_tasks"]) or any(
            len(selected[task]) != cfg["training_rows_per_task"] for task in cfg["training_tasks"]):
        raise ValueError("Bundle released-row selection differs from frozen training scope")


def audit(root, run_id=None, hours=8, epoch=None):
    root = Path(root).resolve()
    current = time.time() if epoch is None else epoch
    report = {"report_type": "r003-carryover-file-audit-v1", "observed_at": now(),
              "run_id": run_id, "status": "BLOCKED", "blockers": [], "native_runs": [],
              "unstarted_inventory": [], "eligible_head_inventory": [], "attempts": [],
              "retained_attempt_inventory": [],
              "dispatch_supported": False, "resume_command": None,
              "scientific_verdict": "NONE", "gpu_usage_seconds": None,
              "native_schema_check": "STATIC_FILE_BINDINGS_ONLY; no private runner execution",
              "scope": "Retained files only; no process, GPU, scorer or scientific verification"}
    observed = {}

    def blocker(code, detail):
        report["blockers"].append({"code": code, "detail": str(detail)})

    def load(relative):
        path = contained(root, relative)
        before = digest(path)
        value = read_json(path)
        if digest(path) != before:
            raise ValueError("Evidence changed while reading: " + str(relative))
        observed[path.relative_to(root).as_posix()] = before
        return value

    def check_ref(ref):
        path = contained(root, ref["path"])
        expected = ref["sha256"]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid SHA256 reference: " + str(ref["path"]))
        if digest(path) != expected:
            raise ValueError("Missing or changed pinned file: " + str(ref["path"]))
        observed[path.relative_to(root).as_posix()] = expected

    try:
        if run_id is None:
            run_id = load("runs/latest-candidates.json")["run_id"]
            report["run_id"] = run_id
        if not isinstance(run_id, str) or not IDENTITY.fullmatch(run_id):
            raise ValueError("Invalid existing batch ID")
        if isinstance(hours, bool) or not math.isfinite(hours) or not 0 < hours <= 8:
            raise ValueError("Retained cumulative cap must be within eight hours")
        cfg = load("configs/candidates.json")
        for key, value in LIMITS.items():
            if isinstance(cfg[key], bool) or cfg[key] != value:
                raise ValueError("Frozen resource limit changed: " + key)
        names = cfg["controls"] + cfg["selected"]
        if (len(cfg["controls"]) != 6 or len(cfg["selected"]) != 12 or len(set(names)) != 18
                or len(cfg["parked"]) != 8
                or set(cfg["selected"] + cfg["parked"]) != {f"I{i:02d}" for i in range(1, 21)}):
            raise ValueError("Frozen candidate/control inventory changed")
        batch = Path("runs/candidate-batches") / run_id
        summary = load(batch / "summary.json")
        window = load("setup-receipt.json")
        window_id = window.get("window_id") or hashlib.sha256(
            str(window["started_epoch"]).encode()).hexdigest()[:32]
        if not isinstance(window_id, str) or not re.fullmatch(r"[0-9a-f]{32}", window_id):
            raise ValueError("Invalid retained window identity")
        marker = Path("runs/windows") / window_id
        claim = load(marker / "candidate-batch.json")
        if (summary["run_id"] != run_id or summary["inventory"] != names
                or summary["window"]["started_epoch"] != window["started_epoch"]
                or summary["window"]["window_id"] != window_id
                or claim["run_id"] != run_id or claim["window_id"] != window_id
                or claim["started_epoch"] != window["started_epoch"]
                or claim["max_preparation_attempts"] != 1
                or claim["max_method_and_control_attempts"] != 18 or claim["retry_count"] != 0):
            raise ValueError("Original batch/marker/clock identity does not match")
        remaining = remaining_seconds(window, hours, epoch=current)
        report["budget"] = {"window_id": window_id, "started_epoch": window["started_epoch"],
                            "cumulative_cap_seconds": hours * 3600,
                            "original_clock_elapsed_seconds": max(0, current - window["started_epoch"]),
                            "remaining_original_seconds": remaining,
                            "reserved_tail_seconds": 120,
                            "remaining_dispatch_seconds": max(0, remaining - 120),
                            "candidate_native_elapsed_seconds": 0,
                            "limits": LIMITS.copy(), "automatic_retries": 0}
        if remaining <= 120:
            blocker("ORIGINAL_BUDGET_EXHAUSTED", "No dispatch budget after the retained 120-second tail")
        if summary["status"] in {"RUNNING", "PLANNED"}:
            blocker("BATCH_ACTIVE_OR_UNKNOWN", summary["status"])
        # Reconcile the shared r001/Wan marker without assuming that its process ended.
        if contained(root, marker / "attempt.json").is_file():
            earlier = load(marker / "attempt.json")["run_id"]
            if not isinstance(earlier, str) or not IDENTITY.fullmatch(earlier):
                raise ValueError("Invalid earlier shared run identity")
            receipt = load(Path("runs/attempts") / earlier / "receipt.json")
            if receipt["run_id"] != earlier or receipt["status"] not in TERMINAL - {"timeout"}:
                blocker("EARLIER_SHARED_RUN_ACTIVE_OR_UNKNOWN", earlier)
        native_ids = summary["native_run_ids"]
        if not isinstance(native_ids, list) or len(set(native_ids)) != len(native_ids):
            raise ValueError("Duplicate or invalid native run inventory")
        for native_id in native_ids:
            if (not isinstance(native_id, str) or not IDENTITY.fullmatch(native_id)
                    or not native_id.startswith(run_id + "-")):
                raise ValueError("Native run belongs to another batch")
        attempts_root = contained(root, "runs/attempts")
        actual_ids = {p.name for p in attempts_root.glob(run_id + "-*") if p.is_dir()}
        if actual_ids != set(native_ids):
            blocker("UNRECONCILED_NATIVE_RUNS", {"listed": native_ids, "on_disk": sorted(actual_ids)})
        consumed = {}
        observed_trials = set()
        preparation = []
        for native_id in native_ids:
            native_path = Path("runs/attempts") / native_id
            # Preserve failed/partial records even if a later compatibility check fails.
            for attempt_file in contained(root, native_path).glob("*/attempt.json"):
                try:
                    record = load(attempt_file.relative_to(root))
                    report["retained_attempt_inventory"].append({
                        "trial_id": record.get("trial_id"), "native_status": record.get("status"),
                        "exit_code": record.get("exit_code"), "path": attempt_file.relative_to(root).as_posix(),
                        "binding_verified": False})
                    if record.get("trial_id") in names + ["prepare"]:
                        if record["trial_id"] in observed_trials:
                            blocker("LIFETIME_TRIAL_CEILING_EXCEEDED", record["trial_id"])
                        observed_trials.add(record["trial_id"])
                except (OSError, ValueError, KeyError, TypeError) as error:
                    blocker("UNREADABLE_RETAINED_ATTEMPT", error)
            try:
                plan = load(native_path / "plan.json")
                receipt = load(native_path / "receipt.json")
                if (plan["schema_id"] != "experiment-run-plan"
                        or receipt["schema_id"] != "experiment-run-receipt"
                        or plan["schema_version"] != "1.0.0"
                        or receipt["schema_version"] != plan["schema_version"]
                        or plan["run_id"] != native_id or receipt["run_id"] != native_id
                        or plan["plan_digest"] != canonical_digest(plan)
                        or receipt["plan_digest"] != plan["plan_digest"]
                        or plan["purpose"] != "engineering" or plan["evidence_mode"] != "developmental"
                        or receipt["purpose"] != plan["purpose"]
                        or receipt["evidence_mode"] != plan["evidence_mode"]
                        or plan.get("protocol_ref") is not None or plan.get("protocol_digest") is not None
                        or receipt["status"] not in TERMINAL - {"timeout"}):
                    raise ValueError("Native plan/terminal receipt binding changed")
                jobs = {job["trial_id"]: job for job in plan["jobs"]}
                if len(jobs) != len(plan["jobs"]) or not jobs or not set(jobs) <= set(names + ["prepare"]):
                    raise ValueError("Native plan exceeds the frozen trial inventory")
                if "prepare" in jobs and len(jobs) != 1:
                    raise ValueError("Preparation must remain a separate single attempt")
                ceiling = 7200 if "prepare" in jobs else 600
                limits = plan["limits"]
                if (limits["max_retries_per_trial"] != 0 or limits["max_attempts"] != len(jobs)
                        or limits["max_development_trials"] != len(jobs)
                        or limits["max_confirmation_trials"] != 0
                        or not 0 < limits["attempt_timeout_seconds"] <= ceiling
                        or not limits["attempt_timeout_seconds"] <= limits["wall_time_seconds"] <= 28800):
                    raise ValueError("Native plan exceeds retained per-attempt bounds")
                for job in jobs.values():
                    config_refs = [ref for ref in job["input_refs"] if ref["path"] == "configs/candidates.json"]
                    if len(config_refs) != 1 or job["seed"] != cfg["seed"] or not job["code_refs"]:
                        raise ValueError("Missing frozen configuration, seed or implementation refs")
                    for ref in job["input_refs"] + job["code_refs"]:
                        check_ref(ref)
                elapsed = receipt["resources"]["seconds"]
                if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("Invalid recorded native elapsed seconds")
                report["budget"]["candidate_native_elapsed_seconds"] += elapsed
                receipt_attempts = receipt["attempts"]
                listed_paths = {a["attempt_path"] for a in receipt_attempts}
                directories = {p.relative_to(root).as_posix() for p in contained(root, native_path).iterdir() if p.is_dir()}
                if listed_paths != directories or len(listed_paths) != len(receipt_attempts):
                    raise ValueError("Unknown/staged attempt or duplicate terminal record")
                for record in receipt_attempts:
                    path = Path(record["attempt_path"])
                    if path.parent != native_path or load(path / "attempt.json") != record:
                        raise ValueError("Receipt disagrees with retained attempt.json")
                    name = record["trial_id"]
                    if name not in jobs or record["status"] not in TERMINAL - {"budget_exhausted"} or record["retry_index"] != 0:
                        raise ValueError("Nonterminal, unexpected or retried attempt")
                    job = jobs[name]
                    if any(record[key] != job[key] for key in ("input_refs", "code_refs", "seed", "group", "arm_role")):
                        raise ValueError("Attempt input/code/protocol fields differ from frozen plan")
                    if record["provenance"] != plan["provenance"]:
                        raise ValueError("Attempt provenance differs from frozen plan")
                    for ref in record["output_refs"]:
                        if Path(ref["path"]).parts[:len(path.parts)] != path.parts:
                            raise ValueError("Output belongs to another attempt")
                        check_ref(ref)
                    report["attempts"].append({"trial_id": name, "native_status": record["status"],
                                               "exit_code": record["exit_code"], "attempt_path": record["attempt_path"],
                                               "git_revision": record["provenance"]["git_revision"],
                                               "seconds": record["seconds"], "retry_allowed": False})
                    if name in consumed:
                        raise ValueError("Lifetime trial ceiling exceeded: " + name)
                    consumed[name] = record
                    if name == "prepare":
                        preparation.append(record)
                report["native_runs"].append({"run_id": native_id, "status": receipt["status"],
                                              "plan_digest": plan["plan_digest"], "seconds": elapsed})
            except (OSError, ValueError, KeyError, TypeError) as error:
                blocker("INVALID_OR_UNKNOWN_NATIVE_HISTORY", native_id + ": " + str(error))
        report["unstarted_inventory"] = [name for name in names if name not in observed_trials]
        report["budget"].update(preparation_attempts_used=int("prepare" in observed_trials),
                                head_attempts_used=sum(name in observed_trials for name in names),
                                remaining_preparation_attempts=int("prepare" not in observed_trials))
        if preparation and (preparation[0]["status"] != "completed" or preparation[0]["exit_code"] != 0):
            blocker("PREPARATION_ALREADY_FAILED_NO_RETRY", preparation[0]["attempt_path"])
        bundle = summary.get("bundle")
        if not bundle and preparation and preparation[0]["status"] == "completed" and preparation[0]["exit_code"] == 0:
            refs = [r for r in preparation[0]["output_refs"] if r["path"].endswith("/out/manifest.json")]
            if len(refs) == 1:
                bundle = str(root / refs[0]["path"])
        report["preparation_required"] = not bundle and not observed_trials
        if bundle:
            try:
                candidate = Path(bundle)
                relative = candidate.relative_to(root) if candidate.is_absolute() else candidate
                path = contained(root, relative)
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                from experiments.embedding_heads.bundle import validate_manifest, all_refs, referenced
                manifest = validate_manifest(path)
                producer = path.parent.parent.parent / "attempt.json"
                producer_record = load(producer.relative_to(root))
                check_bundle_protocol(manifest, producer_record, cfg, observed["configs/candidates.json"])
                for ref in all_refs(manifest):
                    file = referenced(path.parent, ref)
                    observed[file.relative_to(root).as_posix()] = digest(file)
                observed[path.relative_to(root).as_posix()] = digest(path)
                observed[producer.relative_to(root).as_posix()] = digest(producer)
                report["bundle"] = path.relative_to(root).as_posix()
                report["budget"]["remaining_preparation_attempts"] = 0
                report["budget"]["preparation_attempts_used"] = 1
            except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
                blocker("INVALID_SHARED_BUNDLE", error)
        elif any(name in consumed for name in names) or preparation:
            blocker("SHARED_BUNDLE_MISSING", "Started heads or used preparation need their original manifest")
        if report["budget"]["candidate_native_elapsed_seconds"] > report["budget"]["original_clock_elapsed_seconds"] + 1:
            blocker("ELAPSED_RECEIPTS_EXCEED_ORIGINAL_CLOCK", "Reconcile timestamps; no additional budget is inferred")
        # A later writer may change evidence while this audit is reading it.
        if {p.name for p in attempts_root.glob(run_id + "-*") if p.is_dir()} != actual_ids:
            blocker("HISTORY_CHANGED_DURING_AUDIT", "Native run inventory changed")
        for path, expected in observed.items():
            if digest(contained(root, path)) != expected:
                blocker("FILES_CHANGED_DURING_AUDIT", path)
        if not report["blockers"]:
            if not report["unstarted_inventory"]:
                report["status"] = "NO_UNSTARTED_HEADS"
            elif bundle:
                report["status"] = "READY_FOR_MANUAL_REVIEW"
                report["eligible_head_inventory"] = report["unstarted_inventory"][:]
            else:
                report["status"] = "ORIGINAL_PREPARATION_REQUIRED"
        report["observed_file_refs"] = [{"path": path, "sha256": sha} for path, sha in sorted(observed.items())]
    except (OSError, ValueError, KeyError, TypeError) as error:
        blocker("MISSING_OR_INVALID_RETAINED_STATE", error)
    report["next_action"] = ("Retain all attempts and reconcile blockers; do not restart the batch" if report["blockers"]
                             else "Review this inventory; guarded dispatch is not implemented, so do not repeat run_candidates.py")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Existing batch ID; default reads runs/latest-candidates.json")
    parser.add_argument("--hours", type=float, default=8, help="Lower the same original cumulative cap; never adds time")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = audit(root, args.run_id, args.hours)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

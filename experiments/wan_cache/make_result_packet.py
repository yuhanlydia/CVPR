#!/usr/bin/env python3
"""Classify actual coverage; successful process exit alone is insufficient."""
import argparse
import json
import math
from pathlib import Path


def read_json(path):
    try:
        return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, ValueError):
        return None


def coverage(root, manifest):
    if not isinstance(manifest, dict):
        return {"complete": False, "reason": "manifest missing"}
    try:
        prompts = manifest.get("prompts", [])
        steps = manifest.get("force_steps", [])
        expected = {(p["benchmark_index"], step) for p in prompts for step in steps}
    except (KeyError, TypeError):
        return {"complete": False, "reason": "malformed inventory"}
    if not expected or len(expected) != len(prompts) * len(steps):
        return {"complete": False, "reason": "empty/duplicate inventory"}
    observed, invalid = [], []
    try:
        lines = (root / "metrics.jsonl").read_text().splitlines()
        for line in lines:
            row = json.loads(line)
            key = (row["benchmark_index"], row["force_step"])
            calls = row.get("forced_calls", [])
            valid = len(calls) == 2 and {c["branch"] for c in calls} == {"cond", "uncond"}
            valid = valid and all(c["step_idx"] == key[1] and c["forced_cache"] is True for c in calls)
            for name in ("terminal_mse", "reference_guided_local_output_rms"):
                number = row.get(name)
                valid = valid and isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number) and number >= 0
            (observed if valid else invalid).append(key)
    except (OSError, ValueError, KeyError, TypeError):
        return {"complete": False, "reason": "missing/malformed raw metrics"}
    parity = True
    for prompt in prompts:
        reference = read_json(root / f"prompt_{prompt['benchmark_index']:05d}/reference_probe.json")
        parity = parity and reference is not None and set(reference.get("native_parity_checked_branches", [])) == {"cond", "uncond"}
    complete = parity and not invalid and len(observed) == len(expected) and set(observed) == expected
    return {"complete": bool(complete), "expected_runs": len(expected), "valid_rows": len(observed),
            "unique_rows": len(set(observed)), "invalid_rows": len(invalid),
            "missing": sorted(expected - set(observed)), "unexpected": sorted(set(observed) - expected),
            "real_reference_parity_present": bool(parity)}


def write_packet(root, commit, exit_code, run_status="unknown"):
    root = Path(root)
    host = read_json(root / "host.json")
    summary = read_json(root / "summary.json")
    manifest = read_json(root / "manifest.json")
    preflight = read_json(root / "preflight.json")
    receipt = read_json(root / "native-receipt.json")
    evidence = coverage(root, manifest)
    native_ok = (receipt is not None and receipt.get("status") == "completed"
                 and receipt.get("provenance", {}).get("git_revision") == commit
                 and run_status == "completed")
    if exit_code == 0 and native_ok and evidence["complete"] and summary and preflight and preflight.get("status") == "PASS":
        status = "COMPLETE_ENGINEERING"
    elif run_status in {"budget_exhausted", "timeout"}:
        status = "CARRYOVER_BUDGET_BOUNDARY"
    else:
        status = "FAILED_OR_INCOMPLETE"
    smi = ((host or {}).get("nvidia_smi_query") or {}).get("stdout") or "unavailable"
    lines = ["# Round 002 Result Packet", "", f"- status: **{status}**",
             f"- executed_git_commit: {commit}", f"- exit_code: {exit_code}",
             f"- native_run_status: {run_status}", "- scientific_verdict: **NONE**", ""]
    for title, value in (("Coverage", evidence), ("GPU", smi), ("Preflight", preflight),
                         ("Frozen configuration", manifest), ("Diagnostic summary", summary)):
        rendered = value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False)
        lines += [f"## {title}", "", "    " + rendered.replace("\n", "\n    "), ""]
    lines += ["## Evidence", "", "- host.json, preflight.json, manifest.json, metrics.jsonl, summary.json",
              "- prompt_*/reference_probe.json, run.log, stderr.log",
              "- plan.json, native-receipt.json, attempt.json",
              "- Videos remain in the native attempt's workspace/out/large/; never copy them into a result commit.", "",
              "This is engineering coverage. Native VBench scoring, matched perturbation controls,",
              "compute-matched baselines and prospective decision thresholds are required before any",
              "scientific KILL/CONTINUE or research gate decision.", ""]
    root.mkdir(parents=True, exist_ok=True)
    (root / "RESULT.md").write_text("\n".join(lines))
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--exit-code", type=int, default=-1)
    ap.add_argument("--run-status", default="unknown")
    args = ap.parse_args()
    print(write_packet(Path(args.round_dir), args.commit, args.exit_code, args.run_status))


if __name__ == "__main__":
    main()

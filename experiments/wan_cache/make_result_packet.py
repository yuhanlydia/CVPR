#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--exit-code", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.round_dir)
    host = read_json(root / "host.json")
    summary = read_json(root / "summary.json")
    manifest = read_json(root / "manifest.json")
    preflight = read_json(root / "preflight.json")
    smi = ((host or {}).get("nvidia_smi_query") or {}).get("stdout") or "unavailable"
    status = "COMPLETE" if args.exit_code == 0 and summary else "FAILED_OR_INCOMPLETE"
    if summary and summary.get("budget_exhausted"):
        status = "CARRYOVER_BUDGET_BOUNDARY"

    lines = [
        "# Round 002 Result Packet",
        "",
        f"- status: **{status}**",
        f"- executed_git_commit: {args.commit}",
        f"- exit_code: {args.exit_code}",
        "",
        "## GPU",
        "",
        "    " + smi.replace("\n", "\n    "),
        "",
        "## Preflight",
        "",
        "    " + (json.dumps(preflight, indent=2) if preflight else "preflight missing").replace("\n", "\n    "),
        "",
        "## Frozen configuration",
        "",
        "    " + (json.dumps(manifest, indent=2) if manifest else "manifest missing").replace("\n", "\n    "),
        "",
        "## Diagnostic summary",
        "",
        "    " + (json.dumps(summary, indent=2) if summary else "summary missing").replace("\n", "\n    "),
        "",
        "## Evidence",
        "",
        f"- {root / 'run.log'}",
        f"- {root / 'host.json'}",
        f"- {root / 'preflight.json'}",
        f"- {root / 'manifest.json'}",
        f"- {root / 'metrics.jsonl'}",
        f"- {root / 'prompt_*/reference_probe.json'}",
        f"- {root / 'large/'} is local-only and should not be committed.",
        "",
        "Execution is not a scientific verdict; apply the preregistered decision rule.",
        "",
    ]
    (root / "RESULT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

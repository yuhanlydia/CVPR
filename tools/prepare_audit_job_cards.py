"""Prepare pinned inner-job cards, NOT a dispatch plan or a new budget.

Local runs this after source/native acceptance. The actual scientific protocols,
resource admission and one remote run_harness owner remain mandatory.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import digest, read_json, write_json
from experiments.embedding_heads.audit_controls import registry
from experiments.embedding_heads.bundle import (
    validate_manifest, load_training, all_refs, referenced, preparation_receipt,
    check_original_source, UPSTREAM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "upstream", "python", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config).resolve()
    bundle = Path(args.bundle).resolve()
    upstream = Path(args.upstream).resolve()
    interpreter = Path(args.python)
    if not interpreter.is_absolute():
        raise ValueError("Actual absolute native Conda interpreter required")
    for path in (config_path, bundle, upstream):
        path.relative_to(root)
    check_original_source(upstream)
    config = read_json(config_path)
    arms = registry(config)
    manifest = validate_manifest(bundle)
    training = load_training(bundle, manifest, config["head"]["temperature"])
    if ([t["task"] for t in manifest["evaluation"]] != config["evaluation_tasks"] or
            manifest["train"]["revision"] != config["training_revision"]):
        raise ValueError("Native group/input identity changed")
    def ref(path):
        path = Path(path).resolve()
        return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}
    inputs = [ref(config_path), ref(bundle), ref(preparation_receipt(bundle))]
    inputs += [ref(referenced(bundle.parent, item)) for item in all_refs(manifest)]
    code = sorted((root / "tools").glob("*.py"))
    code += sorted((root / "experiments/embedding_heads").glob("*.py"))
    code += sorted(p for p in upstream.rglob("*") if p.is_file() and ".git" not in p.parts
                   and (p.suffix in {".py", ".yaml", ".yml"} or p.name == "LICENSE"))
    code_refs = [ref(p) for p in code]
    cards, blocked = [], []
    for arm in arms.values():
        reason = None
        if arm["id"] == "I01" and training.radius is None:
            reason = "INDEPENDENT_INTERVAL_CALIBRATION_NOT_AVAILABLE"
        elif arm.get("conditional") == "released_real_multi_positive_rows" and not (
                training.positive.sum(1) > 1).any():
            reason = "RELEASED_REAL_MULTI_POSITIVE_ROWS_REQUIRED"
        if reason:
            blocked.append({"arm": arm["id"], "reason": reason, "scientific_verdict": "NONE"})
            continue
        for task in manifest["evaluation"]:
            name = task["task"]
            job = {"trial_id": arm["id"] + "__" + name,
                   "command": [str(interpreter), "tools/run_audit_method.py", "--bundle",
                       bundle.relative_to(root).as_posix(), "--config", config_path.relative_to(root).as_posix(),
                       "--method", arm["id"], "--task", name, "--upstream",
                       upstream.relative_to(root).as_posix(), "--out", "out"],
                   "cwd": ".", "input_refs": inputs, "code_refs": code_refs,
                   "output_paths": ["out/result.json", "out/head.json", "out/head.npz"] +
                       ["out/" + name + suffix for suffix in ("_pred.jsonl", "_score.json", "_replay.json")],
                   "seed": config["seed"], "group": name, "arm_role": arm["id"]}
            cards.append({"arm": arm["id"], "task": name, "family": arm["family"], "native_job": job})
    packet = {"schema": "cvpr.audit-job-cards.v1", "status": "DRAFT_JOB_CARDS_ONLY",
              "dispatch_ready": False, "execution_started": False, "scientific_verdict": "NONE",
              "upstream_revision": UPSTREAM, "skill_revision": config["skill_revision"],
              "bundle": bundle.relative_to(root).as_posix(), "config_sha256": digest(config_path),
              "cards": cards, "blocked": blocked,
              "total_arms_registered": len(arms), "planned_group_jobs": len(cards),
              "cpu_only_scoring": True, "fit_repeated_per_task": True,
              "required_before_dispatch": [
                  "actual per-group native child protocols/arm requirements and live scorer qualification",
                  "actual Local semantic acceptance and complete G01",
                  "actual remaining cumulative attempt/time/spend budget and host CPU/RAM admission",
                  "existing-method gate scope review; no false top15 or novelty certification",
                  "one remote run_harness owner; no bare command launch or budget reset"]}
    write_json(args.out, packet)
    print("Draft job cards:", args.out)
    print("Group jobs:", len(cards), "; blocked records:", len(blocked), "; execution_started=False")


if __name__ == "__main__":
    main()

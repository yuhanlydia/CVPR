"""Bundle existing receipts/logs/JSON evidence; never recompute a benchmark."""
import argparse
import tarfile
from pathlib import Path
from common import digest, read_json, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id")
    a = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    run_id = a.run_id or read_json(root / "runs/latest.json")["run_id"]
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("Invalid run ID")
    run = root / "runs/attempts" / run_id
    if not run.is_dir():
        raise FileNotFoundError(run)
    files = sorted(p for p in run.rglob("*") if p.is_file() and not p.is_symlink()
                   and (p.suffix in {".json", ".jsonl", ".log", ".yaml"})
                   and "sources" not in p.relative_to(run).parts)
    bundle = root / "runs" / (run_id + "-return.tar.gz")
    with tarfile.open(bundle, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    summaries = [read_json(p) for p in files if p.name == "summary.json" and p.parent.name == "out"]
    small = root / "rounds/r001/returns" / (run_id + ".json")
    write_json(small, {"run_id": run_id, "qualification": summaries,
                      "receipt_present": (run / "receipt.json").is_file(),
                      "bundle_sha256": digest(bundle), "evidence_files": len(files),
                      "scientific_verdict": "NONE", "gpu_experiment_status": "read_from_returned_receipts"})
    print(f"Small report: {small.relative_to(root)}")
    print(f"Evidence bundle (no weights/images/embedding pickles): {bundle.relative_to(root)}")


if __name__ == "__main__":
    main()

"""Read-only identity check of the four complete pinned Research Autopilot skills.

This checks source bytes only. It installs nothing, launches no workload and
does not verify mathematical truth, native evaluation or scientific readiness.
Generated on the Web; Local acceptance has not been executed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def inspect_source(skill_root, lock):
    root = Path(skill_root).expanduser().absolute()
    parent = root.parent
    problems = []
    if root.name != "research-autopilot" or not root.is_dir() or root.is_symlink():
        problems.append({"path": str(root), "reason": "invalid_research_autopilot_root"})
    checked = 0
    for name in lock["skills"]:
        directory = parent / name
        if not directory.is_dir() or directory.is_symlink():
            problems.append({"path": name, "reason": "missing_or_symlinked_skill_directory"})
    for entry in lock["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source-lock path")
        target = parent / relative
        if not target.is_file() or target.is_symlink():
            problems.append({"path": entry["path"], "reason": "missing_or_symlinked_file"})
            continue
        content = target.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()
        checked += 1
        if len(content) != entry["size"] or actual != entry["git_blob_sha"]:
            problems.append({"path": entry["path"], "reason": "source_identity_mismatch",
                             "actual_git_blob_sha": actual})
    return {"repository": lock["repository"], "required_revision": lock["revision"],
            "checked_files": checked, "required_files": len(lock["files"]),
            "status": "SOURCE_BYTES_MATCH" if not problems else "SOURCE_MISMATCH",
            "problems": problems,
            "scientific_readiness": "NOT_ESTABLISHED",
            "local_tests_and_runtime": "NOT_CHECKED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-root", required=True, help="Actual complete research-autopilot directory")
    parser.add_argument("--lock", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/autopilot-source.json")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    report = inspect_source(args.skill_root, lock)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "SOURCE_BYTES_MATCH" else 2


if __name__ == "__main__":
    raise SystemExit(main())

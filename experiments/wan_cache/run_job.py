"""A single staged engineering job, owned by the installed native watchdog."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-wan-root", required=True)
    p.add_argument("--source-vbench-root", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--wan-root", required=True)
    p.add_argument("--vbench-json", required=True)
    p.add_argument("--output-dir", default="out")
    args, probe_args = p.parse_known_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    preflight = subprocess.run([
        sys.executable, str(scripts / "preflight.py"),
        "--wan-root", args.source_wan_root, "--vbench-root", args.source_vbench_root,
        "--ckpt-dir", args.ckpt_dir, "--output", str(root / "preflight.json")], check=False)
    if preflight.returncode:
        raise SystemExit(preflight.returncode)
    result = subprocess.run([
        sys.executable, str(scripts / "probe_cache_propagation.py"),
        "--wan-root", args.wan_root, "--ckpt-dir", args.ckpt_dir,
        "--vbench-json", args.vbench_json, "--output-dir", str(root), *probe_args], check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

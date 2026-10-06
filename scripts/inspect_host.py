#!/usr/bin/env python3
import argparse, json, os, platform, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def run(cmd):
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=15)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as e:
        return {"cmd": cmd, "error": repr(e)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="artifacts/host/host.json")
    args = ap.parse_args()

    out = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "executables": {k: shutil.which(k) for k in ["nvidia-smi","git","python","python3","nvcc"]},
        "env": {
            k: os.environ.get(k) for k in [
                "CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES","CONDA_DEFAULT_ENV",
                "VIRTUAL_ENV","SLURM_JOB_ID","SLURM_JOB_GPUS"
            ]
        },
        "nvidia_smi_query": run([
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]) if shutil.which("nvidia-smi") else None,
        "nvidia_smi_full": run(["nvidia-smi"]) if shutil.which("nvidia-smi") else None,
        "torch": None,
        "git": run(["git","rev-parse","HEAD"]) if shutil.which("git") else None,
    }

    try:
        import torch
        out["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                    "capability": list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as e:
        out["torch"] = {"error": repr(e)}

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(path)

if __name__ == "__main__":
    main()

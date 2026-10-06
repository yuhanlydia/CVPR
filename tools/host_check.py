"""Inspect the actual GPU/software; fail instead of silently changing a protocol."""
import argparse
import importlib.metadata
import shutil
import subprocess
import sys
import time
from pathlib import Path
from common import git, read_json, write_json, now


def inspect(root):
    import torch
    import transformers
    from transformers import Qwen3VLModel
    assert Qwen3VLModel is not None
    cfg = read_json(root / "configs/batch.json")
    if not torch.cuda.is_available():
        raise RuntimeError("No usable CUDA GPU. No model experiment started.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one device with CUDA_VISIBLE_DEVICES")
    prop = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    if capability < (7, 5):
        raise RuntimeError("This batch requires Turing or newer CUDA hardware")
    if free < cfg["minimum_free_vram_gib"] * 1024 ** 3:
        raise RuntimeError("Insufficient free VRAM for qualification; release other GPU jobs")
    if shutil.disk_usage(root).free < cfg["minimum_free_disk_gib"] * 1024 ** 3:
        raise RuntimeError("At least 35 GiB free disk required for models/images/logs")
    src = root / "sources/Qwen3-VL-Embedding"
    if git(src, "rev-parse", "HEAD") != cfg["upstream_commit"] or git(src, "status", "--porcelain"):
        raise RuntimeError("Upstream checkout must be clean and at the declared commit")
    # Exercise SDPA on the real device without any benchmark claim.
    x = torch.zeros((1, 1, 16, 64), device="cuda", dtype=torch.float16)
    y = torch.nn.functional.scaled_dot_product_attention(x, x, x)
    torch.cuda.synchronize()
    assert y.shape == x.shape
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    return {"observed_at": now(), "device_name": prop.name, "capability": list(capability),
            "free_vram_bytes": free, "total_vram_bytes": total,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda, "python": sys.version,
            "packages": packages, "sdpa_software_check": "passed",
            "model_memory_qualified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finish-setup", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    info = inspect(root)
    write_json(root / "runs/host.json", info)
    if args.finish_setup:
        receipt = read_json(root / "setup-receipt.json")
        receipt.update(status="completed", completed_epoch=time.time(), host=info)
        write_json(root / "setup-receipt.json", receipt)
    print(f"GPU: {info['device_name']}; FP16/SDPA check passed. Model memory remains unmeasured.")


if __name__ == "__main__":
    main()

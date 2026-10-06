#!/usr/bin/env python3
"""Engineering-only preflight for Round 002.

This validates environment/source/checkpoint prerequisites. Passing it is not
scientific evidence and does not advance any research claim.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

EXPECTED = {
    "wan": "9737cba9c1c3c4d04b33fcad41c111989865d315",
    "vbench": "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490",
}


def git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan-root", required=True)
    ap.add_argument("--vbench-root", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--output", default="artifacts/round_002/preflight.json")
    args = ap.parse_args()

    import torch

    wan_root = Path(args.wan_root).resolve()
    vbench_root = Path(args.vbench_root).resolve()
    ckpt = Path(args.ckpt_dir).resolve()

    checks = {}
    wan_sha = git_sha(wan_root)
    vbench_sha = git_sha(vbench_root)
    checks["wan_revision"] = {
        "observed": wan_sha,
        "expected": EXPECTED["wan"],
        "ok": wan_sha == EXPECTED["wan"],
    }
    checks["vbench_revision"] = {
        "observed": vbench_sha,
        "expected": EXPECTED["vbench"],
        "ok": vbench_sha == EXPECTED["vbench"],
    }

    required = [
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "config.json",
        "diffusion_pytorch_model.safetensors",
    ]
    missing = [name for name in required if not (ckpt / name).exists()]
    config_candidates = list(ckpt.glob("config*.json"))
    dit_weights = ckpt / "diffusion_pytorch_model.safetensors"
    checks["checkpoint"] = {
        "path": str(ckpt),
        "exists": ckpt.exists(),
        "missing_required_files": missing,
        "num_config_json": len(config_candidates),
        "dit_weights_bytes": dit_weights.stat().st_size if dit_weights.exists() else None,
        "ok": ckpt.exists() and not missing and bool(config_candidates) and dit_weights.exists(),
    }

    cuda = torch.cuda.is_available()
    gpu = None
    if cuda:
        props = torch.cuda.get_device_properties(0)
        cc = list(torch.cuda.get_device_capability(0))
        free_b, total_b = torch.cuda.mem_get_info(0)
        gpu = {
            "name": props.name,
            "compute_capability": cc,
            "total_memory_gb": total_b / 1024**3,
            "free_memory_gb": free_b / 1024**3,
            "torch_bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "round002_dtype": "float16",
            "fp16_policy_ok": True,
        }
    checks["cuda"] = {"available": cuda, "gpu": gpu, "ok": cuda}

    vbench_json = vbench_root / "vbench" / "VBench_full_info.json"
    checks["vbench_prompt_json"] = {
        "path": str(vbench_json),
        "exists": vbench_json.exists(),
        "ok": vbench_json.exists(),
    }

    ok = all(v.get("ok", False) for v in checks.values())
    report = {
        "status": "PASS" if ok else "BLOCKED",
        "scientific_evidence": False,
        "checks": checks,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()

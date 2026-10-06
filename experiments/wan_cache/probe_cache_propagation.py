#!/usr/bin/env python3
"""Controlled single-step cache perturbation study for Wan2.1-T2V-1.3B.

Scientific use:
  Compare a TeaCache-style *local* cache-error proxy against the *terminal*
  consequence of forcing exactly one cache reuse at a selected diffusion step.

This script intentionally does not implement the proposed new cache policy.
Round 002 is a falsification/diagnostic round: if downstream amplification is
nearly constant, propagation-aware scheduling is not justified.

Run from the CVPR repository; pass --wan-root to a pinned Wan2.1 checkout.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.cuda.amp as amp


def rel_l1(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    num = (a.float() - b.float()).abs().mean()
    den = b.float().abs().mean().clamp_min(eps)
    return float((num / den).item())


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    diff = cand - ref
    mse = float(diff.square().mean().item())
    mae = float(diff.abs().mean().item())
    data_range = 2.0  # Wan decoded tensors are saved with value_range=(-1, 1)
    psnr = float(10.0 * math.log10((data_range * data_range) / max(mse, 1e-12)))
    if ref.ndim != 4:
        temporal_mse = float("nan")
    else:
        ref_dt = ref[:, 1:] - ref[:, :-1]
        cand_dt = cand[:, 1:] - cand[:, :-1]
        temporal_mse = float((cand_dt - ref_dt).square().mean().item())
    return {
        "terminal_mse": mse,
        "terminal_mae": mae,
        "terminal_psnr": psnr,
        "terminal_temporal_gradient_mse": temporal_mse,
    }


def configure_probe(model: torch.nn.Module, force_steps: set[int]) -> None:
    model._cache_probe = {
        "call_idx": 0,
        "force_steps": set(int(x) for x in force_steps),
        "prev_residual": {"cond": None, "uncond": None},
        "prev_modulated": {"cond": None, "uncond": None},
        "logs": [],
    }


def make_probe_forward(wan_model_module):
    sinusoidal_embedding_1d = wan_model_module.sinusoidal_embedding_1d

    def probe_forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        state = self._cache_probe
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        patched = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in patched]
        )
        tokens = [u.flatten(2).transpose(1, 2) for u in patched]
        seq_lens = torch.tensor([u.size(1) for u in tokens], dtype=torch.long)
        if int(seq_lens.max()) > int(seq_len):
            raise ValueError(f"sequence length {int(seq_lens.max())} > {seq_len}")
        xh = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in tokens
        ])

        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float()
            )
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        context_lens = None
        context_h = self.text_embedding(
            torch.stack([
                torch.cat([
                    u,
                    u.new_zeros(self.text_len - u.size(0), u.size(1)),
                ])
                for u in context
            ])
        )
        if clip_fea is not None:
            context_h = torch.cat([self.img_emb(clip_fea), context_h], dim=1)

        kwargs = {
            "e": e0,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": self.freqs,
            "context": context_h,
            "context_lens": context_lens,
        }

        call_idx = int(state["call_idx"])
        step_idx = call_idx // 2
        branch = "cond" if call_idx % 2 == 0 else "uncond"
        # TeaCache4Wan2.1 uses e (not e0) for 1.3B unless retention steps are enabled.
        modulated = e.detach()
        prev_modulated = state["prev_modulated"][branch]
        prev_residual = state["prev_residual"][branch]

        raw_mod_delta = None
        if prev_modulated is not None:
            raw_mod_delta = rel_l1(modulated, prev_modulated)

        base_hidden = xh
        forced = step_idx in state["force_steps"] and prev_residual is not None
        residual_delta = None
        local_output_delta = None

        if forced:
            xh = base_hidden + prev_residual
        else:
            for block in self.blocks:
                xh = block(xh, **kwargs)
            current_residual = (xh - base_hidden).detach()

            if prev_residual is not None:
                residual_delta = rel_l1(current_residual, prev_residual)
                # This measures the immediate denoiser-output error that would have
                # occurred if the previous residual had been reused at this step.
                cached_hidden = base_hidden + prev_residual
                out_full = self.unpatchify(self.head(xh, e), grid_sizes)[0]
                out_cache = self.unpatchify(self.head(cached_hidden, e), grid_sizes)[0]
                local_output_delta = rel_l1(out_cache, out_full)
                del cached_hidden, out_full, out_cache

            state["prev_residual"][branch] = current_residual

        out = self.unpatchify(self.head(xh, e), grid_sizes)
        state["prev_modulated"][branch] = modulated.clone()
        state["logs"].append({
            "call_idx": call_idx,
            "step_idx": step_idx,
            "branch": branch,
            "timestep": float(t.flatten()[0].item()),
            "forced_cache": bool(forced),
            "raw_modulated_rel_l1": raw_mod_delta,
            "residual_rel_l1": residual_delta,
            "local_denoiser_output_rel_l1": local_output_delta,
        })
        state["call_idx"] = call_idx + 1
        return [u.float() for u in out]

    return probe_forward


def load_vbench_subset(path: Path, num_prompts: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("VBench info JSON must be a non-empty list")
    n = min(num_prompts, len(data))
    # Deterministic prospective sampling across the released benchmark ordering.
    idxs = np.linspace(0, len(data) - 1, num=n, dtype=int).tolist()
    items = []
    for idx in idxs:
        row = data[idx]
        prompt = row.get("prompt_en") or row.get("prompt")
        if not prompt:
            raise ValueError(f"No English prompt at VBench index {idx}")
        items.append({
            "benchmark_index": int(idx),
            "prompt": str(prompt),
            "dimension": row.get("dimension", []),
        })
    return items


def default_force_steps(sampling_steps: int) -> list[int]:
    fracs = [0.10, 0.30, 0.50, 0.70, 0.90]
    steps = sorted({
        max(1, min(sampling_steps - 1, int(round(f * (sampling_steps - 1)))))
        for f in fracs
    })
    return steps


def safe_video_name(prompt_index: int) -> str:
    return f"vbench_{prompt_index:05d}.mp4"


def save_video(cache_video, video_cpu: torch.Tensor, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_video(
        tensor=video_cpu[None],
        save_file=str(path),
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )


def branch_average(logs: list[dict[str, Any]], step: int, key: str):
    vals = [
        r[key] for r in logs
        if r["step_idx"] == step and r.get(key) is not None
    ]
    return float(np.mean(vals)) if vals else None


def run_generation(
    pipeline,
    cache_video,
    cfg,
    prompt: str,
    seed: int,
    size: tuple[int, int],
    frame_num: int,
    sampling_steps: int,
    shift: float,
    guide_scale: float,
    force_steps: set[int],
    video_path: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    configure_probe(pipeline.model, force_steps)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    video = pipeline.generate(
        prompt,
        size=size,
        frame_num=frame_num,
        shift=shift,
        sample_solver="unipc",
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=seed,
        offload_model=True,
    )
    elapsed = time.time() - start
    video_cpu = video.detach().float().cpu()
    save_video(cache_video, video_cpu, video_path, int(cfg.sample_fps))
    peak = (
        int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available() else None
    )
    meta = {
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": peak,
        "probe_logs": pipeline.model._cache_probe["logs"],
    }
    del video
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return video_cpu, meta


def corr(xs, ys) -> dict[str, float | None]:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"pearson": None, "spearman": None}
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    pearson = float(np.corrcoef(x, y)[0, 1])
    xr = np.argsort(np.argsort(x)).astype(np.float64)
    yr = np.argsort(np.argsort(y)).astype(np.float64)
    spearman = float(np.corrcoef(xr, yr)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        r for r in rows
        if r.get("terminal_mse") is not None
        and r.get("reference_local_denoiser_output_rel_l1") not in (None, 0)
    ]
    local = [r["reference_local_denoiser_output_rel_l1"] for r in usable]
    terminal = [r["terminal_mse"] for r in usable]
    mod = [
        r["reference_raw_modulated_rel_l1"] for r in usable
        if r.get("reference_raw_modulated_rel_l1") is not None
    ]
    mod_terminal = [
        r["terminal_mse"] for r in usable
        if r.get("reference_raw_modulated_rel_l1") is not None
    ]
    amplification = [
        r["terminal_mse"] / max(r["reference_local_denoiser_output_rel_l1"], 1e-12)
        for r in usable
    ]
    return {
        "num_forced_runs": len(rows),
        "num_usable_pairs": len(usable),
        "corr_local_denoiser_error_to_terminal_mse": corr(local, terminal),
        "corr_teacache_input_proxy_to_terminal_mse": corr(mod, mod_terminal),
        "empirical_amplification": {
            "mean": float(np.mean(amplification)) if amplification else None,
            "std": float(np.std(amplification)) if amplification else None,
            "min": float(np.min(amplification)) if amplification else None,
            "max": float(np.max(amplification)) if amplification else None,
            "max_over_min": (
                float(np.max(amplification) / max(np.min(amplification), 1e-12))
                if amplification else None
            ),
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wan-root", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--vbench-json", required=True)
    p.add_argument("--output-dir", default="artifacts/round_002")
    p.add_argument("--num-prompts", type=int, default=3)
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--force-steps", default="")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--frame-num", type=int, default=81)
    p.add_argument("--shift", type=float, default=8.0)
    p.add_argument("--guide-scale", type=float, default=6.0)
    p.add_argument("--base-seed", type=int, default=20261006)
    p.add_argument("--max-wall-hours", type=float, default=7.5)
    return p.parse_args()


def main():
    args = parse_args()
    wan_root = Path(args.wan_root).resolve()
    sys.path.insert(0, str(wan_root))

    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video
    import wan.modules.model as wan_model_module

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "large").mkdir(exist_ok=True)

    prompts = load_vbench_subset(Path(args.vbench_json), args.num_prompts)
    force_steps = (
        [int(x) for x in args.force_steps.split(",") if x.strip()]
        if args.force_steps else default_force_steps(args.sampling_steps)
    )

    cfg = WAN_CONFIGS["t2v-1.3B"]
    pipeline = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
    )
    pipeline.model.forward = types.MethodType(
        make_probe_forward(wan_model_module), pipeline.model
    )

    run_start = time.time()
    budget_s = args.max_wall_hours * 3600.0
    rows: list[dict[str, Any]] = []
    manifest = {
        "wan_root": str(wan_root),
        "ckpt_dir": str(Path(args.ckpt_dir).resolve()),
        "vbench_json": str(Path(args.vbench_json).resolve()),
        "sampling_steps": args.sampling_steps,
        "force_steps": force_steps,
        "num_prompts_requested": args.num_prompts,
        "size": [args.width, args.height],
        "frame_num": args.frame_num,
        "shift": args.shift,
        "guide_scale": args.guide_scale,
        "base_seed": args.base_seed,
        "max_wall_hours": args.max_wall_hours,
        "prompts": prompts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for item in prompts:
        if time.time() - run_start >= budget_s:
            break
        idx = item["benchmark_index"]
        prompt = item["prompt"]
        seed = args.base_seed + idx
        prompt_dir = out / f"prompt_{idx:05d}"
        prompt_dir.mkdir(exist_ok=True)
        ref_video_path = out / "large" / "reference" / safe_video_name(idx)

        reference, ref_meta = run_generation(
            pipeline, cache_video, cfg, prompt, seed,
            (args.width, args.height), args.frame_num, args.sampling_steps,
            args.shift, args.guide_scale, set(), ref_video_path,
        )
        (prompt_dir / "reference_probe.json").write_text(
            json.dumps(ref_meta, indent=2) + "\n"
        )

        for step in force_steps:
            if time.time() - run_start >= budget_s:
                break
            variant_path = out / "large" / f"force_{step:03d}" / safe_video_name(idx)
            variant, var_meta = run_generation(
                pipeline, cache_video, cfg, prompt, seed,
                (args.width, args.height), args.frame_num, args.sampling_steps,
                args.shift, args.guide_scale, {step}, variant_path,
            )
            metrics = tensor_metrics(reference, variant)
            row = {
                "benchmark_index": idx,
                "dimension": item["dimension"],
                "seed": seed,
                "force_step": step,
                **metrics,
                "reference_raw_modulated_rel_l1": branch_average(
                    ref_meta["probe_logs"], step, "raw_modulated_rel_l1"
                ),
                "reference_residual_rel_l1": branch_average(
                    ref_meta["probe_logs"], step, "residual_rel_l1"
                ),
                "reference_local_denoiser_output_rel_l1": branch_average(
                    ref_meta["probe_logs"], step, "local_denoiser_output_rel_l1"
                ),
                "variant_elapsed_seconds": var_meta["elapsed_seconds"],
                "variant_peak_cuda_memory_bytes": var_meta["peak_cuda_memory_bytes"],
                "forced_calls": [
                    x for x in var_meta["probe_logs"] if x["forced_cache"]
                ],
                "video_path": str(variant_path),
            }
            rows.append(row)
            with (out / "metrics.jsonl").open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            del variant
            gc.collect()

        del reference
        gc.collect()

    summary = summarize(rows)
    summary["elapsed_seconds"] = time.time() - run_start
    summary["completed_prompt_indices"] = sorted({r["benchmark_index"] for r in rows})
    summary["completed_force_runs"] = len(rows)
    summary["budget_exhausted"] = (time.time() - run_start) >= budget_s
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

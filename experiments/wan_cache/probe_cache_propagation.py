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
from runtime_compat import configure_runtime
from probe_stats import summarize


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
    if ref.ndim != 4 or ref.shape[1] < 2 or ref.shape != cand.shape:
        raise ValueError("Expected aligned C,T,H,W videos with at least two frames")
    else:
        ref_dt = ref[:, 1:] - ref[:, :-1]
        cand_dt = cand[:, 1:] - cand[:, :-1]
        temporal_mse = float((cand_dt - ref_dt).square().mean().item())
    if not all(math.isfinite(v) for v in (mse, mae, psnr, temporal_mse)):
        raise ValueError("Non-finite video error; this run is invalid")
    return {
        "terminal_mse": mse,
        "terminal_mae": mae,
        "terminal_psnr": psnr,
        "terminal_temporal_gradient_mse": temporal_mse,
    }


def configure_probe(model: torch.nn.Module, force_steps: set[int], guide_scale: float) -> None:
    model._cache_probe = {
        "call_idx": 0,
        "force_steps": set(int(x) for x in force_steps),
        "prev_residual": {"cond": None, "uncond": None},
        "prev_modulated": {"cond": None, "uncond": None},
        "guide_scale": guide_scale,
        "pending_cond_delta": None,
        "logs": [],
    }


def make_probe_forward(wan_model_module):
    sinusoidal_embedding_1d = wan_model_module.sinusoidal_embedding_1d

    def probe_forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        state = self._cache_probe
        if clip_fea is not None or y is not None:
            raise ValueError("Round 002 is qualified only for T2V")
        native_context = context
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
        context = [u.to(dtype=self.text_embedding[0].weight.dtype) for u in context]
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
        teacache_rescaled_proxy = None
        if prev_modulated is not None:
            raw_mod_delta = rel_l1(modulated, prev_modulated)
            # Published TeaCache4Wan2.1 polynomial for the 1.3B checkpoint.
            coeff = np.asarray([
                -5.21862437e4, 9.23041404e3, -5.28275948e2,
                1.36987616e1, -4.99875664e-2,
            ], dtype=np.float64)
            teacache_rescaled_proxy = float(np.poly1d(coeff)(raw_mod_delta))

        base_hidden = xh
        forced = step_idx in state["force_steps"] and prev_residual is not None
        residual_delta = None
        local_output_delta = None
        guided_output_rms = None

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
                output_error = (out_cache.float() - out_full.float()).detach()
                if branch == "cond":
                    state["pending_cond_delta"] = output_error
                else:
                    cond_error = state["pending_cond_delta"]
                    if cond_error is None:
                        raise RuntimeError("Missing conditional branch in CFG pair")
                    guided_error = output_error + state["guide_scale"] * (cond_error - output_error)
                    guided_output_rms = float(guided_error.square().mean().sqrt().item())
                    state["pending_cond_delta"] = None
                    del cond_error, guided_error
                del output_error
                del cached_hidden, out_full, out_cache

            state["prev_residual"][branch] = current_residual

        out = self.unpatchify(self.head(xh, e), grid_sizes)
        parity = None
        if call_idx < 2 and not state["force_steps"]:
            # Check both CFG branches on the actual released prompt/checkpoint.
            # This is local compatibility parity, not a leaderboard score replay.
            native_out = self._native_forward(x, t, native_context, seq_len)
            parity = all(torch.allclose(a.float(), b.float(), rtol=1e-3, atol=1e-5)
                         for a, b in zip(out, native_out)) and len(out) == len(native_out)
            if not parity:
                raise RuntimeError("Instrumented forward fails real-input native parity")
            del native_out
        state["prev_modulated"][branch] = modulated.clone()
        state["logs"].append({
            "call_idx": call_idx,
            "step_idx": step_idx,
            "branch": branch,
            "timestep": float(t.flatten()[0].item()),
            "forced_cache": bool(forced),
            "raw_modulated_rel_l1": raw_mod_delta,
            "teacache_rescaled_proxy": teacache_rescaled_proxy,
            "residual_rel_l1": residual_delta,
            "local_denoiser_output_rel_l1": local_output_delta,
            "guided_local_output_rms": guided_output_rms,
            "native_forward_parity": parity,
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
    configure_probe(pipeline.model, force_steps, guide_scale)
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
        "native_parity_checked_branches": [r["branch"] for r in pipeline.model._cache_probe["logs"]
                                         if r.get("native_forward_parity") is True],
    }
    del video
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return video_cpu, meta


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
    run_start = time.monotonic()
    budget_s = args.max_wall_hours * 3600.0
    if args.num_prompts < 1 or args.sampling_steps < 2:
        raise ValueError("Use at least one prompt and two sampling steps")
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        raise ValueError("Positive dimensions must be multiples of 16")
    if args.frame_num < 5 or (args.frame_num - 1) % 4:
        raise ValueError("Frame count must be 4n+1, n>=1")
    if not math.isfinite(args.max_wall_hours) or not 0 < args.max_wall_hours <= 7.5:
        raise ValueError("This model window is bounded by at most 7.5 hours")
    if not math.isfinite(args.guide_scale) or args.guide_scale < 0 or args.base_seed < 0:
        raise ValueError("CFG and seed must be finite/nonnegative")
    if not math.isfinite(args.shift) or args.shift <= 0:
        raise ValueError("Shift must be finite/positive")
    wan_root = Path(args.wan_root).resolve()
    sys.path.insert(0, str(wan_root))

    import wan
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import cache_video
    import wan.modules.model as wan_model_module

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("manifest.json", "metrics.jsonl", "summary.json")):
        raise FileExistsError("Use a fresh isolated attempt; do not merge old metrics with a new manifest")
    (out / "large").mkdir(exist_ok=True)

    prompts = load_vbench_subset(Path(args.vbench_json), args.num_prompts)
    force_steps = (
        [int(x) for x in args.force_steps.split(",") if x.strip()]
        if args.force_steps else default_force_steps(args.sampling_steps)
    )
    if not force_steps or len(set(force_steps)) != len(force_steps) or any(
            step < 1 or step >= args.sampling_steps for step in force_steps):
        raise ValueError("Force steps must be unique and within 1..sampling_steps-1")

    import copy
    cfg = copy.deepcopy(WAN_CONFIGS["t2v-1.3B"])

    # FP16 autocast while retaining native FP32 islands; never globally half-cast.
    cfg.param_dtype = torch.float16
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
    runtime = configure_runtime(pipeline.model, wan_model_module)
    pipeline.model._native_forward = pipeline.model.forward
    pipeline.model.forward = types.MethodType(
        make_probe_forward(wan_model_module), pipeline.model
    )

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
        "runtime": runtime,
        "native_parity_tolerance": {"rtol": 1e-3, "atol": 1e-5, "calls": "first conditional/unconditional real reference pair"},
        "scientific_evidence": False,
        "prompts": prompts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for item in prompts:
        if time.monotonic() - run_start >= budget_s:
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
            if time.monotonic() - run_start >= budget_s:
                break
            variant_path = out / "large" / f"force_{step:03d}" / safe_video_name(idx)
            variant, var_meta = run_generation(
                pipeline, cache_video, cfg, prompt, seed,
                (args.width, args.height), args.frame_num, args.sampling_steps,
                args.shift, args.guide_scale, {step}, variant_path,
            )
            metrics = tensor_metrics(reference, variant)
            forced_calls = [x for x in var_meta["probe_logs"] if x["forced_cache"]]
            if len(forced_calls) != 2 or {x["branch"] for x in forced_calls} != {"cond", "uncond"} or any(
                    x["step_idx"] != step for x in forced_calls):
                raise RuntimeError("Single-step intervention contract violated")
            row = {
                "benchmark_index": idx,
                "dimension": item["dimension"],
                "seed": seed,
                "force_step": step,
                **metrics,
                "reference_raw_modulated_rel_l1": branch_average(
                    ref_meta["probe_logs"], step, "raw_modulated_rel_l1"
                ),
                "reference_teacache_rescaled_proxy": branch_average(
                    ref_meta["probe_logs"], step, "teacache_rescaled_proxy"
                ),
                "reference_residual_rel_l1": branch_average(
                    ref_meta["probe_logs"], step, "residual_rel_l1"
                ),
                "reference_local_denoiser_output_rel_l1": branch_average(
                    ref_meta["probe_logs"], step, "local_denoiser_output_rel_l1"
                ),
                "reference_guided_local_output_rms": branch_average(
                    ref_meta["probe_logs"], step, "guided_local_output_rms"
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
                fh.write(json.dumps(row, allow_nan=False) + "\n")
            del variant
            gc.collect()

        del reference
        gc.collect()

    summary = summarize(rows)
    summary["elapsed_seconds"] = time.monotonic() - run_start
    summary["completed_prompt_indices"] = sorted({r["benchmark_index"] for r in rows})
    summary["completed_force_runs"] = len(rows)
    summary["budget_exhausted"] = (time.monotonic() - run_start) >= budget_s
    summary["expected_force_runs"] = len(prompts) * len(force_steps)
    summary["coverage_complete"] = len(rows) == summary["expected_force_runs"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

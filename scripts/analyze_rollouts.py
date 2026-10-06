#!/usr/bin/env python3
"""Analyze GRPO/RLVR rollout groups without training.

Expected JSONL: one record per prompt/group.
Required field: rewards (list[float]).
Optional fields: token_counts, prompt_id, metadata.
"""
import argparse, json
from pathlib import Path

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def var(xs):
    if not xs:
        return float("nan")
    m = mean(xs)
    return sum((x-m)**2 for x in xs) / len(xs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="?")
    ap.add_argument("--input", dest="input_path")
    ap.add_argument("--output")
    args = ap.parse_args()
    src = args.input_path or args.jsonl
    if not src:
        ap.error("provide JSONL path positionally or with --input")

    groups = []
    for line in Path(src).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rewards = [float(x) for x in r["rewards"]]
        toks = r.get("token_counts")
        groups.append({
            "prompt_id": r.get("prompt_id"),
            "n": len(rewards),
            "mean_reward": mean(rewards),
            "reward_variance": var(rewards),
            "zero_signal": max(rewards) == min(rewards),
            "tokens": sum(toks) if toks else None,
        })

    zero = sum(g["zero_signal"] for g in groups)
    total_tokens = sum(g["tokens"] or 0 for g in groups)
    informative_tokens = sum((g["tokens"] or 0) for g in groups if not g["zero_signal"])

    binary_pred = []
    for g in groups:
        p, G = g["mean_reward"], g["n"]
        if 0 <= p <= 1:
            binary_pred.append(p**G + (1-p)**G)

    report = {
        "groups": len(groups),
        "zero_signal_groups": zero,
        "zero_signal_fraction": zero / len(groups) if groups else None,
        "total_rollout_tokens": total_tokens or None,
        "informative_rollout_tokens": informative_tokens or None,
        "informative_token_fraction": informative_tokens / total_tokens if total_tokens else None,
        "mean_reward_variance": mean([g["reward_variance"] for g in groups]) if groups else None,
        "mean_binary_zero_signal_plugin_prediction": mean(binary_pred) if binary_pred else None,
        "per_group": groups,
    }

    payload = json.dumps(report, indent=2)
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(payload + "\n")
    print(payload)

if __name__ == "__main__":
    main()

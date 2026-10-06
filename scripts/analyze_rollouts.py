#!/usr/bin/env python3
"""Summarize existing GRPO/RLVR logs; this is not a benchmark evaluator.

JSONL: one record per prompt/group, rewards=list of finite numbers (G >= 2),
optional token_counts=list of G nonnegative integers. Equal reward removes the
group-normalized reward contrast, not necessarily the KL/other gradient terms.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def variance(xs):
    center = mean(xs)
    return sum((x - center) ** 2 for x in xs) / len(xs)


def summarize(records, binary_success_probability=None):
    if binary_success_probability is not None and not (
        math.isfinite(binary_success_probability) and 0 <= binary_success_probability <= 1
    ):
        raise ValueError("binary success probability must be finite and in [0, 1]")
    groups, plugin_predictions, iid_predictions = [], [], []
    for line_number, record in records:
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number}: expected a JSON object")
        rewards = record.get("rewards")
        if not isinstance(rewards, list) or len(rewards) < 2:
            raise ValueError(f"line {line_number}: rewards must contain at least two outcomes")
        if any(isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) for r in rewards):
            raise ValueError(f"line {line_number}: rewards must be finite numbers")
        rewards = [float(r) for r in rewards]
        tokens = record.get("token_counts")
        if tokens is not None:
            if not isinstance(tokens, list) or len(tokens) != len(rewards):
                raise ValueError(f"line {line_number}: token_counts must match rewards length")
            if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in tokens):
                raise ValueError(f"line {line_number}: token counts must be nonnegative integers")
        is_binary = all(r in (0.0, 1.0) for r in rewards)
        group = {"source_line": line_number, "prompt_id": record.get("prompt_id"),
                 "n": len(rewards), "mean_reward": mean(rewards),
                 "reward_variance": variance(rewards),
                 "zero_signal": max(rewards) == min(rewards),
                 "binary_rewards": is_binary,
                 "tokens": sum(tokens) if tokens is not None else None}
        groups.append(group)
        if is_binary:
            probability, size = group["mean_reward"], group["n"]
            plugin_predictions.append(probability ** size + (1 - probability) ** size)
            if binary_success_probability is not None:
                p = binary_success_probability
                iid_predictions.append(p ** size + (1 - p) ** size)
    observed = [g for g in groups if g["tokens"] is not None]
    total_tokens = sum(g["tokens"] for g in observed)
    informative_tokens = sum(g["tokens"] for g in observed if not g["zero_signal"])
    all_tokens_observed = len(observed) == len(groups)
    zero = sum(g["zero_signal"] for g in groups)
    binary = sum(g["binary_rewards"] for g in groups)
    return {
        "groups": len(groups), "zero_signal_groups": zero,
        "zero_signal_fraction": zero / len(groups) if groups else None,
        "zero_signal_definition": "equal group rewards; reward contrast only, not all optimizer gradients",
        "mean_reward_variance": mean([g["reward_variance"] for g in groups]),
        "binary_reward_groups": binary, "nonbinary_reward_groups": len(groups) - binary,
        "mean_binary_zero_signal_plugin_prediction": mean(plugin_predictions),
        "binary_plugin_prediction_groups": len(plugin_predictions),
        "binary_plugin_interpretation": "descriptive plug-in using the same group; not an unbiased or independently calibrated prediction",
        "external_binary_success_probability": binary_success_probability,
        "mean_binary_zero_signal_iid_prediction": mean(iid_predictions),
        "iid_prediction_assumptions": "binary outcomes, conditional independent identically distributed rollouts, externally supplied p",
        "groups_with_token_counts": len(observed), "groups_missing_token_counts": len(groups) - len(observed),
        "token_coverage_fraction": len(observed) / len(groups) if groups else None,
        "total_rollout_tokens": total_tokens if observed else None,
        "informative_rollout_tokens": informative_tokens if observed else None,
        "informative_token_fraction": informative_tokens / total_tokens if total_tokens and all_tokens_observed else None,
        "informative_token_fraction_observed_subset": informative_tokens / total_tokens if total_tokens else None,
        "token_fraction_scope": "all groups" if all_tokens_observed else "only groups with recorded token_counts; missingness may be biased",
        "per_group": groups,
        "scientific_verdict": "none; existing-log descriptive diagnostic only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="?")
    parser.add_argument("--input", dest="input_path")
    parser.add_argument("--output")
    parser.add_argument("--binary-success-probability", type=float,
                        help="Independent externally known p; used only for truly binary groups")
    args = parser.parse_args()
    src = args.input_path or args.jsonl
    if not src:
        parser.error("provide JSONL path positionally or with --input")
    path = Path(src)
    hasher = hashlib.sha256()
    def records():
        with path.open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                hasher.update(raw)
                if raw.strip():
                    try:
                        record = json.loads(raw)
                    except (ValueError, UnicodeDecodeError) as error:
                        raise ValueError(f"line {number}: invalid JSON") from error
                    yield number, record
    try:
        report = summarize(records(), args.binary_success_probability)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    report["source_sha256"] = hasher.hexdigest()
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(destination)
    print(payload)


if __name__ == "__main__":
    main()

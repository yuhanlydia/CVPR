# Round 001 — Resource + Signal Readiness

## Scientific purpose

Before selecting a 1–7B multimodal training recipe, establish two facts:

1. What GPU/CUDA/PyTorch environment is actually available on the SSH host?
2. If prior RLVR/GRPO rollouts exist, how much rollout compute is spent on groups with zero reward variance?

This round is diagnostic only. It does not claim novelty or model improvement.

## Required outputs

- `artifacts/round_001/host.json`
- optionally `artifacts/round_001/rollout_signal.json`
- `artifacts/round_001/RESULT.md`

## Go / stop rule

Proceed to model-specific code only after:
- the GPU inventory is observed from the actual host;
- a model fits the memory envelope with explicit headroom;
- the first experiment has a native benchmark and a qualified simple baseline.

For the RL-information route, continue only if observed rollout logs show a meaningful fraction of wasted or low-information compute *after* accounting for existing dynamic-sampling baselines. Otherwise deprioritize it.

## Local-agent instructions

Run:

```bash
python3 scripts/inspect_host.py --output artifacts/round_001/host.json
```

If rollout logs exist:

```bash
python3 scripts/analyze_rollouts.py \
  --input /path/to/rollouts.jsonl \
  --output artifacts/round_001/rollout_signal.json
```

Then write `artifacts/round_001/RESULT.md` with exact commands, commit SHA, timing, exit codes, GPU summary, raw artifact paths, and any deviations. Commit and push the packet back to this repository.

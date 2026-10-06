# CVPR Research Autopilot

Web research/design -> GitHub -> local agent -> SSH GPU -> GitHub results -> web review.

## Current operating contract

- Web side (this repo): research selection, mathematical derivation, experiment design, code generation.
- Local agent: pulls an approved commit, connects to the GPU host over SSH, executes only the finite plan in this repo, collects raw outputs, and pushes a result packet back.
- GPU host: execution only. Do not invent new research branches during a run.
- Review boundary: after each finite batch, return a commit + result packet. New branches are selected after review.

## Round 001

Round 001 is intentionally hardware-agnostic. It does **not** train a model yet.

1. Inspect the actual GPU host and Python/CUDA stack.
2. If prior GRPO/RLVR rollout logs exist, analyze reward-information efficiency with the included diagnostic.
3. Save raw machine-readable artifacts under `artifacts/round_001/`.
4. Commit/push the result packet.
5. Web research then chooses the first model-specific 1–7B experiment.

Why: Research Autopilot requires real resource evidence before sizing the 8-hour queue, and avoids spending GPU time before the decisive diagnostic is specified.

## Local-agent start

Read [LOCAL_AGENT.md](LOCAL_AGENT.md), then execute:

```bash
python3 scripts/inspect_host.py --output artifacts/round_001/host.json
```

If the local agent itself is not running on the GPU host, use:

```bash
GPU_HOST=user@hostname GPU_PROJECT_DIR=/path/to/CVPR \
  bash scripts/run_remote.sh \
  "python3 scripts/inspect_host.py --output artifacts/round_001/host.json"
```

For an existing rollout JSONL:

```bash
python3 scripts/analyze_rollouts.py \
  --input /path/to/rollouts.jsonl \
  --output artifacts/round_001/rollout_signal.json
```

See [research/ROUND_001.md](research/ROUND_001.md) for the scientific purpose and stop/go rule.

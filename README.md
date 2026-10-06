# CVPR Research Autopilot

Web research/design -> GitHub -> local agent -> SSH GPU -> GitHub results -> web review.

## Current hardware envelope

- 1 × NVIDIA RTX 2080 Ti
- 22 GB reported VRAM
- 1–7B models are in scope
- Turing GPU: use FP16 for the Wan DiT path; do not assume native BF16

## Active experiment: Round 002

Round 002 is the first real model experiment. It uses Wan2.1-T2V-1.3B and asks a falsifiable question:

> Does the same-sized local cache approximation cause very different final damage depending on the diffusion timestep where it is injected?

The experiment keeps prompt, seed, solver, CFG and sampling budget fixed, forces cache reuse at exactly one timestep, then resumes exact computation. It measures local approximation error and terminal decoded-video error.

The mathematical and kill/continue contract is in:

- research/ROUND_002_WAN_CACHE_PROPAGATION.md
- research/SOURCES_ROUND_002.md
- research/RESOURCE_BRIEF.md

## One-time source setup

From this repository:

~~~bash
INSTALL_WAN_DEPS=1 DOWNLOAD_WAN_MODEL=1   bash experiments/wan_cache/setup_sources.sh
~~~

If the checkpoint is already present, omit DOWNLOAD_WAN_MODEL.

Then set:

~~~bash
export WAN_ROOT=$PWD/external/Wan2.1
export VBENCH_ROOT=$PWD/external/VBench
export WAN_CKPT=$PWD/external/Wan2.1-T2V-1.3B
~~~

## Execute the 8-hour-bounded round

~~~bash
bash experiments/wan_cache/run_round_002.sh
~~~

Defaults:

- 3 released VBench prompts selected deterministically from the official prompt JSON
- Wan2.1-T2V-1.3B
- 832×480
- 81 frames
- 50 sampling steps
- CFG 6
- shift 8
- five single-step cache interventions
- 7.5-hour internal wall-time boundary

Override examples:

~~~bash
NUM_PROMPTS=1 MAX_WALL_HOURS=2 bash experiments/wan_cache/run_round_002.sh
FORCE_STEPS=5,15,25,35,45 bash experiments/wan_cache/run_round_002.sh
~~~

## Result handoff

Generated MP4s live under artifacts/round_002/large/ and are intentionally gitignored.

Commit/push the small evidence files:

- artifacts/round_002/host.json
- artifacts/round_002/manifest.json
- artifacts/round_002/metrics.jsonl
- artifacts/round_002/summary.json
- artifacts/round_002/prompt_*/reference_probe.json
- artifacts/round_002/run.log
- artifacts/round_002/RESULT.md

The web supervisor then reads that exact result commit and applies the pre-registered KILL / CONTINUE decision before writing any propagation-aware cache method.

## Repository workflow

1. Web supervisor writes a pinned research design and executable code.
2. Local agent pulls the exact commit and uses the user's existing SSH access.
3. GPU host executes only the frozen batch.
4. Local agent commits raw small evidence and a result packet.
5. Web supervisor reads the exact returned commit, verifies the run, and writes the next round.

Do not invent a new research branch on the GPU host during a frozen batch.

# Resource Brief

## Confirmed accelerator envelope

- GPU: 1 × NVIDIA GeForce RTX 2080 Ti
- Reported VRAM: 22 GB
- Project model scope: 1–7B
- 7B models: **in scope**
- Default precision on this Turing-class GPU: FP16 rather than BF16

## What "7B is in scope" means

We may use a 7B multimodal model as a primary research model when the actual measured memory fits. The default training strategies are:

- LoRA / QLoRA
- frozen-backbone or partial-module training
- gradient checkpointing / small micro-batches when needed
- quantized inference for rollout/evaluation when useful

We do **not** interpret 7B support as permission to assume full-parameter Adam training fits in 22 GB.

## Research scheduling implication

Use 2B/4B for cheap falsification when a mechanism can be tested there, but do not downgrade the main scientific target solely to save memory. A promising effect should be confirmed on a 7B-class model when the benchmark and implementation support it.

The local agent should still run `scripts/inspect_host.py` so each execution packet records actual driver/CUDA/PyTorch state and usable VRAM.

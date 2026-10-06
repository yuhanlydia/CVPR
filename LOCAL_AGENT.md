# Local Agent Contract

You are the execution worker for this research project.

## Responsibilities

- Pull the exact approved GitHub commit.
- Use the user's existing SSH setup to reach the authorized GPU host.
- Run the frozen experiment in this repository; do not invent a new research direction during execution.
- Preserve exact commit, commands, environment, GPU inventory, config, logs, elapsed time and raw metrics.
- Keep generated videos local unless the user explicitly requests otherwise.
- Commit/push small result artifacts back to the same project repository.
- Do not interpret execution success as a scientific conclusion.

## Active task: Round 002

Read these first:

- research/RESOURCE_BRIEF.md
- research/SOURCES_ROUND_002.md
- research/ROUND_002_WAN_CACHE_PROPAGATION.md

### Setup

~~~bash
INSTALL_WAN_DEPS=1 DOWNLOAD_WAN_MODEL=1   bash experiments/wan_cache/setup_sources.sh

export WAN_ROOT=$PWD/external/Wan2.1
export VBENCH_ROOT=$PWD/external/VBench
export WAN_CKPT=$PWD/external/Wan2.1-T2V-1.3B
~~~

If the model already exists, point WAN_CKPT to it and do not download it again.

### Execute

~~~bash
bash experiments/wan_cache/run_round_002.sh
~~~

The runner will:

1. record host/GPU state;
2. load pinned Wan2.1-T2V-1.3B;
3. force the DiT path to FP16 for RTX 2080 Ti compatibility;
4. draw a deterministic prospective subset from the released VBench prompt file;
5. run an exact reference for each prompt;
6. force one cache substitution at selected diffusion steps;
7. compare local denoiser error against final decoded-video error;
8. stop starting new interventions at the configured wall-time boundary;
9. write metrics and the summary under artifacts/round_002/.

### Return to GitHub

Do not add artifacts/round_002/large/.

Commit the remaining Round 002 artifact files and push them. In the commit message include "Round 002 results".

The next web round must read the exact result commit before choosing KILL or CONTINUE.

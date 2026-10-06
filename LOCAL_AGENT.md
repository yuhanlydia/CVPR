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
python3 -m venv .venv-wan
source .venv-wan/bin/activate
INSTALL_WAN_DEPS=1 DOWNLOAD_WAN_MODEL=1 bash experiments/wan_cache/setup_sources.sh

export WAN_ROOT=$PWD/external/Wan2.1
export VBENCH_ROOT=$PWD/external/VBench
export WAN_CKPT=$PWD/external/Wan2.1-T2V-1.3B
~~~

If the model already exists, point WAN_CKPT to it and do not download it again.
Use the existing complete Research Autopilot skill on this host. Set
RESEARCH_AUTOPILOT_ROOT if it is installed outside the two documented locations.

### Execute

~~~bash
bash experiments/wan_cache/run_round_002.sh
~~~

The default is one released prompt, one step-5 intervention, and at most two
hours for initial engineering qualification. The inherited 3-prompt/5-step
inventory remains a conditional broader plan, not an automatic next attempt.

The runner will:

1. record host/GPU state;
2. load pinned Wan2.1-T2V-1.3B;
3. use FP16 autocast/efficient SDPA with native FP32 weights;
4. draw a deterministic prospective subset from the released VBench prompt file;
5. check native forward parity on the first real CFG pair and run a full reference;
6. force one cache substitution at selected diffusion steps;
7. compare local denoiser error against final decoded-video error;
8. use the native watchdog's hard deadline, including loading, and retain partial evidence;
9. write a unique result packet under artifacts/round_002/RUN_ID/.

One retained eight-hour clock covers setup and execution. One attempt and no
automatic retries; do not launch the optional Qwen route in the same window.

### Return to GitHub

Do not add artifacts/round_002/large/.

Commit the remaining Round 002 artifact files and push them. In the commit message include "Round 002 results".

The next web round reads the exact result commit and validates coverage/receipts.
Native VBench scoring and stronger scientific controls remain pending; this
engineering qualification cannot choose scientific KILL/CONTINUE.

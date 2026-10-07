# CVPR：小模型多模态研究

最新增量：[r004的8组实验](rounds/r004/EXPERIMENT_DESIGN.md)覆盖所有历史原型，
新增25个对照配置和Local接受/收集代码；43条登记中42条预计可计算，I01仍条件阻塞。
[运行交付](rounds/r004/WEB_HANDOFF.md)、[数学推导](rounds/r004/MATH_CARDS.md)、
[配置](configs/audit_controls_20261007.json)已补齐。
本轮使用作者最新[a8343ae](research/AUTOPILOT_DELTA_2026-10-07_a8343ae.md)，
全部新增文件为generated_unexecuted，未运行项目/tests/模型，科学/资源接受仍待Local。
现有方法审计不人为重开20→15；下方前次发现流程和历史状态保留。


最新长期目标：按 [最新版 Research Autopilot](research/AUTOPILOT_UPGRADE_2026-10-07.md)
完成约20数学候选逐项审查、全池排序选前15，再落实方法代码与完整原生实验设计。
目前保留20张历史卡、12个开发原型及6个对照；当前前15选择与科学准入尚未完成。

## Local Codex: start here

先读 [LOCAL_AGENT_RUNBOOK.md](LOCAL_AGENT_RUNBOOK.md)、
[当前 Web handoff](rounds/r003/WEB_HANDOFF.md) 和 [AGENTS.md](AGENTS.md)。
[源码获取与完整skill安装](LOCAL_AGENT_RUNBOOK.md#下载与完整-skill-安装)
旧1b4b8029绑定保留；[最新99d8ac0适用性审查](research/AUTOPILOT_DELTA_2026-10-07_99d8ac0.md)
与完整新来源锁已新增。
[Download datasets and models](docs/NATIVE_ASSET_RUNBOOK.md#download-datasets-and-models)
给出保留输入的不可变文件清单、实际下载/校验/抽取源码、Conda与离线原生接口命令。
[20卡逐项审查](research/reviews/R003_REVIEW_2026-10-07.md)已补15项条件构造与5项pending，
[完整审查优先级排序](research/reviews/r003-ranking-20261007.json)不是verified top15；
科学前提、完整G01/额外强对照和Local接受仍待补齐。
新增文件为**generated_unexecuted**，Web未执行新代码、tests或模型。

[Web代码完成目标](LONG_TERM_TASK.md) 与 [续做清单](research-backlog.json)
由Web GPT尽快完成代码与完整运行命令并交付**main**，用户local agent负责接受/执行。
旧每日8点任务已停用，当前为一次性代码完成任务。
全程不用Docker。活动Local任务保持原版本、原始记录和累计预算。

- [历史20张数学卡](rounds/r003/IDEAS.md)、[来源](rounds/r003/SOURCES.md)、[原计划](rounds/r003/PLAN.md)
- [历史12项及6对照配置](configs/candidates.json)
- [旧运行说明](docs/LOCAL_AGENT_HANDOFF.md)，须同时满足新版接受前提
- tools/plan_candidate_carryover.py：只读审计，自动续跑未实现

载体仍为冻结Qwen3-VL-Embedding-2B，保留ScienceQA、ChartQA、MSCOCO_i2t的
完整原生候选、标签、分母与官方RankingMetrics。历史小头不等于完善的CVPR实验；
数学、代码、Local接受、GPU与科学结果分别报告。

## 前一批基线与 Wan 资格化记录

Web research/design -> GitHub -> local agent -> SSH GPU -> GitHub results -> web review.

## Current hardware envelope

- 1 × NVIDIA RTX 2080 Ti
- 22 GB reported VRAM
- 1–7B models are in scope
- Turing GPU: use FP16 for the Wan DiT path; do not assume native BF16

## Active experiment: Round 002

Status at review of `2f4e96d`: model code and prerequisite checks are committed;
no GPU execution or scientific result is recorded. This review branch adds
bounded engineering qualification. CUDA loading, memory and kernel support
remain to be checked on the user's actual host.

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
python3 -m venv .venv-wan
source .venv-wan/bin/activate
INSTALL_WAN_DEPS=1 DOWNLOAD_WAN_MODEL=1 bash experiments/wan_cache/setup_sources.sh
~~~

If the checkpoint is already present, omit DOWNLOAD_WAN_MODEL.

Use this separate environment: the optional Qwen route uses NumPy 2, while the
pinned Wan checkout requires NumPy below 2. Setup excludes FlashAttention 2,
uses the cu126 PyTorch 2.8 wheel and retains a bounded setup clock. The real
checkpoint bytes are checked against the pinned public HF revision before loading.

Then set:

~~~bash
export WAN_ROOT=$PWD/external/Wan2.1
export VBENCH_ROOT=$PWD/external/VBench
export WAN_CKPT=$PWD/external/Wan2.1-T2V-1.3B
~~~

The GPU host must have the user's existing complete Research Autopilot skill.
Set `RESEARCH_AUTOPILOT_ROOT=/absolute/path/to/research-autopilot` if it is not
installed in `~/.agents/skills/research-autopilot` or `~/.codex/skills/research-autopilot`.
The skill's private source is not included in this repository.

## Execute the 8-hour-bounded round

~~~bash
bash experiments/wan_cache/run_round_002.sh
~~~

Defaults:

- first qualification: 1 released VBench prompt selected from the official prompt JSON
- Wan2.1-T2V-1.3B
- 832×480
- 81 frames
- 50 sampling steps
- CFG 6
- shift 8
- one intervention at step 5; preserve the inherited 50 steps, 81 frames and resolution
- first qualification: up to 2 hours including preflight and loading; absolute model cap 7.5 hours
- one attempt, zero automatic retries and a GPU lock shared with the optional Qwen route
- setup and execution retain the same original eight-hour clock

FP16 refers to DiT autocast and SDPA. Native FP32 weights are retained because
Wan's time, output and normalization operations require them. The Turing adapter
fails if the memory-efficient SDPA kernel is unavailable; it never silently uses
quadratic math attention. Real reference inputs check both CFG branches against
the native forward under the same adapter before interventions.

Override examples:

~~~bash
NUM_PROMPTS=1 MAX_WALL_HOURS=2 bash experiments/wan_cache/run_round_002.sh
# Only after reviewing real qualification/novelty evidence, freeze a broader window:
NUM_PROMPTS=3 MAX_WALL_HOURS=7.5 FORCE_STEPS=5,15,24,34,44 bash experiments/wan_cache/run_round_002.sh
~~~

## Result handoff

Generated MP4s stay in the isolated native attempt's `workspace/out/large/`.

Commit/push the small evidence files:

- `artifacts/round_002/RUN_ID/`: host, preflight, manifest, metrics, summary, raw reference probes and logs
- `artifacts/round_002/RUN_ID/`: frozen plan, native receipt, attempt and RESULT.md

`COMPLETE_ENGINEERING` requires the entire frozen prompt/step inventory and real
reference parity; timeout or partial output is carryover/incomplete. Scientific
KILL/CONTINUE remains blocked until native VBench scoring, stronger controls,
prospective numerical decision thresholds and the closest-work audit are ready.

Review evidence: [research/REVIEW_2026-10-06.md](research/REVIEW_2026-10-06.md).
Originality blocker: [RA-CFGCache](https://arxiv.org/html/2609.36433v1).
Treat the current question as a baseline/reproduction candidate.
Eight conditional research cards: [rounds/r001/IDEAS.md](rounds/r001/IDEAS.md).
The optional 2B embedding qualification route is in [docs/BASELINE_PACKAGE.md](docs/BASELINE_PACKAGE.md);
7B remains in scope after real resource calibration. Run one route per retained window.

## Repository workflow

1. Web supervisor writes a pinned research design and executable code.
2. Local agent pulls the exact commit and uses the user's existing SSH access.
3. GPU host executes only the frozen batch.
4. Local agent commits raw small evidence and a result packet.
5. Web supervisor reads the exact returned commit, verifies the run, and writes the next round.

Do not invent a new research branch on the GPU host during a frozen batch.

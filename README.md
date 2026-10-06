# CVPR：小模型多模态研究

当前批次是 **20 个数学候选 → 筛选 12 项 → 开发原型 → 有界连续队列**。

**最新职责：助手只准备 GitHub 代码和命令，用户的 local agent 执行。**
长期任务继续逐步补代码；不主动连接 SSH/GPU 或运行实验，全程不使用 Docker。
直接转发 [local agent 运行说明与完整命令](docs/LOCAL_AGENT_HANDOFF.md)。
见 [长期代码任务](LONG_TERM_TASK.md) 和 [未完成任务清单](research-backlog.json)。

- [20 张数学卡与筛选理由](rounds/r003/IDEAS.md)
- [官方源码读取记录与近作边界](rounds/r003/SOURCES.md)
- [运行、预算、对照和返回命令](rounds/r003/PLAN.md)
- [冻结的 12 项及 6 个对照](configs/candidates.json)

载体为冻结 Qwen3-VL-Embedding-2B，独立公开训练数据拟合小头，测试保留完整原生候选与
官方 RankingMetrics。全部候选的自然失败/原创性/科学资格仍待实际证据；
数学推导、代码完成、CPU 检查、GPU 实验分开报告。

实际基线与原始训练图像就绪后，从工程根运行：

~~~bash
.venv/bin/python tools/run_candidates.py \
  --baseline-out "$BASELINE_OUT" \
  --train-image-root "$MMEB_TRAIN_IMAGES" \
  --hours 8 --execute
.venv/bin/python tools/collect_candidates.py
~~~

每项失败/单项超时后继续其他项，所有结果/错误留下独立记录，零自动重试。
共享输入缺失则逐项阻塞；累计 8 小时到期、用户取消或未清理进程时停止并保留未完成清单。
当前没有真实 GPU 结果。CPU CI 状态以对应 GitHub commit 的检查记录为准；
完整私有 Research Autopilot 未安装的环境会明确跳过原生运行器集成检查。

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

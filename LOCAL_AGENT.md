# 最新流程：GitHub 代码 → 用户的 local agent 执行

用户明确要求网页助手只生成 code 与命令，不代跑实验；由用户的 local agent 运行。
最新代码交付到 main；拉取 main 或本对话给出的精确 main commit，记录 git rev-parse HEAD。
先读 docs/LOCAL_AGENT_HANDOFF.md、LONG_TERM_TASK.md 和 research-backlog.json。
已有活动任务保持它的原执行版本，完成/协调后再更新，不覆盖本地修改或旧运行记录。
网页侧长期任务只修改/检查/交付代码和命令，不 SSH、不加载模型、不派发 GPU。
local agent 按本对话实际交付 commit 和已有授权执行下面有限批次，不使用 Docker/其他容器。
准备 manifest 保持真实 native attempt 布局及 producer receipt；保留原始累计时钟、
已完成/失败记录和全部原生预测；将小报告与返回包送回本对话。
tools/plan_candidate_carryover.py 可只读审计旧 batch 的预算、版本与尚未尝试清单。
它不派发任务；跨窗口自动续跑尚未实现，不删除 marker/receipt 或复跑整批来绕过时限。

# 当前任务：r003 候选与有限开发批次

最新用户授权是核对并实现 20 → 10–15 → code → 单项报错继续，本批固定筛选 12 项。
先读 rounds/r003/IDEAS.md、SOURCES.md、PLAN.md 和 configs/candidates.json。
执行 tools/run_candidates.py 的有限范围，保留全部失败/阻塞/超时/未完成项，零自动重试。
复用原始 setup-receipt.json 时钟，不删除 marker 或 receipt 来扩大累计预算。
使用已安装的完整 Research Autopilot，保留 GitHub 交付目标与不向 HF 发布输出的偏好。

12 个小头是用户要求的开发原型，当前没有 Natural Gate 0/IPCG/Gate A PASS。
不得把 exit 0、CPU 测试或单任务开发分数称为科学验证。测试数据不得进入拟合/参数选择。
所有评分继续用实际官方 MMEB 数据、候选、标签和 RankingMetrics。

以下 r001/r002 记录保留为历史/资格化入口；当前入口以上面 r003 为准。


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

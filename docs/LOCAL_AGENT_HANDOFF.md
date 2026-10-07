# 新版入口优先（2026-10-07）

先读[Local runbook](../LOCAL_AGENT_RUNBOOK.md)、[当前Web handoff](../rounds/r003/WEB_HANDOFF.md)
及[升级审查](../research/AUTOPILOT_UPGRADE_2026-10-07.md)。下方保留历史原型命令；
当前验证池/前15选择、完整G01与Local接受未完成，旧命令本身不能认定新版批次准入。
已有运行保持原版本与原始预算，不因来源升级重跑或重置。

## 以下为历史r003命令与证据契约

# 给 local agent：r003 代码和用户侧运行命令

最新职责：网页助手只生成/审查代码与命令并交付 GitHub；实际安装、SSH、模型、训练特征、
原生评测和收集由用户的 local agent 执行。全程不使用 Docker 或其他容器。
当前默认交付入口是本仓库 main；local agent 拉取 main，并记录实际执行的 commit。
review/wan-readiness-20261006 与 PR #1 保留历史。本轮代码属于开发原型，GPU/科学验证仍待实际证据。

## 可直接转发的执行说明
请读取本文件、LOCAL_AGENT.md、rounds/r003/PLAN.md、IDEAS.md、SOURCES.md 和
configs/candidates.json。拉取本对话给出的实际交付 commit，记录 git rev-parse HEAD。
目标是固定 20 个数学候选中选定的 12 个开发原型与 6 个对照，保持官方完整数据、候选、标签、
分母和 RankingMetrics。先核对已有环境、完整私有 Research Autopilot、原始图像、运行状态及累计预算。
有已有 setup-receipt/attempt 时保留，不自动重置、不重复完成任务；有活跃任务不切换它的代码。
按下方命令执行，单项失败继续独立项，收集全部状态/预测/失败记录；不要静默改参数/seed/分辨率。
返回小报告和原始证据包，GPU 执行与真实性能仅按实际 receipt 报告。
任何缺失资源请定位具体路径/依赖，不用假图片、假标签或伪造校准替代。
不把私有 skill、模型权重、图片和大型缓存提交到公开 GitHub。

## 1. 拉代码
首次执行且没有旧运行记录时，新目录使用：

~~~bash
git clone --branch main --single-branch https://github.com/yuhanlydia/CVPR.git CVPR-r003
cd CVPR-r003
git rev-parse HEAD
~~~

本对话最终交付的 main SHA 是本轮实际版本；确认 clone 的 HEAD 对应它。
已有干净 main 且没有活动任务时，可用 git pull --ff-only origin main 更新；记录 git rev-parse HEAD。
已有干净 checkout 可 fetch 后选择那个 SHA；有本地修改/运行记录时先保留并核对兼容性。
不能靠新目录或 worktree 丢弃旧累计预算与 attempt。

## 2. 原生环境与输入
GPU 机器需要 Linux、Python 3.11+、Git、实际可用 NVIDIA/CUDA 环境。
完整 Research Autopilot 默认查找 ~/.agents/skills/research-autopilot 和
~/.codex/skills/research-autopilot；其他已有位置用下面变量指定：

~~~bash
export CUDA_VISIBLE_DEVICES=0
export RESEARCH_AUTOPILOT_ROOT=/actual/installed/research-autopilot
export MMEB_TRAIN_IMAGES=/actual/original-MMEB-train
bash tools/bootstrap.sh
~~~

变量中的 /actual/... 必须替换为真实路径；默认目录已被正确查找到时可省略 skill 变量。
MMEB_TRAIN_IMAGES 根下应保留原始 images/... 路径。当前程序不自动下载 47 GB 训练图片包。
最新配置还要求 data/train-metadata/ScienceQA/*.parquet 和
data/train-metadata/A-OKVQA/*.parquet 为 training_revision 固定版本的原始元数据，
由 local agent 提供真实来源；缺文件会阻塞，不会用模拟数据填充。
当前固定两项训练各 128 行，三项完整评测为 ScienceQA、ChartQA、MSCOCO_i2t；
已有只含 ScienceQA 的旧 baseline/bundle 不满足当前配置，不能直接复用。
bootstrap 只用 Python venv，PyTorch 官方 cu126 wheel，固定直接依赖和原始 Qwen 源码。
已有合格环境/源码/时钟时复用；不要为了“重装”重新开始预算。

## 3. 原生 2B 基线
没有合格原始 baseline 时执行一次：

~~~bash
.venv/bin/python tools/run_window.py --hours 2 --execute
.venv/bin/python tools/collect.py
~~~

2 小时指原始 setup 起点后的累计上限，为 r003 留出剩余时间，包含下载/加载/资格化。
有原始官方 eval 图像根可给 run_window.py 增加 --image-root /actual/original-MMEB-eval。
基线失败或不完整则不启动依赖原型，保留日志和 collect.py 的失败返回。
已有真实 completed r001 baseline 时不重复运行；直接使用对应原始 workspace/out。
这不是强制所有基线在 2 小时完成的性能承诺。

从真实最新 native receipt 读取 completed qualify 的输出位置（不是猜文件夹）：

~~~bash
export BASELINE_OUT="$(.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path.cwd().resolve()
latest = json.loads((root/"runs/latest.json").read_text())
receipt = json.loads((root/"runs/attempts"/latest["run_id"]/"receipt.json").read_text())
found = [a for a in receipt["attempts"] if a["trial_id"] == "qualify"
         and a["status"] == "completed" and a["exit_code"] == 0]
if len(found) != 1:
    raise SystemExit("Need one actual completed native qualify attempt")
out = (Path(found[0]["cwd"])/"out").resolve()
out.relative_to(root)
if not (out/"summary.json").is_file():
    raise SystemExit("Actual baseline summary missing")
print(out)
PY
)"
~~~

原型准备程序还会核对真正 qualifier 的 code hash、模型/数据、原生完整分母与 replay，
不是仅凭这个路径认为基线合格。若复用较早而不是 latest 的 baseline，由 local agent
读取那个实际 receipt 后设置 BASELINE_OUT，不自动选部分/失败输出。

## 4. 运行 12 原型和 6 对照，然后无论单项成功/失败均收集
~~~bash
set -o pipefail
mkdir -p runs/logs
candidate_exit=0
.venv/bin/python tools/run_candidates.py \
  --baseline-out "$BASELINE_OUT" \
  --train-image-root "$MMEB_TRAIN_IMAGES" \
  --hours 8 --execute 2>&1 | tee runs/logs/r003-console.log || candidate_exit=$?
.venv/bin/python tools/collect_candidates.py
~~~

8 小时继续同一个原始 setup 时钟；不是基线结束后另加 8 小时。
18 项各最多 600 秒，准备最多 7200 秒，一次尝试、零自动重试。
普通单项失败/专属输入缺失/单项超时继续其他独立项。
原型 exit 非零可以包含已正确记录的 BLOCKED 或失败项，所以仍执行 collector。
用户取消、全局预算结束或未知活动进程停止执行，保留未完成项。
上面 shell 的 candidate_exit 记录退出码；若需脚本级返回，可在收集后使用 exit "$candidate_exit"。

已有真实原生 preparation 输出时可用 --bundle /actual/CVPR/runs/attempts/.../workspace/out/manifest.json，
替代两个准备参数。manifest 必须保持原始 native attempt 布局和实际 producer receipt；
不接受手工拼装的 manifest 或移到任意目录的无来源缓存。
重复候选命令不会自动跨窗口续跑：当前仍需保留旧记录、明确有限 scope 的兼容续做支持。
任何未实现功能不能靠删除 marker 或新窗口 ID 绕过。

## 5. 已有记录的只读续做审计

已有 r003 batch 时先读取真实 latest；需要检查历史批次时再给 --run-id 传入实际打印的 ID：

~~~bash
carryover_exit=0
.venv/bin/python tools/plan_candidate_carryover.py || carryover_exit=$?
~~~

默认只在 stdout 输出 JSON，不创建 setup receipt、marker、attempt 或新窗口。
也可用 --hours 2 等降低同一个原始累计上限，不能加时。
退出码 2 表示缺记录、过期预算、未知/非终态 run、版本/文件损坏或其他阻塞；
退出码 0 只表示本次文件审计无阻塞，不表示 native/GPU 验证通过或研究成功。
报告保留已观察的失败/超时/中断尝试及其 locator，只把从未尝试项列为清单。
已有来源绑定的 manifest 且兼容时列出 eligible_head_inventory；没有准备记录时明确
ORIGINAL_PREPARATION_REQUIRED。准备已经失败不会重新批准一次尝试。
原生 receipt 中的 elapsed seconds 与原始累计墙钟分别报告，GPU 使用秒数仍为 unknown。
文件可在审计后变化，因此该清单仅供复核，不是活动进程检查或派发授权。
本步没有 resume/execute 参数，守卫式续跑执行器仍待实现；不要据此重复 run_candidates.py。
完整私有 native runner 保持用户侧依赖，本命令不复制它，也不把文件一致性当作执行真实性证明。

## 6. 查看与返回
启动输出会打印本批 summary 路径；stdout.log/stderr.log 在各原生 attempt 下。
collector 打印实际文件名：
- rounds/r003/returns/RUN_ID.json：所有状态与探索比较的小报告。
- runs/RUN_ID-return.tar.gz：原始 attempt/receipt/日志/预测与小头证据，不含权重/图片/大型缓存。

将这两个文件返回本对话，或把小报告和实际运行代码 SHA 提交到约定研究分支后提供准确 locator。
额外保留 runs/logs/r003-console.log 与 r001 的返回包。不要只返回最高分。
如果只有 setup/输入错误，也返回相应实际日志和已有 receipt，不声称模型已经运行。

## 交付检查与未验证范围
本轮补上累计 deadline 状态、真实 preparation/code/output receipt 绑定和预测完整性校验；
工程文件完整性测试不等于真实 native producer 证明。
仓库已有 CPU CI 检查编译、数学/队列/文件完整性与 CLI；GPU、模型、数据加载和原生评分由
local agent 实际执行后验证。Natural Gate 0、IPCG、Gate A 和原创性仍待研究证据。

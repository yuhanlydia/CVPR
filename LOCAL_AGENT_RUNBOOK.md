# Local Codex：当前入口

本次交付是最新版 skill 的核对与长期任务续做入口，状态为
**generated_unexecuted / 科学准入待补**。网页侧没有运行新代码、测试或模型。
读取 [Web handoff](rounds/r003/WEB_HANDOFF.md)、
[升级审查](research/AUTOPILOT_UPGRADE_2026-10-07.md)、
[长期目标](LONG_TERM_TASK.md) 和 [当前清单](research-backlog.json)。

## 主机与版本

控制电脑负责 local agent、GitHub 和 SSH；GPU 主机负责原生环境、harness 和实验。
已有记录是单张 2080 Ti、报告 22 GB；本轮未检查主机。
SSH alias、远端项目/运行器/Conda 路径、GPU UUID 与可用容量仍由 Local 恢复实际值，
不要把网页工作区路径当作远端路径。已有任务保持原运行版本。

CVPR 交付到 main，实际执行记录 git rev-parse HEAD。skill 固定为
Yunbo-max/Research_Autopilot @ 1b4b8029b399d8a1d1607b481ea2d1a22d632233，
[来源锁定](configs/autopilot-source.json) 包含四个完整 skill 的文件身份。
未来版本另建审查记录，不静默替换运行中的版本。

## 下载与完整 skill 安装

以下是用户控制电脑上的源码获取/安装说明；网页侧未执行这些命令。
已有克隆、安装或运行记录先核对并保留，不重复创建同名目录。
私有源码需要已有 GitHub 读取权限；若权限缺失，在用户侧使用 GitHub 的官方账户连接，
凭据不写入代码、日志或本对话。源码不提交到 CVPR。

在 CVPR checkout 根目录，首次获取：

~~~bash
mkdir -p sources
git clone https://github.com/Yunbo-max/Research_Autopilot.git sources/Research_Autopilot
git -C sources/Research_Autopilot checkout --detach 1b4b8029b399d8a1d1607b481ea2d1a22d632233
git -C sources/Research_Autopilot rev-parse HEAD
~~~

完整安装按该固定版本的 sources/Research_Autopilot/docs/INSTALL_LOCAL.md 操作：
四个目录 research-autopilot、writing-top-tier-papers、designing-pipeline-figures、
designing-experiment-figures 必须为真实相邻目录；备份放在 skill 扫描目录之外。
保留已有 hooks/settings，/skills 与 /hooks 的发现/信任按真实客户端完成。
远端 GPU 主机只需要已固定的 Python runtime 与项目文件，不需要另装 Codex/GPT。
源代码传输通过已有 SSH，远端路径尚未恢复时不生成猜测的 launch 命令。

从 CVPR 根目录，Local 用实际已安装 skill 根目录做只读源码身份检查：

~~~bash
python3 tools/check_autopilot_source.py --skill-root "$RESEARCH_AUTOPILOT_ROOT"
~~~

RESEARCH_AUTOPILOT_ROOT 必须是实际 research-autopilot 目录，三个依赖在其父目录下。
退出 0 仅表示源码文件匹配，退出 2 会列出缺文件或版本差异；不创建 receipt、时钟或 attempt。
随后在 Local 的实际 skill 源码上执行其 research_nodes.py check；
这也是来源/绑定检查，不是模型或科学验证。

模型与数据的完整下载卡尚未补齐，属于长期任务 LT_NATIVE_HANDOFF：
Qwen/Qwen3-VL-Embedding-2B 的完整权重/processor；
ziyjiang/MMEB_Test_Instruct 的 ScienceQA、ChartQA、MSCOCO_i2t 原生 test；
TIGER-Lab/MMEB-eval 原始 images.zip；
TIGER-Lab/MMEB-train @ 0c3f4b828d347c4e8508339f99530f6c820061fd 的
ScienceQA、A-OKVQA 原始元数据与所需训练图片。
强基线、控制、scorer 的额外输入也必须进入下载清单。
在 immutable revisions、文件覆盖、加载路径和资源记录补齐前，本入口不称为可派发实验方案。

## 接受、运行与返回

旧 [r003 命令](docs/LOCAL_AGENT_HANDOFF.md) 保留用于历史追溯，
不能据此认定新版 skill 的数学/选择/code/design 边界已经通过。
源码核对、全套数学审查与前 15 排序、Natural Gate 0/IPCG、G01、
Local 软件/原生评分资格和 actual harness plan 均需各自证据。
没有这些证据时先进行记录与审查；不通过裸命令启动新的候选科学批次。

完整新实验 runbook 将在 LT_NATIVE_HANDOFF 补齐：
每个 benchmark 的原生 scorer/分母、全套比较、精度依据、输入获取、
harness 命令、输出/日志、故障指向、E04、独立确认与结果包。
当前原生接入/跨窗守卫式续跑未完成；只读 carryover 审计不派发或重置预算。

故障定位：

| 现象 | 实际文件/配置 | 当前动作 |
|---|---|---|
| skill 不完整/版本不符 | configs/autopilot-source.json；check_autopilot_source.py 的 JSON | 核对固定 clone 与四个相邻目录，保留旧版；不假称安装更新成功 |
| 缺训练元数据/图片 | configs/candidates.json；tools/prepare_candidate_bundle.py | 补固定原始输入与完整获取卡；不制造图片或标签 |
| 模型/OOM/精度问题 | tools/host_check.py；tools/source_adapter.py；真实 stderr | Local 通过受限 harness 核对真实主机；改配置需记录影响 |
| 分母/候选/评分不符 | tools/qualify.py；tools/replay_native.py；官方 RankingMetrics | 保留 raw prediction 与 native receipt；修原生路径后重新资格化 |
| 中断/预算到期 | tools/window_budget.py；tools/plan_candidate_carryover.py | 保留所有 attempt 与原始累计时钟；不重复整批或删除 marker |

已有真实返回包按原始执行版本保留。返回实际 source/skill SHA、测试范围、
全部结果与失败、官方 replay/E04 记录和未完成项；只返回最高分不满足审查。


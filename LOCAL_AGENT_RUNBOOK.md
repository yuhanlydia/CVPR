# r004最终目标：全部72配置

用户2026-10-08明确等待全部72配置结果。小批仅是执行顺序，不能作为最终完成。
先读[最新执行政策](rounds/r004/EXECUTION_POLICY.md#2026-10-08补充最终目标仍是全部72配置)，
保持72项×三个原生任务队列，累计报告已评分/部分/失败/阻塞/待跑；条件阻塞不能计为完成。
不重写可用工具或重复已通过检查；在实际授权内持续carryover，保留旧证据与原预算计数。

# 当前优先政策：先跑通、再完善

用户已明确放宽推进节奏。先读[执行政策](rounds/r004/EXECUTION_POLICY.md)，其分阶段顺序优先于下方旧的一次性全量接受要求。
允许先做明确范围的小批开发实验、先保存点估计与真实预测，之后补完整任务/CI/诊断。
不等待72配置全部ready；必要原生有效性、旧成果保护、唯一执行owner和实际预算仍保留。

# 当前缓存阻塞修复

旧prepare缓存遇到源码哈希不匹配时，先读[单个EOF空行兼容修复](rounds/r004/PREPARE_CACHE_COMPATIBILITY.md)。
仅允许明确current/legacy SHA对；保留原receipt/cache，Local真实验收仍pending。
旧18次方法额度不重置，新实验仍需独立有限授权。

# 当前部署恢复入口

若GPU主机缺run_harness.py或GitHub断连，先读[r004部署恢复](rounds/r004/DEPLOYMENT_RECOVERY.md)。
控制电脑可通过已有SSH传送保留本地提交的Git bundle和完整固定skill源码；
r004来源核对须显式--lock configs/autopilot-source-a8343ae.json。
未知pool/owner/累计预算保持pending，不新建pool重置用量。

# 当前入口：r004扩展代码

先读[当前扩展Local交接](rounds/r004/LOCAL_EXTENSION_HANDOFF.md)和[逐臂三任务完整度矩阵](rounds/r004/EXTENSION_COVERAGE.json)。
29新增配置已生成代码，与原43条通过显式--extension-config连接；共72条登记，
包含所有历史方法/控制。48 primary＋124 secondary contrasts覆盖三个完整原生任务。
状态generated_unexecuted；Web未运行代码/tests/模型，实际Local/native/G01/资源接受及结果pending。
skill固定最新a8343ae。旧默认43臂和活动attempt保持原绑定；原18次/累计硬界不重置。
此当前入口适用于新增接受/执行/修复；下面原交接与历史记录保留。

# Local Codex：当前入口

## 当前r004：完整审计与控制交付

先读[r004 Web handoff](rounds/r004/WEB_HANDOFF.md)和
[8组实验设计](rounds/r004/EXPERIMENT_DESIGN.md)；
新source lock为configs/autopilot-source-a8343ae.json，
作者固定a8343aeb4f51303e2eb651081d4fe51c24c5ed3f，四目录297文件，旧活动绑定不替换。
保留全部12原型+6控制，增加25个有明确对照问题的配置；I01仍缺独立校准。
Local工具：qualify_audit_controls.py、prepare_audit_job_cards.py、
run_audit_method.py、collect_audit_controls.py；实际argv、输出、输入卡/故障在r004 handoff。
缓存兼容时为0 GPU CPU任务，无新模型/数据；重建缓存属于真实1 GPU准备和预算接受。
配置dispatch_ready=false。Web没有执行新代码/tests或模型；未知host、累计budget、
native per-group protocol/G01接受不预填。旧18次/硬时钟不重置。
当前为M审计/修复/已知控制；不将旧发现池追认为verified top15，也不为审计凑新20张卡。
下方前次审查、来源与运行记录保留；新增任务以r004的精确交付commit/入口为准。


## 本轮交付（2026-10-07）

已新增[逐卡数学/源码审查](research/reviews/R003_REVIEW_2026-10-07.md)、
[实际20项 batch](evidence/r003-math-20261007/batch.json)和
[完整20项审查优先级排序](research/reviews/r003-ranking-20261007.json)。
15项条件性形式构造经人工审查，I07/I08/I17/I19/I20的方法构造仍 pending；
实际只读 before-code 检查退出2，未通过全池/前15选择，全部 code/design/results/verdict 为 false。
该排序不是 verified selection；Parent/Natural Gate0/IPCG、联合来源/native证据、G01仍待补齐。

[Download datasets and models](docs/NATIVE_ASSET_RUNBOOK.md#download-datasets-and-models) 是本轮
完整保留输入的获取/校验/抽取入口：[不可变文件锁](configs/native-assets.lock.json)包含模型全部18文件、
三项完整test shards、eval ZIP及两项train shards/ZIP。新源码
tools/acquire_native_assets.py、tools/setup_native_env.sh 与 prepare_assets.py/qualify.py 的锁定离线分支
均为 **generated_unexecuted**。详见同页的顺序命令、真实 loader 路径、日志、故障和接受条件。
下载完成不等于软件/原生/science接受；新强对照和未来top15额外资源没有冒称已覆盖。

作者 main 在本轮更新到99d8ac079682e91f61ab59bfd3e760eccb8b5726；
[适用性差异审查](research/AUTOPILOT_DELTA_2026-10-07_99d8ac0.md)与
[当前完整四skill身份锁](configs/autopilot-source-99d8ac0.json)已新增。
旧1b4b8029来源锁、审查证据与活动执行保持不变。
Web未运行任何新代码/tests/模型；Local先恢复真实host/path/剩余预算，接受这次精确main源码。

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

下方为前次交付的待补记录；本轮已补上述保留输入下载卡，未来top15/强对照额外资源与完整G01仍属LT_NATIVE_HANDOFF：
Qwen/Qwen3-VL-Embedding-2B 的完整权重/processor；
ziyjiang/MMEB_Test_Instruct 的 ScienceQA、ChartQA、MSCOCO_i2t 原生 test；
TIGER-Lab/MMEB-eval 原始 images.zip；
TIGER-Lab/MMEB-train @ 0c3f4b828d347c4e8508339f99530f6c820061fd 的
ScienceQA、A-OKVQA 原始元数据与所需训练图片。
强基线、控制、scorer 的额外输入也必须进入下载清单。
保留输入的immutable revision/文件/获取路径现已补齐，Local实际资源和接受、额外比较输入与G01仍缺，因此不是可派发的新科学方案。

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

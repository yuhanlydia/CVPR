# r003 当前 Web 交付：20卡审查与不可变原生输入

科学/源码审查基于f2388ba8e3c4056fd4b19cfa3081aa1cbe7aebce；交付整合parent为
340ea51bf7a33ff7f657308cef9e02a5ddfe981d，其并发人工审查/decision完整保留。
本文件不预填自身未来commit；实际交付由GitHub条件更新及精确回读确定。

本轮实物为 **generated_unexecuted**：
- [20卡与人工语义审查](../../research/reviews/R003_REVIEW_2026-10-07.md)，每卡保留假设、实质推导、构造/缺口、预测、证伪、简单替代及原源码审计；
- [当前20项batch](../../evidence/r003-math-20261007/batch.json)、
  [实际before-code报告](../../evidence/r003-math-20261007/before-code-report.json)：
  15/20条件形式构造，5项pending，exit2，零selected，全池/selection未通过；
- [完整20项审查优先级排序](../../research/reviews/r003-ranking-20261007.json)：
  first15是review tranche，不是verified top15；
- [不可变模型/数据文件锁](../../configs/native-assets.lock.json)与
  [Download datasets and models](../../docs/NATIVE_ASSET_RUNBOOK.md#download-datasets-and-models)：
  全部模型18文件、3项完整test shards、原始eval ZIP、2项train shards/ZIP及官方hash；
- tools/acquire_native_assets.py、tools/setup_native_env.sh、requirements-assets.txt、
  configs/environment-native.yml；prepare_assets.py/qualify.py新增锁定离线loader分支；
- [当前作者升级审查](../../research/AUTOPILOT_DELTA_2026-10-07_99d8ac0.md)与
  [99d8ac0完整四skill来源锁](../../configs/autopilot-source-99d8ac0.json)。
  旧1b4b8029锁/实际审查/活动任务不替换。

先读[AGENTS](../../AGENTS.md)、[Local runbook](../../LOCAL_AGENT_RUNBOOK.md)、
[详细获取/顺序命令/日志/故障/接受条件](../../docs/NATIVE_ASSET_RUNBOOK.md)与
[实际backlog](../../research-backlog.json)。
Local恢复已有SSH/实际远端路径、Conda、活动任务与原始剩余预算后，
使用单一已接受remote run_harness执行源码/软件/输入/原生资格化。
Web未执行生成代码、软件tests、模型或benchmark，也未检查用户SSH/GPU。

I03距离与双线性分数不等价，I16精确RBF不改变单位点积排序，
I17 MMSE不能直接推出旧检索权重；历史程序保留，未被认证或改写成新方法。
I07/I08/I17/I19/I20关键构造仍pending；实际论文/作者源码/native联合证据和
Parent/Natural Gate0/IPCG、完整G01、前15选择、新强对照及Local接受均未关闭。
禁止把15条形式审查、下载资产数或旧CI记录当成科学选择/方法效果。
没有完整科学先决条件时，新候选代码/G01保持阻塞；本次获取接口属独立维护。

训练/test、完整ScienceQA/ChartQA/MSCOCO_i2t、所有原始候选/标签/分母/官方scorer保留。
不降低为smoke、不推断原生分母、不给未知GPU成本造数字。
原始18项/零重试/累计时钟与全部旧attempt/marker/失败保持不变。
未完成的outer harness接受与跨窗守卫派发不声称已有。

## 前次交付记录（保留）

以下是前次来源维护/handoff记录，当前入口和状态以上述内容为准。

# r003 Web handoff：最新版 skill 与 scheduled 长期目标

日期：2026-10-07。来源 main parent：
43c972104a7e022d5356bd73dc1612fd895865c4。
实际交付 commit 由本轮 GitHub 记录及回读确定；本文件不预填自身未来 SHA。

本轮只完成来源核对与长期任务续做入口，generated_unexecuted。
使用 Yunbo-max/Research_Autopilot @
1b4b8029b399d8a1d1607b481ea2d1a22d632233；
不复制私有 skill 源码到 CVPR，四目录身份在
[来源锁定](../../configs/autopilot-source.json)。

先读 [Local runbook](../../LOCAL_AGENT_RUNBOOK.md)、
[源码获取与完整安装](../../LOCAL_AGENT_RUNBOOK.md#下载与完整-skill-安装)、
[升级审查](../../research/AUTOPILOT_UPGRADE_2026-10-07.md)、
[长期目标](../../LONG_TERM_TASK.md) 与 [清单](../../research-backlog.json)。

现有 [20 张卡](IDEAS.md)、[源码/近作记录](SOURCES.md)、
[历史计划](PLAN.md)、[12+6 配置](../../configs/candidates.json)、
experiments/embedding_heads/{heads,bundle}.py 与 tools/run_candidates.py
均保留原身份，不被追认成验证后的前 15 个方法。
本轮没有新增候选方法或修改实验参数，没有 Web 测试/GPU/科学结果。
训练与 test 仍须分离；ScienceQA、ChartQA、MSCOCO_i2t 完整原生覆盖不能缩减。

具体待办：20 张数学卡逐项语义审查与来源绑定、完整排序/前 15 选择；
自然失败与原创性/IPCG；方法边界 packet；构造到源码；
完整 G01/下载卡/资源测量与原生 harness；
Local 软件/native scorer qualification 与 E04/确认。
边界未闭合时禁止新科学准入，但当前来源维护与代码交付可继续。

长期定时任务继续向 main 交付数学、代码和命令；Local 执行实际实验。
保留所有旧尝试、失败、版本和累计限制；当前自动跨窗口派发仍未实现。


## 用户纠正（2026-10-07）

Web GPT负责赶快完成代码，Local负责真实执行；不按每天上午8点推进。
旧每日任务已停用，当前为一次性尽快启动的Web代码完成任务，实际provider身份见
[任务记录](../../research/scheduled-long-goal.json)。此状态只确认安排，不宣称代码已经完整或任务已执行。


## 2026-10-07 实际进度审查

新增 [20 卡数学/源码审查](../../research/MATH_SOURCE_AUDIT_2026-10-07.md)。
人工逐卡核对完成，但当前验证池、排名/前15、完整方法/G01交付尚未完成。
I03 距离目标与点积打分不等价；I09 直接对照混淆目标与曲率；
I17 恢复可靠度不直接证明最优排名；I07/I08/I20 缺完整可执行构造。
旧代码和配置未改，不把该审查作为方法边界通过或新版科学派发入口。

新观察作者 HEAD 为 99d8ac079682e91f61ab59bfd3e760eccb8b5726；
本次使用模块已在 checkpoint 绑定，原完整安装锁保持 1b4b8029，不替换活动版本。
一次性 automation 的 last_run_time/next_run_time 均为 null，未暴露执行/完成回执；
不宣称 hosted 后台任务活跃。Web 当前人工审查与代码的自动完成是不同状态。

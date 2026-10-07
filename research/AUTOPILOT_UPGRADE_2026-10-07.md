# Research Autopilot 最新版审查与长期目标

本轮用户明确要求使用 https://github.com/Yunbo-max/Research_Autopilot 最新 skill，
随后要求使用 scheduled 计划长期在后台推进。
项目目标保持 1–7B 多模态生成/感知、数学机制、真实原生 benchmark 和可复现代码；
GitHub 交付仓库为 yuhanlydia/CVPR 的 main，实际实验由用户 local agent 执行。

## 实际核对

2026-10-07 读取作者仓库 main，HEAD 为
1b4b8029b399d8a1d1607b481ea2d1a22d632233（2026-10-06 21:14:22 UTC），
tree bb86d610beae96497f0b5717586af00e12515d19。
完成整个 source package 298 个 Git blob 的字节身份核对；这是来源核对，
不代表运行 tests、84 个节点任务或模型。实际使用当前 SKILL.md 及
workflow-harness、method-verification、local-code-handoff、
repository-round-trips、local-installation-and-logging、
local-execution-harness 等当前模块，避免旧 reader 条目与新模块混用。

同时读取 CVPR main @ 43c972104a7e022d5356bd73dc1612fd895865c4；
保留该提交的三项完整评测、两项各 128 行训练、32 维投影和历史尝试策略。
本轮未运行生成代码、软件 tests、GPU 或 benchmark，不新造通过记录。

## 新版明确要求与当前缺口

| 要求 | 当前证据与下一步 |
|---|---|
| 约 20 数学候选逐卡审查、完整排序选前 15 后再写方法代码 | r003 有 20 张历史卡、12 实现；没有当前验证池/完整排名/前 15 选择；逐项审查并建立来源绑定，不增加三个名字补数 |
| verify_methods.py 的 code / experiment-design / dispatch 边界 | 当前没有可用 method-batch 及对应审查 packet；缺失保持 pending，不添加假的通过 flag |
| Web 只生成，Local 接受和执行 | 本次是 web_supervisor；新文件 generated_unexecuted；已有历史 CI 不能替新版本背书 |
| literal main 与精确 commit 回读 | 保留当前 main parent，条件更新与回读，不强推或覆盖并发改动 |
| README / AGENTS / Local runbook / WEB_HANDOFF | 本轮加入当前维护入口；完整实验 handoff 仍由 LT_NATIVE_HANDOFF 补齐 |
| 所有模型、数据、强基线与 scorer 下载命令 | 历史入口仍要求手动训练元数据/图片；待补 immutable revision、完整文件、原生 loader 路径、权限/容量/校验卡 |
| 原生环境与一个 remote harness 执行 owner | 用户禁止 Docker；优先既有 Conda；已有 venv/inner runner 是历史资产，最新 outer harness 的部署/接受尚未确认 |
| 完整 G01、E04 与独立确认 | 未完成；工程通过、计数或小头实现不能成为论文结果 |

## 不可被计数替代的目标

长期成功条件是保留/筛选有实质机制与重要问题的候选，交付完整、可由 Local
资格化的真实模型方法与原生比较。32 维小头与 128 行训练只是历史开发范围；
不通过简单增大样本量、添加三种小头或宣称复杂数学来冒充 CVPR 完整实验。
通用 CCA、白化、Nyström、DRO 等已有机制的复用需碰撞审查和自然失败证据。
如果合格想法不足，记录缺口及证伪/暂存理由，继续所需调查，不补占位方法。
本轮不新开启无证据的科学方向、不保证 CVPR 录用或 Google 工作。

## 下一合法步骤

复用 I01–I20 的有效数学与原始源码记录，先审查各卡假设、推导、
构造、独特预测、证伪条件、最简单替代、近作与现有 benchmark；
建立真实 reviewed pool，排序与选择后再补方法/code/G01 的接受证据。
定时任务每次从实际 main 的最早未完成依赖继续，向 main 写入可读进度与交付。


# 长期任务：只交付研究代码和命令

## 最新长期目标与执行规则（2026-10-07）

用户最新纠正：这里是Web GPT，目标是赶快完成代码，不是每天上午8点推进。
已停用旧每日任务，改为尽快启动的一次性Web代码完成任务；不设每日/每小时循环。
Web在该次任务内持续完成已授权代码与完整运行命令，实际实验交给用户Local agent。
保留持久进度与实际交付状态；未完成的工作不能写成已完成，不用周期提醒代替代码交付。

来源：Yunbo-max/Research_Autopilot @ 1b4b8029b399d8a1d1607b481ea2d1a22d632233。
每轮先读取作者最新源码与 CVPR 的实际 main，核对适用变更；保留旧版与活动任务的版本。
见 [升级审查](research/AUTOPILOT_UPGRADE_2026-10-07.md)、
[来源锁定](configs/autopilot-source.json) 与 [当前 Web handoff](rounds/r003/WEB_HANDOFF.md)。

目标：1–7B 多模态生成/感知、有后果的数学机制、实质原创性、原生 benchmark、
完整可复现代码与可信比较。约20数学候选 → 逐卡审查 → 全池排序选前15 →
满足当前科学边界的方法代码与完整G01 → Local接受/执行 → E04/原生评分/独立确认。
现有20卡、12原型与6对照保留为历史，不能追认为验证后的前15选择。
不足记录缺口，不能补三个普通小头凑数，不能以32维/128行旧原型冒充完整CVPR实验。
后续从证据收敛，不每轮重新生成20个方向。

按 research-backlog.json 的最早就绪依赖推进：source/Local接受；
数学/文献/实际源码/benchmark逐卡审查；完整排序/选择packet；
构造到代码、Natural Gate0/IPCG/方法边界、全套G01与强对照；
全部模型/数据/baseline/scorer下载卡；一个原生remote harness的实用命令与Local接受；
用户返回结果后的E04/确认与收敛。每次保存实际可审查进展与下一动作。

Web角色为web_supervisor，只分析、生成与源码审查、向literal main交付并回读；
不执行生成代码、软件tests、模型或benchmark，不连接用户SSH/GPU，不下载大型模型/数据。
新文件为generated_unexecuted；读取历史CI仅限其真实SHA，不能据此认证新代码。
Local控制会话位于用户电脑，以已有SSH操作GPU主机；所有可执行项目任务由一个
run_harness owner运行，优先既有原生Conda。旧venv/inner runner是待审查复用的历史资产。
全程禁止Docker、docker compose、Podman、Singularity、Apptainer和其他容器。
当前完整实验runbook、模型/数据获取卡、G01与outer harness资格仍待补齐。

保留原生完整ScienceQA、ChartQA、MSCOCO_i2t比较，不缩减候选/标签/分母或改评分器。
保留所有失败、不利、缺对照与未完成记录；单项失败继续独立就绪项，依赖故障阻塞后代。
不重置历史累计8小时/attempt/marker，8小时报告窗口与硬截止/总计算界限分开。
历史18项预算不自动扩张为15个方法的预算；新方案按真实资源和科学依赖准入。
tools/plan_candidate_carryover.py只读，守卫式跨窗派发尚未实现。

交付始终读取实际main head、保留并发修改、条件更新并精确commit回读；
不强推、不以别的分支或PR代替main。不公开复制私有skill、不上传权重/图片/大缓存或HF输出。
代码交付、Local接受、GPU执行与科学结论分开报告；不保证CVPR录用或Google工作。

## 以下为已被当前规则覆盖的历史安排

上面的当前规则覆盖下方历史段落中CPU执行、12项选择和临时PR等不一致内容，
原记录保留用于追溯既定身份、预算、旧代码与失败证据。


## 最新职责（2026-10-06）
用户明确要求：“你不用跑只用给我生成code和命令”，并由用户的 local agent 执行。
本要求覆盖此前自动接入 SSH/GPU、资格化、训练特征编码和派发实验的安排。
网页助手负责数学/原始源码分析、代码、可核查运行命令、GitHub 提交与回读；
不连接/检查远程主机、不启动模型/训练/benchmark、不下载大型模型/数据。
代码检查使用静态审查和仓库已有轻量 CPU CI，不能冒充真实实验。
用户侧入口和完整命令见 [docs/LOCAL_AGENT_HANDOFF.md](docs/LOCAL_AGENT_HANDOFF.md)。

## 固定范围与推进方式
1–7B 多模态生成/感知；当前载体冻结 Qwen3-VL-Embedding-2B。
现有 20 个数学候选、12 个开发原型、6 个对照和 8 个暂存方向保持身份，
详见 rounds/r003/IDEAS.md、SOURCES.md、PLAN.md 和 configs/candidates.json。
每天上午（Europe/London）的长期任务完成下一项代码/命令交付，
用户于 2026-10-06 明确要求将修改交付到 main。同范围代码/命令经过适用代码检查后更新 main；
保留其他写入者改动、完成记录、失败历史与实际版本。GPU 未运行不阻塞原型代码交付。
实际 GPU 容量不是代码交付的阻塞，缺失输入明确列在用户运行条件中。
等用户返回结果，再读取实际证据进行分析；不自行生成“跑出来的分数”。

## 禁止 Docker
助手和生成的用户运行路径均不用 Docker、docker compose、Dockerfile 或其他容器。
使用原生 Python venv/现有原生环境；local agent 使用用户已有授权 SSH。
CPU CI 使用 Ubuntu 虚拟机与 setup-python，不加入 container/services。
上游只有容器示例时审查原生依赖和 entrypoint；不能实现则报告具体依赖。
用户的完整私有 Research Autopilot 保持私有，不公开复制运行器到 CVPR。

## 用户侧程序的资源与失败语义
r003 仍是一次有限队列：沿用 setup-receipt.json 原始累计 8 小时时钟；
准备最多 1 次/7200 秒，18 个方法/对照各最多 1 次/600 秒，自动重试 0，尾部预留 120 秒。
单项普通失败、专属输入缺失或单项超时记录后继续独立项；
共享输入故障阻塞依赖项，缺对照保留 PENDING_COMPARISON。
取消、累计时限到期、未知活动进程或输入版本失效时停止相应执行并留证据。
长期生成代码不重置用户 GPU 的时钟或批准无限/付费计算。
保留已完成 attempt、源代码/协议/输入版本及兼容缓存；不删除 marker/receipt。
tools/plan_candidate_carryover.py 已提供只读有限 carryover 审计：核对原始时钟、
实际 native plan/receipt、终态/未知目录、全部旧 attempt、配置/代码/输入/输出 SHA。
当前没有跨窗口自动续跑；审计不派发任务，守卫式执行器仍待后续代码，不能重复命令伪造它。

## 证据和交付
每次读取最新 main，再将同范围通过检查的代码/命令提交或合并到 main 并回读；
不强推、不覆写并发工作，不绕过分支保护。临时分支/PR 是整合方式，不是最终交付位置。
保持原生数据/完整候选/标签/分母、
独立训练、官方 RankingMetrics/live replay 和所有失败/不利结果的代码约束。
代码完成、CPU CI、真实运行和科学结论分开记录。
本轮补上全局 deadline 状态、准备 producer/code/output receipt 和预测 hash 校验；
工程文件测试不证明模型数据正确或真实 native producer 执行过。
GPU 实际运行、显存/时间、原生新分数及 Natural Gate 0/IPCG/Gate A 均待用户证据。
结果影响变更保留子版本，已查看开发赢家不能作为前瞻确认。
不自动扩展新科学方向/写作或发布 HF 输出，提议须绑定真实证据。

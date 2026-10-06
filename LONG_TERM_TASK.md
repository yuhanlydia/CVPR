# 长期任务：只交付研究代码和命令

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
当前没有跨窗口自动续跑，相关有限 carryover 支持仍待后续代码；不能靠重复命令伪造它。

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

# CVPR：小模型多模态研究

目标：在 **1–7B、用户报告的单张 22 GB RTX 2080 Ti** 的约束下，寻找有明确数学问题和实际任务价值的多模态感知/生成研究。首个基线使用冻结 Qwen3-VL-Embedding-2B 以测成本、获得真实错误；7B 仍在主研究范围，需适配代码和实际显存/速度核查后用于确认。

这里已准备第一轮的 **8 条候选推导、完整比较草案、8 小时基线与资源核查入口**。模型实验尚未在交付环境运行；没有性能提升、创新性通过或论文录用的结论。

| 内容 | 入口 |
|---|---|
| 8 条候选：数学机制、假设、反例、强对照 | [IDEAS.md](../rounds/r001/IDEAS.md) |
| 第一轮预算、原生评测、条件队列 | [PLAN.md](../rounds/r001/PLAN.md) |
| 已检查的论文/源码、版本与缺口 | [SOURCES.md](../rounds/r001/SOURCES.md) |
| VS Code + SSH 运行、收集、故障处理 | [REMOTE_RUN.md](REMOTE_RUN.md) |
| 交付时实际完成的检查 | [verification.json](../checks/verification.json) |

## 在 GPU 远程终端启动

首次拉取：

```bash
git clone https://github.com/yuhanlydia/CVPR.git
cd CVPR
export CUDA_VISIBLE_DEVICES=0
bash tools/bootstrap.sh
.venv/bin/python tools/run_window.py --execute
.venv/bin/python tools/collect.py
```

运行入口使用本机完整安装的 Research Autopilot 原生有限任务执行器。它会自动查找 `~/.agents/skills/research-autopilot` 或 `~/.codex/skills/research-autopilot`；其他位置通过 `RESEARCH_AUTOPILOT_ROOT` 指定，见运行指南。仓库不会复制或公开私有 skill 源码。

第一轮执行原始公开模型和 MMEB 数据的下载、成本校准，以及 **ScienceQA、ChartQA、条件准入的 MSCOCO_i2t** 原生基线。完整任务保留上游全部样本及每个查询的候选集，使用上游 `RankingMetrics` 并重放评分。这里的 QA 是 MMEB 的答案检索协议，不能写成生成式 QA 正确率。

8 小时从 `bootstrap.sh` 写入开始时间起计算，包含环境准备、模型/数据获取和本轮执行；超时/失败保留记录。用已测成本决定是否准入完整任务，未完成任务列入报告。不同候选的训练不自动启动；先检查真实失败、训练/开发数据、近作和公平对照。

交付目标为本仓库；本轮不向 Hugging Face 发布产物。公开上游模型/数据仍从 HF 获取。模型权重、图片、特征缓存保留在 GPU 机器，结果收集器只打包文本证据和小报告。

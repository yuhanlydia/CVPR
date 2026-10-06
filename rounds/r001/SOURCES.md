# 来源、读取范围和待补证据

核查日期：2026-10-05。公开推荐已检索近作。这里区分实际源码/全文检查与摘要线索；普通关键词检索不等于原创性审查通过。

## 论文—代码—数据的当前连接

1. [Qwen3-VL-Embedding/Reranker 原论文](https://arxiv.org/html/2601.04720v1)：已检查结构、训练目标、多阶段训练和任务平衡分析。本文已包含 contrastive training 与 reranker distillation；本项目的普通蒸馏/假负例过滤不能作为原创性主张。
2. [作者源码固定版本](https://github.com/QwenLM/Qwen3-VL-Embedding/tree/393e2978d27852b0d0230d6994f37f9c15bed73c)：已读取 embedding/reranker 实现、环境清单、评测 main、参数、图像 YAML、下载脚本、图像 QA/i2t/t2i parser、candidate 构造、collator、scorer。精确文件 blob IDs 在 `source-identities.json`。
3. [MMEB_Test_Instruct](https://huggingface.co/datasets/ziyjiang/MMEB_Test_Instruct)：上游图像 parser 实际使用的 test 元数据；已检查公开目录/数据卡与 parser 标签/候选格式，没有在交付环境下载整批。
4. [MMEB-eval 原始图片](https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/tree/main)：已检查官方 archive 路径/尺寸。运行时提取被该原始元数据引用的图片、记录 hash；缺失则失败。
5. [原论文与作者 PCGrad 源码](https://github.com/tianheyu927/PCGrad/tree/c5fbd7c856526373828074f06875230f7f3ee79e)：读取原论文算法/理论条件及 `PCGrad_tf.py`。作者实现是 TensorFlow，尚未在本训练载体资格化。I02 的基础投影有直接经典先例。

## 兼容与资源来源

- [NVIDIA RTX 2080 Ti 产品规格](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/)：标准产品为 11 GB；2026-10-06 仓库已更新为用户报告的 22 GB 卡，规划以该报告为准。实际设备/空闲显存仍由远程预检记录，不能用标准规格否定实际改装卡。
- [FlashAttention 官方说明](https://github.com/Dao-AILab/flash-attention)：常规 FlashAttention 2 CUDA 路线的 GPU/精度要求不能直接用于 Turing。首轮声明 FP16/SDPA 协议变化，未宣称其与 BF16/FA2 数值完全相同。
- Qwen 官方 `pyproject.toml` 与源码：Python 3.11+、Torch 2.8、Transformers 4.57 系列。这里固定直接版本并保留实际解析后的包清单；不是全量跨平台 lock，实际 GPU 集成仍待执行。
- [EasyR1](https://github.com/hiyouga/EasyR1)、[FastVideo](https://github.com/hao-ai-lab/FastVideo)、[Wan2.1](https://github.com/Wan-Video/Wan2.1)：前期读取公开说明与有关配置；未在此 GPU 主机运行。模型参数量不能替代具体显存/训练时间测量，RL/视频训练不准入首轮。

## 近作检索线索：尚无最终碰撞裁决

| 候选 | 直接相关来源 | 当前审查范围/缺口 |
|---|---|---|
| I01 | [Eddy-VL 1.9B](https://arxiv.org/abs/2607.16316) | 搜索返回摘要；本轮全文页面获取失败，不能称已完成机制碰撞审查 |
| I03 | [NeighborRetr](https://arxiv.org/abs/2503.10526)、[2026 CLIP hubness](https://arxiv.org/abs/2604.27674) | 摘要/作者代码入口定位，作者训练代码与原生任务深审待完成 |
| I04 | [Beyond the Query](https://arxiv.org/abs/2609.12437)、[MARVEL](https://arxiv.org/abs/2604.07079) | 前者原文摘要页、后者摘要线索；完整方法/代码和实际低预算对照待核查 |
| I05 | [Debiased Contrastive Learning](https://arxiv.org/abs/2007.00224) | 原文摘要和方法线索；此经典混合校正不得包装为本项目新目标 |
| I06 | [CVPR 2024 时序 UOT](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Temporally_Consistent_Unbalanced_Optimal_Transport_for_Unsupervised_Action_Segmentation_CVPR_2024_paper.html)、[UOT-Gap](https://arxiv.org/abs/2609.10224) | 官方条目/摘要线索，UOT-Gap 页面获取失败；全文、作者代码与定位 scorer 待审查 |
| I07 | [DeepSeekMath](https://arxiv.org/abs/2402.03300)、[困难提示动态采样近作](https://arxiv.org/abs/2608.27982)、[GRPO 标准差分析近作](https://arxiv.org/abs/2607.00152) | 近期检索线索；多模态适用性、估计偏差、实际训练代码和资源待审查 |
| I08 | [VBench 原生项目](https://github.com/Vchitect/VBench)、FastVideo/Wan、误差传播/缓存近作 | 原生 benchmark 入口已定位；具体 prompt revision、模型/评分 recipe 和最低必要对照待资格化 |

I01/I02 的论文/代码连接已比其他路线完整，仍没有真实训练数据、现场教师 qualification 或原创性裁决；这解释优先级，不授予方法执行资格。摘要检索可能遗漏最强竞争者，首轮筛选不宣布“没有前人做过”。

## 为什么首轮只执行基线核查

使用的 [Research Autopilot SKILL.md](https://github.com/Yunbo-max/Research_Autopilot/blob/main/skills/research-autopilot/SKILL.md) 在 Gate A 段规定：

> Require current parent, Natural Gate 0 PASS and IPCG CONCURRENT before implementation or Gate A freezing.

其 G01 说明也允许先做 baseline/evaluator qualification 和 bounded feasibility inspection。这里据此把尚缺真实失败、开发集和原创性证据的方法设计保持为条件候选；GPU 不可达另外独立阻止当前网页执行。没有 skill 产生的新权限问题，也没有把源文件阅读或工程测试标为 scientific PASS。

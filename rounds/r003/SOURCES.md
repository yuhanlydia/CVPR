# 源码与文献读取范围

2026-10-06。研究问题/数学构造与文献存在性分开记录。没有完成所有近作全文比较，
因此所有候选 originality/IPCG 仍为 UNRESOLVED；以下不是“已证明新颖”的列表。

## 标准实现已实际读取
Qwen 官方固定提交 [393e2978](https://github.com/QwenLM/Qwen3-VL-Embedding/tree/393e2978d27852b0d0230d6994f37f9c15bed73c)。

| 文件 | 实际读取/对接对象 | 原始 Git blob |
|---|---|---|
| src/models/qwen3_vl_embedding.py | 官方视觉输入、最后有效 token 池化、单位归一化 | a2d4a73349c4648c5d66633eccd57e02aea1279f |
| src/evaluation/mmeb_v2/eval_embedding.py | 原始 query/candidate 缓存、local/global 完整排名与 torch.sort | fe25a1563d0adbf57b25b707dc0b4ead4ec0eff1 |
| src/evaluation/mmeb_v2/models.py | MMEBEmbeddingModel.encode_input、compute_similarity 的点积语义 | f00799a084e8772c4617d9efcbe62efd7a0d2643 |
| src/evaluation/mmeb_v2/data/datasets/base_eval_dataset.py | 原始标签/candidate schema、候选去重生成 | a6317b8511ff86b48127738c7078a0f47a04b1e1 |
| src/evaluation/mmeb_v2/utils/eval_utils/metrics.py | live RankingMetrics 的 hit/ndcg/precision/recall/f1/map/mrr | 097365b58831711d0b04eef08f9d4232f6486d6c |

只改已有 Turing FP16/SDPA 兼容路径，沿用 r001 记录；scorer 与点积模型文件做原始 blob 校验。
新小头使用 CPU float64 排名，记录并列行数；不宣称与官方 GPU 浮点结果逐位相同或复现榜单。
同投影对照使用完全相同 dtype/排名路径；全维基线重放实际官方已保存输出。

训练字段核对了 [VLM2Vec v1.0 的 src/dataset.py](https://github.com/TIGER-AI-Lab/VLM2Vec/blob/ba1e02b2c29461018b03169278c337b4222c7c3a/src/dataset.py)，
blob fa0c8bf76d53b7a3549c2d3a1e8c1f4258532389。
解析 qry/qry_image_path/pos_text/pos_image_path/neg_text/neg_image_path，
只转换 Phi 图像占位符到 Qwen 的独立图像输入，明确记录此训练适配。
没有资格化为 VLM2Vec 官方训练复现。

[MMEB-train 数据卡](https://huggingface.co/datasets/TIGER-Lab/MMEB-train) 明确
original 与 diverse_instruction 的区别；本批固定 original，
revision 0c3f4b828d347c4e8508339f99530f6c820061fd。
原始训练图像须实际存在，不因缺图生成替代数据；实际内容哈希与测试图像重叠即阻塞。
训练配对不作为测试 benchmark。当前 main 的评测任务按 configs/candidates.json 为
ScienceQA、ChartQA、MSCOCO_i2t，均要求 r001 的实际完整原生缓存；旧单任务缓存不符合此版本。

## 最近工作与碰撞记录
- [Qwen3-VL-Embedding 报告](https://arxiv.org/abs/2601.04720)：已读的方法背景，
  对比学习/蒸馏本身已经是标准路线；I01/I05/I09/I14 不能只靠命名声称原创。
- [UniME-V2 原文](https://arxiv.org/html/2510.13515v2)：实际读取语义软监督、负例处理、
  训练目标与评测段，关联 I01/I05/I09/I14。文中将对称 KL 写作 JS 的地方不直接继承；
  原型数学独立推导，不依赖该命名。
- [UniME-R1](https://arxiv.org/abs/2608.06060)：当前读到摘要、作者代码/RL 文档信息，
  未完成全文算法审查；关联 I04/I07，不能宣称已排除重叠。
- [PCGrad](https://arxiv.org/abs/2001.06782)：已有全文/作者代码阅读记录，
  I02 的子空间保持约束需与梯度冲突投影区分，不能用代理保证整个任务能力。
- I06/I08/I20 的既有 Wan、TeaCache、VBench、RA-CFGCache 来源和碰撞记录
  继续引用 [ROUND_002 来源](../../research/SOURCES_ROUND_002.md)，没有新颖性晋级。

## 已知数学机制，非新贡献
- I10：CCA；本批推导训练协方差正则与两侧变换，不声明发现 CCA。
- I11：[Sagawa group DRO](https://arxiv.org/abs/1911.08731)，当前来源摘要核查；
  平滑最差组、正则和泛化限制独立写入卡，全文强对照仍待完成。
- I12：[Diffusion Maps 原论文](https://arxiv.org/abs/math/0503445)，当前摘要级来源核查；
  有限图归一化/外推公式由本批独立线性代数推导，非新谱一致性定理。
- I13：[协方差收缩/OAS](https://arxiv.org/abs/0907.4698)，摘要级来源核查；
  原型仅固定 alpha 收缩，与 OAS 算法不是同一实现。
- I16：[Nyström 近似的原始研究](https://proceedings.mlr.press/v48/si16.pdf)，方法来源待更深阅读；
  正则核因式分解是独立矩阵恒等式，不是原创算法。
- I03/I14/I17/I18/I19 分别有监督度量、Huber、有噪声线性估计、D-optimal、
  Cauchy-Schwarz 剪枝等已知基础。实际新差异需由真实问题与最强对照建立。

本次 CPU 检查只针对程序/代数；不运行自制评测样本，不把工程夹具计作原生评测资格。

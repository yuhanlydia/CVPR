# r004 更多实验：现有方法审计与强对照

Web源审查截止本轮：CVPR `47fd9dc40cec8615d0f204eb3e91668acdb6e148`，
Research Autopilot `a8343aeb4f51303e2eb651081d4fe51c24c5ed3f`。
**generated_unexecuted；没有修复后的新三任务成绩；不是已准入的自动队列。**

最近修复：I03从双线性改为负Mahalanobis距离；I10/I13/白化对变换后向量单位化。
DEBUG记录I09训练CE .5341→.4788、最大margin步长5.058，但缺可复核原始诊断/新原生成绩。
旧512-row返回完成17/18个arm，包含11个idea、6个控制，I01校准阻塞；
历史8个parked idea没有本轮实现/结果，不能说20个里面18个实验都失败。

这一轮保留12个原型、6个控制，增加25个已知对照/参数配置，共43条登记。
I01仍条件性保留，其余42条预计可计算；多正例覆盖、优化器和输入资格可继续阻塞局部项。
它们属于8组有明确问题的实验，而非再创造25个原创idea。完整推导见
[MATH_CARDS.md](MATH_CARDS.md)，配置见
[audit_controls_20261007.json](../../configs/audit_controls_20261007.json)。

| 组 | 新增配置（数量） | 必须同批/同版本对照 | 要区分的原因 |
|---|---|---|---|
| E01 原生压缩前沿 | MRL 64/128/256/512/1024/2048（6） | PCA32、全部已拟合小头；完整原生2048 replay | 原来32维压缩制造的难度，还是实用准确率/维度前沿 |
| E02 I03评分修复 | 同M的历史dot（1） | 当前I03负距离、PCA32、白化 | 缺candidate能量项的实现偏差，或距离方法本身不适合 |
| E03 范数与容量 | I10/I13/白化raw；PCA16（4） | 三者cos、白化×收缩2×2、PCA32 | 范数先验、收缩作用、16维容量限制 |
| E04 I09公平优化 | 同CE L-BFGS、Armijo、固定半步（3） | 一步I09、原MSE KD、PCA32 | CE/MSE目标差异、一步近似/过冲、真实成本优势 |
| E05 收缩稳定性 | α=.05/.5/.8（3） | .2原I13与0普通白化；同norm/ridge/center | 有稳定收益，还是参数单点/开发选择造成的波动 |
| E06 多正例学习 | inside/outside/clean-single L-BFGS（3） | 原I05、旧C_SINGLE、PCA32；真实多正例覆盖 | 容易正例主导、标签信息增量、旧single的错误负例污染 |
| E07 谱方法几何 | I12 cos/distance，I16 cos（3） | 原raw-dot、精确RBF、PCA32 | 径向能量、谱扩展/近似误差；不能把RBF单调变换当新表达力 |
| E08 群体风险 | 充分优化smooth-DRO/ERM（2） | 两者旧80步、PCA32；训练group风险/权重 | 原相同成绩是没优化够，还是风险目标没有实际收益 |

## G01：目标、数据、比较、统计与完整性

决策目标是分辨实现错误、目标混淆、优化残差、表示限制与真实负结果。
没有提出新的原创机制，不证实自然失败或CVPR贡献；原20卡/前15、Parent/Natural Gate0/IPCG
仍保留原状态。M审计不重启人工凑数20→15。新增协议仍需适用Local边界和原生资格。

训练固定：TIGER-Lab/MMEB-train @ `0c3f4b828d347c4e8508339f99530f6c820061fd`，
ScienceQA/A-OKVQA各512个released行，原始正/负标签、去重与图像/query去重隔离。
旧包报告1019 unique，实际以已验证manifest/rows为准；不用test决定训练行或参数。
训练teacher仍同一个冻结2B的完整向量，没有引入reranker/更大模型或额外标签。

每个arm覆盖全部原ScienceQA、ChartQA、MSCOCO_i2t原生test；
此前包各1000行，实际分母只能读native manifest/producer，不能把1000写成新运行实测。
候选列表/label/原次序/原评价类型完整保留，不能减候选或做handmade eval。
scorer固定Qwen `393e2978d27852b0d0230d6994f37f9c15bed73c`的RankingMetrics：
`src/evaluation/mmeb_v2/utils/eval_utils/metrics.py`
blob `097365b58831711d0b04eef08f9d4232f6486d6c`；
原similarity代码blob `f00799a084e8772c4617d9efcbe62efd7a0d2643`。
使用官方全部task metrics；primary分析是官方per-case hit@1。输出保留full rankings、
score/replay hash、feature dim/norm/tie、parameter hash与实际fit/time。

三group的native协议分开绑定原split、manifest、labels、scorer与arm实现；
新runner可只评分一个task，job卡为每arm×每task，42条当前预计可计算arm对应126个group job，
不是43次有预算的试验。每task重新做确定性训练fit，计入实际成本，记录参数digest一致性；
不把三次相同fit当独立seed。单次runner默认可以覆盖三task，但不能借一个group contract
冒称多benchmark的完整资格。采用哪种已接受原生执行适配须在Local协议冻结时绑定。

固定12个primary对比×3task，共36个开发端点，含白化/收缩×norm的差中差。
另固定每个历史idea对所有原required controls和完整2048的secondary配对对比，
每arm所有原生成绩都返回；I02/I14/I17等同样有自身对照与不确定性。
置信分析只从原生逐样本预测与官方hit_at_k得到配对差值；重复query图片用实际
SHA256聚类，text-only精确输入聚类。2000次cluster bootstrap报告条件性95%区间、
实际cluster数、差异top1/full-ranking行数；缺失真实unit来源则保留点估计并标记CI pending。
独立/可交换cluster是假设，不能从一个map/布尔值认证。候选corpus固定的条件性范围需保留。
区间为描述性；36端点/多个arm不作赢家显著性或自动科学PASS，不能挑test最优α。
1pp是开发中的实用观察阈值；1000行不能自动保证区分1pp，
精度不足保留INCONCLUSIVE，不把小数点差异变成“方法成功”。

未来若要性能/效率主张，须选定freeze的目标/参数/最强控制，进行独立确认，
重新资格化改变后的child protocol；已看过的512开发结果和这轮选择结果都不能充作确认。
2048/64/32维不同容量用于准确率/维度/成本frontier，不可宣称同维公平机制消融。
原生encoder的共享成本与head额外拟合/变换/评分/保存成本分别列出，同时给总成本；
不能只算一行矩阵乘法隐藏teacher/PCA/CPU优化和读取全候选的费用。

## 资源、准入与有限执行

已来源化一张2080 Ti、报告22 GB；本轮未看实时型号/VRAM/UUID/占用。
只要兼容完整缓存存在，所有新小头与重评分是CPU/float64、请求0 GPU，没有新模型下载。
缓存重建是额外1 GPU真实prepare任务，继续原2B/FP16/SDPA/Conda路径并保留原身份；
不能悄悄重置old prepare-attempt或把Web新源接受当作成功缓存。

CPU/RAM/time/attempt均待Local恢复实际预算与host测量，配置用null明确未知。
主要内存：训练teacher 8*n_train*m_train字节、原positive/allowed两张bool掩码；
pair X为8*n_train*32²，sample-space系统8*n_train²；
多正例优化另有多张n_train*m_train浮点临时矩阵。
eval完整输入单task约8*(n_query+n_candidate)*2048字节，再加变换/原生rank输出；
MRL2048评分工作可能比32维高，不能沿用旧22m54保证全部42条按时完成。
这些是构造级估算，实际peak RAM/CPU、fit/scoring/write时间才可准入。

不修改旧configs/candidates.json、18次上限、累计8h硬时钟/marker/receipt。
新增42条是提案，不是授予126次native attempt。Local预算接受应读取原累计已用/剩余，
另含源接受、native scorer资格、软件接受、数据准备、E04和返回余量。
未来已批准周期按实际更短限额且≤24h，8h只作软报告；旧活动计划不迁移。
零自动重试；共享输入/源码/native问题阻塞后代，单arm失败继续独立已准入项。
未知活动任务先reconcile，同ID/bounds保留，不重复launch、清lock或新window重置用量。

优先顺序是减少不确定性：先完整证据/修复后重放与原生维度强基线，再I03及I09公平目标，
接着norm×shrink和其他机制诊断。分批可carryover所有剩余稳定arm/group；
不为凑时间删benchmark/对照。没有实测资源时不承诺一轮全部完成或自动ready。

## E04与结果决策

Local先接受来源、数学到代码语义、实际native producer/score：
[WEB_HANDOFF.md](WEB_HANDOFF.md)有每条工具接口、输出、故障和顺序命令。
执行结果包含模型/数据/source/skill/env哈希、所有actual attempts及stdout/stderr、
完整raw预测与官方live replay、原生分母/候选、训练loss/梯度/步长/eigen/norm/多正例覆盖、
每task成本和未完成项。不得只推一张最高分表；大raw包保留实际host locator/hash，
可读结果与小receipt提交GitHub，不能上传权重/私有skill/受限图片/大缓存。

解释规则：
1. 修复改变排名而强基线仍胜：修复被确认，方法收益未确立；
2. 同CE充分优化追平/胜过I09：削弱“一步曲率带来独特准确率”的解释，另比较真实成本；
3. score几何变化提升某task损伤另一task：保留异质性，不平均掩盖MSCOCO下降；
4. optimizer到界/残差大、multi-positive少、CI精度低、raw缺失：对应缺口，不作方法失败判决；
5. 全部合法版本持续不及强控制：形成有范围负证据，旧ID保留而非换名字；
6. 正向开发信号：仅供下一轮选择和前瞻确认，不自动Advance/Gate A PASS。

完成Web交付是代码/配置/完整问题与比较/命令源码可审查，并向main精确回读。
Local语义/native/资源/G01接受及实际protocol/outer harness派发是未执行的下一阶段；
SOURCE_BYTES_MATCH、PROPERTY_CHECKS_PASS、模型读取成功都不能代替科学资格。

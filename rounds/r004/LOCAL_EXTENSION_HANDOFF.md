# 当前优先政策：先跑通、再完善

用户已明确放宽推进节奏。先读[执行政策](EXECUTION_POLICY.md)，其分阶段顺序优先于下方旧的一次性全量接受要求。
允许先做明确范围的小批开发实验、先保存点估计与真实预测，之后补完整任务/CI/诊断。
不等待72配置全部ready；必要原生有效性、旧成果保护、唯一执行owner和实际预算仍保留。

# Local扩展交接：29配置已生成源码，完整性待真实验收

本次是现有方法M审计与已知控制实现，源检查基线CVPR
5d96050e35ae62c36976c45dc9cb843db6cee177；最新作者skill仍为
a8343aeb4f51303e2eb651081d4fe51c24c5ed3f。
状态generated_unexecuted：Web未运行项目、测试、模型或评分。
交付commit以main回读为准。旧提案文件保留历史，新执行配置是
[configs/audit_extensions_20261007.json](../../configs/audit_extensions_20261007.json)。

## 必读入口与输入

在精确交付commit读[AGENTS](../../AGENTS.md)、
[Local总指南](../../LOCAL_AGENT_RUNBOOK.md)、
[父轮交接](WEB_HANDOFF.md)、[新增数学/设计](ADDITIONAL_EXPERIMENTS_20261007.md)、
[逐臂逐任务完整度矩阵](EXTENSION_COVERAGE.json)。
作者完整四skill/297文件安装与旧活动绑定保留规则沿用父轮，不复制private skill到本repo。
[Download datasets and models](../../docs/NATIVE_ASSET_RUNBOOK.md#download-datasets-and-models)
和[获取/skill安装](../../LOCAL_AGENT_RUNBOOK.md#下载与完整-skill-安装)是实际输入卡。
没有新增模型/数据：同冻结Qwen2B及MMEB-train、三个MMEB原生test、
原evaluation图片、完整2048 baseline缓存、PCA32/teacher/native producer全部沿用。
不得首次推理隐式下载，不减少输入清单；缺失输入按原卡真实准入获取。

先在用户电脑恢复已有SSH、隔离delivery checkout；GPU主机恢复真实Conda、
NATIVE_PY绝对解释器、CVPR_REMOTE_ROOT、RESEARCH_AUTOPILOT_ROOT、
原runtime pool、实际剩余budget和host RAM/CPU。
以下沿用父轮已来源化变量：AUDIT_BUNDLE必须来自完成prepare receipt的原layout；
sources/Qwen3-VL-Embedding是root内已核对393e2978官方源码。
这些host值未在Web观测，不能猜路径或视为已ready。
所有项目命令都只能作为实际准入的remote run_harness任务；控制电脑只做Git/SSH/文件身份操作。
无Docker、无第二harness、无自动重试、旧prepare/18次/累计硬界不重置。

## 数学到代码与完整比较

| 组/配置数 | 构造与关键代码 | 必需对照 | Local语义义务 |
|---|---|---|---|
| E09/5 | extension_objective、pair_selection；query等权all-pairs chunk256、hard、固定3seed native负例 | A_CE_LBFGS、PCA32/白化/full2048 | 解析梯度、query权重、negative ID顺序稳定、native label masks；一对数据标冗余 |
| E10/3 | extension_objective；t=(1-α)+αsigmoid(teacher margin) | α0 ERM、α1 CE两个端点、强控制 | 端点目标/梯度等价、.25/.5/.75导数、teacher冲突诊断 |
| E11/6 | selected_rows、subset_training、fit_audit(parent) | 各头同版本joint parent、强控制 | 原source身份、allowed候选并集、teacher/labels重映射；PCA保持全量 |
| E12/9 | 原query ID的SHA排序嵌套64/128/256每group | 同版本全量parent、强控制 | unique身份、嵌套、同subset三头、候选/labels不变；仅head数据量 |
| E13/2 | whitening_head；固定parent μ的加权锚定二阶矩 | 原I13、白化、PCA32/full2048 | weighted-stack等价、PSD/逆平方根、中心不变、使用label明确 |
| E14/4 | protected_head；SVD rank2/4/16、randomQR rank8 | 原I02 rank8、C_KD无保护、强控制 | R正交、Rᵀdelta、native source rank、query能量；无整group保证 |

源文件[扩展实现](../../experiments/embedding_heads/audit_extensions.py)和
[版本化inventory](../../experiments/embedding_heads/audit_inventory.py)分离原43臂；
原base配置字节未改变，extension绑定其Git blob。
四工具增加显式--extension-config，旧默认43臂仍可用，但新代码不能冒充旧execution SHA。
旧运行须在旧pinned collector读取，不将新源码移入活动attempt。

扩展后72条登记、每条三个任务，最多216个registered group槽；
I01独立校准及多正例/来源/rank等条件可阻塞，数量不是预算授权。
48个primary contrasts×3=144开发端点，124个secondary×3=372，
共516个预登记端点。所有失败/blocked/缺任务/缺对照/缺CI必须返回，
不能以代码有72个ID代表已完成72个实验。
数学/设计中的对比必须同版本fit/scorer，旧最好数字不能顶替新parent。
三个任务原始全候选/label/次序、manifest真实分母、官方全部metrics，
hit@1逐样本paired分析、固定候选corpus下2000次真实image/query cluster bootstrap；
描述性CI，不进行winner显著性或确认主张。

## 有序命令卡（只供Local准入任务）

工作目录为GPU主机精确交付checkout的CVPR_REMOTE_ROOT；
解释器为已恢复的NATIVE_PY。out由harness隔离workspace提供。
先按父轮software/native source acceptance运行原工具，再接受扩展：

~~~bash
"$NATIVE_PY" tools/qualify_audit_controls.py --bundle "$AUDIT_BUNDLE" --config configs/audit_controls_20261007.json --upstream sources/Qwen3-VL-Embedding --out out/audit-semantic-checks.json
"$NATIVE_PY" tools/qualify_audit_extensions.py --bundle "$AUDIT_BUNDLE" --config configs/audit_controls_20261007.json --extension-config configs/audit_extensions_20261007.json --upstream sources/Qwen3-VL-Embedding --out out/extension-semantic-checks.json
~~~

新接受工具只使用真实native训练数据：身份/梯度/端点/负例ID稳定性、
subset重映射/嵌套、加权白化、保护子空间。有限差分在非identity参数点，
不生成科学评估样本。PROPERTY_CHECKS_PASS只证明报告列出的性质；
blocked条件不会因此变ready，仍需逐配置实现语义复核和完整native资格。
保存实际stdout/stderr、exit、env/input/code refs；Web不预填通过。

下一步构建全部base+extension job卡：

~~~bash
"$NATIVE_PY" tools/prepare_audit_job_cards.py --bundle "$AUDIT_BUNDLE" --config configs/audit_controls_20261007.json --extension-config configs/audit_extensions_20261007.json --upstream sources/Qwen3-VL-Embedding --python "$NATIVE_PY" --out out/audit-extension-job-cards.json
~~~

输出DRAFT_JOB_CARDS_ONLY，包含完整输入/代码digest、B_* preflight阻塞原因和三任务argv，
不是native plan、更不是预算批准。先完成parent修复后重放与E01/E04強基线，
优先E09→E10→E14→E13，E11/E12来源/成本合格后做。
parent/control和三任务不可删；未完成内容carryover，不能缩benchmark凑时限。

单臂native job接口示例（放入已准入科学native plan，禁止裸跑）：

~~~bash
"$NATIVE_PY" tools/run_audit_method.py --bundle "$AUDIT_BUNDLE" --config configs/audit_controls_20261007.json --extension-config configs/audit_extensions_20261007.json --method B_MIX_05 --task ScienceQA --upstream sources/Qwen3-VL-Embedding --out out
~~~

相同arm最终必须补齐ChartQA和MSCOCO_i2t。
输出result.json/head.json/head.npz及该task完整_pred.jsonl/_score.json/_replay.json。
result保存base/extension/bundle身份、训练pair/subset身份、参数digest、
fit/setup/elapsed、CPU user/system和Linux process peak RSS（不含child进程）。
head的protected_basis会真实保存并进入参数digest。
CPU cached-head评分请求0 GPU；完整共享encoder/teacher/PCA/缓存生产成本
仍从真实producer计入总成本，不以边际fit时间冒充总成本优势。

依父轮[Scientific native/outer plan绑定](WEB_HANDOFF.md#5-scientific-nativeouter-plan绑定与执行)
构建真实purpose=scientific的每group child contracts、完整arm_requirements、
source/model/data/env、实际CPU/RAM/remaining limits；冻结后原唯一owner执行。
复用已核对run_harness --freeze/--execute --approved-plan-digest接口。
本交付没有构造未知host的ready plan或伪造actual G01/native/预算接受。

## 返回、完整度和故障

作为实际准入collection task，repeat --receipt覆盖各真实native windows：

~~~bash
"$NATIVE_PY" tools/collect_audit_controls.py --bundle "$AUDIT_BUNDLE" --config configs/audit_controls_20261007.json --extension-config configs/audit_extensions_20261007.json --upstream sources/Qwen3-VL-Embedding --receipt "$AUDIT_NATIVE_RECEIPT" --out out/audit-extension-return.json
~~~

collector逐attempt验证实际execution source、base/extension/bundle/input/output身份，
live官方RankingMetrics重放、原始候选/labels/分母，
保留72×3所有状态与全部516开发端点。
collector只保留各行top1与有序完整排名SHA256，避免常驻72份全候选字符串；
全排名原文件及hash仍保留，排名变化数明确按SHA256比较，unit来源每task核验一次。
同arm三任务的参数/pair/subset身份不一致，comparison标INVALID_INCONSISTENT_FIT。
experiment_completeness列出native job回放、planned CI覆盖、fit一致性、
需要充分优化的arm是否真的梯度达标，以及未完成数。
缺校准/未准入也可导致complete=false；不会偷偷移除这些记录。
这些是开发覆盖检查，不等于Gate A/论文有效性/确认。

| 故障 | 读取与修复边界 |
|---|---|
| 原producer/code identity不匹配 | 读父轮source delta及原attempt，保留缓存/receipt；合法兼容性未定则真实预算内prepare，不删guard |
| native ID/mask不匹配 | train_rows与manifest candidate_ids及实际train.npz；不按位置/名称猜或修改原标签 |
| source/unique数/rank不足 | 对应B_*明确BLOCKED；不重复行、造query或降低已登记参数 |
| 梯度/矩阵性质不通过 | 新acceptance报告具体property、真实输入和映射函数；修因后新child source重验，不自动调test |
| solver到界/残差大 | diagnostic termination/gradient/conditional gap；当前结果保留unqualified，不称充分优化 |
| 同fit跨task不一致 | parameter与training/pair/subset digests、source/env/input；比较无效，reconcile实际身份 |
| 缺任务/对照/CI或大跨任务损伤 | return coverage/comparisons/native rows；待补齐或INCONCLUSIVE，不平均掩盖损伤 |
| SSH/预算/重复receipt | 同实际owner/PID/attempt/累计计数reconcile，保留所有失败，不重复launch |

上传小的可读结果/receipt、实际execution SHA/dirty patch、env/resource、
全部negative/unfinished、E04解释及剩余carryover状态到yuhanlydia/CVPR main并回读；
大raw预测保留真实GPU host locator/hash，不上传权重/受限图片/private skill。
开发结果不能自动确认选择出的winner；独立确认协议仍另冻结。

## Web完整性边界

本次交付实现29配置、训练身份验证、两配置绑定、执行入口、
Local性质接受工具、全量job卡、官方回放/paired CI/覆盖收集与导航。
已作[数学到源码审查](CODE_REDESIGN_REVIEW.json)；这是Web自身源审查，非独立复核。
没有运行编译、测试、模型或评估。
仍pending：实际软件运行/逐配置语义接受、各native child资格、
资源/剩余预算/G01派发、三任务全部结果/CI/E04与独立确认。
代码生成完整与科学实验完成是两个分别核验的状态。

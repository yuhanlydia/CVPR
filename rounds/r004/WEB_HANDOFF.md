# r004 Local agent交付：8组审计与25个新增对照

状态：**generated_unexecuted**。Web完成最新源码/数学、控制实现、配置、接受/收集工具与命令，
没有执行项目代码/tests/模型，没有生成新成绩。实际交付SHA以GitHub main回读为准，
此文件不预填自己的未来commit。

先读本commit的[AGENTS](../../AGENTS.md)、[Local总入口](../../LOCAL_AGENT_RUNBOOK.md)、
[完整设计](EXPERIMENT_DESIGN.md)、[数学卡](MATH_CARDS.md)、[来源checkpoint](SOURCE_CHECKPOINT.json)、
[技能差异](../../research/AUTOPILOT_DELTA_2026-10-07_a8343ae.md)。
这是现有方法M审计，不创建25个新原创idea，也不追认旧top15/科学前提通过。
活动Local任务、旧skill、累计时钟、所有failed/blocked/marker保留。

## 1. 源码与完整skill

在用户控制电脑上，用已有已授权GitHub连接读取yuhanlydia/CVPR的实际main交付commit，
建立隔离checkout，保留未提交文件和活动运行。不要直接更新活动workspace。
作者源固定为Yunbo-max/Research_Autopilot：
`a8343aeb4f51303e2eb651081d4fe51c24c5ed3f`。
已有clone在隔离源码目录fetch/check；首次没有clone时才使用以下命令：

~~~bash
git clone https://github.com/Yunbo-max/Research_Autopilot.git sources/Research_Autopilot-a8343ae
git -C sources/Research_Autopilot-a8343ae checkout --detach a8343aeb4f51303e2eb651081d4fe51c24c5ed3f
git -C sources/Research_Autopilot-a8343ae rev-parse HEAD
~~~

按该commit的docs/INSTALL_LOCAL.md完整安装四个真实相邻目录：
research-autopilot、writing-top-tier-papers、designing-pipeline-figures、designing-experiment-figures。
保留其他hooks/settings，旧目录备份在skill扫描目录之外，不用symlink。
用已有SSH传同样固定源码到Linux host；host需要runtime/Conda/project，不装新的LLM agent。
不要覆盖活动task使用的skill/install；新目录接受后再绑定新任务。

下面是实际源码接口。所有项目可执行检查/生成/收集均为单一remote run_harness中的任务。
`NATIVE_PY`、`RESEARCH_AUTOPILOT_ROOT`、`CVPR_REMOTE_ROOT`、`AUTOPILOT_POOL`
由Local恢复实际远端Conda解释器、固定skill目录、隔离project根和原共享pool，不能猜路径/SSH alias。
控制电脑源安装不等于host接受；Web未检查这些实际值。

~~~bash
"$NATIVE_PY" tools/check_autopilot_source.py --skill-root "$RESEARCH_AUTOPILOT_ROOT" --lock configs/autopilot-source-a8343ae.json
~~~

SOURCE_BYTES_MATCH只证明297个源码文件身份，不是软件/native/science通过。
scipy==1.16.3已在requirements.txt；若旧Conda不匹配，在隔离native环境中按原设置任务接受，
不要global升级或修改活动环境。实际pip/Conda freeze、interpreter/BLAS版本随任务保留。

## 2. 数据、模型、scorer、缓存及输入卡

完整不可变下载/校验/抽取命令继续使用
[Download datasets and models](../../docs/NATIVE_ASSET_RUNBOOK.md#download-datasets-and-models)与
[native-assets.lock.json](../../configs/native-assets.lock.json)；旧获取/loader实现也是待Local接受的新分支。
本轮没有新权重、数据、teacher或外部judge。每组输入消费者：

| 输入 | 消费者 | 获取/路径/接受 |
|---|---|---|
| 冻结Qwen/Qwen3-VL-Embedding-2B全部18文件/分片与processor | 原baseline、缓存producer、所有训练teacher和42个arm | 原asset lock完整model卡；仅缓存重建需要实际权重/1 GPU |
| 原ScienceQA/A-OKVQA训练shards、train image ZIP | PCA、I03/I09/白化/多正例/图/DRO等fit；原12全部重放 | 原train卡；不引入新正例，metadata=data/train-metadata，真实图片root由旧接受receipt恢复 |
| 原ScienceQA/ChartQA/MSCOCO_i2t完整test shards、eval ZIP | 每arm×三原生task | 原test/eval-images卡；候选/分母/标签不缩减 |
| 固定Qwen官方代码/RankingMetrics/models | 所有打分、live replay、paired hit@1端点 | sources/Qwen3-VL-Embedding @393e2978d27852b0d0230d6994f37f9c15bed73c，原source/download卡 |
| numpy/SciPy/PyTorch等native依赖 | CPU fit与官方CPU评分/接受 | requirements.txt及原Conda卡，SciPy1.16.3；导出实装环境，不当新model资产 |
| 完成的原生baseline/prepare attempt、manifest/训练与完整2048 eval缓存 | jobcards、全部arm、接受与收集 | AUDIT_BUNDLE指向隔离root内真实原layout的manifest，全部output/code/input hashes和producer接受 |

所有后续命令的`AUDIT_BUNDLE`必须从真实完成的prepare receipt恢复，不从目录名猜。
旧bundle的producer/code hash若不匹配当前prepare源码，严格guard会阻塞。
保留旧缓存/producer及其执行checkout，读实际source delta判断兼容性；不能删guard或改receipt伪造成功。
若不能来源化复用，另建受真实剩余预算准入的原生prepare任务，使用原
tools/prepare_candidate_bundle.py与configs/candidates.json、原baseline-out/train roots；
保留原一次prepare与18-arm预算，不用旧run_candidates重开窗口。
未接受的新producer影响依赖cache任务，不代表相应方法失败。

## 3. Local语义接受（0 GPU，科学成绩仍待跑）

先保留现有代码检查的实际SHA/作用范围，不把DEBUG声称的测试当原始receipt。
新工具只对真实原训练输入和原生eval feature行检查导数/矩阵/归一化等，
不产生人为benchmark样本，不把小范围语义检查当完整三benchmark实验。
作为bounded remote acceptance job，argv为：

~~~bash
"$NATIVE_PY" tools/qualify_audit_controls.py --bundle "$AUDIT_BUNDLE" --config configs/audit_controls_20261007.json --upstream sources/Qwen3-VL-Embedding --out out/audit-semantic-checks.json
~~~

输出named property checks：真实训练CE/ERM/DRO/inside/outside/single的解析梯度，
实际样本空间Newton与dense方程、训练Armijo、I03距离展开、RBF等价、
MRL prefix代数与两视图单位范数。缺真实multi-positive或区间校准保留blocked arm。
PROPERTY_CHECKS_PASS只限这些性质；原生prediction/scorer、完全三task、真实资源与G01另接受。
软件接受结果、stdout/stderr、exit status、源码/env/input refs保留在真实attempt。

原新方法边界仍可用实际checker只读检查，不能把退出2改成通过。
如要重启方法开发，绑定真实r003 batch先执行：

~~~bash
"$NATIVE_PY" "$RESEARCH_AUTOPILOT_ROOT/scripts/verify_methods.py" --root "$CVPR_REMOTE_ROOT" --batch evidence/r003-math-20261007/batch.json --before code --candidate I09
"$NATIVE_PY" "$RESEARCH_AUTOPILOT_ROOT/scripts/verify_methods.py" --root "$CVPR_REMOTE_ROOT" --batch evidence/r003-math-20261007/batch.json --before experiment-design --candidate I09
~~~

这是旧候选发现状态检查，不给本轮M审计制造新20→15。
新方法的before-dispatch/verdict及原Parent/Natural Gate0/IPCG等仍受实际边界约束；
审核M审计适用范围、native/G01证据后才能准入科学任务，不能将benchmark job标成engineering绕过。

## 4. 构建完整内层job卡（没有派发）

在源码/输入接受后，以下0 GPU准备任务生成所有arm×task的精确argv、
输入/代码SHA256、完整预期输出与条件blocked记录：

~~~bash
"$NATIVE_PY" tools/prepare_audit_job_cards.py --bundle "$AUDIT_BUNDLE" --config configs/audit_controls_20261007.json --upstream sources/Qwen3-VL-Embedding --python "$NATIVE_PY" --out out/audit-job-cards.json
~~~

这是DRAFT_JOB_CARDS_ONLY、dispatch_ready=false，不能作为run_harness plan直接执行。
43条登记、25个新增对照，常规42条×三task=126个native group job；I01条件性缺校准排除。
所有3task完整；缓存CPU fit在每group重复、参数digest必须核对，不能算独立重复试验。
r003旧18次授权不自动覆盖126次；实际剩余CPU/time/attempt/spend未知，Web未预填。

一个job的实际接口例如（只能放入已准入native plan，不直接裸跑）：

~~~bash
"$NATIVE_PY" tools/run_audit_method.py --bundle "$AUDIT_BUNDLE" --config configs/audit_controls_20261007.json --method A_CE_LBFGS --task ScienceQA --upstream sources/Qwen3-VL-Embedding --out out
~~~

同arm必须最终补齐ChartQA/MSCOCO_i2t；单task的完成不代替整体比较。
--method的全部43个实际值在配置arms.id里，--task仅允许上述3个真实值。
输出：out/result.json、head.json、head.npz、每task完整_pred.jsonl/_score.json/_replay.json。
--out需新目录，不覆盖已有attempt；seed固定、无测试标签拟合、failed exit1/blocked exit20保留。

## 5. Scientific native/outer plan绑定与执行

Local在真实host/context接受完整设计、资源、科学适用范围后，按已检查的最新runtime API构建：
- native `run_experiments.make_plan(root, run_id=..., jobs=..., provenance=..., limits=...,
  purpose="scientific", evidence_mode="developmental", protocol_ref=...)`；
- 每个group对应自己的原native contract/真实sample/labels、全部原metrics/scorer，
  scientific protocol的arm_requirements覆盖job卡的确切arm_role；实现refs含控制模块与配置。
  在required_groups保留三个benchmark；不能把一个ScienceQA contract借给ChartQA或MSCOCO。
- actual code/model/data/env refs、source/skill SHA、真实同一剩余累计budget/零重试、
  per-attempt/总时间与每条成本；不从job卡数量自动生成授权。
- 用 `run_harness.make_plan(root, batch_id=..., tasks=..., pool_dir=AUTOPILOT_POOL,
  limits=..., gpus=...)` 包装实际native plans；task resources请求0 GPU，
  CPU/RAM以实际host+测量profile准入。所有task仍由原共享pool一个执行owner控制。
- idea_id沿用现有20个方法或共同M审计ID，实验配置/task不是新增43个idea。
  审计控制可用共同M-audit身份；旧I IDs保留，不越过runtime最大20个idea。
- 接受/集合任务可按其软件/收集作用登记；原生test评分任务始终scientific。
  setup下载如确需运行，以独立真实零GPUprepare/harness任务管理。
- 冻结前检查每条job's input_refs包含其对应group要求的原native sample/labels
  （和原协议任何selection），不是只复制bundle的features_ref就认定native proof齐全。

恢复真实计划与approved digest后，单一已授权owner使用runtime现有入口：

~~~bash
"$NATIVE_PY" "$RESEARCH_AUTOPILOT_ROOT/scripts/run_harness.py" --inspect-host
"$NATIVE_PY" "$RESEARCH_AUTOPILOT_ROOT/scripts/run_harness.py" "$AUDIT_HARNESS_PLAN" --root "$CVPR_REMOTE_ROOT" --freeze
"$NATIVE_PY" "$RESEARCH_AUTOPILOT_ROOT/scripts/run_harness.py" "$AUDIT_HARNESS_PLAN" --root "$CVPR_REMOTE_ROOT" --execute --approved-plan-digest "$AUDIT_APPROVED_PLAN_DIGEST"
~~~

freeze输出必须保存为实际审核后的canonical plan；digest是该plan真实输出，不预填假值。
本交付未生成实际host/G01已通过的native protocol/outer plan，未知路径/容量/累计预算
明确阻塞dispatch；上面的argv是已核对源码接口，不能据此宣称ready、已授权126次或已执行。
保持原SSH transport/Conda与harness指定设备，禁止Docker和替代容器。
8h报告不终止仍在真实硬界内的任务；未来cycle至多24h且取更短剩余限额，旧运行计划不更改。

## 6. 原生重放、配对统计与返回

每个group结束后核对官方scorer replay、完整候选/分母、head source/config、成本及E04；
新版score/参数不能与旧打分成绩混为同一协议。solver success但梯度残差不达标，
不能标“充分优化”；inside-log本来非凸，驻点也不证明全局最优。

作为bounded collection job，重复--receipt覆盖所有已有窗口的实际native run receipt：

~~~bash
"$NATIVE_PY" tools/collect_audit_controls.py --bundle "$AUDIT_BUNDLE" --config configs/audit_controls_20261007.json --upstream sources/Qwen3-VL-Embedding --receipt "$AUDIT_NATIVE_RECEIPT" --out out/audit-return.json
~~~

跨多native run时多写--receipt，不重建/重开已完成job。collector核对native attempt/result/code/input/output
哈希，重新执行官方RankingMetrics，保留所有43条×3task的pending/failed/blocked状态、
12个预先声明primary contrasts/36个task端点、配对差值/条件cluster CI和全排名变化。
缺image/sample/asset身份时CI pending，绝不造unit map。duplicate attempts标歧义待reconcile；
返回软件exit0不代表科学PASS、全比较完成或未来claims资格。

推main的小结果文件应含：实际执行commit/skill/environment/model/data绑定、全部arm状态和成本、
新原生成绩与live replay/E04作用范围、未完成/负结果、不可混用的old/new协议、
剩余累计budget/稳定pending IDs；原始prediction/log/receipt保留真实host locator与SHA256。
需要审查具体失效案例时引用实际released case，不制造或修改eval例子。
不上传权重、受限图片、私有skill或大缓存；完整raw证据若超GitHub大小，保留已验证
授权host路径/hash并在小结果manifest中明确缺失GitHub原始payload的审查范围。

## 故障/证据决策

| 观察 | 先读 | 受限动作 |
|---|---|---|
| source/导入/Conda版本不符 | skill lock、native env receipt、exact stdout/stderr | 隔离接受固定源码/环境；不global升级、不改活动任务 |
| producer/cache guard失败 | manifest/attempt/code refs、实际prepare/qualify delta | 保留旧receipt/cache，来源化兼容复用或另立受预算准入准备；不删guard/编造producer |
| 梯度/Newton/距离/norm检查失败 | audit-semantic-checks、真实native输入与MATH_CARDS | 只修有证据的原因，重查受影响性质；不用测试fixture当成绩 |
| L-BFGS达到上限/Armijo不接受 | gradient_linf、objective calls/history、真实训练F | 保留未收敛/失败，不在test上挑step或扩大预算静默重试 |
| 多正例/校准缺失 | 真实positive coverage、independent calibration refs | 只阻塞依赖arm；不合成正例/区间，不把缺条件判负 |
| native候选/分母/replay/tie差异 | 全_pred/score/replay、original IDs/source/dtype | 保留现象并接受正确原生路径，不删候选或修标签使分数匹配 |
| 未知任务/断SSH/预算到界 | 原pool、remote PID/boot/owner、actualattempts | 先reconcile，不重复launch、不重置时钟或删除locks |
| 局部升/降或CI太宽 | 三task齐全比较、raw与成本、E04/36端点选择范围 | 保留异质性/不确定性，下一轮冻结目标再独立确认 |

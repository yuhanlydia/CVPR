# r004：用 Research Autopilot runtime 接续全部72配置

状态：命令按作者固定源码核对，generated_unexecuted。尚未从本Web会话启动Local队列或GPU任务。
最终目标见[BACKGROUND_GOAL.md](BACKGROUND_GOAL.md)，分阶段顺序沿[EXECUTION_POLICY.md](EXECUTION_POLICY.md)。

## 1. 恢复已有控制器，不重复准备

作者当前main核对仍为 `a8343aeb4f51303e2eb651081d4fe51c24c5ed3f`；
其Python包 `research-autopilot-runtime`、CLI与NativeWorker已包含在该commit，
不是另一个需要迁移的科学版本。依据为该固定版的
`pyproject.toml`、`src/research_autopilot/cli.py`、`remote.py`、
`docs/INSTALL_LOCAL.md`、`docs/LOCAL_AGENT_RUNBOOK.md`及runtime-library参考模块。
这些私有源码不复制进CVPR。

Local已报告四skill共297文件匹配；复用该实际receipt，不重跑未变化的源码检查。
四skill匹配不证明Python控制器已安装，也不证明host上的harness/pool已恢复。
先从现有控制电脑记录恢复实际Controller DB、project/task ID、owner/fence、
runtime可执行文件和NativeTarget；后者保存原SSH alias、远端项目根目录、
skill根目录及Conda解释器。这里不猜主机路径。

若Local已有可用runtime，直接复用。仅在实际缺少库时，按固定作者
`docs/INSTALL_LOCAL.md`将它装入控制电脑的独立Python 3.11+环境；
已有venv与完整固定源码可用、构建依赖已在环境中时，安装卡为：

~~~bash
"$RUNTIME_PYTHON" -m pip install --no-deps --no-build-isolation "$AUTOPILOT_SOURCE"
~~~

两个变量分别必须来自实际控制器venv和固定作者checkout；这不是Web已执行的安装命令。
不安装新的LLM后端来代替现有NativeWorker；GPU主机沿用原Conda/harness。

## 2. 先读队列状态，再恢复同一任务

以下命令在控制电脑执行。RESEARCH_AUTOPILOT_CLI、CONTROLLER_DB、PROJECT_ID、
NATIVE_TARGET、TASK_ID、ORIGINAL_OWNER、ORIGINAL_FENCE均从已有记录恢复，
ORIGINAL_FENCE为整数，不预填新的身份或空DB。

~~~bash
"$RESEARCH_AUTOPILOT_CLI" --db "$CONTROLLER_DB" status "$PROJECT_ID"
"$RESEARCH_AUTOPILOT_CLI" --db "$CONTROLLER_DB" events "$PROJECT_ID" --after 0 --limit 1000
~~~

若status含原任务的reconcile_required，先核对同一host/run：

~~~bash
"$RESEARCH_AUTOPILOT_CLI" --db "$CONTROLLER_DB" inspect "$PROJECT_ID" "$TASK_ID" --owner "$ORIGINAL_OWNER" --fence "$ORIGINAL_FENCE" --adapter native --native-target "$NATIVE_TARGET"
~~~

实际running任务继续观察；unknown状态保留，不能另开worker或删锁。
只有观察证据支持相应outcome，才用同一owner/fence执行reconcile。
任务若已在原harness运行而尚未被runtime接管，保留原运行并先返回其真实身份；
不把它登记成“待启动”再重复launch。

## 3. 一个外层任务包住现有内层DAG

保留[扩展交接](LOCAL_EXTENSION_HANDOFF.md)中的jobcards与原harness。
base43＋extension29总计72配置，ScienceQA、ChartQA、MSCOCO_i2t最多216槽。
沿唯一pool/driver逐批运行，不能创建72个外层harness driver。
小批是队列前缀；不把本次小批完成当作全部72完成。

已有runtime contract直接复用。首次接入时，Local根据实际host与剩余额度
生成可信project/task/target契约，逐个绑定当前真实input/instruction/plan哈希；
这不是另一套候选发现或全量CI验收。project角色为local_executor，
科学任务action为native_harness；task.parameters.harness_plan_ref也必须是实际
pinned input/instruction。输出声明使用控制电脑实际取回的位置。
未知的路径、预算或计划不写成ready；不重置原18次和旧累计计数。

使用已有可审查契约时，实际CLI接口为：

~~~bash
"$RESEARCH_AUTOPILOT_CLI" --db "$CONTROLLER_DB" register-project "$PROJECT_CONTRACT"
"$RESEARCH_AUTOPILOT_CLI" --db "$CONTROLLER_DB" enqueue "$PROJECT_ID" "$TASK_CONTRACT"
~~~

相同ID/相同契约幂等；冲突需核对原内容，不能覆盖。
已有队列且同一pool没有运行中/待对账任务、实际本批计划与资源已准入时：

~~~bash
"$RESEARCH_AUTOPILOT_CLI" --db "$CONTROLLER_DB" worker "$PROJECT_ID" --owner "$ORIGINAL_OWNER" --adapter native --native-target "$NATIVE_TARGET"
~~~

`worker`一次处理一个外层任务，是前台命令；SQLite保留状态不代表进程自动常驻。
在已连接的Local持续任务/既有进程管理机制里执行，保留真实task/thread或PID/start identity。
一个外层任务可以持续调度本批内层DAG；需要后续批次时读取状态和剩余额度后接续。
不用Web reminder、无限shell重试或新pool冒充后台执行。
软报告继续保存进度；实际硬时限与剩余累计预算决定运行上限。

## 4. 取回输出、累计统计和结算

NativeBridge不会自动把host输出传回控制电脑。Local用已有SSH取回声明输出，
与实际receipt哈希逐项比对，再对同一attempt结算。
缺本地输出时即使host已完成，controller也可能仍为reconcile_required。

实际完成证据和本地输出都齐全后，命令为：

~~~bash
"$RESEARCH_AUTOPILOT_CLI" --db "$CONTROLLER_DB" reconcile "$PROJECT_ID" "$TASK_ID" --owner "$ORIGINAL_OWNER" --fence "$ORIGINAL_FENCE" --adapter native --native-target "$NATIVE_TARGET" --outcome completed --reason "$OBSERVED_COMPLETION_EVIDENCE"
"$RESEARCH_AUTOPILOT_CLI" --db "$CONTROLLER_DB" status "$PROJECT_ID"
~~~

OBSERVED_COMPLETION_EVIDENCE填实际同一run/receipt和取回哈希的说明，不复制“通过”模板。
failed/running结算也必须与实际观察一致；无法确认就保留待对账。

继续用[collect_audit_controls.py](../../tools/collect_audit_controls.py)的既有参数，
包括显式--extension-config，以及本执行版本对应的全部实际--receipt。
新版增加cumulative_progress和每配置task_statuses；只将三个任务都通过原生
live replay、同一fit的配置算作完整评分。部分/失败/阻塞/待跑和无效证据单列。
先保存点估计、预测与失败；完整CI/机制诊断随后补齐，不作为首批普遍前置要求。

只改collector不会改变已保存的head/预测，也不要求重跑方法；
收集时仍使用各receipt实际绑定的runner/config/bundle/scorer版本，不能混合不同执行版本。
修复说明和Local最小回归验收见[本轮进展记录](../../research/PROGRESS_2026-10-08.md)。
最终由原integration writer返回GitHub main实际结果commit、72行覆盖、所有实际失败、
task/run/owner/fence身份与剩余预算；Web据此继续分析。

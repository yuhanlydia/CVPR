# 全部72配置：Local长任务目标

状态：目标已保存；后台activation未验证。Web本对话没有暴露durable task launch/status API，
也没有用户控制电脑或GPU主机SSH执行入口。此文件不是started任务或运行receipt。
不要用定时提醒替代实验执行，不把source/checkpoint写入当GPU启动。

## 在已有连接的Local任务中使用以下目标

用Research Autopilot作为我的Local执行控制器，持续完成yuhanlydia/CVPR r004的
全部72个登记配置在ScienceQA、ChartQA、MSCOCO_i2t上的原生开发评估。
小批只决定执行顺序，不缩减最终范围。先跑通再完善，不等全量CI/诊断/审查文件
全部完成才开始；保存真实结果后再补完善。不要写完计划就结束。

从GitHub main至少包含12a6c9fdfd69559b19db574fb74c262e36723550及此goal的实际交付commit恢复。
保留本地a762948所含10个I09提交、实际后续本地更改、旧checkout和结果；
核对当前完整HEAD，不覆盖、不假称当前仍是a762948，也不重新做未改变版本的已通过检查。
先读当前AGENTS.md、LOCAL_AGENT_RUNBOOK.md、rounds/r004/EXECUTION_POLICY.md、
LOCAL_EXTENSION_HANDOFF.md、EXTENSION_COVERAGE.json和WORKFLOW_CHECKPOINT.json。

沿用我的控制电脑、现有SSH、GPU主机原生Conda、实际共享pool和唯一run_harness owner。
已报告297文件source checks通过，复用其真实receipt/版本；
harness实际部署、缓存兼容接受和可用预算必须读当前记录，不能从历史消息猜状态。
缓存EOF单LF修复是bd947cc；按PREPARE_CACHE_COMPATIBILITY.md验证实际producer SHA，
不改旧receipt、不重新准备已有兼容缓存、不把source check算成方法结果。

调用现有tools/prepare_audit_job_cards.py的--extension-config模式恢复72项队列，
用实际准入的native/outer plans与原owner逐批执行，保存所有失败和未完成项。
不重写可用工具，不开第二队列，不裸跑绕过资源追踪；
共享缓存可用时head/scoring是0GPU CPU任务，先依实际CPU/RAM测量合理调度，
不要因为GPU空闲就盲目提高并发。额外下载/重建缓存单独计入实际限额。

前次18次额度用完，不清marker、不重置用量。
若当前已存在针对r004的有限新增授权，直接复用；若确实没有，
先完成全72范围下可审查的有限计划、实际成本估计和授权差额，
只把这个实际缺口返回给我，不把所有文档未完善当另一个开跑门槛。
未知授权不能写成passed。没有实际资源/授权时paused，保存下一步，不伪造启动。

在实际允许的持续运行期间，保持同一owner和stable IDs，按真实依赖继续其余配置；
可通过已授权tmux/nohup保持GPU主机harness跨SSH断连，断连后先reconcile后resume，
不重复launch。软报告不终止健康作业，实际硬时限必须保留。
报告cadence沿已确认规则；这里不新建定时提醒，也不假称Web自动轮询。

每个里程碑和返回都给累计进度：总72、三个任务均已评分的配置数、
部分、failed、blocked、待跑，以及各task完成槽数（最多216）。
保存实际raw预测、官方live replay、同版本必要parent/强控制、成本和来源；
完整CI/诊断可以后补。I01真实校准等条件缺失单列BLOCKED，不能算已跑完。
完整结果包括72行状态与三任务成绩，不只挑两项高分，不用旧版本数字顶替新结果。

实际运行receipt、execution full SHA/dirty patch、env/model/data/source、
剩余预算、所有失败和raw locator/hash在每轮形成小的可读结果包；
由既有控制电脑integration writer推yuhanlydia/CVPR main并读回，避免Web同时改同一结果文件。
不推private skill、权重、图片或大缓存。
完成标准：72项逐一有三任务实际结果或明确可复核失败/真实阻塞，
条件阻塞不宣称全72已成功跑完；任何剩余项都有实际下一步。
实际task/thread locator、PID/start identity、plan digest和logs须从真实host记录，不自行编造。

## Web侧接续

Web仍负责收到结果后的源码/数学诊断和修复main交付，
不把一个Web后台goal当成GPU已连接，也不运行项目代码规避既有Web/Local角色。
恢复本checkpoint与最新结果packet继续，不重建已有设计或创建重复task。

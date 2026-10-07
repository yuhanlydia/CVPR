# Local补充实验实现交接（尚未实现）

源基线：CVPR 12a9252feaac60e69e39087b9d994f99a0162243；
skill a8343aeb4f51303e2eb651081d4fe51c24c5ed3f。
[设计/数学卡](ADDITIONAL_EXPERIMENTS_20261007.md)、
[29配置声明](../../configs/proposed_audit_extensions_20261007.json)
均proposed_unimplemented；不能把B_*传给当前runner。

## 精确适配范围

1. 新建experiments/embedding_heads/audit_extensions.py，
   独立extension_registry只接受29个B_*。保留原43-arm registry和原配置字节，
   不把len43硬改72而破坏旧协议。读取真实train rows/native query_id；
   每个参数来自声明和parent_config，不用test拟合/选配置。
2. 实现query等权all-pairs流式CE、teacher-hard/固定seed native负例、
   teacher-label mix和相应解析梯度；teacher target保持原τ语义。
   记录pair选择/target digest、行权重归一和实际pair数。
3. source/subset helper先筛行再取allowed候选并集，同步重映射四个矩阵
   与真实IDs；重复unique query身份需明确定义，不能用行索引伪造。
   输出selected_row/candidate SHA和保留的原source身份。
4. whitening用固定parent μ、原α/λ，两种锚定二阶矩；
   I02 rank/随机QR需保留实际R、投影残差和查询保护能量。
5. run_audit_method.py添加显式可选--extension-config：
   验证base配置SHA、extension配置SHA、实现/输入/参数SHA；
   仅选B_*时调用extension fit。原默认43-arm路径不改。
6. job-card/collector添加同一个显式扩展参数，完整跟踪新增source refs，
   native三group资格、失败/阻塞/未完成、训练pair/subset身份和参数一致性。
   不能绕过旧manifest producer源码身份检查，不能凭旧JSON生成新合法receipt。
   完整CI分析按设计登记端点，不能只返回winner。

## 实现后的Local接受顺序

A. 先reconcile已有run/receipt/原累计预算，确认修复后parent与强基线结果状态。
   固定实际Conda python、Qwen scorer checkout、bundle和已接受skill source身份。
B. 只在Local作源接受与真实native输入语义检查：
   对CE/mix/all-pairs解析梯度作有限差分、α0/1端点等价，
   pair权重每query总和、native标签/mask保持、候选顺序变更与ID选样稳定；
   source/subset嵌套和candidate remap、PSD/inverse square root、
   I02 R正交与保护残差。使用真实训练包，只认数学/软件性质通过；
   不将PROPERTY_CHECKS_PASS当科学门通过。
C. 若多pair不存在，E09 all-pairs标冗余；native负例语义不合法则hard-negative阻塞；
   unique queries不足则对应subset阻塞；缺合法来源则E11/E12阻塞。
   单臂问题不伪造数据；共享producer/scorer/source问题阻塞全部后代。
D. 实测CPU/RAM/fit/scoring/write成本及剩余attempt/time，
   同版本运行required parents和完整2048强对照，三任务native contracts逐一接受。
   生成完整科学purpose plan而非engineering绕行；实际G01/资源/native接受后才freeze。
E. 原唯一outer run_harness执行，有效digest绑定实际原生job cards；
   不开第二harness、不Docker、不自动重试、不重置预算。
   本提案不授予87次jobs。未准入项carryover，不能删benchmark/强对照压时间。
F. 推送完整三任务score、native分母、原始预测host locator/SHA、
   官方live replay、所有attempt/失败/成本、paired CI与不确定性到GitHub；
   不推权重/图片/大缓存/作者private skill源。缺少证据只标pending。

## 当前可做与不可误用的命令边界

现有r004已实现工具的精确命令保留在[WEB_HANDOFF.md](WEB_HANDOFF.md)。
它们只能处理原43臂配置：
- tools/qualify_audit_controls.py：现有数学/软件性质，非29扩展验收；
- tools/prepare_audit_job_cards.py：旧43臂draft，不是本29臂派发；
- tools/run_audit_method.py：旧注册ID；B_*现在会拒绝；
- tools/collect_audit_controls.py：旧coverage/receipt合同，不冒充新增结果。

在扩展接口和资格实际落地前，没有可靠的新增运行命令。
因此这里不提供看似可执行的B_*命令或伪造已完成receipt。
Local实现后必须将精确argv、真实环境路径、输出契约和schema补回同一交接，
再进入原native/outer harness准入；任何参数字段变化需新child protocol。

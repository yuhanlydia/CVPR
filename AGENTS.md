# 当前增量：29配置扩展代码（2026-10-07）

先读[当前扩展Local交接](rounds/r004/LOCAL_EXTENSION_HANDOFF.md)和[逐臂三任务完整度矩阵](rounds/r004/EXTENSION_COVERAGE.json)。
29新增配置已生成代码，与原43条通过显式--extension-config连接；共72条登记，
包含所有历史方法/控制。48 primary＋124 secondary contrasts覆盖三个完整原生任务。
状态generated_unexecuted；Web未运行代码/tests/模型，实际Local/native/G01/资源接受及结果pending。
skill固定最新a8343ae。旧默认43臂和活动attempt保持原绑定；原18次/累计硬界不重置。
此当前入口适用于新增接受/执行/修复；下面原交接与历史记录保留。

# CVPR project instructions

## 当前增量：r004现有方法审计（2026-10-07）

先读[当前r004 Web handoff](rounds/r004/WEB_HANDOFF.md)、
[8组完整实验设计](rounds/r004/EXPERIMENT_DESIGN.md)、
[数学卡](rounds/r004/MATH_CARDS.md)和
[a8343ae来源核对](research/AUTOPILOT_DELTA_2026-10-07_a8343ae.md)。
新任务固定作者a8343aeb4f51303e2eb651081d4fe51c24c5ed3f，完整297文件身份锁为
configs/autopilot-source-a8343ae.json；旧来源/活动任务不迁移。
25个新对照加18个历史记录组成43条登记，常规42条预计可评分、I01校准条件保留。
这是M路线审计/修复/收敛/已知控制，不是新方法发现或已通过top15/原创性。
新源码与命令全部generated_unexecuted；Local在已有SSH/Conda和单一remote run_harness
接受真实源码/cache/native/G01/资源后执行。jobcards不是native/outer dispatch plan。
不修改旧18次/累计硬界/marker，不给126个group job自动授予新预算。
以下r003记录与新发现的科学前提继续保留。


Read [LONG_TERM_TASK.md](LONG_TERM_TASK.md), [LOCAL_AGENT_RUNBOOK.md](LOCAL_AGENT_RUNBOOK.md),
the [current Web handoff](rounds/r003/WEB_HANDOFF.md), and
[the skill upgrade audit](research/AUTOPILOT_UPGRADE_2026-10-07.md) at the actual pinned
commit before setup, acceptance, repair or new method development.
[Acquisition and source installation](LOCAL_AGENT_RUNBOOK.md#下载与完整-skill-安装)
is the source-maintenance entry. Read the current
[complete retained-input download/acceptance cards](docs/NATIVE_ASSET_RUNBOOK.md#download-datasets-and-models),
[twenty-card review](research/reviews/R003_REVIEW_2026-10-07.md) and
[latest author delta](research/AUTOPILOT_DELTA_2026-10-07_99d8ac0.md).
The old source lock remains for active bindings; use configs/autopilot-source-99d8ac0.json
for newly accepted current-author installation. New acquisition/loader source is
generated_unexecuted; full top-15/scientific prerequisites are still pending.

The Web role is `web_supervisor`: inspect primary papers, actual code and native
benchmark protocols; perform mathematical review; generate code/commands; publish
scoped changes to literal main and verify remote content. Web does not run
generated code, software tests or experiments. Label new deliverables
`generated_unexecuted`; read existing CI receipts only at their actual revision.

The Local control agent runs on the user's computer and operates the GPU host
using existing authorized SSH. Restore actual paths, environment and resource
identities before setup. Use the complete pinned skill and one remote
`run_harness.py` owner for executable setup/test/model/scoring/collection work.
No Docker or substitute containers. Prefer the existing native Conda environment;
keep historical venv runs at their original revision.

The 20 cards and 12 implementations in r003 are retained developmental history.
They are not a verified top-15 selection or complete CVPR experiments.
Audit/reuse justified mathematics and implementations, preserve IDs and negative
history, and close current math/selection/originality/code/G01 prerequisites
before admitting new method plans. Never manufacture reviews, scores or receipts,
pad the candidate pool or reduce native benchmark coverage to pass a check.

Keep active Local jobs at their pinned source/skill/protocol revision.
Do not overwrite concurrent main changes, force-push, reset cumulative budgets,
delete attempts/markers, retry unknown live jobs or upload private skill contents,
model weights, restricted images or large caches to this repository.

# CVPR project instructions

Read [LONG_TERM_TASK.md](LONG_TERM_TASK.md), [LOCAL_AGENT_RUNBOOK.md](LOCAL_AGENT_RUNBOOK.md),
the [current Web handoff](rounds/r003/WEB_HANDOFF.md), and
[the skill upgrade audit](research/AUTOPILOT_UPGRADE_2026-10-07.md) at the actual pinned
commit before setup, acceptance, repair or new method development.
[Acquisition and source installation](LOCAL_AGENT_RUNBOOK.md#下载与完整-skill-安装)
is the current source-maintenance entry; full model/data download cards are pending.

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


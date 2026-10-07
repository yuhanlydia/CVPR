# r004部署恢复：缺harness / 远端GitHub断连

来源：Local报告本地main的HEAD a762948包含Web 8ff091e及原10个I09提交，
尚未push/启动；GPU主机空闲，harness/pool/已接受预算未恢复，隔离checkout网络clone失败。
这是用户提供的本地状态，Web没有读取本地a762948或远端主机。
本记录是source-reviewed恢复交接，未执行任何安装、测试或实验。

## 1. 找对runtime来源

run_harness.py属于Yunbo-max/Research_Autopilot固定a8343aeb4f51303e2eb651081d4fe51c24c5ed3f：
skills/research-autopilot/scripts/run_harness.py。
CVPR仓库不打包这个runtime，也不要求GPU主机运行Codex/GPT。
完整四个sibling技能和runtime scripts/schemas按
[原来源安装卡](../../LOCAL_AGENT_RUNBOOK.md#下载与完整-skill-安装)部署；
不是只复制run_harness.py或SKILL.md。不得把作者private源码提交到CVPR。

先在控制电脑查已安装技能/已固定作者clone；
在GPU主机只读核对旧运行记录里的runtime root、pool_dir和owner。
原pool_dir恢复前不得另建第二pool或删除锁。
作者源码的默认pool是~/.local/state/research-autopilot，
这个默认值只能作为查找线索，不证明旧任务用了它。
Python runtime要求3.11+，模型/scorer继续真实原生Conda。
已有owner/未知PID先reconcile；GPU空闲不等于没有owner或累计用量。

## 2. 远端GitHub失败时用控制电脑传输

不需要remote GitHub凭据，也不需要反复clone。
控制电脑已有完整项目Git时，使用git bundle带上本地main全部已提交祖先，
包括10个I09提交；不得只复制8ff091e而丢弃local改动。
控制电脑原仓库与远端旧checkout/结果不修改。

控制电脑（D:/cvpr/CVPR，Git操作，不是科学任务）：
先读取git status --porcelain、git branch --show-current、git rev-parse HEAD，
确认main实际完整SHA、dirty状态与8ff091e祖先关系。
有未提交修改时先保留并明确部署范围，不声称bundle包含这些修改。
在新的传输文件位置执行：

~~~powershell
git -C D:/cvpr/CVPR merge-base --is-ancestor 8ff091e08a8248fec00ea6f306ce9a1b0c5f896b refs/heads/main
git -C D:/cvpr/CVPR bundle create D:/cvpr/CVPR-r004-main.bundle refs/heads/main
git -C D:/cvpr/CVPR bundle verify D:/cvpr/CVPR-r004-main.bundle
Get-FileHash D:/cvpr/CVPR-r004-main.bundle -Algorithm SHA256
~~~

每条命令分别检查exit code；已有同名bundle先核对/用新文件名，不覆盖未知记录。
通过已有SSH/scp传到实际恢复的远端staging位置，核对SHA256。
SSH alias、staging和新checkout路径必须从已有配置/记录恢复，不能把示例路径当主机事实。
在远端新隔离目录git clone --no-checkout本地bundle，
checkout --detach到控制电脑记录的完整HEAD（不是仅a762948短串），
读取rev-parse HEAD和merge-base确认祖先。不得覆盖旧checkout。
这些是source transfer/Git元数据操作，不启动项目脚本。

作者clone同理在控制电脑固定a8343ae，用git archive完整该revision到tar，
经已有SSH传输核对文件SHA后在新的runtime source目录解包；
保留skills/四个目录、scripts/schemas和INSTALL_LOCAL.md的真实层级。
若控制电脑只有完整安装目录，则按其真实四sibling目录打包并依锁验证；
若作者源也缺失，先在有现有读取权限的控制电脑获取固定源。
不在CVPR里存private skill源码，不把远端断网当需要另一台GPU。

## 3. 正确锁文件与bootstrap检查

恢复RESEARCH_AUTOPILOT_ROOT为远端实际research-autopilot目录；
其parent包含另三个真实sibling。保持旧任务旧版本绑定。
来源工具tools/check_autopilot_source.py默认lock是旧configs/autopilot-source.json；
r004新接受必须显式--lock，不能用默认旧锁误诊a8343ae：

~~~bash
"$NATIVE_PY" tools/check_autopilot_source.py --skill-root "$RESEARCH_AUTOPILOT_ROOT" --lock configs/autopilot-source-a8343ae.json
~~~

这是只读源码身份核对，SOURCE_BYTES_MATCH不证明runtime/数学/实验接受。
它应在原规则的实际准入setup/acceptance任务里执行；不将项目脚本裸跑作为绕行。
harness自身bootstrap按作者local-execution-harness.md允许的只读主机检查：

~~~bash
"$HARNESS_PY" "$RESEARCH_AUTOPILOT_ROOT/scripts/run_harness.py" --inspect-host
~~~

HARNESS_PY是实际远端Python3.11+解释器；NATIVE_PY是实际Conda解释器，
相同或不同均需记录，不能沿用控制电脑的python路径。
先核对来源/脚本存在及旧owner/pool，再inspect；此命令不启动科学workload。
安装runtime、查路径或inspect成功，不等于已验收科学plan。

## 4. 预算和缓存对账后才准入

恢复旧window_budget/marker、prepare/native attempt receipts、I09各次真实耗时和失败；
把10个Git提交与实际attempt分开，不能按commit数推算compute用量。
核对累计授权、已用、剩余、真实host CPU/RAM、允许的setup/acceptance范围。
缺记录就是未知，不造“剩余216次”或新8小时凭据。
若旧额度用尽，返回最小完整比较及实际成本估计，需要真实新增有限授权，
不能因为72条设计登记而自动扩张预算。

缓存必须绑定原真实完成producer、当前prepare代码身份及原layout，
所有manifest refs/hash、teacher/PCA/train labels、完整native候选/分母均接受。
本地I09修改与Web代码merge不代表数学/源码已自动兼容：
复核a762948实际diff；结果影响改动须child protocol和同版本parent/control。
不重新跑已接受且版本未变的检查，不把旧版本结果顶替新版证据。

随后按[扩展Local交接](LOCAL_EXTENSION_HANDOFF.md)的有序命令，
在唯一remote harness中完成软件/native/G01/resources接受，生成job卡，
绑定真实scientific native child与有限outer plan，冻结实际digest后才启动。
先恢复parent修复后重放和E01/E04强对照；任何第一批均保留三个任务和必要控制。
当前没有ready plan，本记录不授予启动或扩张旧预算。

## 返回

先返回实际项目full SHA/dirty diff、runtime root/source checks、Python、
原pool/owner状态、缓存身份、旧累计已用/剩余和各阻塞原因，
不用再只返回“找不到harness”。
若有待准入finite plan，返回其完整比较、资源/成本、必要授权差额，
不写假passed receipt。完成后再沿原结果返回/push/readback合同。

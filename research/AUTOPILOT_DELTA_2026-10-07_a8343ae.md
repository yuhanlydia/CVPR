# 作者最新版来源核对：a8343ae（2026-10-07）

本轮实际读取 Yunbo-max/Research_Autopilot @
`a8343aeb4f51303e2eb651081d4fe51c24c5ed3f`（观察时间为本轮检索）。
本地插件目录的旧快照不冒称作者最新安装。完整四skill文件身份锁为
[configs/autopilot-source-a8343ae.json](../configs/autopilot-source-a8343ae.json)，
来自完整未截断Git tree的297个blob；锁定文件身份不代表所有文件已读、已安装或测试通过。
实际本轮读取范围在[r004 source checkpoint](../rounds/r004/SOURCE_CHECKPOINT.json)。

本轮适用规则：
- Web生成完整来源/数学/代码/设计，标记generated_unexecuted；不执行项目代码/tests或模型数据下载。
- 当前M路线的审计/修复/收敛不人为重启20→15；新发现仍需要验证全池、排名前15和原科学前提。
- 沿用已来源化单GPU数量，Local恢复真实型号/VRAM/UUID/占用。缓存小头和评分请求0 GPU，
  若需要重建缓存则是另一个真实1 GPU准备任务，不能当免费控制。
- Local使用原生Conda、已有SSH和单一remote run_harness owner；禁止Docker/容器。
- 8小时是未来批次软报告节奏，Local周期至多24小时且受真实更短限额约束；
  旧运行计划的8h硬界、18 attempt、marker/receipt不自动更改或重置。
- 所有修复后评分/目标/参数变化均属开发子协议，旧成绩不能充当新独立确认。

相对99d8ac0，作者存在runtime/controller/lease相关维护；本轮实读新run_experiments、
run_harness和verify_methods入口/模式/plan schema，未声称项目适配已运行通过。
a8343ae提交说明为native交接测试增强与saved来源一致性；前一1aa4298是心跳CI增强。不能把作者CI或旧项目单测
当作这次CVPR新增代码/native协议的接受证据。没有在Web运行任何CI。

新来源锁与旧autopilot-source.json/99d8ac0锁并存。活动任务继续原固定版本；
Local隔离接受新完整四目录后才把新任务绑定到新来源。私有skill内容不复制入CVPR。
r004的43条arm登记是42条当前预计可计算配置加I01条件保留，并非43个原创idea。
Parent/Natural Gate0/IPCG、原20卡全池/前15、Local原生资格仍按其原边界待关闭。

## 交付前最新版增量核对

先前读取的1aa4298与当前a8343ae已做真实Git compare和完整tree身份比较。
全部SKILL.md、references、scripts、schemas的blob完全一致，适用已完成读取；
本轮再次读取a8343ae entry，blob78d9f31070225056012a5cb95f0d6db4133f1b38。
新的skill UI YAML采用等义格式、test_harness.py改为基于真实进程ready标记的交接测试，
EXPORT_MANIFEST/VALIDATION/审查记录更新。297文件锁采用a8343ae真实新blob身份。

作者记录已保存/验证其个人skill来源树，未在Web查用户的实际安装。
作者文档保留此前失败和19项scoped测试范围，不能给CVPR新增实验认证。
科学规则、生产scheduler/validator和本文生成代码的接口不变；
没有因为test-only更新重跑任何项目/作者测试，也没有改活动Local来源。

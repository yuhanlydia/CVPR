# Round 003 执行与返回

用户最新要求为助手只生成代码/命令并交付 GitHub，由用户的 local agent 运行；不使用 Docker。
完整交付与运行命令见 [LOCAL_AGENT_HANDOFF](../../docs/LOCAL_AGENT_HANDOFF.md)。
上层代码任务与续做待办见
[LONG_TERM_TASK](../../LONG_TERM_TASK.md) 和 [research-backlog](../../research-backlog.json)。
下列命令仍对应一次有限原生批次，不能靠定时重复命令延长累计时间。

## 现状与授权边界
用户要求检查并补齐“20 数学候选 → 10–15 筛选 → 读官方 code 后实现 → 单项报错继续”。
本批固定 20 张卡、12 个原型、6 个对照；8 项暂存，不启动 RL/视频训练。
研发原型不是正式研究批准：Natural Gate 0、IPCG、Gate A 全部未通过，真实性能未知。
先前 33 个 CPU 检查只对应 cf862cd 版本；新增验证看 GitHub 检查记录，不沿用旧通过数。

## 资源与 G01 设计卡
- 问题：固定小模型预算下，训练小头能否保留/改善原生多模态候选排序？
  真实失败尚未观察，所以该问题为 developmental，不承诺新方法必要性。
- 载体：冻结 Qwen3-VL-Embedding-2B，FP16/SDPA，单卡串行；用户报告 22 GB 2080 Ti，
  实际空闲显存/时间/内核可用性须由 host + native r001 资格化读取。
- 训练：MMEB-train@0c3f4b8 的 original；当前 main 配置固定 ScienceQA/A-OKVQA 各前 128 行。
  不按性能挑训练行、不从测试拟合 PCA/教师/分组/参数。
- 评测单位：当前 main 配置要求 r001 已完成的官方完整 ScienceQA、ChartQA、MSCOCO_i2t
  查询与原始全部候选；不截尾排名、不补标签。旧单任务缓存不满足此版本。
  核查原始 official parser/labels/split，live RankingMetrics 与实际持久化预测重放。
- 端点：原生 hit@1；每条方法卡的必需对照在 configs/candidates.json。
  全维原始基线、相同 32 维恒等头、平方 KD、ERM、单正例、白化、RBF；
  I12 还需 I16 的同锚点对照。
- 不确定性：每任务配对原始查询 bootstrap，固定 seed=42，2000 次，95% 百分位区间。
  查询可交换假设、同数据挑选多个候选的选择偏差须保留；这不是跨任务/独立 seed 确认。
  1 pp 是预先记录的探索实际差异尺度，不据此自动 PASS/KILL。
- 成本：计入环境准备、数据/模型获取、训练特征编码、拟合、完整评分、保存、hash 与返回。
  r001 测得的峰值/时间不自动证明训练输入同成本；新 preparation 回传实际峰值。
- 总界：保留 setup-receipt.json 的原始累计 8 小时；准备最多 1 次/7200 秒，
  12+6 项各最多 1 次/600 秒、重试为 0；尾部预留 120 秒。
  这些是硬上限，尚不是测得的容量模型。大任务可能 CARRYOVER，不保证 8 小时全完成。
- 判决：只有开发观察；缺原生资格/对照/校准/精度时 PENDING/BLOCKED，
  崩溃是工程故障，不是科学 KILL。不得按 exit 0 晋级。

## 准备真实基线
先安装已有完整 Research Autopilot（设置 RESEARCH_AUTOPILOT_ROOT），并按
[REMOTE_RUN](../../docs/REMOTE_RUN.md) 的 r001 命令得到真实完整 baseline 输出。
Qwen 的 sources checkout 保持固定提交且干净。模型/数据是公开 HF 输入下载，
不上传任何输出至 HF。不要删除 setup-receipt.json 或 attempts 来延长窗口。

准备程序只读取自己被实际 native receipt 标识的 r001 生成缓存；不接受随手构造的 pickle。
训练图像必须是原始 MMEB 图像，并在根下保留 images/...；本程序不自动下载 47 GB 全包。
缺图、原始字段不兼容、训练测试实际图像/输入重叠都会阻塞共享准备。

## 单命令启动有限批次
从工程根目录运行；BASELINE_OUT 是原始 receipt 中 completed qualify attempt 的 cwd/out，
例如 runs/attempts/r001-.../qualify-a1-.../workspace/out。
MMEB_TRAIN_IMAGES 是已有原始训练图像根目录，不能指向 eval images。

~~~bash
export RESEARCH_AUTOPILOT_ROOT=/actual/installed/research-autopilot
export BASELINE_OUT=/actual/CVPR/runs/attempts/r001-.../qualify-a1-.../workspace/out
export MMEB_TRAIN_IMAGES=/actual/MMEB-train
.venv/bin/python tools/run_candidates.py \
  --baseline-out "$BASELINE_OUT" \
  --train-image-root "$MMEB_TRAIN_IMAGES" \
  --hours 8 --execute
~~~

已有真实准备缓存时，用 --bundle /actual/CVPR/.../out/manifest.json 替代两个准备参数。
manifest 和其所有输入必须位于工程内，便于 native runner 按实际哈希隔离每次尝试。
去掉 --execute 只显示库存与剩余原始时间，不运行模型。

同一个 retained window 的候选批次只能启动一次。允许明确新增本批范围，但必须确认此前
attempt 有终态 receipt，并取得与 r001/Wan 共用的 whole-device lock；原始时钟不重置。
重复命令不会自动续跑、重试或换 seed。下一窗口/修复需要保留原记录并另行明确 scope。

## 只读有限 carryover 审计（2026-10-07）

~~~bash
.venv/bin/python tools/plan_candidate_carryover.py
~~~

默认读取真实 latest batch；--run-id 只选择已有记录，--hours 只降低原始累计上限。
不创建/修改 marker 或 receipt、不派发实验；失败/超时/中断都消耗原来的一次尝试。
核对已保留 native plan/receipt/attempt 的文件绑定、配置/代码/输入/输出 SHA、
旧窗口身份与 120 秒尾部；未知目录、缺终态、版本变化和已过期预算阻塞清单资格。
即使输入损坏，也保留已观察失败的 locator，不将其重新列为未尝试。
本步只是审计器，自动续跑和活动进程核验仍未实现；exit 0 不代表科学验证。

## “报错不停”的具体含义
- 每项独立 subprocess、cwd、stdout/stderr、模型头、原始预测、分数、receipt。
- 单项普通异常、OOM 非零退出、缺方法专属输入、单项 timeout：记录后继续其余项。
- I01 没有独立真实校准区间，I05 没有原始多正例：明确 BLOCKED；不伪造资格。
- 准备失败：所有依赖它的项逐项 BLOCKED_SHARED_INPUT，不重复运行同一坏输入。
- 必需对照失败：其他独立项仍执行，但受影响的比较在返回报告标记 PENDING_COMPARISON。
- 用户取消、全局 8 小时到期、未清理子进程/未知以前 job、改动了冻结输入：停止并留证据，
  不能把“不停”理解成无限重试或忽略资源/协议异常。

父程序有累计时限 alarm，native watchdog 负责每个 subprocess/process group 与 GPU lock；
不会因为父程序退出让已知 child 逃离 attempt 时限。读取安装的真实 native runner，
不公开复制私有 skill 代码，也不使用假 host 来冒充验证。

## 查看和返回
启动时打印 ledger。实时日志位于 runs/attempts/r003-...-prepare 或 -heads 下；
完成/失败项各有 attempt.json，尚未完成的没有终态 receipt。

~~~bash
.venv/bin/python tools/collect_candidates.py
~~~

保存全部项状态、所有完整比较、原始失败日志与预测。小报告在 rounds/r003/returns，
原始返回压缩包在 runs/；不包括模型权重、图片和大型编码缓存。
把小报告与对应压缩包返回当前对话再核查；不能只返回最好的一行分数或截图。
本地/CI 没有 GPU 时，模型加载、原生数据/scorer 的实际运行、显存与总耗时均 NOT_RUN。

## 新增工程检查
GitHub CI 只安装 NumPy，检查矩阵/约束、预算/状态、文件来源与只读续做清单；
实际数量和通过范围以本次交付 commit 的 candidate-engineering 日志为准。
2 个真实子进程续跑集成检查仅在 RESEARCH_AUTOPILOT_ROOT 指向实际完整 skill 时运行；
没有安装则 SKIP，不能把 CI 绿色解读为这两项已通过或 GPU 已运行。
私有 skill 不公开复制进仓库或 CI。真实训练机可运行：

~~~bash
.venv/bin/python -m unittest discover -s tests -p 'test_candidate*.py' -v
~~~

旧单任务 ScienceQA 输出无法支持 I11 的跨组最差性能声明；当前多任务配置尚无已核查结果，
也不能证明视频/RL/跨域泛化。
对应完整科学比较仍需增补原生资格、相应任务与独立确认，本批只保留开发观察。

## 本次交付的证据来源修复
准备 manifest 必须保留实际 completed native prepare 的 attempt.json、生产脚本 SHA 和
真实 output SHA；方法 staging 同时携带这个 producer receipt。
全局累计 alarm 被 native runner 处理为 interrupted 后，父程序仍按真实 deadline 标注预算 carryover。
collector 复核实际 native result/prediction SHA 和 child scorer replay SHA，
损坏比较标为 INVALID_EVIDENCE；共享来源损坏仍返回失败/原始日志，不给它评分。
这些是代码语义与工程文件完整性检查，真实 GPU/native 执行由 local agent 验证。

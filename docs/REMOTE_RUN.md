# 在 VS Code 的 SSH 远程终端运行

当前 r003 优先使用 [LOCAL_AGENT_HANDOFF](LOCAL_AGENT_HANDOFF.md)。网页助手只交付代码与命令，
用户的 local agent 执行；下文为 r001 原生基线入口，必须拉取包含这些入口的 review 分支。

当前网页会话没有 GPU，也没有连接你的 SSH 主机。下面的命令在 GPU 机器执行。该机器需要 Linux、Python 3.11+、Git、可用 CUDA 驱动、可访问公开 GitHub/HF/PyTorch 资源，预留至少 35 GiB 磁盘和 9 GiB 可用显存。这些是启动检查门槛，真实模型峰值仍需校准。

## 1. 拉取与环境准备

新目录：

```bash
git clone --branch review/wan-readiness-20261006 --single-branch https://github.com/yuhanlydia/CVPR.git
cd CVPR
export CUDA_VISIBLE_DEVICES=0
bash tools/bootstrap.sh
```

已有干净 checkout 使用 `git pull --ff-only`；有未提交修改或正在运行的任务时，保留它们并用另一个目录拉取。bootstrap 在项目 `.venv` 安装固定的直接依赖，使用 PyTorch 官方 cu126 wheels，固定 Qwen 源码 commit。不会安装 FlashAttention。驱动/wheel 不匹配时预检失败，保留失败输出，先修环境再运行模型。

bootstrap 最长 1 小时，写入真实设备、包版本和 FP16/SDPA 软件检查。当前规划按用户报告的 22 GB 2080 Ti，7B 仍可选；模型是否适合实际空闲显存由真实加载校准决定。完整下载可能包含约 7.13 GB 的官方图片压缩包；仅解出本轮实际引用的图片。下载也占本轮时间。

## 2. 指定已有 Research Autopilot

入口自动查找以下完整安装目录：

- `~/.agents/skills/research-autopilot`
- `~/.codex/skills/research-autopilot`

如果你的 skill 在另一个位置：

```bash
export RESEARCH_AUTOPILOT_ROOT=/absolute/path/to/research-autopilot
```

路径必须含 `scripts/run_experiments.py`、其依赖脚本和 `schemas/`。本轮仓库没有捆绑另一人的私有 skill，也不会替你克隆那个私有仓库。若 GPU 机器尚未安装，请在你有权限的本地 Codex 中安装已有的完整 skill；skill 源码的取得权限与本研究仓库权限分开。

## 3. 执行本轮

```bash
.venv/bin/python tools/run_window.py --execute
```

`run_window.py` 先检查设备/源码和本轮剩余时间，形成注册的原生执行器 plan，绑定配置、环境与代码文件 hashes，串行运行一个基线核查任务。全卡文件锁由原生 watchdog 继承，超时杀死整个受管进程组；每轮最多一次 attempt，无自动参数搜索或 OOM 重试。锁约束此入口启动的任务；它不能替代系统 GPU 调度器，因此预检还会检查真实空闲显存。

原生 plan 的类别是 engineering/developmental：它核查资源、数据和原生基线，不产生方法证实或 Gate PASS。下载时锁定 HF 实际 commit，此后所有原始数据读取使用该 revision，模型权重/图片 hashes 进入 receipt。未事先假装这些动态资产已经下载或验证。

如果已经有官方 MMEB 图片目录，可复用并记录每张引用图片的 hash：

```bash
.venv/bin/python tools/run_window.py --image-root /absolute/path/to/original-MMEB-images --execute
```

该目录应直接包含原始数据路径，比如 `ScienceQA/...`、`ChartQA/...`。缺失图片会失败；不会用空图片、NULL 文本或自制样本替代。默认运行最多剩余 8 小时；`--hours 4` 可把当前窗口缩短到总共 4 小时。时间不足的任务被记录为 pending。

## 4. 收集与返回

```bash
.venv/bin/python tools/collect.py
```

输出两个路径：

- `rounds/r001/returns/RUN_ID.json`：小报告，可按你的 GitHub 工作流 commit/push。
- `runs/RUN_ID-return.tar.gz`：原生预测、样本身份、评分、重放、配置、stdout/stderr、失败和资源记录；通过本聊天附件返回。

默认读取 `runs/latest.json`。强制中断导致没有 latest 时，运行时打印的 run ID 仍可用于 `tools/collect.py --run-id RUN_ID`；缺少完整 receipt 会明确标记。收集器不会执行上传的代码或加载外部 pickle，也不会上传权重和原始图片。

结果在 `runs/attempts/RUN_ID/.../workspace/out/`，完整特征缓存也在该工作区，可用于后续合法的缓存研究。失败/超时的残余缓存不是合格 checkpoint，本入口不自动复用；新的训练或评测配置需要形成新的协议。不要为赶满 8 小时而重新运行已经完成的基线。

## 故障和预算

环境准备、网络或 CUDA 失败：保留终端输出及 `setup-receipt.json`，不报告模型结果。下载超时：asset revision/日志仍保留，不用替代数据。真实推理 OOM：保存 stderr、硬件和配置，停止本 attempt；降低图像分辨率会改变协议，需要另行比较，不能静默修改。原生分母、候选集或评分重放不一致：该结果无效，不进入方法选择证据。

整个首轮只授权当前 8 小时。该入口不自动建立第二个 8 小时窗口，也不是后台日报/SSH 服务。准备重新开一轮时，先审查这轮结果和剩余任务，保留原窗口记录；不要删除或覆盖时间记录来掩盖已经消耗的计算。

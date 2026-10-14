# Knowledge Tracing Research

基于固定版本 pyKT 的独立知识追踪研究项目。目标是提出有明确研究问题和证据支持的原创改进，并在可比协议下超越强基线。**三个首期数据集已准备并核验，69 项自动检查和 9 次真实数据 GPU 前向检查通过；首批训练待确认，尚无训练结果或 SOTA 声明。**

首期数据集为 ASSIST2009、ASSIST2015、NIPS34；以 2026-10-29 为首轮结果汇报节点，提前验证完成就提前交付。困难场景主线只在验证阶段决定，不能看过最终测试再选择有利指标。

## 运行环境

本机已验证 Windows + WSL2 Ubuntu 24.04、AMD RX 9070 XT 16 GB、Python 3.12、PyTorch ROCm 7.2.1。代码固定使用 `third_party/pykt` 的上游提交 `eec087e0230520e7ae7bbd4f5433d71612df62fb`，不依赖不断变化的 PyPI 最新版。

在 Windows PowerShell 中运行：

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'bash /mnt/d/doc/knowledge_trace_graduation_project/knowledge-tracing/scripts/bootstrap_env.sh'
wsl -d Ubuntu-24.04 -- bash -lc 'bash /mnt/d/doc/knowledge_trace_graduation_project/knowledge-tracing/scripts/kt.sh doctor'
```

本机 bootstrap 创建 `/home/ml/.venvs/knowledge-tracing`，通过 `.pth` **只读复用**现有 `/home/ml/.venvs/pytorch` 的 GPU 运行库，在新环境中安装少量辅助包，不覆盖既有 torch。它是本机节省磁盘的环境方案，不是假设其他机器具备同一路径。迁移机器时先安装适配该 GPU 的 torch，再设置 `KT_RUNTIME_PYTHON` 和 `KT_ENV`。

以下命令在 WSL 的项目目录运行，使用上述新环境的 Python：

```bash
python -m ktlab data prepare all
python -m ktlab data inspect all
python -m ktlab status
python scripts/verify_upstream.py
python -m pytest -q
```

数据来源、许可证与实际校验结果见 [数据说明](docs/DATASETS.md)。下载文件、处理数据、权重、逐样本预测、审批记录和缓存不进入 Git。

## 实验工作流

1. 固定数据、学生划分、允许输入和评测协议，验证文件哈希。
2. 用实际数据与当前源码生成批次清单，展示模型、参数、预算和停止条件。
3. 用户确认该批次后，记录与清单哈希绑定的审批，再执行训练。项目初始化授权不代替逐批训练确认。
4. 仅用训练与验证数据选择模型、检查点和研究方向。
5. 冻结候选、分析方案及检查点，在已确认批次内开放最终测试。

```bash
python -m ktlab batch prepare-smoke
python -m ktlab batch inspect configs/batches/B001-smoke.json
# 下一条必须已有用户对这份具体清单的确认及对应审批记录：
python -m ktlab train --batch configs/batches/B001-smoke.json --approval approvals/B001-smoke.json
python -m ktlab report
```

第一批 proposal 只涉及 ASSIST2015 的 AKT/simpleKT 各 5 次优化更新；目的是验证完整训练管线，不评价模型质量。运行时间上限与具体配置以生成的清单为准。源码或数据清单改变会使旧审批失效。

## 评测与研究记录

- [评测协议](docs/PROTOCOL.md)：主指标、五折与随机种子、因果边界、与官方协议的差别。
- [前沿研究证据表](docs/RESEARCH_LEDGER.md)：论文状态、作者报告、可复现代码、输入与协议限制。
- [候选研究方向](docs/RESEARCH_DIRECTION.md)：待验证假设、最近邻方法和否证条件，不冒充已证实创新。
- [初始化交付](reports/INITIALIZATION_2026-10-14.md)：实际完成项、验证范围及首批训练申请。
- [KTST 接入审计](docs/KTST_INTEGRATION.md)：官方实现、题级集合表示和协议差异。
- `reports/`：只记录已经执行并验证的阶段事实。`runs/` 存本机运行证据。

主实验采用统一题目级验证和测试、完整精度概率及真实最佳验证检查点。它与上游默认的重复 KC 验证、四位概率舍入及检查点阈值存在明确差异，不能无标记地与论文最高 AUC 直接排名。

AKT/simpleKT 为基础对照；MoC-KT、FA-KT、KTST 等作为近期对照候选。模型可用性、输入条件、实现审计和本机验证结论以研究表和实际检查报告为准。没有运行的模型不会被标记为已复现。

## 许可

本项目代码使用 MIT 许可证。固定上游源码保留 [pyKT MIT 许可证](third_party/pykt/LICENSE)。数据集有独立使用条件，仓库代码许可证不授予数据再分发权。

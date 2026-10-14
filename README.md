# Knowledge Tracing Research

基于固定版本 pyKT 的独立知识追踪研究项目。目标是提出有明确研究问题和证据支持的原创改进，并在可比协议下超越强基线。**B001/B002 已完成运行校准；B003已完成ASSIST2009的AKT和simpleKT（验证AUC分别0.779116、0.764001）。MoC缓存兼容性错误经17项人工GPU检查修复后，独立B003-R1已启动剩余10组。** [首两组比较](reports/B003_FIRST_TWO_BASELINES_2026-10-15.md)保留2/12结果；[故障与剩余实验恢复](reports/B003_FAILURE_AND_RECOVERY_2026-10-15.md)另行留痕，尚无整批比较、最终测试或 SOTA 结论。

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

2026-10-14 最新指令优先于旧的逐批批准约定：用户已授权启动 B003，并持续授权本机项目内后续执行，无需逐批再次询问。每次训练最多50轮，连续3轮验证 AUC 无严格改善即早停；仍须保存具体清单、哈希、预算和真实授权依据。付费云资源不在默认授权范围。

1. 固定数据、学生划分、允许输入和评测协议，验证文件哈希。
2. 用实际数据与当前源码生成批次清单，展示模型、参数、预算和停止条件。
3. 将最新持续授权依据与该清单哈希绑定并记录，再执行训练；本机项目内批次不再逐次请求批准。
4. 仅用训练与验证数据选择模型、检查点和研究方向。
5. 冻结候选、分析方案及检查点后，按持续授权开放最终测试并保存解封记录，无需再次批准；测试标签仍不得参与选模。

```bash
python -m ktlab batch inspect configs/batches/B003-shared-convergence-baselines.json
# 执行前保存落实最新限制的清单，以及绑定其哈希的持续授权记录：
python -m ktlab train --batch configs/batches/B003-shared-convergence-baselines.json --approval approvals/B003-shared-convergence-baselines.json
python -m ktlab report
```

B001、[B002](configs/batches/B002-one-epoch-calibration.json) 已在各自冻结版本上完成，原审批仅覆盖对应历史批次。不要用当前代码重跑历史清单：源码或数据清单改变后，应建立新清单并绑定最新持续授权。B002 训练与验证循环累计775.15秒，记录起止跨度1065.44秒，峰值已分配显存2.166 GiB；KTST独立参考轨与其他模型分表报告。

[B003具体清单](configs/batches/B003-shared-convergence-baselines.json)已获启动授权：四模型×三集，fold0/seed42，按最新指令改为最多50轮、连续3轮验证 AUC 无严格改善早停、完整共同验证。执行须绑定落实上述设置的新清单；实际启动和完成状态以运行记录为准。AS09/AS15每次4小时、NIPS34每次8小时，累计上限64小时。这是固定配置收敛探索，不是64小时的预计耗时或SOTA承诺。

当前共享协议版本已通过174项自动回归及15项真实数据GPU无梯度检查，详见[验收记录](reports/SHARED_PROTOCOL_VALIDATION_2026-10-14.json)。这些检查不代替新协议下实际完成的训练或正式结果。

## 评测与研究记录

- [评测协议](docs/PROTOCOL.md)：主指标、五折与随机种子、因果边界、与官方协议的差别。
- [前沿研究证据表](docs/RESEARCH_LEDGER.md)：论文状态、作者报告、可复现代码、输入与协议限制。
- [完整证据卡](docs/evidence/README.md)：KTST逐折强配置、MoC全文缺口、UKT/KeenKT主表、KeenKT无梯度因果反例，以及频次收缩原创性判断。
- [候选研究方向](docs/RESEARCH_DIRECTION.md)：待验证假设、最近邻方法和否证条件，不冒充已证实创新。
- [初始化交付](reports/INITIALIZATION_2026-10-14.md)：实际完成项、验证范围及首批训练申请。
- [B001 结果与 B002 方案](reports/B001_RESULTS_AND_B002_PLAN_2026-10-14.md)：已完成训练的证据及下一批具体预算。
- [B002 结果与下一批方案](reports/B002_RESULTS_AND_B003_PLAN_2026-10-14.md)：12次已完成运行、成本与证据边界。
- [B003 首个模型的误差分层](reports/B003_AKT_ASSIST2009_ERROR_STRATA_2026-10-15.md)：AKT/ASSIST2009 的固定历史、训练题频及可见支持分层；稀疏题值得后续核对，尚不据此确定原创模型。
- [B003 首两组基线比较](reports/B003_FIRST_TWO_BASELINES_2026-10-15.md)：AKT与simpleKT相同验证事件的配对结果、学生抽样区间与固定误差分层。
- [共享事件开发协议](docs/SHARED_EVENT_PROTOCOL.md)：不同模型使用同一目标事件和允许历史的验证规则。
- [已保存验证预测的离线分析](docs/SAVED_PREDICTION_ANALYSIS.md)：收据核对、单/多KC分层、按学生配对重采样；仅探索性验证，不加载模型或解封test。
- [KTST 接入审计](docs/KTST_INTEGRATION.md)：官方实现、题级集合表示和协议差异。
- [KTST 协议复核](docs/KTST_PROTOCOL_REVIEW.md)：独立参考流程与未来统一比较的要求。
- `reports/`：只记录已经执行并验证的阶段事实。`runs/` 存本机运行证据。

主实验采用统一题目级验证和测试、完整精度概率及真实最佳验证检查点。它与上游默认的重复 KC 验证、四位概率舍入及检查点阈值存在明确差异，不能无标记地与论文最高 AUC 直接排名。

AKT/simpleKT 为基础对照；MoC-KT、KTST 已完成单轮训练校准，FA-KT 的完整依赖仍待解决。B002 的 KTST 使用 `ktlab-ktst-reference-dev-v1`，与 v1 不混排 AUC。新增 `ktlab-v2-shared-event-context-kc200-dev` 统一开发集的目标、允许历史及 epoch0/每轮最佳验证检查点选择；保留各模型原训练表示，明确披露 KC/事件 loss 等差异。旧权重不能重标为新协议成果。共享协议的最终测试入口当前仍关闭；完成候选、检查点与分析方案冻结后，可按持续授权实施解封并留痕，无需再次批准。

## 许可

本项目代码使用 MIT 许可证。固定上游源码保留 [pyKT MIT 许可证](third_party/pykt/LICENSE)与 [KTST MIT 许可证](third_party/ktst/LICENSE)。数据集有独立使用条件，仓库代码许可证不授予数据再分发权。

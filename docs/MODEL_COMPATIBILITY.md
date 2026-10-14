# 模型兼容性与验证边界

核查日期：2026-10-14。上游固定为 pyKT `eec087e0230520e7ae7bbd4f5433d71612df62fb`，`third_party/pykt` 源码保持原样。

| 模型 | CPU | 本机 RX 9070 XT / ROCm | 当前限制 |
| --- | --- | --- | --- |
| AKT | 已验证前向 | 已验证前向及 CPU/GPU 数值一致性 | ASSIST2015 使用 `n_pid=0`，不伪造题目 ID |
| simpleKT | 已验证前向 | 已验证前向及 CPU/GPU 数值一致性 | 同上；序列长度不能超过配置的位置编码长度 |
| MoC-KT (`mockt`) | 上游原版不支持 | 已验证前向及固定长度下的答案扰动检查 | 构造器硬编码 `.cuda()`；使用序列总长度选择卷积分支 |
| FA-KT (`fa_kt`) | 原版完整 MoE 未验证 | 原版完整 MoE 未验证 | 默认专家包含 Mamba，本环境缺少 `mamba_ssm`；还需要真实时间戳 |
| KTST mean/MHSA | 已验证前向与原loss | 三数据集前向、CPU/GPU一致性、因果/排列/padding/重载通过 | 独立reference-dev协议；完整测试适配尚未实现，未训练 |

表中“已验证”描述随机初始化模型的前向检查。另有用户批准的 B001：AKT/simpleKT 在 ASSIST2015 各完成5次更新，见[批次报告](../reports/B001_RESULTS_AND_B002_PLAN_2026-10-14.md)。MoC-KT 与 KTST 尚无优化更新，所有模型均未计算测试集预测；这些检查不能支持精度或 SOTA 声明。

## 原样加载与设备适配

`src/ktlab/adapters.py` 使用 `importlib` 在按源码路径和设备区分的独立模块命名空间中加载模型，不执行 `pykt.models.__init__`。因此 AKT 的运行不依赖其他模型的可选库。模型构造、注意力、损失正则项等数学实现没有被替换。

唯一的模块变量重绑定是上游 `device`：将其显式设为用户选择的 PyTorch 设备，并同步该命名空间中的辅助模块。这属于设备放置适配，记录在运行元数据中。没有修改全局 `torch.Tensor.cuda` 或其他 PyTorch 方法。

MoC-KT 构造器中的显式 `.cuda()` 不受这一变量控制，所以 CPU 请求会得到明确的不可用错误。多 GPU 时必须选择其默认设备，不能假定未注册为 buffer 的 ALiBi 张量会随 `.to()` 移动。

数据以 CSV 行偏移索引保留在 CPU，每个 batch 才转移到 GPU；不加载上游 GPU pickle 缓存。有效样本、shifted targets 和 padding 的含义保留；`selectmasks=0` 明确排除，`-1` 为 padding。此次准备的上游文件使用 `1/-1`。

## MoC-KT 的长度信息假设

上游 `MOCKTDataset` 将一条完整输入序列中非 padding 的响应数量记为 `ground_seq`。MoC-KT 的 `FrequencyLayer` 根据这个总长度与 `sequence1/sequence2` 的比较选择卷积分支。

因此，即使当前和未来答案扰动不影响之前的预测，改变未来序列长度、使总长度跨过分支阈值，仍可能改变之前的预测。当前通过的测试只支持“固定序列长度时未观察到答案泄漏”，不支持“所有情况下都满足在线前缀不变性”的结论。

本项目保留这一原版行为并单独披露。若研究改成只使用当前前缀长度，必须作为独立消融/修正模型重新命名和评测，不能将结果标成原版 MoC-KT 复现。

## FA-KT 的依赖与输入边界

FA-KT 默认的 MoE 包含 CNN、LSTM、attention、Mamba。缺少 Mamba 时，适配器报错，不自动删掉专家或用其他层替换。上游的 `nomoe`、`onlyattn` 等选项属于显式消融，必须在配置和结果名称中保留；消融运行成功不等于完整 FA-KT 已复现。

时间特征沿用 `DktForgetDataset.calC` 的 log2 分桶公式，输入必须含真实时间戳；不为 ASSIST2015 等缺失时间戳的数据生成虚构时间。固定 gap 词表大小默认为 64，越界会报错，不暗中裁剪。完整 FA-KT 还需要在目标 ROCm 环境验证 Mamba 及其底层 kernel。

## 协议与测试范围

当前引擎协议名为 `ktlab-v1-causal-valid-full-precision-repeat-kc-train`。它保留上游重复 KC 训练格式，但使用按题构造的因果验证序列、未舍入的 late-mean 概率和真实最高验证 AUC checkpoint。它与上游默认的重复 KC 验证和 checkpoint 阈值存在区别，结果不能无标记混入官方协议榜单。

官方 question-level 测试的同题 KC 隔离来自 `expand_question`：每个 KC 共享当前题之前的历史，而不是通过 late fusion 事后消除答案泄漏。历史题仍按多个 KC 重复表示；当前训练流程也没有改成每题只更新一次。

`tests/test_engine.py` 的检查已在专用 WSL 环境通过，涵盖：

- AKT/simpleKT 当前及未来答案扰动、尾部 padding 不变性。
- AKT/simpleKT checkpoint 重载精确一致，CPU/GPU 前向容差 `rtol=2e-4, atol=2e-5`。
- 原版 MoC-KT 在本机 GPU 的有限输出与固定长度答案扰动检查。
- 跨 batch 的同题多 KC 聚合、全精度概率导出和独立指标重算。
- 数据文件哈希变更拒绝、因果验证文件要求、训练批准门槛和独立测试集解封门槛。
- 随机权重 checkpoint 的 `evaluate_run(valid)` 完整接口，无优化器更新。

另对真实 ASSIST2015 第 0 折验证集读取一个 32 行 batch：855 个有效目标，AKT/simpleKT 在 GPU 的输出均为 `[32, 199]` 且有限。该检查没有报告随机权重的 AUC，也没有访问测试集进行预测。

`max_validation_batches` 只能与 `debug_smoke=true` 同时使用，指标带 `partial_debug=true`，不能作为实验结论或参与最终测试。正式测试须单独解封，并验证候选配置、数据清单、checkpoint、评测源码和分析计划哈希。

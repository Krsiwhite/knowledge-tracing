# UKT / KeenKT 证据卡与公开代码因果性审计

核查日期：2026-10-14。论文数字均为作者报告，未在本项目复现。本文的合成审计仅运行随机未训练模型的 CPU 前向；没有学生数据、梯度、优化器、测试集或质量指标。

**执行状态更新（2026-10-15）：** B003 已于2026-10-14 23:56（香港时间）启动，本次更新时正在首组 AKT / AS09，尚无终局结果；最多50轮、连续3轮验证 AUC 无严格改善早停，项目内持续授权生效，无需再次逐批批准。上述合成审计范围保持原样，UKT/KeenKT 不在 B003 内。

## 论文结果与协议

UKT 为 AAAI 2025 正式论文（官方发表日期2025-04-11）；高斯分布表示、Wasserstein attention 和不确定性对比学习已经覆盖“概率状态加抗噪”的宽泛动机。Table 1 位于 PDF 第6页 / 印刷页27910。作者采用20%留出、其余80%做5折，最多200轮；搜索维度、学习率、dropout、层数、heads及对比损失权重，但未报告实际完成 trial 数。[官方记录](https://ojs.aaai.org/index.php/AAAI/article/view/35007)、[论文全文](https://ojs.aaai.org/index.php/AAAI/article/view/35007/37162)

| UKT Table 1，AUC（保留原表±项） | ASSIST2009 | ASSIST2015 | NIPS34 |
| --- | ---: | ---: | ---: |
| UKT | 0.8563 ± 0.0014 | 0.7267 ± 0.0007 | 0.8035 ± 0.0004 |
| 同表 AKT | 0.8474 ± 0.0017 | **0.7282 ± 0.0004** | 0.7960 ± 0.0003 |
| 同表 simpleKT | 0.8413 ± 0.0018 | 0.7237 ± 0.0005 | 0.7966 ± 0.0000 |

AS15 的 UKT 低于同表 AKT，不能写“全部数据集最好”。这三列不涵盖论文的全部六个数据集。

KeenKT 为 AAAI 2026 正式论文（官方发表日期2026-03-14），采用 Normal-Inverse Gaussian 状态表示、去噪重构及分布对比学习。Table 2 位于 PDF 第5页 / 印刷页19289。正文报告学生5折、d128、hidden256、batch128、学习率候选0.01/0.001/0.0001中选0.001、patience10；实际搜索 trial 数仍未知。[官方记录](https://ojs.aaai.org/index.php/AAAI/article/view/39004)、[论文全文](https://ojs.aaai.org/index.php/AAAI/article/view/39004/42966)

| KeenKT Table 2，AUC（保留原表±项） | ASSIST2009 | ASSIST2015 | NIPS34 |
| --- | ---: | ---: | ---: |
| KeenKT | 0.8606 ± 0.0011 | 0.8214 ± 0.0006 | 0.8162 ± 0.0011 |
| 同表最高列出对照 | UKT：0.8563 ± 0.0014 | RobustKT：0.7805 ± 0.0007 | UKT：0.8035 ± 0.0004 |

两篇主表并非本项目 v2 事件协议成绩。AS15 较大的表面差值不能替代代码与数据审计，也不能推算某个代码问题对已发表数字的贡献。

## 固定代码与可比较性

核验的作者仓库提交为 UKT `86ad9d8f76ee2c761e116fcadc43705f66ffd0b6`、KeenKT `911e4e460b0e2ddb25142fb3c7e46123c1043690`。[UKT 快照](https://github.com/UncertaintyForKnowledgeTracing/UKT/tree/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6)、[KeenKT 快照](https://github.com/HubuKG/KeenKT/tree/911e4e460b0e2ddb25142fb3c7e46123c1043690)

两个公开 `split_datasets.py` 的 Git blob 均为 `95e5e6b0e35dc329a1dcc479f3e1f7d3dbb689cb`：代码按学生序列使用 seed1024 固定20%测试，其余80%做5折；默认长度200，多KC先展开。由此可以确认公开处理脚本具备固定 test 的流程，不能继续仅因 KeenKT 正文写“5折”而断言它没有固定 test。但同脚本不等于原始输入版本、输出数据哈希及论文实际运行索引完全相同。[UKT 预处理](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/preprocess/split_datasets.py)、[KeenKT 预处理](https://github.com/HubuKG/KeenKT/blob/911e4e460b0e2ddb25142fb3c7e46123c1043690/KeenKT_Code/preprocess/split_datasets.py)

仍需闭环的项目包括：实际数据映射与分块/滑窗路径、事件或KC评分单位、multi-KC 聚合、checkpoint 规则、每集最优参数及搜索预算。命令行默认参数不是论文逐数据集最佳配置。后续已完成[UKT 作者/pyKT 的限定合成审计](UKT_CAUSALITY_2026-10-14.md)：198项不变量检查在预设容差内，18项历史阳性控制敏感；尚未接入共享验证 adapter，也未验证训练分支。

## KeenKT：两个可定位的不可见答案依赖路径

固定源码文件为 `KeenKT_Code/models/KeenKT.py`（Git blob `181e57d6a39d0622d4a56af2be00f2a5c97e9353`）和 `utils.py`（`925d9527a76bd0de70e75af7170981ee1ec6fabe`）。审计启动前核对原始文件字节；上游缓存源码保持原样，诊断干预只在独立进程内存中进行。[模型源码](https://github.com/HubuKG/KeenKT/blob/911e4e460b0e2ddb25142fb3c7e46123c1043690/KeenKT_Code/models/KeenKT.py)

1. `SEBlock.forward` 在第29行对整个时间维求均值，产生所有位置共享的 gate；第367–368行在 eval 前向也调用它。未来位置的状态包含此前已观察回答，把它们汇总回当前位置会形成不可见答案路径。
2. `TransformerLayer` 的 `mask=0` 是严格下三角。第478行起的 `uattention` 用有限值 `-1e32` 填首行所有 logits 后做 softmax，因此空首行成为均匀权重；传入的 `zero_pad` 未被处理。首行汇总整段回答 value；第二层又把上一层状态作为 key，让信息影响较早预测。仅移除 SE gate 不足以阻断这条路径。

位置检查：`forward` 用原始首位置与 shifted 序列拼回完整序列，输出不再移位。下列 `preds[:,2]` 就是原始第3个事件的预测；翻转回答索引2是当前答案，索引5是未来答案。丢掉输出首位置并不消除这两条依赖。

### 合成反例与诊断对照

构造8个事件，概念号0至7，回答为0/1交替；CPU，随机初始化，seed42/2026/3407，d32、两层、4 heads、无dropout、`n_pid=0`、`stoc_qid`。固定同一份模型权重，分别翻转历史索引1、当前索引2、未来索引5；始终看索引2预测。`train=False`，扩散与对比学习训练分支未执行。重复同输入前向差值为0。

| 前向变体 | 六个当前/未来扰动中的最大概率绝对变化 |
| --- | ---: |
| 原样上游实现 | 0.0005574226 |
| 只以 Identity 替换 SE gate | 0.0007932186 |
| 只将 attention 空首行输出清零 | 0.0003084540 |
| 两项诊断干预同时使用 | 0 |

共36项：3 seeds × 4变体 × 3扰动。组合对照在三个 seed 上仍对允许的历史答案扰动有非零响应，排除了“模型完全忽略回答”这种解释。对照数值之间不是可加性贡献分解，也不代表修正后模型更准。

证据：[完整机器报告](../../reports/KEENKT_CAUSALITY_AUDIT_2026-10-14.json)、[可重跑脚本](../../scripts/audit_keenkt_causality.py)。脚本禁止 optimizer 构造和 backward/autograd 调用。复查只需从固定 raw 源码地址下载两文件，在独立包目录放置空 `__init__.py`，再指定目录与尚不存在的输出文件：

```bash
python scripts/audit_keenkt_causality.py --source-dir .cache/research/keenkt_911e4e4 --output reports/keenkt-audit-new.json
```

**证据边界：** 这构成该快照、上述配置、整段序列前向不满足因果性的具体反例；不测量作者训练权重，不推断论文成绩被抬高多少，不断言所有配置受同等影响，也不代表已经完成 KeenKT 复现。诊断修正不是原模型的新训练结果。逐目标只提供合法前缀并隔离真实当前答案能够阻断这些输入来源，但会改变执行路径，仍须完整协议验证。

## 对本项目的决定

- KeenKT 暂不作为可直接训练和排名的强基线；先保留作者报告数字，完成模型级因果审计及路径说明后，另行决定原版参考轨或明确命名的因果修正版。不能静默修改后继续称“官方复现”。
- UKT 是假设A必须考虑的近邻，尚未完成本项目共享 adapter 接入和训练；后续完成接入审计并登记批次后可按持续授权执行。“不确定性、去噪、分布对比”本身不足以构成原创点。
- 原创模型和所有接入基线都要检查当前标签隔离、未来答案不变性、历史扰动阳性控制、全padding/空历史边界；通过这些只能证明指定条件下的正确性，不能替代性能证据。
- B003 没有 UKT/KeenKT。原合成审计没有改变其配置、源码指纹或当时的授权记录；后续执行依据用户最新持续授权，不把审计结果当作训练成绩。

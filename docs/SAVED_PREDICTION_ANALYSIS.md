# 已保存共享验证预测的离线分析

[分析脚本](../scripts/analyze_shared_predictions.py) 只分析已经保存的完整 v2 验证预测。它不导入 `ktlab`、PyTorch 或模型，不读取原始数据、权重文件、私人审批记录，不重新前向或训练，也不自动发布结果。

2026-10-14 本轮仅用人工 CSV 和人工收据验收工具；**尚未分析真实 B003，B003 仍须取得具体批次批准。** 下列命令是未来正式工件产生后的用法，不表示这些运行已完成。

## 输入与用法

每次限同一数据集、同一验证 fold、同一训练 seed；至少两个不同模型，每个模型只接受一个运行。必须指定参考运行。输入是各运行根目录的 `valid_predictions.csv`，工具只额外读取同目录的：

- `valid_predictions.csv.manifest.json`：预测字节哈希、开发集目标及历史摘要、独立评测结果。
- `metadata.json`：完成状态、记录的训练更新数、批次/源码/检查点/验证身份及最佳 epoch。
- `resolved_config.json`：实际保存配置。
- `valid_metrics.json`：训练引擎最终导出记录。

还须显式提供公开 batch manifest；不读取其中 `dataset_manifest` 或源数据路径，也不读取审批文件。各输入的**实际字节 SHA256 和大小**进入聚合报告；报告不包含路径、学生 ID、事件 ID、学生级指标或逐样本记录。

```bash
python scripts/analyze_shared_predictions.py \
  --batch configs/batches/B003-shared-convergence-baselines.json \
  --predictions \
    runs/B003-akt-assist2009-f0-s42/valid_predictions.csv \
    runs/B003-ktst_mean-assist2009-f0-s42/valid_predictions.csv \
  --reference B003-akt-assist2009-f0-s42 \
  --bootstrap-replicates 500 --seed 42 \
  --output reports/B003-assist2009-saved-valid-analysis.json
```

`--seed` 固定的是 bootstrap 抽样随机数；训练 seed 来自已保存配置，另列在报告中。对另外两集分别调用；本工具不混合不同数据集、fold 或训练 seed，不平均不同模型的预测，也不执行最终多 seed/fold 确认分析。

## 在计算指标前核验什么

只接受 `ktlab-v2-shared-event-context-kc200-dev`、`split=valid`、KC 预算200。必须有 `selected_targets = expected_total = eligible_total`，且 `max_targets=null`、`selected_for_debug=false`、`selection_limited=false`。拒绝 train、test、旧 v1/KTST 参考轨、抽样、重复 ID、缺目标、非法概率/标签、调试配置及缺失的训练导出身份。

训练收据必须为 `completed`、实际记录更新数大于零且不超过批准清单记录的上限。batch canonical SHA 必须与配置和 metadata 中保存的 approved batch SHA 一致；配置哈希、模型及实现摘要、数据清单摘要、验证来源/目标/历史摘要也须相互一致。清单中的模型参数必须包含于实际保存配置，允许引擎填入额外默认值。

**epoch 0 可以是合法的最终最佳候选。** 若训练确实记录了更新，但后续候选都没有改善初始验证 AUC，工具保留该运行，输出 `selected_checkpoint_stage=initial_untrained`。其它阶段区分完整 epoch 与受预算截断的部分 epoch；不会只留下“训练后改善”的模型。

每条 CSV 用 `(uid,event_id)` 精确配对，要求题号、标签、KC 数、目标 SHA 和历史 SHA 全部一致。标签单独生成汇总 SHA；不误认为不含当前标签的 target/context SHA 已经校验了标签。至少一份输入必须保留原始导出顺序，以重新核算冻结的 target/context 序列摘要；其它输入可有不同顺序，其预测按 ID 重新对齐。任何输入的 CSV 实际字节哈希仍必须匹配自己的 sidecar。所有输入都打乱而失去冻结顺序时拒绝，不猜测原始学生行顺序。

工具从 CSV 重算指标，并与 sidecar、最终 `valid_metrics.json` 及 metadata 的最佳验证 AUC 对照。结束时再次核对全部已读输入的字节哈希，防止分析过程中工件变化；校验失败不生成最终报告，已存在的输出不会被覆盖。

### 收据核验的明确限度

当前 `evaluate_shared` sidecar 在训练引擎补充选择策略之前写出，保留 `model_training_and_checkpoint_selection_not_verified=true`；随后 `valid_metrics.json` 包含同一结果及 `checkpoint_selection_policy` / `partial_debug`。工具按这个实际导出流程联核两者，不要求 sidecar 存在尚未导出的字段。

当前没有一份独立收据直接绑定“checkpoint SHA → prediction SHA”。工具只能确认运行目录中保存的检查点身份、批次身份、配置和评测记录相互一致，**不读取或验证检查点字节，不完整证明真实训练身份，也不独立认证私人批准**。报告因此固定写明 `training_identity_fully_proven=false`、`weights_read_or_verified=false`、`direct_checkpoint_to_prediction_hash_binding_available=false`。这些本地收据没有数字签名，不能抵抗所有文件一起被伪造；原始目标/历史语义也不能仅靠不透明摘要重新证明。

## 指标、分层与区间

总体和 query 单 KC / 多 KC 两层均输出事件加权 AUC、NLL、Brier、ACC 和 **15 个固定等宽区间的 ECE**，与现有 `binary_metrics` 口径相同。ACC 阈值为0.5；ECE 区间左闭右开，最后一个区间包含1。NLL 将概率截到 float64 machine epsilon 与 `1-epsilon`，避免概率端点产生无穷。空层指标为 `null`；单类层 AUC 为 `null`，不硬填0.5。层内给出事件和学生总数，不导出学生记录。

相对指定 reference，输出总体 `candidate - reference` 的 AUC/NLL/Brier 差值和95% percentile 区间：AUC 正值有利，NLL/Brier 负值有利。每次从 K 个学生中有放回抽 K 次，将被抽学生的**全部事件**按其抽中次数加权；所有模型共享这份权重。AUC 使用预先排序的分数及相同分数分组，每个模型只需排序一次；NLL/Brier 先积累学生内损失总和，再按事件总权重归一。它不是每个学生指标的等权平均。

默认500次，固定抽样 seed42；可显式选择200次。单类 draw 的 AUC 不可计算：计入 `single_class_auc_draws` / `undefined_draws`，不补抽；NLL/Brier 仍使用该 draw。因此 AUC 区间以抽到双类为条件，需同时看 `usable_draws`；少于两个有效 draw 时 CI 为 `null`。低于200次会给出抽样次数偏少标记。两种 KC 分层目前仅作描述，不另做分层区间或事后挑选显著层。

学生内多事件相关，不能把全部作答当成独立个体抽样。方法依据可参见 [相关诊断数据 AUC 的非参数 bootstrap，第2.3节](https://pmc.ncbi.nlm.nih.gov/articles/PMC10728486/)，其 cluster 方案保留被抽对象的全部观测；原论文应用于动物诊断，与知识追踪不同，不能据此承诺本数据的精确覆盖率。

**这些是探索性区间。** 相同验证集已经用于选择 checkpoint，本工具没有修正由选模造成的偏差；没有多重比较校正，不产生 p 值或正式显著性判定。不能把学生抽样不确定性改写成多 seed 稳定性、重训不确定性、最终测试证据或超越 SOTA。有限验证样本上的模型选择偏差背景见 [Cawley 与 Talbot，JMLR 2010](https://www.jmlr.org/beta/papers/v11/cawley10a.html)。

## 成本和未提供的分析

只用 CPU、NumPy 和标准库。最多2000次 draw，且 `事件数 × 模型数 × draw数 ≤ 500,000,000`；超限明确拒绝，需主动减少次数，不自动采样或伪造区间。当前 B003 清单的 NIPS34 四模型、221348个目标、500次为442696000，处于这个计算预算以内；这只是规模计算，未读取或分析它的预测。内存保存指定 CSV 的配对字段和分数，不展开 bootstrap 复制后的巨型表。

CSV 没有每事件的历史长度或训练题目频次，本工具明确不提供对应分层；不读取原始数据补齐，也绝不把验证集中出现次数当成训练频次。若以后需要，应先另行定义并审计这些字段的训练来源。

人工验收命令：

```bash
python -m pytest -q tests/test_saved_prediction_analysis.py
```

[测试](../tests/test_saved_prediction_analysis.py) 只生成小型人工 CSV/JSON，检查工件拒绝规则、按 ID 配对、epoch0 保留、相同模型零差值、单类 draw、概率端点、已知指标、与显式复制学生事件的 bootstrap 对照，以及不读取权重/源数据、不输出个人记录的边界。不会运行实际模型。

2026-10-14 实际验收结果：**51 项通过，pytest 计时 1.67 秒**。另经独立静态复核确认两个现有训练引擎的导出 schema 与本工具匹配。真实 B003 工件尚未产生，因此这不是实际训练产物的端到端验收。

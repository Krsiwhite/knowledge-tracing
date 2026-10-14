# 已保存共享验证预测的离线分析

[分析脚本](../scripts/analyze_shared_predictions.py) 只分析已经保存的完整 v2 验证预测。它不导入 `ktlab`、PyTorch 或模型，不读取原始数据、权重文件、私人审批记录，不重新前向或训练，也不自动发布结果。

2026-10-14 本轮仅用人工 CSV 和人工收据验收工具；**尚未分析真实 B003。用户最新指令已授权启动 B003：每次最多50轮，连续3轮验证 AUC 无严格改善早停；本机项目内后续执行无需逐批再次批准。** 清单、哈希与持续授权依据仍须记录。下列命令是未来正式工件产生后的用法，不表示这些运行已经启动或完成。

## 输入与用法

每次限同一数据集、同一验证 fold、同一训练 seed；至少两个不同模型或不同合法 UKT 训练模式。普通模型每个模型只接受一个运行；UKT 每种合法 recipe 只接受一个运行。必须指定参考运行。输入是各运行根目录的 `valid_predictions.csv`，工具只额外读取同目录的：

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

### UKT 训练模式的比较身份

UKT 的 `config.model` 仍是 `ukt`。工具只接受已经进入实际配置哈希的明确 `ukt_training_recipe`，并核对 `model_config.use_CL`、`model_config.use_uncertainty_aug`，不从运行目录名或自填显示名推断变体：

| `ukt_training_recipe` | `use_CL` | `use_uncertainty_aug` |
|---|---|---|
| `ukt-no-cl` | false | true 或 false；关闭 CL 时不启用增强 |
| `ukt-cl-same-responses` | true | false |
| `ukt-author-first-two-from-last-86ad9d8-v1` | true | true |

两个开关都必须显式保存为 JSON boolean，或兼容官方脚本的整数0/1；拒绝缺失值、字符串、浮点数及其它整数。recipe 缺失、未知或与开关冲突时拒绝，即使其它收据哈希相互一致。非 UKT 模型不能携带非空的 UKT recipe。配置来自引擎解析后默认值也可以，但仍须通过既有 batch 子集匹配和实际配置哈希检查。

输出中每个 run 保留 `model`，新增 `ukt_training_recipe`（普通模型为 `null`）及 `comparison_label`（普通模型等于模型名，UKT 为 `ukt / <完整recipe>`）。`comparison_identity(config)` 提供相同三字段给其它离线汇总工具；调用方须先完成收据校验，该 helper 本身不认证配置来源。`_read_run` 与 `Receipts` 的调用和返回接口保持不变。

重复检查使用 `(model, recipe)`，不会使用可自填的 `comparison_label`。因此作者增强 CL 与无 CL 可以配对；同一 recipe 即使更换 run_id、显示名、模型宽度或其它超参仍视为重复。统计仍以原 run_id 为键，reference 也仍是明确的 run_id。不同 recipe 获准配对只表示身份可区分，不证明架构、批次、优化器或预算已经匹配；关于 CL 效果的消融结论仍须另核这些设置。原有 split、协议、seed、标签、目标/历史/来源及工件哈希校验全部保留。

## 在计算指标前核验什么

完整批次的轮次、选点和资源汇总可用 `scripts/summarize_shared_batch.py --repo-root <训练工件所在仓库> --batch <公开批次清单> --output <新报告路径>`。它复用本工具的收据与保存预测校验，再读取 `initial_valid_metrics.json` 和 `history.jsonl` 重放严格最佳选点、早停及更新上限，核对终局元数据；要求清单内所有运行完成，否则不写完成报告。epoch0不累计stale，同分不替换最佳，完整最后一轮恰好达到更新上限可合法标为 `max_train_steps`。

该汇总按数据集/协议/fold/seed保留分组，不做跨集指标平均；输出实际轮次、选中轮次、停止原因、更新、耗时和峰值显存，软时间上限超出仅如实标记。它不读取授权记录、源码树、权重、原始数据或test，也不把保存收据的一致性当作独立训练证明。首次人工文件验收为 **26项通过，0.52秒**；加入下面的预定导出容差后，**21项新增或受影响检查通过，17项未重复执行，0.49秒**。尚未对未完成的B003产生正式汇总。

只接受 `ktlab-v2-shared-event-context-kc200-dev`、`split=valid`、KC 预算200。必须有 `selected_targets = expected_total = eligible_total`，且 `max_targets=null`、`selected_for_debug=false`、`selection_limited=false`。拒绝 train、test、旧 v1/KTST 参考轨、抽样、重复 ID、缺目标、非法概率/标签、调试配置及缺失的训练导出身份。

训练收据必须为 `completed`、实际记录更新数大于零且不超过绑定授权依据的清单上限。batch canonical SHA 必须与配置和 metadata 中保存的 approved batch SHA 一致；该字段继续用于留痕，不要求用户重新逐批确认。配置哈希、模型及实现摘要、数据清单摘要、验证来源/目标/历史摘要也须相互一致。清单中的模型参数必须包含于实际保存配置，允许引擎填入额外默认值。

**epoch 0 可以是合法的最终最佳候选。** 若训练确实记录了更新，但后续候选都没有改善初始验证 AUC，工具保留该运行，输出 `selected_checkpoint_stage=initial_untrained`。其它阶段区分完整 epoch 与受预算截断的部分 epoch；不会只留下“训练后改善”的模型。

每条 CSV 用 `(uid,event_id)` 精确配对，要求题号、标签、KC 数、目标 SHA 和历史 SHA 全部一致。标签单独生成汇总 SHA；不误认为不含当前标签的 target/context SHA 已经校验了标签。至少一份输入必须保留原始导出顺序，以重新核算冻结的 target/context 序列摘要；其它输入可有不同顺序，其预测按 ID 重新对齐。任何输入的 CSV 实际字节哈希仍必须匹配自己的 sidecar。所有输入都打乱而失去冻结顺序时拒绝，不猜测原始学生行顺序。

工具从 CSV 重算指标，并与 sidecar、最终 `valid_metrics.json` 及 metadata 的最佳验证 AUC 对照。结束时再次核对全部已读输入的字节哈希，防止分析过程中工件变化；校验失败不生成最终报告，已存在的输出不会被覆盖。

历史选点和最终导出是两次独立前向，GPU 浮点计算不保证逐位一致。**在查看 B003 终局指标之前**，固定 `SELECTED_EXPORT_ABS_TOL=1e-6`：仅对“metadata 记录的选点 AUC 与最终 CSV 重算 AUC”要求有限且绝对差不超过此值，不使用相对容差或四舍五入。每个 run 明确输出 `selected_valid_auc_recorded`、`export_minus_selected_auc`，报告另列 `selected_export_auc_comparison` 说明阈值和适用范围；不把小差值改写成零。超过阈值即拒绝，不能据结果再调阈值。此规定是预定的数值边界，不表示已发现真实 B003 异常。

公开 helper `selected_export_delta(selected, exported, metric="auc")` 核验有限数值及上述绝对容差，返回有符号的 `exported-selected`，供其它离线汇总工具比较分别前向的历史与导出指标。**同一 CSV 的指标重算仍使用原来的 `rel_tol=1e-10, abs_tol=1e-12`；收据中直接复制的数值仍要求一致**，身份、文件哈希和选择策略校验均不放宽。

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

2026-10-15 增补 UKT 人工工件：合法 recipe 配对及明确输出、同 recipe/普通模型重复拒绝、未知 recipe 与开关矛盾、缺失/非法开关、配置哈希篡改、epoch0 保留，以及 `_read_run` 原返回接口兼容性。

该次纯文件验收结果：**76 项通过，pytest 计时 1.70 秒**；没有导入实际模型、读取真实数据或训练。

随后加入预定的选点／导出差异检查，补测小漂移通过且报告原差值、超界拒绝、非有限值拒绝，以及同一 CSV／复制收据仍严格核验：**89 项通过，pytest 计时 1.85 秒**。仍只使用人工文件。

2026-10-14 实际验收结果：**51 项通过，pytest 计时 1.67 秒**。另经独立静态复核确认两个现有训练引擎的导出 schema 与本工具匹配。真实 B003 工件尚未产生，因此这不是实际训练产物的端到端验收。

# UKT 接入与验收

日期：2026-10-15。实现位于独立 `research/ukt-integration` 分支；B003 正在使用的主仓源码保持不变。接入固定 pyKT 模型 `eec087e0230520e7ae7bbd4f5433d71612df62fb`，并适配作者 UKT `86ad9d8f76ee2c761e116fcadc43705f66ffd0b6` 的实际训练增强。详见主分支的[静态训练契约](https://github.com/Krsiwhite/knowledge-tracing/blob/main/docs/evidence/UKT_TRAINING_CONTRACT_2026-10-15.md)。

## 实际行为

`model="ukt"` 默认使用 `emb_type="stoc_qid"`、`atten_type="w2"`、CL 开启、作者增强开启。解析后的模型参数和 `ukt_training_recipe` 写入配置及检查点身份；显式 recipe 与开关矛盾时拒绝训练。

| 模型开关 | 保存的 recipe | 训练目标 |
| --- | --- | --- |
| `use_CL=False` | `ukt-no-cl` | 原回答的 masked shifted BCE |
| `use_CL=True, use_uncertainty_aug=False` | `ukt-cl-same-responses` | BCE 加原 NCE，第二视图仍走模型但响应相同 |
| `use_CL=True, use_uncertainty_aug=True` | `ukt-author-first-two-from-last-86ad9d8-v1` | BCE 加原 NCE，第二视图使用固定作者增强 |

作者增强在非空、左侧连续有效前缀域内，令原/shifted 响应的首字段都为 `1 - 最后有效回答`，其他字段不变；这是其循环的实际结果。没有改成随机翻转或全位置翻转。输入响应不原地修改，空/内部空洞 mask、非二元响应及非零 padding 明确拒绝。原 NCE 的符号、固定长度池化和权重保持不变。

普通评测与共享事件验证均关闭 CL/增强。共享验证采用既定 T=200、目标答案占位、每 KC 独立查询与全精度均值，不把目标兄弟 KC 的答案加入历史。推理拆批不能当成 CL 训练拆批。

保留项目训练表示与验证选点规则，并非复现作者整套实验。Adam weight decay、batch、shuffle、grad clip 仍须在后续实验清单中显式确定；没有因接入 UKT 就将它们静默改为作者配置。作者 loader 的 `selectmasks != -1` 与项目 `selectmasks == 1` 在含 0 排除标记时不同，测试明确保留此差异。AS15 无独立题 ID，应使用 `n_pid=0` 与接口零占位，不引入伪造题目参数。

## CPU-UKT-001 验收

在用户持续授权下，先登记人工数据执行清单及实现/测试 SHA，再执行一次。结果为 **27 passed in 15.80s，exit 0，零跳过**；实际 5 次 backward、3 次 Adam.step，受硬上限 8/4 约束。仅 CPU 单线程、d16/2层、人工张量和 CSV；原作者源码按 SHA 核验。没有真实学生数据、现有训练权重、GPU 或最终测试集。

覆盖：短二元序列穷举与 padding、长度 2/199/200、作者 loader 逐字段等价、非法输入拒绝、3 种 recipe 的原作者模型概率与原 `cal_loss` 等价、有限梯度/真实更新、非零随机题目难度下的共享目标标签隔离、多 KC 均值与推理拆批，以及真实两轮人工引擎的最佳权重、recipe 指纹及最终验证导出。原作者与接入模型的概率比较使用零容差；损失容差为 rtol=1e-7、atol=1e-9。

实现 SHA256：`97eaebf70bf700b0dcb22624c963bc6933d5af01e7133b7e512d5e277dc9e892`。测试文件 SHA256：`29e63e6b87ebcba7a1955429659f44f6dfb7228e7c8049ec52c3672594f8e475`。执行清单 canonical SHA256：`110a99115c568a1d57811ad3575ab0ab092ecab25f79170d9d3fb3580102b8c7`。stdout SHA256：`cbc945c9c3776ec43acc5cd97e4e2962da6edfc821740a41f01814705e35bd3b`。

私有授权记录与原始日志保留在 ignored `approvals/`、`.cache/`。运行测试需明确设置 `KTLAB_RUN_UKT_CPU_UPDATES=1`；作者 oracle 默认缓存缺失时跳过，显式设置 `KTLAB_UKT_AUTHOR_LOADER` / `KTLAB_UKT_AUTHOR_MODELS` 后缺失或哈希不符则失败。本次两个路径均显式设置且没有跳过。公共作者文件仅为测试 oracle，不是生产代码依赖。

相关旧路径的 CPU 回归为 **45 passed、2 skipped，79.51s**（`test_engine.py`、`test_shared_engine.py`、`test_shared_adapters.py`）；运行器禁止梯度、优化器构造与 GPU 初始化。两项跳过均为必须使用 GPU 的原 MoC-KT 检查，未将跳过记作通过。

随后对 fold0 对应的全部训练序列做了不导入 torch 的[输入域检查](../reports/UKT_TRAINING_INPUT_AUDIT_2026-10-15.json)：AS09/AS15/NIPS34 分别3016/12344/6043行均满足连续前缀、二元响应、padding与ID范围；有效 shifted 监督位置没有0排除标记，因此这批训练数据上的作者/项目监督掩码一致。AS15全部行使用无独立题号的接口占位。该检查只统计训练fold，不执行模型；脚本为 `scripts/audit_ukt_training_inputs.py`。

这些结果证明指定人工条件下的接入行为与当前训练文件的输入域。尚未进行 UKT 的真实数据训练或 GPU 训练兼容性验收，没有模型质量或 SOTA 结论。下一步为[已准备的 B004](../reports/B004_UKT_PLAN_2026-10-15.md)，继续遵守最多50轮、连续3轮无严格验证 AUC 提升早停。

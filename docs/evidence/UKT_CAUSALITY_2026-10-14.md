# UKT 作者实现与 pyKT 实现：有限合成因果性审计

日期：2026-10-14。**结论仅限本卡覆盖的随机未训练前向：没有观察到超过预设容差的当前/未来回答依赖；两份源码在本次显式配置下输出一致。** 这不是对所有 UKT 配置、训练过程或数据管线“无泄漏”的证明，不是接入共享验证成功的声明，也不提供任何 AUC、复现或 SOTA 结论。

交付物：[可复现脚本](../../scripts/audit_ukt_causality.py)、[机器可读审计记录](../../reports/UKT_CAUSALITY_AUDIT_2026-10-14.json)。本轮没有梯度、优化器构造、学生数据、已训练权重或测试集访问；没有修改 `src`、vendor、配置、lock 或训练批准。

## 1. 固定来源与隔离方式

| 来源 | 固定提交 | 模型文件 SHA256 |
|---|---|---|
| 作者 UKT | `86ad9d8f76ee2c761e116fcadc43705f66ffd0b6` | `8b092537174bbe459d4d2e93333d66b16c77bd94456b2b06dc549af6d1183651` |
| pyKT | `eec087e0230520e7ae7bbd4f5433d71612df62fb` | `edf78e409e19d7aaf70d8450bab8e10254928cd72f224bac1974b12a0e7c4d55` |

直接来源：[作者模型](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/models/ukt.py)、[作者 utils](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/models/utils.py)、[pyKT 模型](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/ukt.py)、[pyKT utils](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/utils.py)。作者文件仅保存在忽略的 `.cache/research/ukt_86ad9d8`；本地 pyKT 两文件另与固定远端原始文件核对，SHA256 相同。

两份模型及相对 `utils` 以各自私有模块命名空间载入，不运行上游包初始化器，不共享同名模块。CPU 设备选择在导入前固定；禁用字节码写入。原源码及数学计算没有替换。报告保存模型和 utils 的运行前后摘要，以及审计脚本摘要；源码前后相同。程序阻止 `backward`、autograd 求导和优化器构造，记录的调用尝试均为 0。

## 2. 输出与回答对齐

合成序列为 8 个概念 `[0,1,2,3,4,5,6,7]`，回答 `[0,1,0,1,0,1,0,1]`。输入字典按原接口拆成前 7 位与后 7 位；模型以 `cat(首回答, shifted回答)` 恢复完整回答，题目/概念也按同样方式恢复。脚本显式检查该重建，未额外平移输出。

`preds[:,t]` 对应题目位置 t 的回答概率。作者评估调用先取 `preds[:,1:]`，再与 `shft_rseqs` 对齐；所以审计原输出位置 2，翻转回答位置 2 是当前回答扰动，翻转位置 5 是未来回答扰动，位置 1 则是允许历史阳性控制。不能把原输出位置 2 误解释为预测位置 3。[作者 forward](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/models/ukt.py#L119)、[作者评估对齐](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/models/evaluate_model.py#L100)、[作者损失标签](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/models/train_model.py#L31)。这些上游训练文件仅静态读取，没有导入或执行训练函数。

## 3. 实际检查范围与结果

运行环境：Python 3.12.3、PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`，**实际计算设备 CPU**、float32、单线程、`eval()`、`inference_mode()`、`train=False`。硬编码绝对容差 `1e-6` 在执行前确定，没有按观察结果调大。

共同小配置：`d_model=32`、2 blocks、4 attention heads、`d_ff=64`、输出层 64/32、dropout 0、`stoc_qid`、`separate_qa=False`、位置编码最大长度 200。三个配置为 w2/无题 ID、w2/有题 ID（`n_pid=12`）、dp/无题 ID；每份源码各用 seed 42、2026、3407，共 **18 个模型实例**。CL 与不确定性增强开关为 true，但 `train=False`，训练分支不执行。

每个实例有 12 项记录，共 **216 项**；其中 198 项要求保持不变，18 项是历史阳性控制。

| 检查 | 对照范围 | 18 个实例的结果 |
|---|---|---|
| 重复相同输入 | 全 8 个输出 | 全部逐位相同，最大差 0 |
| 当前回答翻转 | r2 翻转，检查输出 0–2 | 全部逐位相同 |
| 单个未来回答翻转 | r5 翻转，检查输出 0–2 | 全部逐位相同 |
| 全部未来回答翻转 | r3–r7 翻转，检查输出 0–2 | 全部逐位相同 |
| 未来题号/概念改变 | 仅改位置 3–7，检查输出 0–2 | 全部逐位相同 |
| 空历史首位置 | r0 翻转，只检查输出 0 | 全部逐位相同 |
| 最后位置当前回答 | r7 翻转，只检查输出 7 | 全部逐位相同 |
| 整段与截至目标的前缀 | 8 位与前 3 位，比较输出 0–2 | 全部逐位相同 |
| 前缀当前 dummy 0/1 | 仅前 3 位输入，r2 由 0 改为 1 | 全部逐位相同 |
| 右 padding 到 200 | 同 8 个有效位置，比较其全部输出 | 最大差 `7.450580596923828e-7`，均在预设容差内；并非逐位相同 |
| 右 padding 内容改变 | 固定 200 长度及有效 mask，改变 192 个 padding 槽的题号/概念/回答 | 前 8 个输出全部逐位相同 |
| 历史阳性控制 | r1 翻转，检查输出 2 | 18/18 大于容差；绝对差范围 `0.0003315210` 至 `0.0173290670` |

右 padding 的数值差单独保留；本卡不将容差内一致写成数学上的严格相等。全部报告中没有应保持不变检查超过容差的反例。历史阳性控制说明这些检查没有仅因模型完全忽略回答而通过，但不证明其学习能力或效果。

## 4. 作者与 pyKT 的实际差异

对 9 组“配置 × seed”，两版本同种子初始化后的 `state_dict` 逐位相同，所有审计记录中所比较的概率也逐位相同。没有加载训练权重、跨模型平均预测或构造 ensemble。

- 去除注释和通常的开头 docstring 后，`base_emb`、`forward`、Architecture、TransformerLayer、MultiHeadAttention、实际生效的 attention/uattention、Wasserstein 距离/对比损失及位置编码的 AST 相同；utils 的完整 AST 也相同。这是指定定义的结构证据，不能代替整个仓库等价性证明。
- 构造器**签名并不相同**：pyKT 改变了部分参数的排列，`nheads` 默认 4→8、`loss1/2/3` 默认 0.5→0。这些名称未参与本次预测计算，实际 attention heads 用显式 `num_attn_heads=4`。因此本卡只声称本次显式关键字配置一致，不能保证旧位置参数调用等价。[作者签名](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/models/ukt.py#L25)、[pyKT 签名](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/ukt.py#L32)。
- 报告中的 `UKT.__init__.body` AST 比较为 false：`super().__init__()` 后的独立说明字符串有修改，脚本只去除标准开头 docstring，没有删除这种中途字符串表达式。静态 diff 中这是构造器 body 的差异；不能把这个 false 自动解释成预测数学改变。
- 作者文件里较早的同名旧 attention 会被后定义覆盖；pyKT 将该旧版本注释掉。另删除了不在预测调用路径中的绘图/打印函数，并改写说明文字。报告按 Python 最后同名定义生效的规则比较 attention。[作者两次定义附近](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/models/ukt.py#L542)。

## 5. 与 KeenKT 反例的关系及未覆盖边界

两份 UKT 的 w2 与 dp 分支均在 `zero_pad=True` 时显式将首行 attention 权重清零，再乘 value；这阻断了 KeenKT 审计所定位的“空历史行 softmax 成均匀分布、跨层 key 传播回答”的具体路径。UKT 前向也没有 KeenKT 的全序列 SE gate；CL 的整段池化只在训练分支使用，另一个 `temp` 平均值没有送入预测 head。[作者 w2 首行清零](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/models/ukt.py#L700)、[pyKT w2 首行清零](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/ukt.py#L699)、[作者 dp 首行清零](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/models/ukt.py#L644)。

以下仍未验证，不能由本卡结果外推：

- backward、训练损失与 CL/增强分支，以及真实训练之后的参数状态；没有评估梯度路径或统计泄漏。
- 题难度向量保留原版零初始化；有题 ID 配置没有覆盖学习后非零难度的情况。未覆盖 `separate_qa=True`、其他 embedding 变体、其他层数/维度、GPU、混合精度和 train mode。
- 真实学生数据、官方完整预处理/多 KC 事件映射、共同目标/历史构造、任何已训练 checkpoint 或质量指标。原多 KC 展开训练是否使同题兄弟答案进入某训练位置仍是数据层问题，不能用本次模型单事件前向检查消除。
- “200”只覆盖右 padding 到最大长度；没有覆盖 200 个真实交互、左 padding、中间缺失 mask、滑动截断丢弃历史或位置重置。本次核心序列 batch size 为 1，未审计跨样本混合。

**后续决策：** UKT 可以继续作为待接入候选做数据及共享验证 adapter 审核；本次结果不自动赋予 B003 资格，也不修改已冻结批次。新的训练仍需独立批次批准。

## 6. 复现命令

先从上述固定 commit 下载作者 `models/ukt.py`、`models/utils.py` 到 `.cache/research/ukt_86ad9d8/`。脚本会拒绝哈希不符的源码和已存在的输出，不需要下载数据或模型权重。以下命令在仓库根目录运行；如已存在原报告，请选择新的诊断输出名，不覆盖本卡证据。

```bash
/home/ml/.venvs/knowledge-tracing/bin/python scripts/audit_ukt_causality.py \
  --author-source-dir .cache/research/ukt_86ad9d8 \
  --pykt-source-dir third_party/pykt/pykt/models \
  --output reports/UKT_CAUSALITY_AUDIT_2026-10-14.json
```

Windows 本机应通过 `wsl -d Ubuntu-24.04 -- bash -lc` 使用该环境。此次审计没有新增运行依赖。

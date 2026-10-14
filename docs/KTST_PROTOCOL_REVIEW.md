# KTST 接入的独立协议审查

审查日期：2026-10-14。范围：独立 `knowledge-tracing-ktst` 工作树；不修改 main/B001，不训练，不接触最终测试指标。本文区分已有源码事实、正在实现的接入约定和未来协议建议；不是接入全部通过的验收报告。

**结论：本轮 KTST 不能并入 v1 排名。** 当前接入约定为 `ktlab-ktst-reference-dev-v1`，只读取训练/验证数据，测试直接拒绝。可以复用学生划分和预处理产物，但分块、评分目标、历史、训练 loss 单元和 checkpoint 候选均有差异。只有统一目标/历史清单通过审计后，才能另建共同比较轨；既有 v1 结果不能直接改名迁移。

## 1. 证据与本轮范围

已审查本项目 `data.py` 的 `_validation_files/audit_causal_validation`、`adapters.py::SequenceDataset/forward_model`、`engine.py::dataset_for_split/train/evaluate_run`、`evaluation.py::evaluate_loader`，以及固定 pyKT `split_datasets.py` 的 `add_qidx/expand_question/generate_question_sequences`。这些实现说明现有 v1 的实际目标与历史，不能只凭“maxlen=200”推断可比。

KTST 固定官方提交为 [`e6331a6f45e597f286d1272fbb96e7bb67bcaa52`](https://github.com/kainbr/kt_set_transformers/commit/e6331a6f45e597f286d1272fbb96e7bb67bcaa52)。原论文已发表于 [JEDM 18(1)](https://jedm.educationaldatamining.org/index.php/JEDM/article/view/961)。论文中“测试完整学生序列”需要结合脚注 7 阅读：评分遍历学生序列，但每次使用最多 200 个 KC 的滑窗历史，不是允许无限历史的全长 attention。这个区别会直接影响窗口边界和可评分事件。[论文第 5.1 节及脚注 7](https://jedm.educationaldatamining.org/index.php/JEDM/article/download/961/272)

与接入负责人对齐的当前约定：

- 数据只用 `train_valid_sequences.csv` 和对应原始学生序列核验，保留原 200-KC 分块，再按 `is_repeat` 合并为题目事件；边界的不完整集合保留、统计并披露。官方 KC 去重是 `list(set(l))`，参考转换逐值对齐这一实现，不自行换成排序；原顺序与集合等价性分别记录/验证。
- 模型保留原 `set_dense`、概率 `[B,T]`、内部 shift 和 `BCEWithLogitsLoss`；第一事件和 padding 不参与 loss。AS2015 的 q=0 是常量占位，`num_questions=1`，不是恢复真实题号。
- theta 参数独立学习率组；禁止 `shared` ALiBi 变体；官方核心源码不改动，导入隔离由 wrapper 处理。
- 本轮不实现或宣称已验证官方测试滑窗，不把参考轨变成当前 v1 的一个模型名称。完整测试仍需后续协议实现、冻结产物和独立解封。

上述为接入合同。具体实现完成后仍须通过下文检查，不能把设计约定当成实测证据。原实现审计和官方配置来源见 [KTST_INTEGRATION.md](KTST_INTEGRATION.md)。

## 2. 为什么不能直接合并排名

| 维度 | 现有 v1 | 本轮 KTST reference-dev-v1 | 可比性影响 |
| --- | --- | --- | --- |
| 学生/折 | 固定学生 80/20，再对开发集分 5 折 | 复用同一划分并核验 | 相同划分是必要条件，尚不充分 |
| 训练样本 | KC 展开，过去同题 KC 的回答仍可出现在该题后续 KC 训练位置 | 原分块内合并事件，通常每题一个 loss | 多 KC 题的训练权重、信息路径和样本数不同 |
| 验证目标 | 由安全展开和 `window=False` 派生；首位、尾段与 mask 决定目标 | 训练分块合并后每段第一事件不评分 | 学生相同也可能不是同一批题目事件 |
| 可见历史 | 目标兄弟 KC 移除；更早历史仍展开，边界可能截断历史题的 KC 集合 | 历史按事件聚合；分块边界可能留下 partial set | 历史事件数、KC 子集及重复权重都要审计 |
| 长度单位 | 200 个展开位置，目标占一个位置 | 先按 200 KC 分块再合并，事件数通常小于 200 | 不能同时写成“200 道题” |
| 输出 | 每目标 KC 一个概率，完整精度 late mean | 每事件直接一个概率 | KTST 不能被重复展开后 late mean 冒充原模型 |
| 优化 | probability BCE 加上原模型正则；现有 Adam 配置 | 原 logits BCE；theta 独立学习率 | 相同学习率数字不等于相同优化设定 |
| checkpoint | 当前 v1 从训练后的验证开始选择 | 官方参考允许更新前 epoch 0 入选 | 接入必须记录其候选集合，不能隐去差异 |
| 测试 | 当前 v1 非滑窗文件 | 本轮禁止；官方后续目标是修正后的 KC 预算滑窗 | 当前不存在可以与 v1 对照的 KTST 测试结果 |

重复 KC 训练是 v1 已披露的历史语义。KTST 的事件训练解决了其中一部分表示问题，但该差异本身已经存在于 KTST，不能写成本项目原创。将不同训练表示纳入未来共同评测是允许的，前提是公开训练单位、有效目标数、预算和消融；不能宣称所有模型采用相同训练目标。

**事件 ID 也不能直接复用字符串做等号比较。** v1 题级 `qidxs` 来自调用时的学生行顺序；AS2015 当前 `event_id` 更是 CSV 的 `row_id * n + 位置`。它们都是有意义的产物内部定位符，但不天然等于另一 adapter 的源事件 ID。真正再次作答同一题必须是新事件；只看题号相等会误合并。

## 3. 最小合理的共同 target manifest（建议，尚未实现）

建议未来使用一个独立版本，例如 `ktlab-v2-shared-event-context-kc200`。先从原始开发学生序列生成清单，验证通过后才考虑测试版本；不覆盖 v1 文件。清单在模型预测之前冻结，不能根据哪个模型能输出或哪个事件得分更好再取交集。

最小结构由三部分组成，避免为每个模型重复保存整份数据：

| 部分 | 必需字段及含义 |
| --- | --- |
| 元数据 | schema/protocol version、dataset、源文件与 ID 映射 SHA256、预处理/adapter 版本、学生划分摘要、split/fold、预算规则、目标集合摘要、上下文摘要 |
| 源事件表 | `uid + source_event_ordinal` 稳定键，真实 question ID 或 null，按规则去重的 KC 集合，原 KC 顺序/展开位置，源记录位置；label 保存在对应 split 的受控数据侧 |
| 目标/上下文表 | 目标源事件键、是否评分与排除原因、有序历史事件键及每个事件实际可见的 KC 子集、边界完整性、历史预算用量、各 adapter 输出位置到源事件的映射 |

清单不是模型输入张量。其历史中允许已完成事件的真实回答；当前及未来回答不能出现在模型可用的上下文或统计特征中。目标标签仅供 loss/指标消费，不用于决定哪些目标被选择。测试标签不写入公开或默认开发报告。

最小共同历史政策建议：对每个目标题，取其之前的**完整事件后缀**，按 `sum(历史 KC 数) + 当前题 KC 数 <= 200` 分配预算；所有目标 KC 共用这个历史，不因预测第几个 KC 改变窗口。KTST 合并历史集合；KC 模型按固定原顺序展开同一批历史事件，逐目标 KC 预测后聚合。这样 KTST 的当前集合与展开模型的目标集合都被计入同一预算，且没有模型获得额外完整历史。

这是有意区别于 v1 和官方 KTST 的新共同协议，不是声称与它们逐位等价。若单个事件超过预算、历史为空、KC 缺失或真实字段不合法，应在模型运行前按照固定规则排除并计数；不得静默截断当前题集合。首批建议不引入 partial 历史事件，以降低共同允许信息定义的歧义。若保留 partial 历史，则必须在清单中记录具体可见 KC 子集，不能只有事件 ID。

验收时至少输出：两种 adapter 的目标键集合完全相同；每目标允许的历史源事件及 KC/回答可见性完全相同；总事件数、排除原因、KC 数、历史长度分布；旧 v1 键到新源事件键的映射。**相同 target hash 不能替代相同 context hash。** 模型内部的表示、位置编码和更新方式可不同，这才是要比较的部分。

## 4. 本次接入的验收门槛

| 门槛 | 必须得到的证据；通过前的处理 |
| --- | --- |
| 版本隔离 | KTST 原文件/许可证/lock 校验；main/B001 未修改；运行元数据和 checkpoint 使用参考轨协议名，不写 v1 |
| 原始事件映射 | 逐学生核对序列与原 KC 流；只依 `is_repeat` 合并；真实重复作答保留；记录 partial_start/partial_end、缺失 KC 数、零目标分块数 |
| set_dense 一致 | 与官方转换函数逐值对齐；`[B,T,Cmax+2]`、KC padding、题号、答案列和 mask 类型正确；AS2015 dummy 分支独立验证 |
| 预测/标签对齐 | wrapper 保留原 `[B,T]`，外层直接用同形状 score_mask 排除首位；不重做模型内部历史 shift；指标、loss、导出使用一致的有效位置；禁止通过广播掩盖 `[B,T]` 与 `[B,T-1]` 不一致 |
| 无目标 batch | 合并后可能只剩一题，首题又被排除；在调用会计算原 loss 的 forward 前处理整批空目标，不能让空 BCE 产生 NaN 后改写为 0；单行空目标不得污染同批有效行 |
| 因果性 | 修改目标题答案不改变该题概率，修改未来答案/内容不改变过去概率；当前答案变更可以改变当前 loss；过去答案在允许位置可以影响后续预测 |
| 集合/边界 | mean/MHSA 分别检查 KC 置换不变性、重复 KC 处理、相邻同题真实重复作答、跨 200-KC 边界、首题/最后题及 padding；不能用 mean 的通过代替 MHSA |
| 原 loss/优化组 | 独立计算选定 logits 的 BCEWithLogits 与模型 loss 对齐；不是二次叠加 probability BCE；theta/base 两组覆盖每个可训练参数恰好一次，学习率来源记录 |
| 模型独立性 | 禁止 shared ALiBi；同进程构建两个实例的参数不共享；隔离导入不覆盖其他模块；checkpoint 保存/重载预测一致 |
| 重评身份 | checkpoint/metadata 绑定模型变体、fold/seed、resolved config、数据和实现摘要；重评不能改配置或源码后静默覆盖同一 run 的验证结果；官方 lock 与实际源码字节验证 |
| 资源兼容 | CPU、可用时 ROCm 无梯度前向有限且容差符合测试；数据留在 CPU、只移动 batch；不以静态“无 CUDA 扩展”代替实测 |
| 验证完整性 | 同一 `(uid,源事件)` 在计分输出中至多一次；不完整集合按参考轨明确记录；验证只有批准范围内的开发折；debug 限量结果明确不能作为正式比较 |
| 测试封锁 | 当前参考开发轨的 test 请求明确失败；不能回退到 v1 文件假装复现 KTST 滑窗；未来开启需冻结完整协议、配置和 checkpoint |

特别注意：`padding_mask` 在模型外部数据接口的 true/false 含义必须与原实现一致；不能把“有效位置为 true”直接当成 PyTorch attention 的“屏蔽位置为 true”。同理，原 loss 已屏蔽首事件，模型已完成回答历史的移位，外层保持 `[B,T]` 并使用 score_mask 即可，不应再次移位而丢掉第二事件。

已知上游 trainer 的 `set_dense` 路径存在 flat 专用变量未定义问题，使用本项目 recorder 绕过该 trainer 是已披露的执行修复，不应更换模型数学操作。官方训练与测试实现位置分别为 [trainer](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/logic/trainer.py)、[set_dense 测试读取器](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/data/pykt/eval_dataset_format_set_dense.py)、[模型](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/model/KTST.py)。

本表是验收要求；合成/无梯度检查通过仅说明实现性质，没有批准的训练仍不能做 backward/optimizer step，也不能据此前向概率宣称模型质量。

## 5. 原创研究在 KTST 之后还需证明什么

KTST 已覆盖集合表示、事件原生建模、集合聚合和可学习 ALiBi；其附录还检验了题 embedding 的零初始化。把这些组合重新命名不构成新的研究贡献。官方论文分数只作来源明确的参考，当前既没有本地 KTST 训练结果，也不能仅凭论文标题把它当成所有协议的全局 SOTA。[论文第 4 节、附录 B](https://jedm.educationaldatamining.org/index.php/JEDM/article/download/961/272)

后续保留两个**待证伪假设**，先用共同协议验证问题，再决定是否值得发明新结构：

1. **历史证据强度与状态修正分开处理。** 在相同可见历史下，短历史或稀疏概念支持是否导致对单次异常回答的过度更新？最近邻包括 [UKT](https://ojs.aaai.org/index.php/AAAI/article/view/35007)、[KeenKT](https://ojs.aaai.org/index.php/AAAI/article/view/39004) 和 [MoC-KT](https://pykt.org/mockt)。KTST 应作为无新增证据模块的强骨干对照，另设简单门控/残差平滑的同参数对照。只有自然数据上的预登记场景有稳定收益，且不只来自更长历史或人工噪声，才支持继续；不能将学到的权重称为真实粗心/猜测概率。
2. **训练证据不足时的题目特有参数收缩。** 比较动态收缩与 KTST 原零初始化、weight decay、固定题/KC 混合和简单频次规则。统计只用当前训练折；AS2015 关闭独立题目分支，因此它是共同概念骨干的检验，不能声称该分支在三数据集都有贡献。AKT/simpleKT/QIKT 和 KTST 是最近邻；若手工频次规则达到同样效果，或收益依赖全数据统计，则不支持新增复杂结构。严格 unseen-item 另行登记，不把学生划分的 warm-item 成绩叫作冷题泛化。

这些方向沿用 [RESEARCH_DIRECTION.md](RESEARCH_DIRECTION.md) 的问题假设，不代表已完成新颖性检索或已决定最终模型。第一优先级是识别哪一种真实误差在 KTST 之后仍然存在；“修复公平评测后 AUC 变了”属于协议贡献证据，不能自动归因于架构创新。

## 6. 轨道命名与后续发布条件

- **本轮：`ktlab-ktst-reference-dev-v1`。** 保留官方表示/分块训练验证语义，由独立 wrapper/engine 执行；只允许开发集，无测试结果。
- **未来官方协议参考：`ktlab-ktst-reference-kc200-window-test-v1`（建议名，未启用）。** 还需独立实现并核验官方修正滑窗、目标计数、优化/配置来源与剩余偏差；保留 reference 标签，不称原论文逐位复现。
- **未来共同排名：`ktlab-v2-shared-event-context-kc200`（建议名，未启用）。** 所有参评模型重新按共同 manifest 验证、选择 checkpoint、冻结配置后再申请最终测试。既有 v1 或 KTST 参考轨 checkpoint 不自动获得新轨资格。

若当前 adapter/engine 与这些约定冲突，先修正或更新独立协议版本，不修改 main/B001 以掩盖冲突。所有实际梯度训练、扩大预算以及最终测试解封继续逐批与用户对齐。

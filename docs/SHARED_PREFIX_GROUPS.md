# ASSIST2015 共享前缀验证执行方案

日期：2026-10-15（Asia/Hong_Kong）。状态：S008 限定合成 CPU 验收已执行一次并通过，实际结果见第6节；尚无真实 GPU 速度或模型质量结论。开发基线为 `1bc4111`，独立分支 `perf/shared-prefix-groups`。显式执行标识为 `shared_validation_execution="single_kc_prefix_groups_v1"`；缺省继续使用原逐目标路径，不修改已冻结批次、检查点选择记录或既有结果。

本方案只减少同一允许历史的重复计算，不改变 [共享事件协议](SHARED_EVENT_PROTOCOL.md) 的目标、历史、标签、指标或模型数学，也不是新的 KT 模型。首版限定 ASSIST2015 的单 KC、无真实 question ID 开发验证，支持 AKT、simpleKT、UKT、KTST mean/MHSA 的下述路径；拒绝 MoC-KT、多 KC、含真实 question ID 及其他数据集。配置与 predictor 双重检查路径白名单：AKT/simpleKT 的 `emb_type=qid`、UKT 的 `emb_type=stoc_qid` 且 `atten_type=w2`、KTST 的 `attn_variant=learnable_alibi_monotonic_q_k`。最终测试入口不在本轮范围内。

## 1. 可以共用的前缀

对目标事件序号 `t`，其 canonical 历史从 `s=history_start_index` 开始，到 `t-1` 结束，查询在张量位置 `p=t-s`。只将原始目标顺序中同一学生、相同 `s`、连续目标序号的上下文分为一组。每组选择最后一个、也是最长的上下文，保留固定 `T=200`，不同组可以按32行进行一次 forward，再按各目标自己的 `p` 取出预测并恢复原始目标顺序。

`(uid, s)` 只是分组键，不足以证明前缀一致。planner 必须逐项验证：

- 每个目标的 `history` 恰等于最长上下文 `history[:p]`，含事件身份、顺序、KC、question 和过去回答；长度、`target_index` 与 `history_start_index` 相互一致。
- 非末目标的 query 与目标事件身份恰对应最长上下文 `history[p]` 的 query 部分；最后目标对应最长上下文自身 query。所有事件仅一个 KC，question 为 null，事件不能缺失、重复或重排。
- 当前标签不参与分组资格、前缀匹配或输入构造。中间目标的真实回答只从最长上下文已经允许的 `history` 取得；最后查询回答仍填 dummy 0。严禁通过读取各目标 `label` 拼接历史。

例如预测第2、3、4次作答，可使用同一个物理输入 `[第1次真实回答，第2次真实回答，第3次真实回答，第4次dummy]`。第2次预测只能读第1次回答，第3次预测可以读前两次回答。物理行中存在后续答案不等于允许早期预测使用它，须由已核验的因果路径保证。若把第2、3次回答都置为 dummy，反而会破坏第3、4次目标原有的合法历史。

窗口起点改变时必须分组：单 KC 下 `s=max(0,t-199)`。序号199的目标起点为0，序号200的目标起点为1，不能共用一个长行，也不能复用包含被丢弃早期历史的 hidden states。相对距离相同不足以证明缓存可复用；simpleKT/UKT 的窗口相对位置也会改变。因此每个学生通常只有起点0的前199个合格目标可以形成长组，满窗后的各目标通常各自成组。

## 2. 已审模型的数学条件

本节是固定源码的静态依赖分析，不替代执行验收，也不泛化到任意同名模型配置。

| 路径 | 早期输出不依赖后续事件的依据 | 必须保留的条件 |
| --- | --- | --- |
| [AKT](../third_party/pykt/pykt/models/akt.py) | interaction encoder 因果可见当前回答；预测 cross attention 严格只见过去。distance 在因果 mask 后做行内 softmax/cumsum/sum，`detach` 不改变前向可见信息 | 首版白名单限定 `qid` 路径；丢弃附带正则/loss |
| [simpleKT](../third_party/pykt/pykt/models/simplekt.py) | 严格过去注意力，逐位置 LayerNorm/FFN，固定正弦位置取前缀 | 相同窗口起点；保留原接口一次 shift；输出索引不能再次 shift |
| [UKT](../third_party/pykt/pykt/models/ukt.py) | 两种注意力距离均在 mask 后做行内计算；全序列 `temp` 不参与预测；CL 池化只用于训练分支 | 必须显式 `train=False`，同时 `eval()`；相同位置，不能调用 CL 训练路径 |
| [KTST mean/MHSA](../third_party/ktst/pkg/model/KTST.py) | encoder 因果计算后右移一次，decoder 因果；mean/MHSA 只聚合单事件的 token；ALiBi 使用位置差 | 保留原内部 shift 与 start token；丢弃原 forward 附带的 dummy BCE |

统一使用 `eval` 和 `no_grad`，不改变参数、设备、dtype、因果 mask 或 padding 合同。固定 `T=200` 限制长度变化，但 batch 形状和浮点归约仍可能产生小数值差异；只允许使用预先登记的数值容差，不能先承诺逐位相同。

MoC-KT 的可见前缀长度会选择不同卷积分支，不能直接取最长上下文的所有位置代替逐目标推理，因此首版明确拒绝。多 KC 还存在同题兄弟答案与逐 KC 投影问题，不能从本次单 KC 结论外推。

## 3. 原目标、摘要和指标必须保留

分组后的每个概率仍属于其原目标事件，而非组的最后目标。输出必须保留原始逐目标顺序、uid/event_id、label、question、KC 数量、target/context digest；逐目标 digest 不得被最长行的 digest 取代。指标层才读取 label，仍每原事件计分一次，不做跨目标平均或漏掉组内较早目标。

沿用原数据读取前后 source 校验、所选目标/上下文流式摘要和数量核对。任何前缀不一致、缺失/重复输出、概率非有限或超出 `[0,1]` 均失败，不截交集或静默回退后报告全量通过。执行 recipe 应写入 resolved config、运行/验证收据，默认原路径保持兼容；新执行路径不能反向改写旧批次身份或声称旧批次已经用它执行。

## 4. 已有索引证据与可预期的计算规模

依据 [只读索引审计报告](../reports/AS15_SHARED_PREFIX_GROUPS_INDEX_2026-10-15.json)，文件 SHA256 为 `56ad9e12206e619f463cf2b41bbc2842c750e5daa42f3a07cfb4ef55e24988e9`。它仅统计 ASSIST2015 fold0 valid 的 canonical 开发上下文，没有模型 forward、梯度、optimizer、测试集或训练权重访问；本说明只读取已保存报告，不重新扫描数据。

| 项目 | 原逐目标执行 | 按连续前缀分组的计算估计 |
| --- | ---: | ---: |
| 目标事件数 | 102,776 | 102,776，保持不变 |
| forward 行数 | 102,776 | 5,042 |
| 每批32行时的批数 | 3,212 | 158 |
| 固定 T200 的 padding 后位置数 | 20,555,200 | 1,008,400 |
| 有效位置数 | 4,271,917 | 501,244 |

报告含3,055名学生、3,055个起点0组和1,987个截窗单目标组，最大组长199。`20.38397` 是原行数/分组行数的**压缩比**，约95.09%的行数减少；不是模型或端到端实测提速。数据检索、前缀核对、摘要、CPU搬运、输出和指标仍需处理全部102,776个目标。要实现所列158批，必须将不同组再批处理，不能只在原逐目标 minibatch 内做局部合并。报告也没有证明各模型预测等价。

## 5. S008 预登记与可检验验收

执行前登记的预期覆盖为5个 CPU 原模型（`d=16`）、85次原模型 forward；KC 模型使用 `n_blocks=2`，KTST encoder/decoder 各2层、聚合1层，覆盖跨层传播。硬上限为12个模型、128次 forward、0次梯度调用、0次 optimizer/参数更新，进程限时300秒。仅人工合成上下文和临时合成权重的保存/重载，不读取真实数据、已训练权重或测试集。显式开关为 `KTLAB_RUN_PREFIX_GROUPS_CPU_ACCEPTANCE=1`；Root 已先冻结并登记最终 source/test 哈希及预算，再执行一次。该设计预算继续保留，实际消耗另见第6节。

必须核验的结果：

1. 每个支持模型的 grouped 输出与原逐目标、固定 T200 路径在预登记容差内一致，包含 batch1/多组、首个合格目标、满200位置与不同组大小；计数明确覆盖哪些配置，不能称所有配置均已证明。
2. 组内中间回答翻转时，该位置及更早输出不变；后续目标允许使用变化后的历史。未来 query/KC/回答扰动不改变较早输出。毒化独立 `label` 访问仍可完成 planner/predict，指标单独消费标签。
3. 201事件窗口边界强制分组；同uid/start但历史回答、事件身份、query、顺序、target_index不一致必须拒绝，不能仅比组键或 opaque hash。
4. 纯文件/fixture 路径验证全部原目标一一输出，原始顺序、逐事件摘要与指标一致，读源变动仍失败；拒绝 MoC、多 KC、非 concept-only、非 AS15 和未知 execution recipe。
5. 缺省旧路径行为不变；训练和最终测试不得因新 flag 隐式切换。执行身份如写入 checkpoint/config，加载规则须明确兼容边界，不能给旧检查点重新贴执行身份。

S008 通过后，才有依据在 GPU 空闲时另行冻结 S009 受控开发验证计时方案；截至本次更新，S009 尚未准备或执行。拟议计时的两条路径使用同一组预先固定的随机初始化权重/新建模型，明确不读取、不追加复测已完成训练的权重。性能结论需同时报告原/新执行路径、初始化身份、相同完整目标摘要、概率数值差和端到端耗时；不以随机模型产生质量结论，索引压缩比也不能替代计时证据。

## 6. S008 实际验收记录

依据已保存的 [S008 冻结清单](../configs/checks/S008-shared-prefix-groups-cpu.json) 和 [S008 验收报告](../reports/S008_SHARED_PREFIX_GROUPS_CPU_ACCEPTANCE_2026-10-15.json)，本次更新仅读取收据并整理结果，没有重跑检查。执行身份为：

- implementation SHA256：`6765534b76c4fad5a50bd2b5217f3ec43dec64798bcc09a535eb9d65cf50e251`。
- manifest canonical SHA256：`70fd90fdc0182b0699e93bcb27e53bd6eb658b0c51cd55e11f432ec157a82867`。
- 验收报告文件 SHA256：`34fe501ee50cc4b8fb287bfcb8f5d3883e537abc500461e7ed3fcb5db0237de0`；日志、绑定文件和私有收据摘要见报告。

单次运行结果为 **53 passed，0 failed/errors/skipped/warnings**；pytest 耗时53.84秒，进程耗时55.89126089秒，未超时。实际构造5个双层 CPU 原模型、共85次完整 forward（每模型17次）；`autograd.grad`、backward、optimizer 构造及参数更新均为0。使用 seed42、CPU单线程、随机初始化和人工合成数据，没有 GPU、真实数据、测试集或既有训练权重访问。

五模型各235个合成目标的 grouped 对原逐目标 oracle 最大绝对误差均为0（登记容差 `atol=1e-6, rtol=0`），历史回答和当前 KC 正控差均大于 `1e-8`。检查覆盖标签/当前与未来回答隔离、201事件窗口、输出顺序与摘要、source变化/超时拒绝、epoch0及执行身份绑定、缺省行为和临时合成权重重载；详细断言与数值见报告。

结论仅覆盖清单中的五种双层 CPU 小配置及人工输入，不能外推任意超参、GPU 数值或真实数据。真实模型对照未计算质量 AUC；文件 fixture 的指标一致性不是研究成绩。S008 不证明20.384倍实际提速、不修改任何训练批次；S009 GPU 计时仍未准备或执行。

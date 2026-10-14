# 假设 B 原创性审计：按训练证据收缩题目特有参数

核查日期：2026-10-14。范围：只读论文、作者代码和本项目已固定源码；未训练、未构建优化器、未修改 src/config。B003 仍须单独批准。本页不是新模型命名或实现承诺。

## 直接判断

**当前表述具有可检验的工程价值，但原创性偏弱，尚不足以作为独立新模型贡献。** “稀疏题依赖共享 KC、充足数据题保留个性”“融合题目和概念”“频次/证据越少收缩越强”均有明确先例。尤其 2026 年 SLC 已直接在 KT 中使用逐题 empirical-Bayes 收缩；简单加入 `n/(n+τ)` 或 sigmoid 频次门不能支持“首次”的主张。[SLC 原文 §3](https://arxiv.org/html/2606.14123v1)、[作者核心代码](https://github.com/xiaoran-y/SLC/blob/f87e4ecd40e3a52c7a1c8fcd4e6524b28412c707/posthoc/item_drift.py)

仍值得检验的窄问题是：**在相同事件目标、完整历史和训练预算下，仅用当前训练 fold 的题目次数约束题目残差，能否稳定改善真实长尾题的 NLL/校准与事件 AUC，并优于正确调参的普通正则及手工频次收缩。** 这是待证伪的增量假设，不是已确认新颖性；即使有效，也须解释收益来自何处。

证据等级在本页表示核查深度，不是论文质量：**A** 为原文及相关作者/固定实现均已静态核对；**B** 为原文可查，但实现或其数据来源未闭环；**C** 为本审计推论、建议或待验证假设。作者报告的性能不等于本项目复现结果。没有搜到完全同式的方法不构成新颖性证明。

## 四组最接近先例

### 1. SLC：直接的题目收缩先例，2026，A

完整题名为 *Recovering Stranded Discrimination in Knowledge Tracing: Per-Item Bias Correction via Empirical-Bayes Shrinkage*，Xiaoran Yan、Cheng Tang、Atsushi Shimada。arXiv 于 2026-06-12 提交；[arXiv 元数据](https://arxiv.org/abs/2606.14123)和[作者仓库](https://github.com/xiaoran-y/SLC)自报 ECML PKDD 2026 Research Track 接收，本轮未独立核到会方录用名录。

SLC 冻结 KT 主干，用独立 calibration 窗口的响应估计逐题 logit 偏差，再加到主干预测。原文 §3.5 的静态收缩由题目有效信息量与先验精度共同决定；§3.1 明确 calibration 与之后的 test 分开。**它收缩的是输出偏差，中心为零；不是训练期将 item embedding 收缩到 KC 中心。** 这个差别允许比较，但不能抹去题目稀疏性、部分汇聚和收缩的已有贡献。[原文 §3.1、§3.5–3.6](https://arxiv.org/html/2606.14123v1)

作者 commit `f87e4ecd40e3a52c7a1c8fcd4e6524b28412c707` 的 `posthoc/item_drift.py::fit_dynamic_item_bias` 已核对：先累加 `p(1-p)` 为 `n_eff`，用残差和及先验方差得到 `static_bias`；另有可选 `shrink_tau`，按 `n_eff/(n_eff+shrink_tau)` 继续缩放。默认 `shrink_tau=0`，不能说论文默认使用了这层额外门。**这里的 n_eff 是预测相关 Fisher 信息，不是原始出现次数。**[固定源码](https://github.com/xiaoran-y/SLC/blob/f87e4ecd40e3a52c7a1c8fcd4e6524b28412c707/posthoc/item_drift.py)

剩余差异：训练期、KC 条件中心、仅次数输入，以及多 KC 集合的一致性。均需实验支撑。如果拟比较 SLC，不能在当前 validation 标签上拟合校正器再在同一批标签上报效果；须另行批准训练 fold 内的校准划分或交叉拟合，并给所有方法同等数据预算。原文时序校准成绩不能直接加入本项目学生划分榜单。

### 2. QIKT → Q-MCKT：题/KC 联合建模和长尾题专门处理，A/B

[QIKT 原文](https://arxiv.org/html/2302.06885)已同时建模题目敏感及概念状态，并组合多个预测分量。固定 pyKT 的 `qikt.py::predict_one_step` 提供配置权重的 logit 组合，以及非 IRT 模式的加权概率平均；不是由当前题目训练频次控制的混合。[固定 QIKT 源码](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/qikt.py)

**Q-MCKT 是比只对比 QIKT 更贴近假设 B 的基线。** 其 2024 arXiv v3 明确把少交互题目的过拟合作为问题：按频次阈值确定低频 anchor，从同 KC 的较高频题中按难度选正负样本，做题目对比学习；题和概念两路还有专家门控。它已覆盖“用丰富的相关证据改善稀疏题”，但不是单纯次数门控制 item 残差。[Q-MCKT §4.2–4.4](https://arxiv.org/html/2403.07322v3)

作者仓库 commit `ee1698638888c5cddd577326aa30cadd5538f42d` 中，模型实际名为 `qikt+`：`qikt_plus.py::Aggregate_MoE.forward` 做 softmax 专家加权；`get_positive_pairs` 读取 `q_pos_neg`；`contrastive_loss_func` 对题目向量施加距离约束。[模型源码](https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/models/qikt_plus.py)、[模型注册](https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/models/init_model.py)

2026-10-14 补充审计已闭环生成路径：`KTQueDataset.__load_data__` 先筛训练 folds，再统计频次/正确率；标准干净重建路径没有混入验证或测试响应。**缓存没有数据/配置指纹，负例也没有论文所述的高频过滤**；统计来自切块后保留的位置，不能默认为全部原始事件。历史运行缓存及阈值选择来源仍未知。不能从 global accuracy 措辞判泄漏，也不能把源码隔离升级为历史实验的运行证明。详见[固定行号审计与接入取舍](QMCKT_STATISTICS_AUDIT_2026-10-14.md)。

出版状态：arXiv v3 标注 TKDD accepted，[作者主页](https://www.zitaoliu.com/index.html)列为 TKDD 2025；本轮 ACM DOI 页面访问返回 403，因此方法核查依据为作者 v3 和代码，未用出版页补写未经核实的细节。[元数据](https://arxiv.org/abs/2403.07322)

### 3. AKT / simpleKT：共享概念基底加题目差异项，A

[AKT](https://arxiv.org/abs/2007.12324)与[simpleKT](https://arxiv.org/abs/2302.06881)已经提供最直接的结构对照：概念表示加题目特有的 Rasch 风格变化。当前固定实现中，AKT 的题目参数是标量，simpleKT 默认 `qid` 是向量，`qid_scalar` 才用标量；二者题目参数均零初始化。因此“概念基底 + 题目残差”或“先共享、再学题目差异”本身没有新增结构原创性。[AKT 源码](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/akt.py)、[simpleKT 源码](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/simplekt.py)

固定 `AKT.forward` 还返回所查到题目参数平方和乘 `l2` 的正则项；这按本次输入中的参数出现次数累加，不能不加说明地等同于每个题目只计一次的全局层级先验。`simpleKT.forward` 默认路径没有返回该 AKT 正则项，不能因为构造函数含 `l2` 就说两者正则相同。对照应记录 loss 的实际归一化和 penalty 的计数单位。

剩余差异只能落在“训练次数如何改变残差约束以及为什么有效”，不能仅落在新增题/KC 两路。若改 simpleKT 的 scalar/vector 参数化，同时又加收缩，必须分开消融，防止把降参数收益归给频次机制。

### 4. DKTT：频繁题保留差异、稀疏题借用技能的早期明确动机，B

Pu 等人的 *Deep Knowledge Tracing with Transformers*（AIED 2020）已明确提出频繁题与稀疏题共享技能的取舍。方法 §3 的交互表示是 `softmax(W_row)S`，用共享潜在技能向量表示交互；**原式不是显式的次数函数 g(n)**。因此它否定“首次提出这一动机”，但不能被描述为已实现完全相同的次数门。[作者论文 PDF，§1、§3](https://yuchuang1979.github.io/assets/papers/aied20.pdf)

可找到[第一作者公开代码](https://github.com/scott-pu-pennstate/dktt_light/tree/d0f41f586778c5c8177c7740c8c668c6b137bdc0)，本轮 HEAD 为 `d0f41f586778c5c8177c7740c8c668c6b137bdc0`。README 自称 light variation，不能把它未经核对地当成论文全量原版；本轮只确认出处，未运行或完成模型侧逐行复现。[固定 README](https://github.com/scott-pu-pennstate/dktt_light/blob/d0f41f586778c5c8177c7740c8c668c6b137bdc0/Readme.md)

## 层级 IRT、KTST 与 cold-item 的边界

层级 Bayesian IRT 中，题目参数部分汇聚到群体均值、由方差控制收缩，是已建立的统计机制；非线性 IRT 下还需检验先验和偏差，不能直接把高斯收缩直觉当成性能定理。近期可核对 [König 等，OH2PL，2023 卷期](https://link.springer.com/article/10.3758/s13428-022-02000-5)和[König 与 Alexandrowicz，2024](https://journals.sagepub.com/doi/10.1177/01466216241227547)。本项目候选若没有先验、似然、后验或对应估计过程，应称“频次条件正则/收缩启发式”，不称 Bayesian uncertainty。

当前已有 KTST 也必须保留为控制：固定 `KTST.py::_reset_parameters` 可将题目 embedding 置零，`AggregationMean` 结合题目与 KC 表示。因此仅把随机题目初始化换成零，也不是新的稀疏建模思想。[KTST 固定源码](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/model/KTST.py)

Cold-start 文献必须逐项区分：csKT 针对新学生短序列，不等于新题目零响应；PICKT 的 2026-09 v2 确实研究新题，使用难度、文本和知识图关系，输入条件超出本项目当前仅 ID/KC/响应设置。本轮核到 PICKT v2 的官方摘要及版本变更，未审计全文实验或代码，故只作范围提醒，不能据其摘要宣称同信息条件的 SOTA。[csKT 出版页](https://www.sciencedirect.com/science/article/abs/pii/S0957417424028550)、[PICKT v2 官方记录](https://arxiv.org/abs/2512.07179v2)

## 简单次数门为何可能只是重参数化（本审计推论，C）

设候选对每个题目使用

\[
e_q=c_{K(q)}+g(n_q)u_q,\qquad g(n_q)>0,
\]

其中 `u_q` 是每题无约束的自由参数，`n_q` 在训练前固定。令 `v_q=g(n_q)u_q`，就得到 `e_q=c_{K(q)}+v_q`。**在此条件下，固定次数门没有改变可表示函数集合。** 差异可来自初始化、优化路径或正则，而不是新增表达能力；如果同一参数还用于未门控分支，或 g 随样本变化，则需重新分析，不能套用此结论。

若对 `u_q` 施加 L2，则换元后的惩罚为

\[
\lambda\|u_q\|^2=\frac{\lambda}{g(n_q)^2}\|v_q\|^2.
\]

这给出了必须做的“次数加权正则”对照。目标函数等价不表示 Adam/SGD 轨迹逐步相同，报告时须区分。`g(0)=0` 能定义未见题的共享回退，但零频题没有题目响应证据，不能据此恢复其特有难度；同 KC 的未见题可能仍无法区分。原始次数也不等于参数可靠性：重复来自同一学生的回答、极端难度以及 KC 标注质量都会影响有效信息。

## 剩余可检验差异与最小消融

先用 B003 共同验证结果确认真实失败模式，再决定是否申请新实验；不以尚未得到的结果宣称长尾已是主要瓶颈。如下是设计要求，不是已批准批次。

| 对照 | 要排除的替代解释 |
| --- | --- |
| 原 AKT/simpleKT/KTST，按原设置及同预算调参 | 弱基线或欠收敛造成的假收益 |
| 共享 KC 骨干、无题目残差；共享加自由残差 | 收益只是增加/删去题目参数 |
| 与候选相同的零初始化、普通 L2/weight decay | 初始化或通用正则已足够 |
| 固定混合，以及手工 `g(n)=n/(n+τ)` | 学习门是否真超过简单先验；τ 只用开发验证选择 |
| 同中心、同归一化的次数加权 L2 | 门控的收益是否可由重参数化后的正则解释 |
| 候选门与参数量匹配的普通门；固定种子打乱题目频次分配 | 新增参数与题目次数信号各自的贡献 |
| 相同收缩强度：零/全局中心与 KC 中心；单 KC 与多 KC 分层 | 收益是否依赖可靠 KC 共享结构，错误共享是否伤害头部题 |
| Q-MCKT（训练 fold 内重建统计）及公平校准条件下的 SLC/静态 ridge | 是否超过真正针对稀疏题的近邻，而不仅超过基础骨干 |

首轮应优先手工门和正则控制；若其结果相当，就保留简洁方法与负结果，不增加复杂结构。若要使用 Fisher 信息、后验方差或正确率，已经超出“仅次数”的最小假设，应单列消融与数据拟合流程，不能悄悄扩大输入。

统计必须在**当前训练 fold 的原始事件**上生成，KC 展开、重叠窗口或训练 epoch 数不能把一个真实响应重复计入 `n_q`；学生真正重复答同一道题仍是不同事件。另报每题独立学生数，作为次数相关性的诊断。保存 fold、来源文件 SHA、题号映射和频次表 SHA；validation/test 的次数、正确率都不得参与拟合。未见题的映射及 KC 关联可见范围须预先声明。

建议预先冻结 `n=0` 和训练频次分位组，主报全体事件 AUC、NLL、Brier，另报各组指标、事件数、题目数和学生数；低频组不具备两类标签时 AUC 标不可计算，不删除该组。可补题目等权 NLL，以免头部题遮蔽长尾。比较按相同 event_id 配对，多种子/多 fold 检查方向；置信区间应考虑同学生相关性。阈值和胜负标准不能看最终 test 后再改。

否证条件：普通正则或手工门已达到相同效果；AS09 与 NIPS34 方向不稳定；少量尾部收益伴随明显总体/头部退化；或优势仅在全数据统计、额外信息或不匹配预算下出现。此时最多支持一个工程控制结果，不支持复杂新模型或普遍 SOTA 主张。

## ASSIST2015 的硬限制

已直接核对本项目 `data/prepared/assist2015/manifest.json`：`num_q=0`、`num_c=100`；固定上游预处理把 `sequence_id` 作为技能，把题目列表写为 NA。[官方固定预处理](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/preprocess/assist2015_preprocess.py)

因此 AS15 **不能检验独立 item 参数收缩、item 频次效应或 cold-item 泛化**。不能将 KC 号改名为题号，也不能把 adapter 的 dummy 0 当成真实题库。本假设的题目分支在 AS15 应关闭；该集只检验共同 KC 骨干及总体训练管线。若仍宣称三集均证明题目收缩贡献，结论不成立。AS09/NIPS34 的标准学生划分也是 warm/mixed-item 开发实验；严格未见题实验须另行登记、冻结题目划分及历史可见性。

## 本轮证据与未完成项

本轮通过公开论文/作者代码和 `git ls-remote ... HEAD` 核对上述远端 commit；没有克隆、安装或运行第三方项目。GitHub 未认证 API 限流，已改用公开 raw 文件及只读 Git 引用查询；未绕过权限。pyKT 和 KTST 使用本项目已锁定 commit 的源码作实现证据。

Q-MCKT 正负题统计的标准 fold 数据流已在同日补充审计中闭环；仍未完成其历史缓存/阈值设定来源核验、SLC 全评价脚本的逐行协议复核、PICKT 全文和实现复核，以及在本项目协议下复现任何近邻。未获得这些证据前，不能将作者报告的 AUC 与本项目数值直接相减，也不能把本页认作穷尽性的新颖性检索。

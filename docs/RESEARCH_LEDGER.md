# 知识追踪研究证据账本

核查日期：2026-10-15。第一轮结果目标日期：2026-10-29。

本文件记录候选方法和比较条件，**不提供论文结果的统一复现排名**。B001 链路检查与 B002 单轮运行校准已完成；B003已完成6/12组后因系统更新重启中断：[ASSIST2009四模型](../reports/B003_FIRST_FOUR_BASELINES_2026-10-15.md)中KTST mean验证AUC为0.792165；ASSIST2015的AKT和simpleKT分别以0.731876与0.726691完成。按用户最新要求，B007优先训练KTST历史KC×回答交互版及等参数对照，剩余基线暂后排，见[最新进度](../reports/B003_INTERRUPTION_AND_B007_PRIORITY_2026-10-15.md)。尚无自有变体完成成绩或本项目最终测试结果。下列文献表中的优势属于论文或作者报告，和本机共享协议结果分开；代码公开、成功导入和训练复现是三种不同状态。

2026-10-14 最新授权：启动 B003，每次最多50轮，连续3轮验证 AUC 无严格改善早停；后续本机项目内批次无需再次请求批准。仍须登记具体清单、哈希、预算及持续授权依据。最终测试先冻结候选、检查点和分析方案，再按持续授权执行；付费云资源不在默认授权内。此处记录授权，不表示训练已经启动或完成。

[KTST训练历史覆盖统计](../reports/KTST_TRAINING_CONTEXT_COVERAGE_2026-10-15.md)已一次完成：在当前题与块内历史均完整的训练目标中，AS09有30.12%、AS15有1.76%、NIPS34有41.88%可用共同KC200后缀补回更早历史。它支持后续优先考察NIPS34的配对训练适配，不是预测收益或原创结论；抽样控制、原完整监督骨干及成本均需比较。B007保持冻结，该后续适配尚未建训练批次。

本轮补充的[完整证据卡](evidence/README.md)覆盖 KTST 逐折官方配置、MoC 代码与全文缺口、UKT/KeenKT 主表及 KeenKT 合成因果审计、题目收缩近邻。KeenKT 指定公开前向存在当前/未来答案依赖，尚不能直接进入比较；简单频次收缩的宽泛原创性也不足。具体证据和限制见各卡。

10-15执行进展：B003已在本机运行，尚未完成。UKT的[训练契约](evidence/UKT_TRAINING_CONTRACT_2026-10-15.md)已核实，[独立接入分支](https://github.com/Krsiwhite/knowledge-tracing/pull/1)通过27项人工CPU训练检查和45项相关CPU回归（另2项GPU专用检查跳过）；真实数据/GPU训练仍待执行。新的[残差更新近邻审计](evidence/CAUSAL_UPDATE_NOVELTY_2026-10-15.md)认为现有假设A不足以选为原创主线，避免将答案趋势重新命名为新信息。

## 1. 本阶段的问题与数据范围

第一阶段研究 ASSISTments2009、ASSISTments2015、NIPS34，主任务是根据历史交互预测下一次作答正确概率。三个数据集采用同一份[项目评测协议](PROTOCOL.md)，先建立可靠强基线，再选择原创研究假设。题目文本、选项、LLM生成解题过程暂不作为共同输入，相关方法放入独立的内容增强比较。

“超过 SOTA”必须同时给出数据版本、输入信息、划分、评测单位、历史长度和调参预算。不同论文的最大 AUC 不能直接组成统一榜单。

## 2. 第一批需要审计的候选

| 方法 | 已核实状态与来源 | 与当前研究的关系 | 尚待完成的核查 |
| --- | --- | --- | --- |
| KTST | JEDM 18(1)，2026-02-07，89–112页；[期刊页面](https://jedm.educationaldatamining.org/index.php/JEDM/article/view/961)、[全文](https://jedm.educationaldatamining.org/index.php/JEDM/article/download/961/271)、[作者代码](https://github.com/kainbr/kt_set_transformers) | 集合式交互表示、可学习ALiBi；8个pyKT数据集包括本项目三个数据集。mean 已完成 B002 本地校准、v2 前向检查；[主表和20份逐折配置已核](evidence/KTST_2026-10-14.md)。 | B003 已获启动授权，按50轮/耐心3执行；固定小配置与作者最优不同，后续须做强配置或同预算搜索。 |
| MoC-KT | TOIS；出版商元数据的online日期为2026-09-15；[DOI](https://doi.org/10.1145/3847658)、[作者团队介绍](https://pykt.org/mockt) | 多尺度因果卷积、自适应分段和距离衰减注意力；已进入pyKT。作者报告4个数据集、26个比较模型。 | 全文实验表中的具体数据集、版本、预算与数值；导入成功不能替代复现。pyKT新闻日期05-08是集成日期。 |
| FA-KT | WWW 2026，7080–7090页；出版商online日期2026-04-12；[DOI](https://doi.org/10.1145/3774904.3792272)、[官方介绍](https://pykt.org/fakt) | 频率感知的异构专家混合，含attention、Mamba、CNN、LSTM；作者报告5个数据集、20个基线。已进入pyKT。 | 具体实验表及配置；检查ROCm运行。当前adapter要求真实timestamps，缺失时间时不纳入共同信息主榜。不能伪造时间或替换专家后仍称原模型复现。 |
| CGMKT | 2026-09-22已有[官方公开介绍与集成](https://pykt.org/cgmkt)；[DOI](https://doi.org/10.1016/j.inffus.2026.104800)对应Information Fusion的2027年3月卷期 | 双图融合与学生group-level mastery；作者报告3个数据集、21个基线。对“显式知识状态”方向必须比较。 | 区分公开可用日期与卷期年份；核对三个数据集、图构建数据范围、状态更新的因果顺序和内存成本。 |
| UKT | AAAI 2025，官方发表日期2025-04-11；[论文](https://ojs.aaai.org/index.php/AAAI/article/view/35007)、[作者代码](https://github.com/UncertaintyForKnowledgeTracing/UKT) | 概率分布表示、Wasserstein attention、不确定性对比学习；六个数据集包括本项目三个数据集，已进入pyKT。[主表已核](evidence/UNCERTAINTY_KT_2026-10-14.md)，AS15 低于同表 AKT；[作者/pyKT限定前向审计](evidence/UKT_CAUSALITY_2026-10-14.md)216项结果及签名差异已记录。 | 真实多KC与共享adapter、训练/增强分支仍待审；在相同输入、划分和预算下重新比较。 |
| KeenKT | AAAI 2026，官方发表日期2026-03-14；[论文](https://ojs.aaai.org/index.php/AAAI/article/view/39004)、[作者代码](https://github.com/HubuKG/KeenKT) | 状态分布、去噪重构和分布对比学习；AS2009、AL2005、BD2006、NIPS34、AS2015、POJ。公开划分代码为20%固定test加80%内5折。 | [固定快照的36项合成控制](evidence/UNCERTAINTY_KT_2026-10-14.md)定位全序列SE和attention空首行两条不可见答案依赖；不据此归因论文成绩，接入前须明确合法前缀路径/修正版及数据对应关系。 |

基础锚点保留 DKT、AKT、simpleKT；QIKT、DTransformer、FoLiBiKT、stableKT、FlucKT、RobustKT按研究方向及验证表现补充。列入候选清单不代表已经具备直接训练条件，仍需完成协议与实现审计并登记预算；满足条件后可按持续授权执行。当前模型集成情况可查[pyKT官方模型文档](https://pykt-toolkit.readthedocs.io/en/latest/models.html)。

## 3. 需要分开比较的前沿方法

| 方法 | 证据和数据 | 不能直接混比的原因 |
| --- | --- | --- |
| TSDR | [IJCAI 2026正式论文](https://www.ijcai.org/proceedings/2026/415)、[全文](https://www.ijcai.org/proceedings/2026/0415.pdf)、[官方代码](https://github.com/DMIRLAB-Group/TSDR)。Spanish、AS17、Slepemapy、AL05、Prob、Linux、Database、Comp、EdNet；训练目标结合propensity、误差填补及时间平滑。 | 仅保留每个学生最后50次交互，学生分层5-fold；和本项目数据集合及窗口协议均不同。用于去偏研究近邻，不直接引用其分数作为三数据集SOTA。 |
| DisKT | [全文](https://arxiv.org/html/2503.02539v1)、[作者代码](https://github.com/zyy-2001/DisKT)。WWW 2025；11个数据集含AS09、AL05、BD06、Statics、EdNet及PTA/语言/地理数据。 | 原文保留最后100次交互；把多KC组合映射为新概念；EdNet、Comp、Slepemapy各抽5000学生。与pyKT数据定义不同。 |
| BAIM | [ACL Findings 2026-07正式论文](https://aclanthology.org/2026.findings-acl.1125/)、[作者代码](https://github.com/sjin4861/BAIM)。XES3G5M、NIPS34；离线RLM四阶段解题表示加历史条件路由。 | 额外使用题目内容，不能直接覆盖AS09/AS15。重建表示的计算量需另计；若使用公开缓存，应核对题目映射、生成模型与PCA拟合范围。 |
| OPERA | [作者团队介绍](https://pykt.org/opera)日期2026-08-24，已列入[CIKM 2026官方program](https://cikm2026.diag.uniroma1.it/program/)。离线生成并审核选项级推理/误概念说明；pyKT提供五个增强变体。 | 面向multiple-choice数据，依赖题干、选项和对应表示。属于内容增强轨道，不能把额外信息收益全部归于预测网络结构。 |
| CDFKT | [官方集成公告](https://pykt.org/cdfkt)日期2026-10-11；[出版社页面](https://www.sciencedirect.com/science/article/pii/S1566253526005245)对应Information Fusion 137，2027年1月。Mamba知识积累、序列步衰减适应性和自适应融合。 | [接入检查](evidence/CDFKT_CAREKT_ADMISSION_2026-10-15.md)：固定源码依赖题目映射文件，无AS15概念独立路径；本机尚未安装两个额外依赖，ROCm兼容性未验证。完整实验表仍缺，不直接加入三集队列。 |
| CareKT | [出版社条目](https://www.sciencedirect.com/science/article/pii/S0045790626004817)列Computers and Electrical Engineering 139 Part B，111411，2026年11月卷期；置信度细化、多视角、辅助监督和focal loss。 | 作者摘要称四集中的三集AUC最好，NIPS34具有竞争力。完整表与[匿名代码入口](https://anonymous.4open.science/r/CareKT)尚未核验；只列近邻，见[接入边界](evidence/CDFKT_CAREKT_ADMISSION_2026-10-15.md)，不填猜测分数或声称全部SOTA。 |
| ACE-KT | [AISTATS 2026/PMLR](https://proceedings.mlr.press/v300/guo26b.html)、[作者代码](https://github.com/AWord992/ACEKT)。AS09、AL05、BD06、NIPS34、EdNet；会议日期05-02至05-05。 | 正文报告在5个数据集中4个最好，EdNet低于最高方法0.014；不能从摘要推导“全部数据集最好”。 |
| Jev / JevKT | [arXiv:2610.11135v1](https://arxiv.org/abs/2610.11135v1)，2026-10-08首发预印本，题名 *Can a System-One LLM Perform Knowledge Tracing When Few or No Learners Are Logged?*；作者给出匿名代码与结果仓库链接。本轮只核官方摘要、版本记录及该链接的存在，未审实现。 | 研究零/少量平台学习者日志与预训练LLM单次概率推理；监督KT在更多日志条件下会追上，是作者摘要已披露的边界。训练样本量、预训练信息和API成本与本项目完整训练集不同，不能用其少样本均值替代三集共同条件SOTA；不调用付费API。 |

补充待核查：KDD2026的[Structured Inductive Bias for Multi-timescale Knowledge Tracing](https://kdd2026.kdd.org/papers/)已在官方名单，但本阶段未核定全文/代码。[RegKT范围核对](evidence/REGKT_SCOPE_2026-10-15.md)现已补齐正文与HEXED 2025官方接收记录：2026-09-18是arXiv上传日期；其ASSIST2009实验筛选最常见500题，不能纳入本项目共同条件排名。保留为IRT辅助监督近邻，不插入当前自有模型实验队列，也不作为已复现SOTA。

## 4. 每篇论文必须补齐的证据卡

正式安排某方法训练前，补齐以下记录，并保存来源链接、访问日期和代码提交：

1. 发表/接收/预印本状态；原文表号；作者报告指标与单位。AUC 0.003的绝对提高是0.3个百分点，不能写成相对提高0.3%。
2. 原始数据来源及版本；学生、交互、题目、KC数量；过滤、抽样、多KC处理和题ID定义。
3. 划分单位、固定测试集与否、validation用途、窗口切分、测试时可见历史及聚合规则。
4. 额外信息：文本、选项、时间、图、预训练表示；是否由全体交互或测试标签产生统计量。
5. 搜索次数、训练轮数、early stopping、种子/fold、参数量、硬件及训练成本。
6. 证据状态依次为“已定位论文”“已读协议”“代码已审计”“无梯度检查通过”“登记训练清单与授权依据”“验证结果”“冻结后测试”；不跳级。

研究账本保留失败复现与负结果。无法获取或无法运行的方法写明缺口，不能用猜测分数补表。

## 5. 十五天内的优先级

- 10-14至10-16：冻结三数据集定义、协议和复现锚点；提取优先候选的完整实验表。先解决泄漏和指标口径。
- 10-17至10-21：登记批次范围与预算后，按持续授权开展基线筛选；结合验证集整体AUC、短历史和长尾题目分层定位稳定失败点。
- 10-22至10-26：对最多两个原创假设做公平验证和核心消融，不预先锁定语义/IRT或复杂模块组合。
- 10-27至10-29：交付第一轮已验证结果、证据边界、成本和下一批建议。完整5-fold×3-seed确认若尚未完成，交付明确的阶段结果并延长，不用单次最好结果冒充最终结论。

每一批梯度训练仍须先记录范围、预算、预期产物和持续授权依据，遵守50轮上限与耐心3；不再逐批询问。日历日期不能代替协议审计、候选冻结或实际结果。

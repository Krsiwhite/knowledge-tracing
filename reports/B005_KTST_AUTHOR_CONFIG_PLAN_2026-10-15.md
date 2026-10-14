# B005：KTST 作者配置在共同开发协议下的对照

2026-10-15。**三项具体训练清单已准备并完成静态核验，尚未启动。** 使用作者固定提交中与本项目同为fold0的mean配置，在ASSIST2009、ASSIST2015和NIPS34各训练一次。目标是检验作者选择的架构与优化设置在共同验证协议下的表现，为后续原创模型补充更充分的比较对象；参数更多不保证成绩更好。

当前B003/R1继续执行，B004的UKT六项对照已排在其后。B005必须在检查前两批终局状态、结果与未解决故障、确认GPU空闲后接续，不与当前训练争用GPU。用户的项目内持续授权适用，无需新增逐批询问；启动前仍绑定本清单与真实授权依据。

## 来源和架构

固定官方提交`e6331a6f45e597f286d1272fbb96e7bb67bcaa52`，配置分别为[AS09 reproduce_65](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/config/reproduce/ktst/benchmark/reproduce_65.yaml)、[AS15 reproduce_15](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/config/reproduce/ktst/benchmark/reproduce_15.yaml)、[NIPS34 reproduce_29](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/config/reproduce/ktst/benchmark/reproduce_29.yaml)。已核对获取收据的文件SHA256和字节数，避免用NIPS的fold4配置代替fold0。

| 数据集 | d_model | encoder / decoder | Transformer heads | FFN / classifier | dropout | 已核验参数量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ASSIST2009 | 128 | 2 / 6 | 8 | 512 / 256 | 0.05 | 4,402,801 |
| ASSIST2015 | 256 | 3 / 4 | 8 | 512 / 256 | 0.20 | 4,969,305 |
| NIPS34 | 128 | 4 / 4 | 4 | 256 / 128 | 0.25 | 1,520,689 |

完整构造参数直接绑定已完成的[CPU参数计数报告](KTST_STRONG_CONFIG_CAPACITY_2026-10-15.md)，本轮不重复构造模型。公共设置为mean集合聚合、`use_bias_emb=true`、`use_zero_init=true`、`use_decoder_only=false`、`bias_before_agg=false`及`learnable_alibi_monotonic_q_k`；保留作者`nhead_agg=4 / num_layers_agg=2`字段，mean分支不因此增加MHSA聚合层。

数据维度沿用同一manifest：AS09为17,737题/123KC，AS15为100KC并使用既有单dummy题约定，NIPS为948题/57KC。训练仍是原200 KC分块再合并题目事件的表示及原BCEWithLogits损失，排除每个分块首事件和padding。没有改为200道完整题重新切分或新的共享事件训练损失。

## 优化、预算和停止条件

三者均用作者Adam优化器、梯度范数裁剪1.0、训练batch128、shuffle=true、drop_last=false；theta参数单独学习率，两组使用同一weight decay，无新增scheduler、AMP或梯度累积。

| 数据集 | 主学习率 | theta学习率 | weight decay | 每轮更新 | 50轮更新上限 | 单次时间软上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ASSIST2009 | 0.00005 | 0.001 | 0.00005 | 24 | 1,200 | 4小时 |
| ASSIST2015 | 0.0001 | 0.0005 | 0.00005 | 97 | 4,850 | 8小时 |
| NIPS34 | 0.0001 | 0.0005 | 0.00001 | 48 | 2,400 | 12小时 |

**合计最多8,450次更新、24小时训练与验证循环软预算，准备时间另计。** 这是上限，不是预计耗时；提前结束即分析交付。步数按已审计训练行数3,016 / 12,344 / 6,043取`ceil(rows/128) × 50`，最后一批分别为72 / 56 / 27行。既有全量审计记录三集`zero_score_chunks=0`。本次核对了当前manifest与该审计的来源摘要；启动前再校验实际数据文件完整性。

全部使用fold0、seed42、最多50轮、连续3轮完整验证AUC无严格提升早停，epoch0参与候选；并列保留更早检查点，最终导出最佳检查点的完整共同验证预测。共享验证batch32、每次前向最多32行，目标和允许历史沿用B003的v2协议。失败或预算耗尽即停止整批，不自动重试、不隐式降低batch或改网络尺寸；必要修订另行留痕并沿用持续授权。没有云费用或测试集访问。

## 比较如何解释

这是**作者配置迁移到共同开发协议**，不是精确复现论文表格。作者配置是200轮、patience10、各自seed33/78/53和分块验证batch64；本项目采用50/3/42、共同事件验证32/32及额外时间/更新上限。当前DataLoader使用独立随机数生成器，作者使用默认全局随机数路径；即使使用相同seed也不能承诺相同训练批次顺序或逐位复现。自有评测器绕过作者set_dense验证入口的既知问题，保留原模型计算。

与B003的KTST相比，本批次同时改变架构、dropout、部分优化参数和训练batch，因此结果只能说明这套配置的表现，不能将增益单独归因于网络深度，也不是相同步数/时间的消融。训练batch从32变128，每轮更新次数减少；两者都允许完整50轮，但计算预算并不相同。B005每数据集只增加一套预先选定配置，没有声称进行作者每fold100次超参搜索。

完成后先逐运行核验各自真实清单、数据/目标/历史与标签，再按已有保存预测流程比较总体AUC、NLL、Brier、Accuracy、ECE、固定分层及实际资源成本；与其他批次比较必须保留源码身份及其差异说明。单fold/seed验证结果仍不足以支持多折多seed稳定性或最终SOTA结论；最终测试须待候选、检查点及分析方案冻结。

## 冻结与核验

- [具体清单](../configs/batches/B005-ktst-author-configs-shared.json)：canonical SHA256 `01c3e0b042af462a0307727868a50997d6ff4ceca57f356f6700b2468d6d55bd`。
- 绑定研究分支实现SHA256 `7bbfc8ed9b60a6759fac2f1c2f0b5752e014cd89ed169dcb0bff57409dcd6b78`；未修改运行代码，上游锁和原数据来源保持不变。该实现包含已检查的UKT接入，与正在执行的main/R1来源分别记录。
- 前置B004 canonical SHA256 `8d4f71eeada8c15576d0aaa6bdcca374e47eacc81e9996745f613c722dcadc6b`；B003-R1 canonical SHA256 `c2c4dbc0ffcfee1bb591948e6dffac26366b723bd13d7c8e484ebff886c15bd3`。
- [本轮准备核验](B005_PREPARATION_2026-10-15.json)：三份YAML/获取收据、参数计数、当前manifest、训练行审计、共享目标与前置清单身份核对通过，现有训练器配置检查三项通过。0模型构造、前向、反向、优化器更新和权重载入；未初始化GPU，未打开序列/测试文件，未创建运行或授权记录。

本轮核验只证明清单一致和配置可表达；更大配置的实际GPU显存、运行时和模型质量须由之后的真实执行确定。准备脚本保存在忽略目录`.cache/prepare_b005.py`，其SHA256在准备记录中留存；清单本身已包含完整可执行配置。

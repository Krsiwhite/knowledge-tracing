# B006：MoC 长度路由训练适配方案（2026-10-15）

状态：**仅准备，未启动**。本轮只读真实开发训练文件并核验收据，0 模型构造、0 前向、0 梯度、0 optimizer、0 GPU；未读取测试文件或已有权重，未创建运行授权记录。

## 要回答的问题

每个数据集先运行 `mockt-three-pass-ground-seq-control-v1`，再运行 `mockt-target-prefix-branches-v1`，顺序为 ASSIST2009、ASSIST2015、NIPS34，共 6 组。control 三次调用原模型均使用整行有效长度，prefix 三次分别使用 1/51/151，再按当前目标的有效 KC 前缀长度（≤50、51–150、>150）选取分支；只加一次原完整 batch 正则项。两者均保留原全局 masked BCE。

**prefix 对 control**用于判断长度路由作用；**prefix 对原 MoC**用于判断适配总收益，包含三次前向、dropout 随机路径与成本差异。两者相同 seed 不意味着每次随机消耗完全相同。仅超过原 MoC 不算项目目标完成：还须面对同一验证协议下最强已完成基线，AS09 当前 KTST mean AUC 为 **0.7921645709548386**；UKT/B005 后续更强时更新比较对象。不预承诺三数据集均有提升。

control 仅对齐三次完整前向、选点掩码及 dropout 调用结构，不保证逐点随机掩码、FLOPs、峰值显存或墙钟相同。原 FrequencyLayer 按分组使用 kernel2/4/6；prefix 每次全行采用一个分支，control 仍按原 ground_seq 分组，卷积成本可能不同。执行后须报告实际成本，不称严格等算力；原单前向 MoC 也不是等算力对照。

这属于**训练适配与受控一致性实验，不是新机制**。仍使用 pyKT 200 个 KC 位置的训练分块、保留重复 KC 展开和分块边界；没有改成完整题事件训练，也没有换成事件级训练损失。验证继续使用完整事件历史的 v2 共享协议，训练和验证粒度差异仍在。

## 配置与预算

六组逐项深复制 main B003-R1 的三个 MoC 配置。每组唯一变化为 `run_id` 与新增 `config.mockt_training_recipe`；模型参数、学习率 0.001、batch32、dropout0.1、优化器/正则、数据 split、共享验证 target/context 身份及单组预算均原样保留。模型为 d128、2 blocks、8 heads、FF256、FC256、kernel4、`qid_conv_ker_noexp`；seq_len200、分支阈值50/150来自固定代码默认值。

固定 fold0/seed42，最多50轮，连续3轮验证 AUC 无严格提升早停（min_delta0）；每轮完整验证，epoch0 可成为合法最佳检查点，平分保留较早点。初始验证不算一次 stale。每个 recipe 只跑这一个固定配置，无额外调参搜索。

| 数据集 | 训练分块行 | 每完整轮更新 | 每 recipe 更新上限 | 每 recipe 软墙钟 | 有效 KC 长度 | prefix/full-row 分支不同的评分目标 |
|---|---:|---:|---:|---:|---:|---:|
| assist2009 | 3,016 | 95 | 4,750 | 4 h | 3–200 | 132,736/218,025 |
| assist2015 | 12,344 | 386 | 19,300 | 4 h | 3–200 | 169,770/426,156 |
| nips_task34 | 6,043 | 189 | 9,450 | 8 h | 3–200 | 627,406/892,456 |

合计更新硬上限 **67,000**；每组 ASSIST2009/ASSIST2015/NIPS34 分别4/4/8小时，双 recipe 合计 **32小时软上限**。该计时从数据/模型准备之后开始，准备另计；检查在 batch/验证边界进行，不能视为精确抢占或保证可在预算内跑完。三次原前向增加成本，目前未用真实模型测量显存和耗时。失败或墙钟耗尽即停止整批，不自动重试或缩减参数；完整第50轮正好触及更新上限可合法标记 `max_train_steps`，不等同于墙钟失败。

## 真实训练输入核验

只解析 `train_valid_sequences.csv` 中 fold1–4；fold0 行在解析序列前跳过。字节 SHA 覆盖整个开发文件，另外仅对 `train_valid.csv` 做字节收据核验，没有重新生成验证目标或运行评测。3 个数据集均为右侧 -1 padding、长度200、至少2个有效 KC，response 为二元，ground_seq 等于有效 response/KC 数；相邻位置 mask 是连续前缀，smask 是其子集，零评分目标行与越界选择均为0，适配后的 masked 输入全为0且重建有效序列逐项一致。ASSIST2015 没有题号，沿用现有 adapter 的 dummy q0、原模型 num_q=0 模式，不把 q0 当真实题号。

上述目标计数是**KC 位置数**，不是独立题事件数。表中分支差异仅衡量既有整行长度与目标前缀长度的路由条件不同，不能据此预测 AUC 改善。重复 KC 和跨200位置边界的部分题仍存在；详细计数、长度直方图和文件 SHA 见[准备审计 JSON](B006_MOCKT_PREFIX_PREPARATION_2026-10-15.json)。

## 证据与执行前置条件

S006 已完成196项人工 CPU 验收，实际2个模型、58次原模型前向、4次 autograd.grad、0 backward、0更新；这是先前检查结果，本次没有重跑。S006在 dropout0 的合成输入上验证 prefix 概率/BCE/梯度和control，不能推广为 dropout0.1 下独立前缀的逐值等价，也不证明真实数据收益。

必须等 **B003-R1、B004、B005 全部成功 completed**，保留原 B003 两条 AS09 已完成结果，不重跑；检查真实 worker/child 已退出、GPU空闲、冻结清单/源码/数据收据仍匹配后才排队。任一前置失败或预算耗尽均阻止后续启动。该准备没有声称这些条件已经成立。执行时记录2026-10-14持续本机项目授权即可，无需再次询问；付费云资源不在范围内。最终测试继续封存，待候选、检查点和分析方案冻结后才按持续授权执行。

- B006 canonical SHA256：`0c82973118722b846358ba0c2dfe437a5cc179ea09f6da070f4074d13fafc394`。
- 固定实现 SHA256：`0e9eb36f2ef023f51400ee74ff7bc3ae699aac0b52fbe1568bc73389b4cf07f0`。
- 原 R1 canonical SHA256：`c2c4dbc0ffcfee1bb591948e6dffac26366b723bd13d7c8e484ebff886c15bd3`。
- S006 canonical SHA256：`f04a5126293b635e00a53ab3d25faba04797a8efc39ad219dcbeb12393a70ea7`。
- B004 canonical SHA256：`8d4f71eeada8c15576d0aaa6bdcca374e47eacc81e9996745f613c722dcadc6b`。
- B005 canonical SHA256：`01c3e0b042af462a0307727868a50997d6ff4ceca57f356f6700b2468d6d55bd`。

[B006 清单](../configs/batches/B006-mockt-prefix-training-shared.json)中的运行顺序与全部配置已固定；后续比较必须按 `(model, mockt_training_recipe)` 区分身份，不把两组同名 MoC 合并。当前为单fold/单seed开发方案，不是SOTA、跨seed稳定性或最终测试结论。

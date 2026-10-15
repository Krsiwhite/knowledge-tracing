# KTST 历史 KC × 回答交互：最小性能实验

日期：2026-10-15。状态：**实现与 S007 合成验收完成；尚无本变体的真实训练或预测质量结果。** 本文登记历史表示的一个小改动及等参数对照，不将其命名为原创模型。真实训练骨干须在 B005 完成后，依据开发验证结果另行冻结；当前不创建训练批次，也不接触最终测试。

## 1. 为什么保留 mean 骨干

Neubauer、Rudolph、Brefeld 的 *Principled Transformers for Predictive Performance in Knowledge Tracing* 正式发表于 JEDM 18(1), 89–112，2026-02-07，DOI `10.5281/zenodo.18518653`。[期刊记录](https://jedm.educationaldatamining.org/index.php/JEDM/article/view/961)

| 数据集 | KTST mean：作者 AUC ± SD | KTST MHSA：作者 AUC ± SD |
| --- | ---: | ---: |
| ASSIST2009 | 0.7993 ± 0.0012 | 0.7871 ± 0.0022 |
| ASSIST2015 | 0.7314 ± 0.0003 | 未报告 |
| NIPS34 | 0.8071 ± 0.0000 | 未报告 |

来源：Table 1（印刷页100 / PDF第12页）及完整 Table 6（印刷页111 / PDF第23页）。这些是**作者结果**，不能与项目开发验证成绩直接判定胜负；四位小数的 SD `0.0000` 不表示真实方差为零。作者在第4.2节讨论小 KC/题目比率下 MHSA 的优化与过拟合问题，因此原生 MHSA 是可做的既有对照，但不是默认更强的升级。[正式 PDF](https://jedm.educationaldatamining.org/index.php/JEDM/article/download/961/271)

链接核对：期刊页的 PDF 按钮指向 `/download/961/271`，返回24页 PDF；HTML 按钮指向 [`/download/961/272`](https://jedm.educationaldatamining.org/index.php/JEDM/article/download/961/272)，返回同一文章的 HTML 正文。两者不是两个 PDF 版本；本文页码均按 `/271`。

## 2. 唯一改动及等参数对照

设事件 t 的去重 KC 集合为 C_t，已观察回答 r_t ∈ {0,1}。原 mean 表示为 `x_t = e_question + mean(e_c)`，历史交互为 `y_t = x_t + e_response`；共同 bias 等原运算保留。当前待预测事件只提供 query，回答只能在其后成为历史信息。[固定官方源码](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/model/KTST.py)

增加零初始化可训练表 `V ∈ R^(num_c × d_model)`，对一个事件定义 `v_t = sum(V_c, c∈C_t) / |C_t|`。比较三种设置：

| 设置 | 历史表示 | 新增参数 |
| --- | --- | ---: |
| 原 mean | `y_t` | 0 |
| `signed_response` | `y_t + (2*r_t - 1)*v_t` | `num_c*d_model` |
| `response_independent` | `y_t + v_t` | `num_c*d_model` |

后两项通过顶层 `ktst_interaction_recipe` 显式区分；缺省为原模型。控制组只让**新增项**不依赖回答，原有 `e_response` 仍保留。它检验收益是否来自新增历史 KC 参数，而不是回答条件交互。两项使用相同初始化、参数量和优化设置；不同时添加门控、辅助任务、对比损失或新的融合规则。

`V` 在原模型构造完成后由全零张量注册为参数，不抽取随机数，也不增加可训练 padding 行。只有真实 KC 参与均值；缺失 KC 槽和 padding 事件的新增项为零。新增项仅进入历史流 y，不能加入当前 query x 或分类器直接拼接的 x。保留原因果 encoder、历史右移及 logits BCE：每训练块排除首事件与 padding，一事件一次监督。新增参数进入普通学习率组；ALiBi 的 theta 分组保留。

## 3. 性能假设与新颖性边界

待检验的性能假设是：在相同 mean 骨干与数据合同下，显式的 KC 条件回答差异，可能比仅共享两个回答 embedding 更易优化。它不要求缺失的题目ID、文本或时间戳，因而适用于三集；ASSIST2015 仍使用原恒定 dummy question，不将 KC 重新命名成题目。

最近邻已经覆盖核心思想：AKT 作者代码中 `separate_qa=True` 使用 KC/回答联合嵌入，而关闭时使用 KC embedding 加两个共享回答 embedding。本实验是这类既有交互参数化向 KTST 集合历史表示的轻量适配，不能声称首次建模 KC × 回答，也不能仅凭符号乘法解释为真实学习、遗忘或心理状态。[AKT 作者实现](https://github.com/arghosh/AKT/blob/master/akt.py#L25-L82)（2026-10-15 查证）

原 KTST 后续非线性层已经能够学习交互，因此本假设也不等于证明原模型缺乏相应表达能力。性能价值与科学新颖性分别判断：若有稳定收益可以报告实用改进；当前证据不足以将该参数化作为原创机制。

## 4. 验收与训练边界

限定合成验收检查：新增表全零且不改变 RNG 状态；共享原参数相同；零残差时概率与原 logits BCE 一致；同参数控制只改变回答乘子；KC 排列和 padding 不改变目标结果；当前/未来答案不能影响对应目标，已允许历史有阳性响应；新增表可获得有限有效梯度并恰好被优化器参数组收录一次。

[S007 验收记录](../reports/S007_KTST_INTERACTION_CPU_ACCEPTANCE_2026-10-15.json)：**182 passed，0 failed / skipped / warnings**，pytest 35.92秒，进程38.00秒；实测10个 CPU d16 模型、52次完整前向、8次 `autograd.grad`，0次 optimizer / 参数更新，未读取真实数据或既有权重。代码指纹为 `3e6e65f50027bfe6ed9d396117d755b3c22506832b5599b22d55d71fdbf31180`，执行前后保持一致。检查覆盖 `num_q=0` 单KC与含题目ID的多KC两种语义、训练/eval的零残差等价、非零残差的独立原 forward 注入参考及全部参数梯度、共享预测路由和保存预测身份。

[具体清单](../configs/checks/S007-ktst-interaction-cpu.json)记录模型/前向/梯度上限12 / 96 / 8及300秒进程上限；执行前已将 source/test 哈希绑定至用户持续授权。真实模型测试需显式设置 `KTLAB_RUN_KTST_INTERACTION_CPU_ACCEPTANCE=1`，普通 pytest 不会意外运行。合成验收只证明指定配置的实现行为，不提供 AUC、真实训练成功或 SOTA 证据；无须重复运行已通过检查。

配置、模型源码、checkpoint、保存预测及比较标识必须保留 recipe 身份；变体不能标为只有 import 绑定差异的原版 KTST，也不能把原 checkpoint 重标为变体。上游源码和锁文件不改；项目本地变体的数学源码与上游来源分别可追溯。

当前接入范围是训练与开发验证。冻结的最终测试实现 `3a4b2789…` 仍只支持原版 KTST 的 vendor 身份，**尚不支持本地交互变体**；本次不能借用该最终测试入口或将变体重标为原版。将来确需最终测试时，必须显式更新兼容评测、验收新的模型身份处理，并按既定规则冻结候选、checkpoint 与分析方案。

本轮保持原200-KC分块后合并事件的训练 loader。块边界的部分 KC、首事件不监督，与 v2 完整事件后缀验证之间的差异**仍然存在**。新增残差没有完成“共同完整事件重训”，也不是原论文完整复现。后续三组必须使用相同训练输入、mask、预算、epoch0 候选和验证选择规则，不将额外训练、窗口变化或更大骨干的收益归因残差。[共同验证协议](SHARED_EVENT_PROTOCOL.md) · [作者配置与项目差异](evidence/KTST_2026-10-14.md)

2026-10-15，用户明确要求优先开展自有模型实验，原“等待 B005”顺序由此调整。[B007 首批实验](../reports/B007_KTST_INTERACTION_PILOT_2026-10-15.md) 直接沿用已完成的 ASSIST2009 KTST mean 骨干，顺序训练 `signed_response` 与 `response_independent`，复用原版完成结果；不重训或复测原版。每次最多50轮、连续3轮验证 AUC 无严格改善早停。若 `signed_response` 不能优于等参数控制，不支持回答条件交互的解释；若仅单折偶然提高，保留探索性结论。最终候选仍须按已登记跨数据集、多fold/seed及封存测试方案确认，不能由本说明预先承诺提升。

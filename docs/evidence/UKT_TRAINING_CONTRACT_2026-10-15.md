# UKT 训练接入契约：固定源码与共同验证边界

日期：2026-10-15。项目审查基点：`26514e8e8b225aa1d4aea6958164b5257e27de6c`。本轮仅静态阅读源码、下载固定提交的参考源码并写本文；没有导入模型、读取学生数据、模型推理、构造优化器或梯度操作。B003 的代码、配置及运行不因本文改变。

**结论：固定 pyKT 已包含 UKT 的模型与 CL 损失分支，但未包含作者的增强数据加载器。仅在当前 adapter 中加一次 `model(batch)`，只能得到预测或无 CL 的训练目标，不能称为完整 UKT 的增强训练。可以实施两个明确区分的模式：`no_cl` 与 `author_aug_cl`。后者需要复现作者加载器的实际增强行为，不能自行添加随机翻转或静默修正其循环。共同验证仍使用现有逐事件、当前答案不可见的上下文；保留训练目标不代表复现作者整套实验协议。**

## 1. 固定来源与可复查路径

pyKT 固定提交 `eec087e0230520e7ae7bbd4f5433d71612df62fb`；作者 UKT 固定提交 `86ad9d8f76ee2c761e116fcadc43705f66ffd0b6`。

| 证据 | 固定源码路径 / 关键行 |
|---|---|
| pyKT 模型构造、输入、输出 | [pykt/models/ukt.py L32–238](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/ukt.py#L32-L238) |
| pyKT NCE 实现 | [ukt.py L252–280](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/ukt.py#L252-L280) |
| pyKT 初始化工厂 | [pykt/models/init_model.py L147–148](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/init_model.py#L147-L148) |
| pyKT 损失及 forward 分支 | [pykt/models/train_model.py L46–53](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/train_model.py#L46-L53)、[L128–134](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/train_model.py#L128-L134)、[L288–340](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/train_model.py#L288-L340) |
| pyKT 数据工厂与普通加载器 | [pykt/datasets/init_dataset.py L212–219](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/datasets/init_dataset.py#L212-L219)、[data_loader.py L81–104 / L121–167](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/datasets/data_loader.py#L81-L167) |
| 作者增强的真实实现 | [datasets/ukt_dataloader.py L85–134](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/datasets/ukt_dataloader.py#L85-L134) |
| 作者 train/valid 加载入口 | [datasets/init_dataset.py L67–91](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/datasets/init_dataset.py#L67-L91) |
| 作者测试时关闭增强 | [datasets/init_dataset.py L34–41](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/datasets/init_dataset.py#L34-L41) |
| 作者训练与参数入口 | [models/train_model.py](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/models/train_model.py)、[train/wandb_train.py](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/train/wandb_train.py)、[train/wandb_ukt_train.py](https://github.com/UncertaintyForKnowledgeTracing/UKT/blob/86ad9d8f76ee2c761e116fcadc43705f66ffd0b6/train/wandb_ukt_train.py) |

本地参考下载目录：`.cache/research/ukt_86ad9d8_training/`，保持上表作者源码的目录层级；旧模型/训练参考另在 `.cache/research/ukt_86ad9d8/`。缓存仅供审查，不是生产依赖或供应链锁定文件。

本次核验的 SHA256：

| 文件 | SHA256 |
|---|---|
| pyKT `pykt/models/ukt.py` | `edf78e409e19d7aaf70d8450bab8e10254928cd72f224bac1974b12a0e7c4d55` |
| pyKT `pykt/models/train_model.py` | `70e16b6ab8f81a2c78d61990af2c959ae148a17aedb5c921a38dcde3129d50de` |
| pyKT `pykt/datasets/init_dataset.py` | `181412f43afcfb36dd5c771003b8b1e99aa0aeec4eafc5dc425d61da8905b16c` |
| pyKT `pykt/datasets/data_loader.py` | `711ce97aaac210f9ac341e5e135b233cb87a646a2a9fee13bc35b43acdebca29` |
| 作者 `datasets/ukt_dataloader.py` | `e6af191d9e814db82d5c4db63e77cf34ac07ee9cb737ec0980c2e112ac322579` |
| 作者 `datasets/init_dataset.py` | `b58d46daf2e24118e47f48b99461a817c73f097ce192a0311713c47f23b44218` |
| 作者 `train/wandb_train.py` | `49bc083cbfe0943d1666305f283ca868088012491d2997d33e6e875fdd5443e1` |

## 2. 模型初始化必须显式确定的参数

工厂的实际调用是 `UKT(num_c, num_q, **model_config, emb_type=emb_type, emb_path=emb_path)`。这里 `n_question` 是 KC 数量，`n_pid` 才是题目数量。AS15 没有独立题号，应为 `n_pid=0`，仍提供形状正确的全零 `qseqs/shft_qseqs`；不能把 KC ID 重新解释为独立题号。该占位是接口适配，不新增题目参数。

当前 `build_model` 的公共默认 `emb_type="qid"` **不能原样用于 UKT**。UKT 的主要随机分布配置应显式记录 `emb_type="stoc_qid"`、`atten_type="w2"`、`use_CL`、`use_uncertainty_aug`、`cl_weight`、`separate_qa`、`seq_len=200`，以及维度、层数、dropout。注意实际 attention 头数由 `num_attn_heads` 决定，签名中的 `nheads` 不是该处的控制项。

构造函数默认 `use_CL=True`，但作者与 pyKT 的 `wandb_ukt_train.py` 均默认 `--use_CL=0`，`--use_uncertainty_aug=1`。所以“使用默认 UKT”没有唯一训练含义，必须把解析后的参数写入配置/元数据。原始 `qid` 也是不同分支：CL 只用均值替代协方差，预测拼接亦不同，不能与 `stoc_qid` 混名。`use_mean_cov_diff`、`loss1/2/3`、`num_layers`、`nheads`、`start` 等签名参数在该固定模型中没有相应有效控制路径；不应把这些字段当作已生效的消融。

`l2` 虽存为 `self.l2`，该 UKT forward 不返回 AKT 式题目难度正则项；不能自行加入 `l2 * difficulty_norm`。作者训练脚本明确给 UKT 配置 Adam `weight_decay=1e-5`（L138–140），固定 pyKT 脚本则落入普通 Adam，默认 `weight_decay=0`（`examples/wandb_train.py` L147–154）。这两种来源的优化器设置需要分开记录，不能从模型的 `l2` 推导出实际 weight decay。

## 3. `dcur` 与返回值契约

令完整窗口长度 `T=200`，批量大小 `B`。进入模型前所有张量都已放到同一设备；模型内部 `.long()` 不会替调用方移动设备。

| 字段 | 形状 / 内容 | 需要它的路径 |
|---|---|---|
| `cseqs`, `shft_cseqs` | `[B,T-1]`，KC 整数 ID，分别对应窗口 `[:-1]` 与 `[1:]` | 所有 forward |
| `qseqs`, `shft_qseqs` | 同形状，题目 ID；AS15 为零占位 | 所有 forward，`n_pid=0` 时不使用题目参数 |
| `rseqs`, `shft_rseqs` | 同形状，二元原始响应；现有 loader 为 float32，模型转换 long | 所有 forward / BCE 标签 |
| `masks` | `[B,T-1]` bool，相邻两位置均为非 padding | forward 总会读；CL 池化实际使用 |
| `smasks` | `[B,T-1]` bool，监督目标选择掩码 | 外部 BCE，不是模型 forward 的输入依赖 |
| `r_aug`, `shft_r_aug` | `[B,T-1]`，作者规则得到的响应视图 | 仅 `train=True && use_CL && use_uncertainty_aug` |

模型按 `cat([seq[:,:1], shft_seq], dim=1)` 重建 `[B,T]`，不是把两个 `[B,T-1]` 直接拼起来。`tseqs/utseqs/ground_seq/qidxs` 对 UKT 数学本身没有作用；官方通用 `model_forward` 仍先访问时间字段，因此直接调用那整个通用函数需要额外空字段。推荐现有私有 import + 小 adapter，不为了复用工厂而导入所有 `pykt.models`。

| 调用 | 返回值 |
|---|---|
| `model(dcur, train=False, qtest=False)` | `preds[B,T]` |
| `model(dcur, train=False, qtest=True)` | `(preds[B,T], concat_q[B,T,4*d_model])` |
| `model(dcur, train=True)` 且 `use_CL=False` | `(preds[B,T], 0, 0)` |
| `model(dcur, train=True)` 且 `use_CL=True` | `(preds[B,T], cl_loss scalar, 0, 0, temp[B])` |

`model.train()/eval()` 控制 dropout；显式 `train` 参数控制是否计算第二视图和 CL，二者不是同一个开关。评测应同时 `eval()`、`train=False`、无梯度。训练应同时训练模式、`train=True`。仅设置 `model.train()` 后调用默认 `model(dcur)` 不会计算 CL。

`temp` 是对 `ELU(cov_d_output)+1` 在特征与时间维度取均值所得的每窗口统计（L217–219），时间平均没有 padding mask；它不参与 `cal_loss`，不是每题预测概率，也不能直接当作已校准置信度。

## 4. 作者增强的逐项语义

作者 `UKTDataset.__load_data__` 先按传入 folds 过滤行，然后产生普通序列张量。`__getitem__` L85–99 先取相邻有效 `masks`，将原序列和 shifted 序列分别乘它，使 padding 对应输入为 0。随后 L101–123 克隆响应再增强；不访问其他学生、不统计全数据正确率，也没有随机采样。

对于**至少两个有效位置、有效区为左侧连续前缀**的二元响应窗口，原源码的确切效果为：

1. 取 `shft_rseqs[masks][-1]` 作为末尾有效原响应 `z`。
2. 若首个原响应等于 `z`，翻转 `r_aug[0]`。
3. 循环的条件始终检查 `shft_r_aug[0]`，不是 `shft_r_aug[i]`。如果该首位等于 `z`，`i=0` 时翻转它，此后条件变假；如果开始就不等于 `z`，整个循环不翻转任何位置。
4. 模型只取 `r_aug` 首位与完整 `shft_r_aug` 重建增强视图。因此**至多前两个原响应被翻转**；长度为 2 时第二个正好也是末尾响应。`r_aug` 的其他位置保持原值，不能按重建的增强窗口重新生成两份 shifted 张量并声称逐字段等价。

在上述合法输入域内，可用等价式实现批级投影：克隆两份已 masked 响应，令 `r_aug[:,0] = shft_r_aug[:,0] = 1 - z`，其他位置不动；`z` 的索引为 `masks.sum(1)-1`，这个索引公式仅对连续前缀 mask 成立。应以原作者逐字段 oracle 确认实现，而不是只比较重建视图。

例如原响应 `[1,1,0,1]` 的增强模型视图是 `[0,0,0,1]`；原响应 `[1,0,0,1]` 的增强视图也为 `[0,0,0,1]`。这不是“按概率随机翻转”，也不是“把所有与最后一题同答案的位置翻转”。将条件中的 `[0]` 改为 `[i]` 会改变已发布实现，需要另列修正版，不属于忠实移植。

全 padding 或仅一个有效位置会导致 `rshft_aug_mask[-1]` 空索引。CPU lazy adapter 应在进入增强前验证至少一个有效相邻对、无内部 padding 空洞、二元有效响应；对不满足域约束的数据明确报错，不能默默改变增强定义。对合法右 padding，增强不应改动 padding 槽。

另一个容易漏掉的差异：作者监督掩码是 `selectmasks[:,1:] != -1`；当前 ktlab 是 `(selectmasks[:,1:] == 1) & masks`。训练文件若仅含 `1/-1` 且 padding 对齐时二者等价；含 `0` 排除标记则不等价。原样比较 fixture 必须揭示这个边界，不能为追求 oracle 相同而破坏项目的明确排除语义。

作者工厂 train/valid 均实例化 `UKTDataset`（默认 `need_aug=True`）；测试实例化 `need_aug=False`。作者 valid 虽生成增强字段，标准预测 `train=False` 不使用它们。项目共同验证无需生成增强字段，当前答案占位 0 也绝不能通过增强逻辑换成真实标签。

## 5. 完整训练目标及不能悄悄改动的细节

固定 pyKT `train_model.model_forward` 对 UKT 显式调用 `train=True`，将 `preds[:,1:]` 对齐到 `shft_rseqs`。`cal_loss` 的目标为：

```text
loss = BCE(preds[:,1:][smasks].double(), shft_rseqs[smasks].double())
       + cl_weight * cl_loss                 # only when use_CL is true
```

没有增强响应的第二份 BCE；监督标签始终是原响应。`y2/y3` 是 0 占位；`temp` 只被向外传递，未加到目标。将 `regularizer=cl_weight*cl_loss` 作为现有 adapter 的第二返回项，可直接复用现有引擎的 BCE 相加接口，不能再在 engine 中重复加一次。无需 detach 该项。

CL 计算两次同一个 architecture，第二次使用增强交互嵌入；`use_uncertainty_aug=False` 时 `target_aug=target`，仍做第二次前向，训练 dropout 可使两视图不同。CL 的池化 mask 是 `cat([True, masks])`，然后 `torch.mean(output * mask, dim=1)`：分母是固定窗口 `T`，不是有效长度。因此动态裁短窗口、改为 masked sum / valid count，都会改变损失。`smasks` 不参与 CL 池化。

NCE 使用温度 1、批内两视图的均值与协方差，协方差先做 `ELU+1`；同视图对角线置负无穷。固定实现里跨视图 `sim12` 对 `1/(1+distance)` 取负号，而同视图没有这个负号（L269–278）。不在接入时重写为常见 InfoNCE 的符号、温度、归一化或拼接。批大小/顺序/是否丢弃末批会改变对比目标；不能将 `max_rows_per_forward` 的推理拆批直接套到 CL 训练。

作者训练批量默认 64，两个工厂的 `DataLoader` 均未启用 shuffle、drop_last（即 False）；具体实验可选不同预算，但要记录。作者 `Adam(weight_decay=1e-5)` 与 pyKT 默认 `Adam(weight_decay=0)` 的区别见第 2 节；原 UKT 分支没有专门梯度裁剪。项目若使用已有 `grad_clip`，应作为训练协议差异披露。

## 6. 两种可运行模式与因果边界

建议保持底层 `model="ukt"`，在冻结配置、日志和分析标签中显式增加训练模式，不靠另起模型名隐藏开关：

| 模式建议 | 约束 / 允许的表述 |
|---|---|
| `ukt_training_mode="no_cl"` | `use_CL=False`；无增强数据依赖；BCE 训练。表述为“UKT 无 CL 消融 / pyKT 脚本默认 CL 关闭配置”，不能称完整增强目标。 |
| `ukt_training_mode="author_aug_cl"` | `use_CL=True, use_uncertainty_aug=True`；固定作者增强算法、原模型 CL 和 BCE；建议作者 Adam weight decay；记录作者数据脚本 commit/SHA。表述为“固定 pyKT 模型 + 作者增强/损失行为，在项目共同验证协议下训练”。 |

`use_CL=True, use_uncertainty_aug=False` 是可定义的第三种消融，但不属于以上两个模式，若实施须独立命名/显式解析。让 `cl_weight=0` 但保留额外视图也不应被自动等同于 `no_cl`，因为 dropout 随机数消耗和计算路径不同。

作者增强读取训练窗口最后响应，是**训练辅助目标使用完整训练窗口**，不是证明评测答案泄漏。正常预测主视图由未增强的原响应构造，第二视图只进入 CL；共同验证 `train=False` 根本不访问增强字段。相应边界是：不能声称 CL 中的每一个中间状态也只受其局部过去监督，不能把这一训练增强用于在线预测时补全未知末尾答案。

UKT architecture 的 retrieval 使用严格过去 mask（`ukt.py` L352、L405–408）。但严格 token 因果性不自动保证多 KC 事件因果性：KC 展开训练行中，同一次题目作答的前一个 KC 可能携带同一个答案。项目的共同验证通过每个目标 KC 独立查询、目标 siblings 全部不进入历史来处理此问题；不应声称它同时改变了既有训练数据表示。

共同验证保留固定 T=200、相同完整事件历史和 query KC 集，每个目标 KC 当前响应固定 0，取目标位置概率后 full precision late mean。只用 `model(batch, train=False)`，既不生成 `r_aug`，也不计算 CL。AS15 只支持 KC 语义，不能用于论证 UKT 题目难度参数的效果。原作者/pyKT 的 checkpoint 选择为 `auc > best + 1e-3`、10 轮无改进、无 epoch0（pyKT `train_model.py` L357–377）；项目应保留共同验证的 epoch0 + strict best + 当前预算早停规则，并明确这是统一协议比较，不是作者报告数值的精确复现。

## 7. 当前代码可直接实施的最小改动

以下仅为实施接口，本文未修改这些代码。

| 位置（项目基点 26514e8） | 最小改动 |
|---|---|
| `src/ktlab/adapters.py` `SUPPORTED_MODELS` / `build_model` L22、L66–103 | 增加 `ukt:UKT`；使用既有隔离 import/device binding；明确 `stoc_qid`、CL 开关与 `seq_len`；记录解析后 model_config 和既有模型源 hash。拒绝模式/开关矛盾。 |
| `SequenceDataset` L180–220 或独立轻量 wrapper | 对训练 `author_aug_cl` 的每条 CPU item，在现有 masked 原响应上增加两字段；复现作者逐字段行为、不改原响应、不生成 CUDA 缓存。`no_cl`、共同验证不生成增强。校验合法窗口域。 |
| `forward_model` L228–248 | UKT 训练时调用 `train=True`，按 use_CL 解包三项/五项，返回 `preds[:,1:]` 与 `cl_weight*cl_loss`（无 CL 则 0）；评测显式 `train=False`。训练状态与参数分支须一致，不能评测时触发需要增强的路径。 |
| `src/ktlab/engine.py` `dataset_for_split` L95–119 / train L308–311 | 给训练 dataset 传解析后的增强模式；保留现有 train fold 文件和监督损失接口；配置记录增强来源、模式、batch、shuffle、weight_decay，不沿用不明确默认。共同验证构造不变。 |
| `engine._shared_config` L140–167 | 允许 `ukt` 进入已经封锁 test 的共同验证路径；保留 epoch0、strict best、完整 validation target/context hash 与源码身份绑定。 |
| `src/ktlab/shared_adapters.py` L23、L275–289 | 允许 UKT 并检查 positional capacity ≥200；现有 `collate_shared_kc` 已满足 UKT 推理字段；新增显式 `model(batch, train=False)` 分支更清楚。取完整 `[B,T]` 的 target_positions，不再 shift 一次。 |
| checkpoint / provenance /分析标签 | 配置指纹必须覆盖训练模式、增强来源及所有解析后模型/优化器字段；新增生产实现须进入已有源码指纹。不得只依赖未跟踪 `.cache` 运行。用单独 run_id 标识模式，避免把两个 UKT 模式误合并为同一个重复模型结果。 |

可以保留现有 `forward_model` 二元返回契约；训练与评测分支可以由明确的调用参数控制，或在此 adapter 内按 `model.training` 分派，但应在测试中覆盖。不要把 CL 当可忽略的普通诊断项，不要仅添加 shared inference allowlist 就宣称训练已经接通。推理 `max_rows_per_forward=32` 与训练 CL batch size 是两个不同设置。

## 8. 实施后验收清单（本轮未执行）

1. **作者数据 oracle**：只用人工 CSV，CPU 隔离导入原 `UKTDataset`，与 lazy adapter 逐字段比较原/shifted q/c/r、masks、smasks、r_aug、shft_r_aug。覆盖末尾 0/1、前两个响应各组合、长度 2、完整 T=200、右 padding；明确 0 selectmask 与 AS15 dummy q 的已声明适配差异。单有效位置/内部空洞应在 adapter 边界被拒绝。
2. **损失公式 oracle**：人工至少两行 batch、固定权重、dropout=0；核对 use_CL on/off 的返回 arity、shape、有限值，以及项目总 loss 等于原始 `BCE(double)+cl_weight*sim`，不增加 temp/y2/y3/L2。更改仅增强字段应允许 CL 变化，同时不改变主预测；不要求训练 dropout 打开时 train=True/False 两条路径逐位相等。
3. **设备和隔离**：CPU 全字段可运行、不因机器有 GPU 就创建 GPU dataset；私有 namespace 不污染全局 pyKT 导入；之后在资源空闲时单独验证 GPU parity。原文件 hash 不变。
4. **共同验证因果性**：翻转独立目标 label 后投影逐字节不变；手动翻 query dummy、未来/右 padding 响应不改变当前概率；历史响应改变可改变预测；多 KC siblings 不入历史；同题重复事件保持不同 event_id；AS15 n_pid=0。补充非零题目难度权重 fixture，避免仅凭零初始化题目分支做结论。
5. **训练接线但不训练的测试**：mock forward/optimizer，证明 epoch0 valid 在优化器构造前、CL 模式训练字段完整、valid 不请求增强、源码/配置/增强模式改变时拒绝重标旧 checkpoint。真正梯度行为需后续单独执行并报告，不能由 forward-only 检查推断。

既有 `reports/UKT_CAUSALITY_AUDIT_2026-10-14.json` 只覆盖两固定来源的小型 CPU 合成推理；该报告列明未测试训练损失、增强/CL、真实学生、多 KC 映射或训练后权重。本卡对训练接口的判断属于固定源码静态证据，不能将旧推理检查升级成“完整 UKT 训练复现已验证”。

# KTST 官方实现审计与接入计划

核查日期：2026-10-14。本文是源码审计和待实现计划；没有下载第二套训练数据、安装 KTST 环境、运行 KTST 前向或执行任何训练。

建议把 **KTST (mean)** 纳入正式基线，在 ASSIST2009 上再加入 **KTST (MHSA)**。它是真正按题目事件建模的模型，能检验研究改进是否优于更合理的多知识点表示。当前 `ktlab` 尚未支持它，不能把增加模型名称到配置文件视为完成接入。

## 版本、论文和许可证

官方仓库默认分支为 `development`，本次固定提交为 [`e6331a6f45e597f286d1272fbb96e7bb67bcaa52`](https://github.com/kainbr/kt_set_transformers/commit/e6331a6f45e597f286d1272fbb96e7bb67bcaa52)，提交日期 2026-05-19。仓库采用 [MIT License](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/LICENSE)，版权行为 2026 ml3；引入源码时应保留许可和版权声明。

论文 *Principled Transformers for Predictive Performance in Knowledge Tracing* 已于 2026-02-07 正式发表于 JEDM 18(1), 89–112，作者 Kai Neubauer、Yannick Rudolph、Ulf Brefeld。仓库 README 的“accepted”表述落后于期刊页面。[期刊记录](https://jedm.educationaldatamining.org/index.php/JEDM/article/view/961)

该论文是值得复现的近期强基线；本次没有穷尽检索所有新论文，因此不把它直接标为“截至今天所有数据集的绝对 SOTA”。

## 数据可复用，但表示不能照搬当前适配器

KTST 仓库包含一份 `pykt-toolkit/` 源码目录。对以下 Git blob 的比对表明，首批三个数据集的预处理和划分实现与本项目固定的 pyKT `eec087e0230520e7ae7bbd4f5433d71612df62fb` 完全相同：

| 相对 pyKT 路径 | 两份源码相同的 blob SHA |
| --- | --- |
| `pykt/preprocess/assist2009_preprocess.py` | `50217aa76c11071bb827806d054c0d4113b78bfb` |
| `pykt/preprocess/assist2015_preprocess.py` | `3199459c5c4f613baacc8f62035d45fbf250271c` |
| `pykt/preprocess/nips_task34_preprocess.py` | `d75ea3cf8c32184c804b7ca12747ec06d52000d8` |
| `pykt/preprocess/split_datasets.py` | `95e5e6b0e35dc329a1dcc479f3e1f7d3dbb689cb` |
| `pykt/preprocess/split_datasets_que.py` | `0edac36acb5df367aa4f397ec8a22788bf8f3e40` |

这证明相关源码相同，不证明任何现成数据文件或原论文实际运行产物的哈希相同。接入时仍须核对原始文件、学生集合、fold 和产物哈希。可复用本项目已准备的数据，不必重新下载大型原始文件。[KTST 的预处理入口](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/preprocess.py)

官方训练/验证读取 `train_valid_sequences.csv`，按 `fold != val_fold_idx` / `fold == val_fold_idx` 分组。先沿用 pyKT 的最多 200 个 KC 位置分块，再依据 `is_repeat` 把同题多个 KC 合成一个事件；不是直接把每个学生重新切成 200 道题。一个分块若以某题的后续 KC 开始，代码把该块首个 `is_repeat` 强制设为 false，因此边界可能保留不完整的题目知识点集合。严格复现轨应保留并统计这个行为。[数据实现](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/data/pykt/__init__.py)

`set_dense` 的 CPU 整数张量为 `[batch, sequence, max_concepts + 2]`：前面的槽位是去重后的 KC 集合，倒数第二列是题号，最后一列是答案；缺失 KC 和 padding 为 -1。有效位置的 mask 为 true。ASSIST2015 没有题号时，官方使用一个常量 dummy question（编号 0，`num_questions=1`），它不是恢复出的真实题目 ID。禁止对每个 KC 独立输出后做 late mean 来冒充 KTST 的题级输出。

`combinatorial_dense` 是另一种变体：将整个 KC 集合编码成一个类别。官方 `preprocess.py:230–279` 从训练/验证和测试的概念集合共同建立词表，并按 Python set 的枚举次序分配 ID。这属于接触测试协变量的词表构造，不是答案统计泄漏，但不符合更严格的仅训练集建词表协议。第一阶段不接入此变体；以后若使用，需固定映射文件哈希并明确 transductive 设定，或单列训练集词表/OOV 修正版。

## 模型与训练语义

核心类为 [`pkg/model/KTST.py::Model`](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/model/KTST.py)。`q_mean_c` 聚合题目 embedding 与 KC 均值；`self_attn_all` 对题目和 KC 集合做 MHSA 聚合。默认使用 `learnable_alibi_monotonic_q_k`，题目 embedding 零初始化，带 bias embedding。

同题的所有 KC 先聚合为一个事件。答案 embedding 只进入历史流，因果 encoder 的输出向右移动一位后才送给 decoder（`Transformer.forward`、`_shift_with_start_token`，约 213–240 行）。因此预测当前题时不需要暴露该题答案给后续 KC。这个性质来自输入事件定义和历史移位，不来自输出阶段的融合。仍应以实际扰动测试验证实现。

官方 `forward(data, padding_mask)` 返回 `(probabilities, loss, 0.0)`。训练 loss 是 logits 上的 `BCEWithLogitsLoss`，排除每段第一个事件及 padding。接入时应保留该 loss，不能让当前引擎统一重算 probability BCE 而不披露数值实现差异。当前引擎应扩展为允许模型提供明确的训练 loss，并单独导出概率用于指标。

[`train.py:60–88`](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/train.py) 将参数名以 `thetas` 结尾的 ALiBi 参数划入独立学习率组。原版使用 Adam，常见主学习率为 5e-5/1e-4，theta 学习率为 1e-4/5e-4/1e-3，并保留 weight decay。当前引擎的单参数组 Adam 不能直接用于严格复现。

官方 trainer 在首次参数更新前验证并保存可能的最佳初始 checkpoint；其后按每轮验证 AUC 的严格改善保存，默认最多 200 epoch、patience 10、梯度裁剪 1。训练集 shuffle=true。当前 `ktlab` 从首轮训练后开始选 checkpoint，且 shuffle 由配置明确控制，这些差别需要写入协议。[trainer](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/logic/trainer.py)

## 已确认的上游执行问题

固定版本 `pkg/logic/trainer.py::Trainer.eval` 仅在 `self.data.format == "flat"` 分支中赋值 `y_pred_pre_q`，但在分支外无条件执行 `y_scores_pre_q.append(y_pred_pre_q...)`，之后也无条件计算对应指标。默认 `set_dense` 和 `combinatorial_dense` 因而会出现未赋值局部变量错误。这是静态源码可确定的控制流问题；本次没有执行训练来触发它。

应使用项目自己的验证/记录器包裹原模型，并明确记录“绕过上游 trainer 的无关 flat 指标分支”；不必改动模型数学实现。若以后需要完整官方脚本复现，应单独提交最小补丁及回归测试，不能隐瞒该修复或把未经运行的仓库称为开箱可用。

可选 attention 变体 `learnable_alibi_monotonic_shared_q_k` 使用模块级 `SharedTheta` 单例，可能在同一 Python 进程的多个模型实例间共享参数。官方 benchmark 的常用配置没有 `shared`。最小接入固定非 shared 变体，每个运行使用新进程，避免跨运行状态污染。[attention 实现](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/model/utils/attentions/LearnableALiBiMonotonicAttention.py)

## 验证与测试协议不能混入一个榜单

| 项目 | 官方 KTST 实现 | 当前 ktlab v1 |
| --- | --- | --- |
| 训练表示 | 200-KC 分块内按题合并，通常每题一次 loss | 重复 KC 序列训练 |
| 验证 | 同样分块内按题合并，跳过每段首题 | 按目标题构造无当前答案的因果历史，KC 概率 late mean |
| 测试 | `test_window_sequences_quelevel.csv` | 非滑窗的 `test_question_file`，AS15 使用对应概念序列 |
| 历史预算 | KTST 重新生成滑窗，整窗累计最多 200 个 KC | 当前固定的 pyKT 非滑窗文件语义 |
| 初始 checkpoint | 更新前也可成为最佳 | 当前从训练后验证开始 |
| 输出单元 | 每道题直接一个概率 | 同题 KC 分数聚合 |

KTST 的 `preprocess.py:161–228` 会覆盖 pyKT 生成的题级测试滑窗：窗口长度按 KC 数累计，窗口中尽量保留完整题目；初始窗口评分多个位置，后续窗口只评分最后一个位置。测试读取器还排除每行首位。这个修正必须保留在独立产物中，不能覆盖本项目已经固定且带哈希的 pyKT 文件。[测试读取器](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/data/pykt/eval_dataset_format_set_dense.py)

建议保留两条明确命名的轨道：

1. **KTST 参考复现轨**：保留其表示、分块验证、theta 学习率、初始 checkpoint 候选和修正滑窗测试。与论文比较时逐项列出剩余偏差。
2. **本项目统一比较轨**：所有模型使用同一学生划分、目标事件集合、允许历史和历史预算；由共同题目事件清单分别派生 KTST 集合输入与其他模型的 KC 输入。保留各模型的原训练表示，并披露 loss 单元不同。只有目标/历史审计通过后才能与当前 v1 合并比较，否则新命名协议。

不能仅将已有因果验证 CSV 按行送入 KTST：该文件的同题 KC 可能分散在多个样本，各自拥有重复的历史；必须从原始事件或带事件标识的中间数据构造正确的集合。测试仍受项目独立解封规则约束。调试阶段只使用训练/验证切片。

## 依赖、AMD 和内存

官方 [`env.yaml`](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/env.yaml) 使用 Python 3.11、PyTorch 2.2、NumPy 1.26.4，并包含 NVIDIA CUDA 11.8、Hydra/Optuna/Ray、TensorBoard 和画图工具。不要在本机照搬这个 conda 环境或安装 CUDA 11.8。

只包装核心模型需要现有 PyTorch 和 Python 标准库；数据/指标需要项目已有的 NumPy、pandas、scikit-learn。完整官方命令行才需要额外 Hydra/OmegaConf 等工具。核心模型与自定义 attention 未发现 Mamba、Triton 或自定义 CUDA 编译扩展，因而静态审计没有发现 NVIDIA 专属硬阻碍；这不等于已验证 ROCm 可运行。

主要兼容性风险是自定义 attention 调用了 `torch.nn.functional` 的私有方法 `_canonical_mask`、`_none_or_dtype`、`_mha_shape_check` 等。必须在当前 PyTorch/ROCm 下做 CPU/GPU 前向、因果性和保存重载验证，不应为绕过错误先替换 attention。[自定义 attention](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/model/utils/attentions/__init__.py)

官方训练与测试读取器会把所有序列堆成完整 CPU Tensor。对于大数据集，这可能成为主机内存和构建时间问题。项目适配器应维持按行索引、按 batch 组装的方式，并用小样本对齐原转换函数输出；不要在 GPU 缓存整个数据集。默认 batch128 能否放入 16GB 显存尚未测量，不能预先保证。

## 最小实施顺序与验收

1. 固定源码、许可证和文件哈希，按清单引入核心模型及其依赖，不复制另一套大型数据。上游使用绝对 `pkg.*` imports，优先在每个运行的专用子进程中加载独立源目录；不能把通用 `pkg` 全局别名混入已有长寿命进程后依赖偶然导入顺序。
2. 新建题目事件数据适配器，支持 `set_dense`、ASSIST2015 dummy question、原200-KC训练分块和可审计事件 ID。测试构造器先只用人工样例和验证折，完整测试产物访问继续由解封规则约束。
3. 扩展模型注册、forward 返回值、模型自带 loss、两个 Adam 参数组、按题预测导出及协议元数据；保留实际输入文件、上游源码、适配器、配置和 checkpoint 哈希。
4. 前向验收：输入转换与官方函数逐项相等；同题 KC 排列不变性；当前/未来答案扰动；未来内容与 padding 扰动；CPU/GPU 容差；保存重载；无当前题答案进历史；目标事件与其他基线一致。对 mean/MHSA 分别检查，不以一个变体的通过代替全部。
5. 所有梯度更新继续等待具体训练批次批准。先提议一个受限 smoke 批次测量显存、步时和完整验证耗时，再确定可承受的正式预算；调试分数不进入论文结果。

可直接作为参数来源的官方配置有：ASSIST09 第0折 [`reproduce_65.yaml`](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/config/reproduce/ktst/benchmark/reproduce_65.yaml)，ASSIST15 第0折 [`reproduce_15.yaml`](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/config/reproduce/ktst/benchmark/reproduce_15.yaml)，NIPS34 第4折 [`reproduce_50.yaml`](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/config/reproduce/ktst/benchmark/reproduce_50.yaml)。这些是不同折、不同种子的配置，不应不加说明地改成第0折后称为原论文精确复现。

工程估计为 2–3 个工作日完成最小适配、协议审计与前向验收；这是计划估计，不是已测耗时。正式训练时间需获批的计时批次确认。官方搜索空间是每折100个 trial，200个 epoch 上限；论文披露完整调参约需1200 GPU小时，所以15天单卡计划应优先使用公开选定配置并明确有限调参预算。[官方搜索空间](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/config/sweep/ktst/search_space.yaml)、[论文附录 A](https://jedm.educationaldatamining.org/index.php/JEDM/article/download/961/272)

接入的硬性前置条件是：绕过或最小修复上游 trainer 的未赋值变量错误；实现真正的题目事件数据路径；明确验证/测试轨道；验证私有 PyTorch API 在本机可用。满足这些条件前，不应启动 KTST 正式训练或宣称已复现其结果。

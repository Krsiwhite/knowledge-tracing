# Q-MCKT 题目频次、正确率与 q_pos_neg 数据来源审计

核查日期：2026-10-14。固定作者仓库版本：`rattlesnakey/Q-MCKT@ee1698638888c5cddd577326aa30cadd5538f42d`。本轮仅静态阅读远端源码和论文；未导入或执行作者代码，未训练、推理或构造优化器，未访问本地学生数据。下面所有源码链接均固定到该 commit，行号为原文件的一基行号。

**执行状态更新（2026-10-15）：** B003 已于2026-10-14 23:56（香港时间）启动，本次更新时正在首组 AKT / AS09，尚无终局结果；最多50轮、连续3轮验证 AUC 无严格改善早停，项目内持续授权生效，无需再次逐批批准。Q-MCKT 不在 B003 内，本页原静态审计事实不变。

## 结论与证据边界

**已闭环定位生成代码。标准五折调用在无陈旧缓存的重建路径中，只用当前训练 folds 的响应计算题目频次、正确率及正负题关系；没有发现该路径将验证或测试响应混入这些统计的证据。** `q_pos_neg` 由 `KTQueDataset.__load_data__` 直接生成，并非缺失的外部生成脚本。关键顺序是：训练入口传入其余四折 → CSV 先按 fold 筛选 → 从筛选结果累计统计 → 将训练 dataset 的关系表传给模型。[调用与返回：init_dataset.py L104–134、L185–188][init-train]，[筛选与累计：que_data_loader.py L139–208][loader-count]

这个结论是**代码级 A 证据**，不是作者全部已发表实验的运行追溯证明。已有缓存缺少数据/配置指纹；论文使用的实际缓存、文件内容、统计分箱阈值的选择数据均未获得运行证据。因此不能把“标准重建路径隔离”升级为“所有历史结果已确认无泄漏”。也不能仅凭论文中的 global accuracy 措辞判定泄漏。

接入仍有两项实质取舍：**原代码频次来自切块后保留的题目序列，而非全部原始训练事件；负例没有强制高频过滤，与论文描述存在差异。** 后续复现必须明确遵循固定代码还是论文规则，不应悄悄修改后沿用同一结果标签。

本页证据等级：A = 固定源码可直接追踪；B = 原文陈述但实际运行来源未证明；C = 本审计推论或接入建议。没有使用任何本地模型分数作证据。

## 1. 完整读写路径

| 环节 | 固定版本的实际行为 | 证据 |
| --- | --- | --- |
| 原始数据入口 | `examples/src/data_preprocess.py` 指定数据路径；AS09 为 `skill_builder_data_corrected_collapsed.csv`，NIPS34 为 `train_task_3_4.csv`。`process_raw_data` 写中间 `data.txt`，随后分别运行 concept 和 question 两条预处理。这里只追踪 question 路径。 | [入口 L7–24、L38–53][pre-entry]；[分派 L3–9、L20–21、L44–50][raw-route] |
| 问题序列划分 | `split_datasets_que.main` 从 `data.txt` 读学生序列，先建立全量 ID 字典，再按学生序列行分出 20% test；余下 80% 分五折。先写未切块 `train_valid_quelevel.csv`，再写切块 `train_valid_sequences_quelevel.csv`。测试写独立文件并标 fold=-1。 | [split_datasets_que.py L145–200][que-split]；[train_test_split/KFold_split L175–212][fold-split] |
| 数据配置 | `train_valid_file_quelevel` 指向切块文件；不是 `train_valid_original_file_quelevel` 指向的未切块文件。AS09、NIPS34 的 folds 均为 0…4。 | [AS09 配置 L66–100][config-as09]；[NIPS34 配置 L301–335][config-nips] |
| 当前训练/验证选择 | `init_dataset4train(..., i, ...)`：验证用 `{i}`，训练用 `set(data_config['folds']) - {i}`，两者读同一 train_valid 文件。CSV 虽先整体读入内存，但用于关系表的行已通过 `df['fold'].isin(folds)` 筛选。 | [init_dataset.py L104–134][init-train]；[loader L139–176][loader-count] |
| 关系表生成 | 仅 `len(folds)>1` 时累计 `q_acc`、KC 分组及正负题；标准五折下这对应训练四折。单折验证和 `[-1]` 测试不会生成该表。 | [loader L171–208][loader-count]；[测试调用 L51–57][init-test] |
| 关系表传入模型 | `init_dataset4train` 返回 `curtrain.q_pos_neg`；`wandb_train.main` 将其放进 kwargs；`init_model` 传给 `QIKTPlus`，内部 `QIKT_Plus_Net` 保存该字典。 | [init_dataset.py L185–188][init-return]；[wandb_train.py L73–83、L119][train-pass]；[init_model.py L99–101][model-pass]；[qikt_plus.py L126–143][model-store] |
| 模型消费 | 训练态从当前 batch 的题号取 unique anchors，按字典取所有正负题 embedding；不是每次重新读取 CSV 或重新拟合正确率。无表项题目跳过；eval 分支不构造这些 pairs。 | [qikt_plus.py L215–268][model-pairs] |

默认 `dpath` 是相对路径 `../../data/<dataset>`，按运行时工作目录解析，并非自动以仓库根目录解析。验证折为 0 时，AS09 训练缓存名称为：

```text
../../data/assist2009/train_valid_sequences_quelevel.csv_1_2_3_4_qlevel.pkl
../../data/assist2009/train_valid_sequences_quelevel.csv_1_2_3_4_q_pos_neg.json
```

折号先排序，接在输入文件路径后。两个缓存都存在才走缓存分支；此时直接读取 pickle 和 JSON，不重新筛选 CSV 或计算统计。只存在其中一个时重建；关系表非空对象（包括空字典）会写 JSON，`None` 不写。验证单折通常没有关系 JSON，因而即使已有 pickle，仍会再次预处理。这是缓存条件的行为，不是验证响应进入训练关系表的证据。[loader L36–60][loader-cache]

固定 commit 的完整 Git tree（229 个条目，`truncated=false`）没有跟踪 `*_q_pos_neg.json`、`*_qlevel.pkl` 或 `train_valid*.csv`。这只能说明源码树里没有这些运行产物，不能说明作者曾使用的外部数据包或磁盘缓存是什么。[固定 Git tree][tree]

## 2. 频次、正确率与多 KC 的精确定义

记 `S_f` 为 loader 按训练 folds 筛出、已经切块并保留下来的序列位置。代码对每个 `q != -1` 的位置累加一次：

- `N_q =` 该题在 `S_f` 中的位置数；同一学生重复作答同一题仍逐次计数，不是独立学生数或去重题数。
- `C_q =` 其中响应等于 1 的位置数；正确率为 `C_q / N_q`。
- 计数不使用 `selectmasks`，也不排除每个 chunk 的第一个事件；唯一显式跳过条件是题号为 -1。标准入口会在更早的 `read_data` 中检查响应属于 `{0,1}`，但 loader 自身不再次验证标签、对齐长度或事件身份。[累计 L184–220][loader-count]；[响应检查 L52–74][read-response]

**多 KC 不会在这条 question loader 内把同一位置再乘上 KC 个数。** 每个位置的概念字符串以 `_` 拆成列表，并补 -1 到 `max_concepts`，但 q/r 仍各一个位置。分组键是排序后的完整 padded KC 列表字符串：要求整组一致，不是仅共享任一 KC，也不是每个 KC 独立建候选池。[loader L143–166][loader-count]、[L184–196][loader-count]

这里有三个需要保存的边界：

1. KC 列表没有去重。正常的相同 KC 集合只因顺序不同会被排序统一；`[a,a,b]` 与 `[a,b]` 则不会被当作同组。-1 padding 也进入字符串键，正常固定宽度下不改变相等关系，但不能把代码称为抽象集合规范化。
2. 同一题若在不同位置出现不同 KC 组合，`kc_q_mapping` 会将它放进多个组合；`q_kc_mapping[q]` 却只记第一次见到的组合，而 `q_acc[q]` 汇总所有位置。构造该题 anchor 时采用第一个组合。源码没有一致性断言；实际数据是否存在这种情况，本轮未读数据，**未知**。
3. AS09 原始处理器在 dropna、按学生分组及排序后直接输出每行题/KC/响应；没有再按 `order_id` 或 `(student,question)` 合并去重。入口选择的是名称含 collapsed 的文件，但不能仅凭文件名保证其内容已经按真实事件去重。NIPS34 的题目 KC 元数据则先用 set 保留 Level 3 KC，学生序列仍保留每个回答行。真实重复作答与数据重复记录是否需要区分，必须以事件 ID 审计，不能简单按题号去重。[AS09 L16–40][raw-as09]；[NIPS34 L34–42、L62–78][raw-nips]

### 切块会改变统计总体

`generate_sequences` 用 `j += maxlen` 做不重叠切块；默认 maxlen=200。不足 3 个事件的末尾残段丢弃，其他残段右 padding。故标准训练输入**不会因滑动窗口重复累计历史**，但长度 201/202 的序列末尾 1/2 个响应不会进入关系统计；每个被保留 chunk 的第一个响应会进入统计，即使它不作为该 chunk 的下一步预测标签。[split_datasets_que.py L9–50][que-chunks]

`generate_window_sequences` 另用于测试窗口文件，标准 `curtrain` 不读它。若接入时擅自改为重叠窗口 CSV，关系表累计没有 mask/事件去重机制，会把重复历史多次计数；这是替换输入后的风险，不能描述成标准路径已经发生的问题。[测试窗口生成 L184–210][que-split]、[训练文件选择][init-train]

## 3. 正负题选择及论文与代码差异

| 项目 | 固定源码规则 |
| --- | --- |
| 高频候选 | `N_q >= question_frequency_threshold` 的题加入 `pos_neg_cand`。 |
| Anchor | 只处理 `N_q < threshold` 的已观测题。训练中完全未见题不在 `q_acc` 中，也不会有关系表项。 |
| 难度类别 | 直接用训练位置正确率；配置断点列表后追加 1。0 单独分入 A，其余区间左开右闭，无平滑、伪计数或后验不确定性。 |
| 正例 | 同完整 KC 组、同正确率类别，移除自身，再与高频集合相交。正例为空则整个 anchor 跳过。 |
| 负例 | 从同完整 KC 组中，按 `neg_level_mapping[anchor_level]` 收集题目。**没有与高频集合求交**，因此低频题也可能进入负例。负例可以为空。 |

上述规则可直接追踪 [loader L211–241][loader-level] 与 [L260–287][loader-pairs]。**负例未过滤高频是确定的实现事实；实际有多少低频负例，未知。** 论文 §4.4.1 则将负例写为从高频正例候选池中选择不同难度类的题，因而此处有明确的纸面与代码差异。[论文 §4.4.1][paper-pairs]

固定配置主要值如下，不能把它们说成代码自动在每折上学习的阈值：

| 数据集 | 高频阈值 | 正确率断点（代码再追加 1） |
| --- | --- | --- |
| ASSIST2009 | 12 | 0.4、0.6、0.7 |
| NIPS34 | 500 | 0.3、0.4、0.5、0.6、0.7 |
| Algebra2005 | 4 | 0.5 |
| EdNet | 22 | 0.4、0.6、0.8 |

来源：[AS09][config-as09]、[NIPS34][config-nips]、[Algebra2005 L57–64][config-algebra]、[EdNet L292–299][config-ednet]。例如 NIPS34 的类别区间与论文列出的六类一致；负类由配置明确枚举，不是运行时按距离推导。AS09/EdNet 的 B→C、C→D 等相邻类也会被列作负类；Algebra 仅两类互为负类。因此不能把“至少相隔两类”泛化为全部固定配置已经满足的性质。[论文 §4.4.1][paper-pairs]

**阈值/类别映射如何选定仍是 B 级未知。** 论文描述参考正确率分布划分类别，并描述五折实验；固定代码只提供常量，未提供按当前训练折拟合这些断点的过程或其来源记录。本轮没有证据证明作者设定常量时用了测试标签，也没有证据证明完全没用。后续可以把固定公开常量作为预先声明的外部超参数，或在训练/开发流程内重新选择，但须标明策略及预算。[论文 §4.4.1、§5.3][paper]

## 4. 缓存完整性与其他接入限制

**缓存最大的缺口是身份不足，不是文件名缺少 fold。** 文件名包含排序后的 folds，但不包含 CSV 内容哈希、题号映射哈希、源码版本、max_concepts、频次阈值、正确率断点或负类映射。若文件路径相同而数据/配置变化，已有两个缓存仍可被直接复用。JSON 只保存每个 anchor 的 positive/negative 列表，不保存 `q_acc`、计数分母、输入来源或生成配置。现有代码无法仅从缓存证明其来自哪些具体学生与响应。[缓存 L36–60][loader-cache]、[输出表 L285–287][loader-pairs]

其他确定行为及其含义：

- `len(folds)>1` 是启用统计的启发式，不是显式“训练”权限标记。标准五折 caller 正确，但手动把验证/测试折一起传入时 loader 不会拒绝；仅一个训练 fold 时反而不生成关系表。未来接口应验证明确的允许 fold 集合。[loader L173][loader-count]
- `correct_ratio_level = self.data_config[...]` 后直接 `.append(1)`，会原地修改传入列表；缓存命中时不做此修改。默认边界重复追加 1 通常不改变正标签区间，但配置序列化会受调用次序/缓存状态影响。新封装应使用规范化配置快照并记录真实有效边界。[loader L211–232][loader-level]
- 全量题/KC ID 字典及 max_concepts 在学生 train/test split 前建立。这是题库/结构可见范围问题，与在训练关系表中混入测试响应不是同一结论。标准 transductive 题库与严格 unseen-item 协议必须分别声明。[split_datasets_que.py L145–169][que-split]
- 作者配置中的 AS15 为 `num_q=0`、input_type 仅 concepts，而 `KTQueDataset` 要求 questions 和 concepts 同时存在。该原始 item 对比路径不直接支持 AS15；不能用 dummy q=0 冒充真实题库。[AS15 配置 L2–9][config-as15]、[loader L27–34][loader-init]

## 5. 接入前最小验收建议（原审计时未执行）

1. **显式定义统计总体。** 代码参考轨应重建“训练 folds、question-level 非重叠切块、丢弃短尾”的原始规则。若共同协议改为训练 canonical events 一次一计，应单列为适配变体，说明补回了原代码丢弃的短尾及可能的数据去重差别，不能暗称与原缓存逐值等价。
2. **先过滤训练成员，再拟合任何响应统计。** 训练 fold 集合、学生/事件成员、源文件、ID/KC 映射、有效参数与生成源码均纳入 hash；保存逐题 `N_q/C_q/ratio/class` 和配对表。验证/测试只做只读查询，不生成或更新统计。
3. **冻结负例规则。** 固定代码版本允许低频负例；论文描述版本强制高频负例。二者作为明确变体比较，不能在复现时静默“修好”再声称原算法。
4. **纯合成回归验收。** 未来实现应证明：改变验证/测试响应不改变训练表；改变训练响应只影响相应题的统计和派生 pairs；KC 排列不变、重复 KC 与不一致题/KC 映射显式处理；重复真实事件保留、重复导出行不误计；201/202 长度边界；阈值上下界；无正例/无负例/训练未见题；配置或数据变更使缓存失效。这些是建议，**本轮未运行这些测试**。
5. **区分静态证据和复现证据。** 接入前仍需用已登记的当前实际训练 fold 检查计数、候选覆盖和 source hashes，后续按持续授权执行；原审计没有数据覆盖率、影响量或性能结论，也没有验证作者历史缓存。

此次补齐了上一张原创性卡的“生成代码及训练 fold 数据流未知”。仍未知的是历史运行产物、阈值设定来源，以及实际数据中的重复/不一致情况；这些不能靠阅读 global 一词推断。Q-MCKT 仍是相关近邻，但公平接入需上述声明和验收。

[init-train]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/datasets/init_dataset.py#L104-L134
[init-return]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/datasets/init_dataset.py#L185-L188
[init-test]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/datasets/init_dataset.py#L51-L57
[loader-init]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/datasets/que_data_loader.py#L27-L34
[loader-cache]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/datasets/que_data_loader.py#L36-L60
[loader-count]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/datasets/que_data_loader.py#L139-L220
[loader-level]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/datasets/que_data_loader.py#L211-L241
[loader-pairs]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/datasets/que_data_loader.py#L260-L287
[pre-entry]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/examples/src/data_preprocess.py#L7-L53
[raw-route]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/preprocess/data_proprocess.py#L3-L50
[que-split]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/preprocess/split_datasets_que.py#L145-L210
[que-chunks]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/preprocess/split_datasets_que.py#L9-L50
[fold-split]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/preprocess/split_datasets.py#L175-L212
[read-response]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/preprocess/split_datasets.py#L52-L74
[raw-as09]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/preprocess/assist2009_preprocess.py#L16-L40
[raw-nips]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/preprocess/nips_task34_preprocess.py#L34-L78
[train-pass]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/examples/src/wandb_train.py#L73-L119
[model-pass]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/models/init_model.py#L99-L101
[model-store]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/models/qikt_plus.py#L126-L143
[model-pairs]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/pykt/models/qikt_plus.py#L215-L268
[config-as15]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/configs/data_config.json#L2-L9
[config-as09]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/configs/data_config.json#L66-L100
[config-nips]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/configs/data_config.json#L301-L335
[config-algebra]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/configs/data_config.json#L57-L64
[config-ednet]: https://github.com/rattlesnakey/Q-MCKT/blob/ee1698638888c5cddd577326aa30cadd5538f42d/configs/data_config.json#L292-L299
[paper-pairs]: https://arxiv.org/html/2403.07322v3#S4.SS4.SSS1
[paper]: https://arxiv.org/html/2403.07322v3
[tree]: https://api.github.com/repos/rattlesnakey/Q-MCKT/git/trees/ee1698638888c5cddd577326aa30cadd5538f42d?recursive=1

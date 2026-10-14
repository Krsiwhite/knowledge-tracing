# KTST 作者强配置的参数规模核验

2026-10-15。**当前 B003 mean baseline 的参数规模明显小于指定的作者 fold0 配置，AS15 差距最大。** 本轮仅在 CPU 单线程逐个构造6个模型并计数：0次前向、反向、优化器构造/更新、权重加载和 GPU 初始化，没有训练或推理结果，也未新增训练清单。

| 数据集 | 架构配置 | 总参数 | 可训练参数 | FP32参数字节 | 相对本集B003 |
| --- | --- | ---: | ---: | ---: | ---: |
| ASSIST2009 | B003 mean | 2,948,777 | 2,948,777 | 11,795,108 | 1.0000× |
| ASSIST2009 | 作者 reproduce_65，fold0 | 4,402,801 | 4,402,801 | 17,611,204 | 1.4931× |
| ASSIST2015 | B003 mean | 675,625 | 675,625 | 2,702,500 | 1.0000× |
| ASSIST2015 | 作者 reproduce_15，fold0 | 4,969,305 | 4,969,305 | 19,877,220 | 7.3551× |
| NIPS34 | B003 mean | 791,337 | 791,337 | 3,165,348 | 1.0000× |
| NIPS34 | 作者 reproduce_29，fold0 | 1,520,689 | 1,520,689 | 6,082,756 | 1.9217× |

数据维度仅取项目已绑定的 manifest：AS09 为17,737题/123 KC，AS15 为0题/100 KC（构造器按既有约定使用1个 dummy question），NIPS34 为948题/57 KC；没有打开任何原始记录、开发序列或测试文件。完整维度、构造参数、字节摘要和限制见 [JSON核验记录](KTST_STRONG_CONFIG_CAPACITY_2026-10-15.json)。

这里的 B003 baseline 指原清单的计划架构；JSON 的 `baseline_run_id` 仅标识这条计划，不表示原 B003 KTST run 已执行。恢复批次 B003-R1 沿用相同 KTST 架构。本轮不读取任一运行的权重或结果。

三集 B003 均为 d128、1 encoder、2 decoder、FF256、分类隐藏层256。作者配置分别为 AS09 d128/2 encoder/6 decoder/FF512，AS15 d256/3 encoder/4 decoder/FF512，NIPS34 d128/4 encoder/4 decoder/FF256/分类隐藏层128。全部保持 `q_mean_c`、`use_decoder_only=false`、`use_bias_emb=true`、`use_zero_init=true` 和原 attention 变体；没有用 NIPS fold4 文件替换。作者配置保留的 aggregation 层数不代表 mean 聚合实际新增了 MHSA 层。

配置来自固定作者提交 `e6331a6f45e597f286d1272fbb96e7bb67bcaa52` 的 [AS09 reproduce_65](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/config/reproduce/ktst/benchmark/reproduce_65.yaml)、[AS15 reproduce_15](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/config/reproduce/ktst/benchmark/reproduce_15.yaml)、[NIPS34 reproduce_29](https://github.com/kainbr/kt_set_transformers/blob/e6331a6f45e597f286d1272fbb96e7bb67bcaa52/pkg/config/reproduce/ktst/benchmark/reproduce_29.yaml)。三份本地 YAML 均逐字节核对缓存获取收据中的大小和 SHA256；模型源文件通过现有 vendor 锁校验。B003 配置绑定原 canonical SHA `765c4d0183fd0b4332d50f6b6f7a206c845e007fb06fa87a640bf286e5e80d44`。

**参数字节不等于运行显存。** 表中只计 FP32 参数本体，未计梯度、Adam 状态、buffer、激活、分配器及完整验证成本；本次没有测量耗时或峰值显存，也没有证明 GPU 可训练性或模型质量。参数更多不能直接推出 AUC 更高。

若后续采用这些架构，项目的最多50轮、patience3、seed42与共同 v2 可见历史，仍区别于作者 YAML 的200轮、patience10及原 seed（AS09 33、AS15 78、NIPS34 53）。这将是项目协议下的作者配置基线，不能称为逐表复现或 SOTA 证据。本轮构造随机种子为0，仅用于确定性初始化，不影响参数数量。

复现入口保存于工作树忽略目录 `.cache/ktst_strong_config_capacity.py`；使用工作树 `PYTHONPATH=src`、单线程及隐藏 GPU 的环境执行。脚本对 forward/backward/优化器/GPU初始化/权重加载安装拒绝断言，并限制数据目录只能读取三份 manifest；退出码0，输入收据在计数结束后再次验证。脚本 SHA256、构造器 SHA256、vendor 锁与缓存收据 SHA256 均保留在 JSON 中。

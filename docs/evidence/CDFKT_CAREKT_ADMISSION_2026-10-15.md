# CDFKT 与 CareKT 的接入边界

2026-10-15。两项方法继续列为前沿候选，尚不加入已冻结的三数据集训练队列：CDFKT 需要明确的题目映射与运行依赖适配，CareKT 尚缺可核验的完整实验表和代码。它们的作者结论不能代替本项目统一协议下的成绩。

## CDFKT

[pyKT 官方公告](https://pykt.org/cdfkt)于 2026-10-11 宣布集成 CDFKT；[作者所在大学的新闻](https://www.jnu.edu.cn/2026/0817/c3135a861865/page.htm)确认论文获 Information Fusion 录用。[出版社条目](https://www.sciencedirect.com/science/article/pii/S1566253526005245)列卷 137、文章 104648、2027 年 1 月。代码公开、接收和卷期是不同事实，不能用未来卷期推断当前没有可用代码。

模型结合 Mamba 序列编码器、按交互步数衰减的编码器及逐特征融合门。[官方文档](https://pykt-toolkit.readthedocs.io/en/latest/models.html#cdfkt)列出 `mamba-ssm`、`causal-conv1d` 依赖。当前 WSL 项目 Python 环境的安装元数据中两者均不存在；此次只检查包元数据，没有安装依赖或导入、运行模型。AMD/WSL 兼容性因此仍待独立检查，不能写成已确认不兼容。

对固定 pyKT 提交 `eec087e0230520e7ae7bbd4f5433d71612df62fb` 的[模型代码](https://github.com/pykt-team/pykt-toolkit/blob/eec087e0230520e7ae7bbd4f5433d71612df62fb/pykt/models/cdfkt.py)作静态检查，得到以下接入条件：

- 构造时读取 `../data/<dataset_name>/question_concept_map.npy`，并建立以题目数为第一维的参数表；该路径与本项目规范化 manifest/data_dir 不是同一接口，需要显式适配。
- `_resolve_dataset_name` 按 `emb_type` 中的 EdNet_large、NIPS34、XES3G5M、AS09 选择数据目录；没有 ASSIST2015 分支。默认 `qid` 也不能自行推断目录。
- 本项目 ASSIST2015 的 `num_q=0`。原接口没有对应的概念独立后备路径，不能把 KC 冒充独立题目 ID 后称为原版复现；若新增概念路径，应单独标记为适配。
- KC embedding 用于构造题目参数表的初值，但该局部 embedding 未保存为模型子模块；后续 `forward` 查的是题目表。不能仅凭初始化方式声称模型一直保持可训练的 KC–题目共享分解。
- `forward(last_pro,last_ans,next_pro)` 的信息边界仍需在实际接入时核对，尤其题级映射、shift、多 KC 聚合和可见历史。静态看到左填充和历史输入，不等于已经通过端到端因果检查。

本次读取的 `cdfkt.py` SHA256 为 `9e318f839884d529b22fb177d6b9be5b585b077a7b147e6d339dc6147fa80075`。没有改动上游文件、当前训练环境或已冻结批次。

出版社正文直接读取返回 403，当前可核原始来源只支持模型说明与“三个数据集”的作者概括；完整结果表、确切三集和划分预算仍未核齐。不用二手转载补成绩。下一步若优先接入，应先取得作者可访问全文/完整表，再做隔离依赖与因果检查。

## CareKT

[出版社条目及可检索摘要](https://www.sciencedirect.com/science/article/pii/S0045790626004817)列题名 *Confidence-aware refinement and learning for class imbalance in knowledge tracing*，Computers and Electrical Engineering 139 Part B，文章 111411，卷期为 2026 年 11 月。摘要介绍多视角表示、置信度引导的局部细化、辅助监督及 focal loss；报告四个基准中的三个取得最佳 AUC，并明确 NIPS34 只是具有竞争力。因此不能将其转述为四集全面最好。

出版条目指向[作者匿名代码入口](https://anonymous.4open.science/r/CareKT)。本轮工具未能打开该入口，出版社正文直接读取也未成功；这不等于作者没有代码。完整表、代码版本、预处理、划分、正类定义与额外输入仍待核验，目前不填具体 AUC、不判断与本项目三集直接可比，也不预登记训练。

它对本项目的具体提醒是：若后续改变类别权重、focal loss、局部预测平均或辅助目标，需将这些已有机制列为近邻，并拆开表示适配与损失变化的对照。当前拟做的 KTST 历史 KC×回答残差保持原 BCE，不同时叠加这些目标。

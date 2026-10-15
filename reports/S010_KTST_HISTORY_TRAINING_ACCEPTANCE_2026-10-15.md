# S010：KTST目标采样与历史对齐适配验收

2026-10-15完成合成人工样例验收。首次S010为209项通过、4项失败；四项均正确拒绝重复UKT配方，但旧测试匹配的报错前缀被新文案改变。仅恢复该前缀并补充KTST身份说明，判断逻辑与测试预期不变；S010-R1只复查这四项，全部通过。不能把首次记录改写成全通过。

两个结果分别保存为[首次收据](S010_KTST_HISTORY_CPU_INITIAL_2026-10-15.json)和[文案修正收据](S010_R1_KTST_HISTORY_READER_MESSAGE_2026-10-15.json)。合并证据覆盖213个不同案例；已通过的209项及模型计算没有重复运行。首次pytest为12.42秒、受控子进程13.76秒；修正子进程2.19秒。日志中故意拒绝已有输出、损坏收据和缺失产物的stderr是负例行为。

## 计算与结果边界

实际只构造6个CPU d16模型，调用36次完整前向及8次autograd.grad；0次真实backward、0个真实optimizer、0次参数更新。模型为原版、signed_response和response_independent，各覆盖有题ID与仅KC两种输入，均使用两层encoder/decoder；fake engine仅检查控制流，不属于模型训练。S010-R1没有模型或梯度调用。

人工CSV覆盖原200-KC切块、完整目标、跨块partial、丢弃尾段、重复KC和held-out折隔离。配对抽样的目标与权重一致，使用独立随机数且保留有放回重复；长历史遵守包含完整当前题的200-KC预算。原分类器真实logit、外部目标BCE、当前/未来答案隔离、历史答案正控制、有效梯度和全优化batch归一均通过。

fake engine覆盖默认旧路径、短/长两recipe、最后不完整epoch，以及微批中途超时后零更新；已经完成的前向行与提交的训练样本分别记录。配置、checkpoint、metadata和保存预测保留recipe/K/helper源码身份，离线reader拒绝不一致收据。旧模型默认分支没有增加训练字段。

这仅证明限定人工样例下实现与合同吻合，不证明真实GPU速度、验证提升或科学原创。没有读取真实训练/验证数据、测试集或已训练权重。真实训练K和批次未由本验收冻结，最终测试仍封存。

## 冻结与修正链

- 训练source：`7066077ebd02a0bd0cecccf3d3a6b3c57df9b8162a1370f2e3cc576cb14fbfef`；核心helper：`c930087ccf7872ba3db7d21229e13f282ea97ca718f2f171c1814356e6f51f50`。两次执行均前后核对，训练source始终未变。
- [S010清单](../configs/checks/S010-ktst-history-training-cpu.json) canonical：`47913ac2b152e0fa2a1343b325ea713f39bf2d7541d173efbb7cdc4142ee8509`。执行上限12模型、128前向、12次grad、0更新、300秒；实际均低于上限并与固定预期计数一致。
- [S010-R1清单](../configs/checks/S010-R1-ktst-history-reader-message.json) canonical：`ccf7d7dba4bda0b22dba79cff796e4d0cd79affcd7ca064e5c4d1a379624cf41`。只允许四项纯文件案例、0模型/梯度/更新、60秒。
- 修正后的分析器SHA：`e93351b1cd6d023a77ad09abc46f54ec5810e5f40dcb03c1f5a5f803206e4040`。R1 runner反向替换这一条字符串后核对S010原分析器SHA，证明没有悄改判断逻辑；其余S010绑定文件逐项保持不变。
- 三份测试SHA、vendor与runner摘要保存在清单中。两个runner分别是[首次入口](../scripts/run_s010_acceptance.py)和[局部修正入口](../scripts/run_s010_r1_acceptance.py)，本机排他attempt已存在，不能原样重跑或覆盖。

执行前均已记录具体清单、哈希、预算及用户原话“启动B003，但每次最多50轮，3次未见则早停。之后的执行不需要我再批准”；没有虚构新确认。根代理、实现代理和独立审查代理在首次执行前做了静态审查；四项文案失败另经独立诊断后才执行局部修正。B007与其他冻结工作树未被改动。

首批性能候选见[AS09配对pilot成本计划](KTST_HISTORY_PILOT_COST_PLAN_2026-10-15.md)；该计划尚不是训练执行清单。

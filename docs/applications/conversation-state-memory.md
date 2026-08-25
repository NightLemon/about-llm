# 对话状态与长期记忆：暂停后怎样安全恢复

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：构建长任务 Agent、对话产品和个性化记忆服务的开发者。
- **先修**：[一次退款修复总览](code-conversation-llmops.md)和基本的状态机、数据库事务概念。
- **首次阅读**：暂停现场 → 三类信息 → 任务检查点 → 记忆写入 → 修正与过期 → 长会话评测。
- **完成信号**：能区分聊天记录、任务检查点和长期记忆，并为每类信息指定来源、作用域、过期与删除规则。
- **卡住时**：先只问“恢复任务时必须重新确认哪些事实”，不要从向量数据库选型开始。

</div>

退款补丁已经生成，目标测试也已经通过。用户此时说：“我晚点再继续发布。”

第二天恢复时，Agent 面前可能有一段很长的聊天记录。它需要知道的却是结构化事实：仓库有没有变化、补丁是哪一版、
哪些测试真的运行过、发布是否获批，以及支付服务是否仍有结果未知的请求。

如果系统只保存一句“退款问题已经修好，等待发布”，其中每个词都可能产生歧义。

## 先拆开三种经常被叫作“记忆”的东西

| 信息 | 本例中的内容 | 主要用途 | 生命周期 |
| --- | --- | --- | --- |
| 原始对话（transcript） | 用户请求、澄清和 Agent 回复 | 回看双方说过什么 | 按会话与隐私策略保存 |
| 任务检查点（task checkpoint） | 补丁、测试、审批和发布阶段 | 恢复尚未完成的工作流 | 任务结束后归档或清理 |
| 长期记忆（long-term memory） | 用户明确允许保存的稳定偏好 | 在未来会话中减少重复设置 | 可查看、更正、删除并受同意和过期约束 |

“用户偏好中文”可能适合长期记忆；“测试刚跑到第 37 个”属于任务检查点；“支付服务已退款”则应重新查询权威系统。
模型在旧消息中读到一句话，不能替代数据库或外部工具的当前状态。

## 任务检查点保存可恢复状态

本例的最小检查点可以写成：

```text
task_id
base revision + current patch/commit identity
current stage + completed transitions
reproduction command + evidence digest
verification commands + result digests
pending questions + approvals
external proposals + receipts + uncertain outcomes
budget + policy + tool versions
```

恢复时按顺序做三件事：

1. 重新读取当前仓库版本和磁盘上的实际 Diff；
2. 检查证据文件、审批和外部回执是否仍然存在且身份一致；
3. 从明确的任务阶段继续，而不是让模型根据聊天摘要猜下一步。

仓库版本已经前进时，旧补丁可能无法应用，旧测试结果也未覆盖新代码。审批参数变化时，旧审批应失效。外部退款结果未知时，
恢复动作通常是查询和对账，而不是直接重试。

任务检查点本身也要防止串线。租户、用户、任务编号、代码版本和策略版本必须进入读取条件；不能先加载所有任务，
再让模型决定哪一条属于当前用户。

## 长期记忆只保存未来确实需要的信息

常见记忆可以分成四类：

| 类型 | 适合保存什么 | 容易犯的错误 |
| --- | --- | --- |
| 工作记忆（working） | 当前任务中的临时约束 | 自动升级为永久偏好 |
| 情节记忆（episodic） | 带时间、来源和过期时间的过去事件 | 用摘要覆盖原始来源 |
| 语义或档案记忆（semantic/profile） | 用户明确确认的稳定事实或偏好 | 未经同意跨会话个性化 |
| 程序记忆（procedural） | 版本化工作流、策略和工具 Schema | 被一次普通对话永久改写 |

写入前先回答：未来用途是什么，用户是否预期，信息是否敏感，来源能否验证，何时过期？模型可以提出候选记忆，
真正写入应由确定性规则或用户确认决定。

每条记录至少保存：

- 值、类型与来源事件；
- 租户、用户、会话或档案作用域；
- 创建、更新与过期时间；
- 置信度以及同意或策略版本；
- 它修正、取代或撤回了哪条旧记录。

无法追溯来源的自由文本总结，不适合作为权威事实。

## 用可执行账本观察一次偏好修正

仓库中的 `ConversationMemoryLedger` 把事实和修正保存成不可变事件。下面先记录“偏好中文”，一分钟后由新事件改为
“English”：

```python
from datetime import datetime, timedelta, timezone

from about_llm.conversation import ConversationMemoryLedger, MemoryKind, MemoryScope

ledger = ConversationMemoryLedger()
now = datetime.now(timezone.utc)
old = ledger.add_fact(
    fact_id="fact-1",
    tenant_id="tenant-a",
    subject_id="user-7",
    key="preferred_language",
    value="中文",
    kind=MemoryKind.WORKING,
    scope=MemoryScope.SESSION,
    source_event_id="message-42",
    created_at=now,
    confidence=1.0,
    policy_version="memory-policy-v1",
    expires_at=now + timedelta(hours=8),
)
new = ledger.correct_fact(
    previous_fact_id=old.fact_id,
    new_fact_id="fact-2",
    tenant_id="tenant-a",
    subject_id="user-7",
    value="English",
    source_event_id="message-43",
    created_at=now + timedelta(minutes=1),
    confidence=1.0,
)
assert ledger.active_facts(
    tenant_id="tenant-a",
    subject_id="user-7",
    now=now + timedelta(minutes=1),
) == (new,)
```

这里没有覆盖原记录。旧事实仍保留在历史中，新事件明确指向它；当前视图只返回新值。这样既能回答“现在使用什么”，
也能解释“为什么发生变化”。

参考实现还检查：

- 同一租户、用户和 Key 只有一个当前值，修正必须指向旧事实；
- 修正与撤回不能跨租户或用户；
- 档案作用域必须带同意凭据，会话事实不会自行升级；
- 过期时间到达后，事实不再进入当前视图，所有时间必须带时区；
- 未来发生的修正不会提前改变过去时间点的视图；
- 写入值会变成规范 JSON 快照，调用方后来修改原对象不会篡改历史；
- 当前视图隐藏已取代、撤回或过期事实，历史视图保留解释链。

## 摘要负责导航，结构化状态负责恢复

摘要会压缩内容，也可能丢掉否定、时间、说话人和不确定性。长任务仍应保留原始事件指针与结构化状态，并定期检查：

- 用户修正后，旧事实已经标记为被取代；
- 待执行、已执行和结果未知的动作没有混在一起；
- 工具报错没有被总结成业务事实；
- 多用户和多租户实体没有串线；
- 摘要版本能够回到对应的来源范围。

向量检索适合帮助找到可能相关的旧内容，但它不能决定当前权威状态。余额、订单、权限、审批和发布阶段，都应从各自的
权威系统重新读取。

## 长会话评测要跨过修正和中断

单轮问答成功无法验证恢复能力。长会话用例至少包括：

1. 用户中途纠正偏好，旧值不再生效；
2. 用户打断任务，恢复后从正确阶段继续；
3. 工具返回错误，摘要没有把错误消息当成业务结果；
4. 外部结果未知，系统先查询而不是重复执行；
5. 用户要求删除记忆，当前视图、历史、缓存和备份按策略处理；
6. 租户和用户切换时，不会检索到别人的记录；
7. 上下文变长后，任务成功、延迟和成本仍在可接受范围。

## 运行仓库中的记忆验证

```powershell
python -m pytest tests/test_conversation_memory.py -q
```

这些测试验证单进程内存账本的来源、过期、修正、撤回、作用域和快照行为。生产服务还需要数据库事务、加密、访问控制、
跨副本一致性、并发冲突、备份删除和定期过期处理。

这个账本没有保存退款任务检查点，也没有连接真实用户或外部数据库。它用一个小实现帮助你看清“当前视图”和“不可变历史”
的区别。

下一步继续[LLMOps 发布与回滚](llmops-release.md)，把已经验证的补丁绑定进完整发布版本。

## 自测

1. 为什么任务检查点不能只保存一段聊天摘要？
2. “用户偏好中文”与“支付服务已退款”为什么需要不同的真值来源？
3. 用户修正偏好后，为什么要保留旧记录而不是直接覆盖？
4. 外部结果未知时，恢复动作为什么通常是查询而不是重试？

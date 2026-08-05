# Safe Agent Runtime

目标：把“模型建议动作”和“系统获权执行”分开。任何模型或 Agent 框架都只能产生 ToolCall；执行内核负责 schema、权限、审批、预算和幂等。

## 已实现的不变量

- 工具注册表拒绝重名和未知工具；
- 参数在审批前校验，避免让用户批准模糊或非法动作；
- 可逆/不可逆工具必须显式 approved；
- 相同 call_id 与相同输入只执行一次；
- 相同 call_id 换参数触发冲突，而不是覆盖；
- 工具调用总数有硬预算；
- handler 错误转为失败结果，不伪装成功；
- 可列出超时 pending，人工确认外部成功、标记放弃或已补偿；
- reconciliation 保留审计事件，放弃/补偿后的重试必须使用新审批的新 call_id。

~~~powershell
pytest tests/test_agent_runtime.py tests/test_sqlite_agent_ledger.py
~~~

## 生产替换点

默认教学运行可使用内存 ledger；SQLiteLedger 提供跨进程持久化、原子 call-id claim 和 pending/completed 状态。若 handler 超时或崩溃，记录保持 pending，后续实例不会盲目重放。

SQLite 只能保护 claim，不能与远程副作用构成一个原子事务。`list_stale_pending` 找出待调查调用；外部审计确认成功后用 `resolve_external_completion` 写入结果，确认未发生或已经逆向操作后用 `resolve_without_completion` 标记 abandoned/compensated。旧 call id 永不重新执行。大规模服务可替换为带唯一约束的数据库，并用业务事务与 outbox 协调。

审批令牌应绑定用户、call_id、工具、参数指纹、过期时间和会话，不能只传一个布尔值。权限由用户身份和资源 ACL 决定，不由模型在参数中自报。

## 后续里程碑

1. Pydantic/JSON Schema 工具协议；
2. planner loop、最大步数和无进展检测；
3. 外部文档提示注入测试；
4. LangChain/LlamaIndex/云工具调用 adapter；
5. trace、回放和 Agent 任务成功率评测。

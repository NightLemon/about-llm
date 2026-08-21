# 实验 0D：Opaque Reasoning 工件与重放边界

**定位**：安全工程选修，预计 60–90 分钟；默认全部离线，不发送请求、不加载模型、不需要 API key。

**实验导航**：[返回总览](../labs.md#lab-0) · [Reasoning 工件安全](../../quality/reasoning-artifact-security.md) · [云 API 契约](../../models/cloud-api-contracts.md) · [项目入口](../projects/cloud-api-contracts.md#run)
{ .doc-nav }

## 开始前

**先修知识**：完成[实验 0C](lab-0c-cloud-budget.md)，能够区分 confidentiality、integrity、provenance、authorization，以及 trusted identity 与 request body 自报字段。

**本页完成后**：你应该能解释为什么 content-only AEAD 无法阻止合法 ciphertext 被拿到错误上下文重放，并能为 subject、tenant、session、branch、predecessor、model audience、expiry、key status 和 single consumption 分别写出负例。

## 第一步：先做预测

运行前填写下表，不要先看 JSON 结果：

| Case | Content-only 预测 | Context-bound 预测 | 原因 |
|---|---|---|---|
| exact context |  |  |  |
| cross subject |  |  |  |
| cross tenant |  |  |  |
| cross session |  |  |  |
| cross model |  |  |  |
| wrong predecessor |  |  |  |
| expired |  |  |  |
| retired key |  |  |  |
| tampered claims |  |  |  |
| second consumption |  |  |  |

不要把“攻击者改密文”当成唯一威胁。论文案例的关键是：攻击者可以拿到未被修改的合法 block，但在错误主体、会话或模型中使用。

## 第二步：运行 replay matrix

~~~powershell
python -m about_llm.integrations.cloud_api_cli reasoning-replay-matrix `
  --output artifacts/cloud-api/reasoning-replay-matrix.json
~~~

最低应观察到：

- `network_performed: false`；
- `real_provider_artifacts_used: false`；
- `plaintext_reasoning_emitted: false`；
- `case_count: 16`；
- `unsafe_acceptance_count: 4`；
- 全部 case 的 `passed: true`。

这里的 `passed` 表示实际行为符合实验预期。四个 content-only 错误上下文被接受是**成功复现了弱协议缺陷**，不是安全门禁通过。查看每一行的 `unsafe_acceptance_demonstrated`，不要只看顶层绿色布尔值。

## 第三步：运行 trajectory release gate

~~~powershell
python -m about_llm.integrations.cloud_api_cli trajectory-release-gate `
  --input projects/cloud-api-contracts/trajectory-release.example.json `
  --output artifacts/cloud-api/trajectory-release-report.json
~~~

安全投影应得到 `passed: true`、`opaque_reasoning_block_count: 0` 与 `unknown_block_count: 0`。把任一 block 的 `type` 临时改成 `thinking` 或未知值后，命令应以 1 退出；报告可以给出 block 数组位置和拒绝类别，但不能回显该 block 的值或未知类型名。

检查 `secret_pii_scan_performed: false`。这不是缺陷隐藏，而是证据边界：字段 allowlist 能保证发布对象没有受支持 schema 之外的 block，却不能证明允许的 visible text、tool arguments/result 和 citation 已无 secret、PII、版权或 consent 问题。

## 第四步：解释 AAD

打开 `src/about_llm/integrations/reasoning_artifact.py`，分别观察两种 associated data：

```text
content-only:
  version + binding_mode + provider + key_id

context-bound:
  上述字段 + artifact/subject/tenant/session/branch/
  predecessor/model audience/issued/expiry claims
```

AEAD 解密成功后，强协议仍要把已认证 claims 与可信 `ReasoningReplayContext` 比较。原因是 claims 即使真实，也可能不属于当前请求。身份和 tenant 必须来自认证控制面，不能从 envelope 外另附的未认证 JSON 或 Prompt 取得。

## 第五步：运行定向测试

~~~powershell
python -m pytest tests/test_reasoning_artifact.py tests/test_trajectory_release.py tests/test_cloud_api_cli.py -q
~~~

测试分开覆盖：

- 弱协议接受 cross subject/tenant/session/branch/predecessor/model/expiry；
- 强协议对每类 scope drift 给出稳定、脱敏的拒绝原因；
- claims 或 ciphertext 任一 bit 漂移导致 authentication failure；
- 同一 key 下 nonce 重复签发被拒绝；
- retired key 不再解密旧 envelope；
- 同一 artifact 第二次消费被 replay ledger 拒绝。
- 发布 gate 拒绝 thinking/signature/unknown block 与 schema drift，且不回显输入名称或值。

## 第六步：制造三个反例

每次只做一个临时改动，运行同一组测试并记录首先失败的 case，随后恢复改动：

1. 从 context-bound AAD 中移除 `context`，观察 claims tamper 不再首先表现为 authentication failure。
2. 从 `_authorize_replay` 中移除 subject 或 predecessor 检查，观察对应跨域 case 被接受。
3. 调用消费函数时不传 `InMemoryConsumptionLedger`，观察 exact same-context 的第二次使用无法被无状态 AEAD 单独发现。

再回答：nonce ledger 与 consumption ledger 为什么不能合并成同一个概念？前者保护同一 key 下的加密安全，后者保护业务 artifact 的一次性使用；两者 identity、生命周期和故障语义不同。

## 第七步：设计生产迁移

为一个假设的旧格式写迁移表：

| 阶段 | 旧 envelope | 新 envelope | 必须记录 |
|---|---|---|---|
| 发布修复 | 有界兼容或立即拒绝 | 签发 | key id、格式、owner |
| 迁移窗口 | 仅原 session owner 可重新签发 | 接受 | archive identity、审批、expiry |
| 窗口结束 | retired key 拒绝 | 接受 | 拒绝率、受影响 session |
| 泄露事件 | 撤销、隔离、删除传播 | 按 scope 调查 | artifact/key/session/公开副本 |

迁移不能只做“新代码上线”：已经公开的旧 block 需要 key retirement；暂停中的合法 Agent workflow 则需要验证所有权后的重签发或明确失效。

## 常见失败

- 把 base64、opaque、signature 和 encryption 当作同义词。
- 用客户端自报 `user_id` 代替 authenticated subject。
- 只绑定 model name，不绑定 tenant/session/predecessor。
- 认为 AES-GCM 自动阻止合法 ciphertext 重放。
- 把单进程 set 写成 durable、多区域 replay protection。
- 发布前只扫描 visible text，保留未知或 encrypted block。
- 把 2026 年 7 月论文结果描述成当前 provider 仍可复现。

## 交付与结论边界

最低交付物：预测表、16 项 replay matrix、一次安全和一次拒绝的 trajectory gate 报告、三个故意失败的测试记录、一份旧格式迁移表，以及不超过五行的结论边界。

合格结论应类似：本实验使用成熟 AES-GCM 实现和本仓库准备的 bytes，说明“认证内容”与“绑定使用上下文”
是不同协议属性。实验不解析真实 provider artifact，也不访问模型或网络，因此不能据此判断当前 API 是否存在
同类漏洞或已实施相同修复。内存 nonce/replay ledger 也不能代表生产环境的 key custody、持久化、并发或多区域一致性。

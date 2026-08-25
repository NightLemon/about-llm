# 实验 0D：一份加密工件为什么还能被错误会话重放

**定位**：安全工程选修，预计 60–90 分钟；默认全部离线，不发送请求、不加载模型，也不需要 API key。

**实验导航**：[返回总览](../labs.md#lab-0) · [Reasoning 工件安全](../../quality/reasoning-artifact-security.md) · [云 API 契约](../../models/cloud-api-contracts.md) · [项目入口](../projects/cloud-api-contracts.md#run)
{ .doc-nav }

## 从一次跨会话重放开始 {#running-example}

假设 Agent 在 `session-a` 中得到一块不应直接展示的推理数据。应用把它加密成 `artifact-001`，然后继续执行
后续步骤。

这份工件原本属于：

```text
用户：subject-a
租户：tenant-a
会话：session-a
分支：main
前一步摘要：111...111
允许使用的模型：model-strong
有效时间：100 ≤ now < 200
```

现在有人拿到**没有被修改过**的完整加密工件，把它放进 `session-b`，或者交给另一个租户和模型继续使用。

这个例子容易让人产生一个直觉：“密文没改，AES-GCM 验证也成功，当然可以使用。”问题在于，加密认证只
回答了内容和关联数据是否保持原样；它不会自动知道当前用户、租户和会话是否有权消费这份工件。

安全消费需要依次回答三道问题：

1. **内容认证**：密文及其关联数据有没有被改动？
2. **上下文授权**：工件声明的用户、租户、会话和模型是否与当前可信上下文一致？
3. **重放检查**：这份一次性工件是否已经消费过？

同一份密文可以通过第一关，却在第二关或第三关失败。本实验就是要把这三道关卡分开看清楚。

## 第一步：运行 16 个固定案例 {#run-matrix}

先运行：

```powershell
python -m about_llm.integrations.cloud_api_cli reasoning-replay-matrix `
  --output artifacts/cloud-api/reasoning-replay-matrix.json
```

报告摘要应包含：

```text
passed: true
network_performed: false
real_provider_artifacts_used: false
plaintext_reasoning_emitted: false
case_count: 16
unsafe_acceptance_count: 4
```

顶层 `passed: true` 的含义是“16 个案例的实际行为都符合各自预期”，不是“16 个案例都安全”。四个弱协议
案例本来就预期被错误接受；程序成功复现这个缺陷，所以这些行同样显示 `passed: true`。

看报告时找到 `unsafe_acceptance_demonstrated: true`。它出现在四种没有修改密文的重放中：

- 换成另一个用户；
- 换成另一个租户；
- 换成另一个会话；
- 换成未获允许的模型。

剩余案例让上下文绑定协议处理正确上下文、分支、前一步摘要、有效期、密钥停用、声明篡改和第二次消费。

### 跟着跨会话案例走一遍

弱协议签发 `artifact-001` 时，AES-GCM 的关联数据只包含：

```text
envelope version
binding mode
provider
key id
```

`session-a` 不在关联数据中。攻击者把原封不动的 envelope 交给 `session-b` 后：

1. 密文没有改变，AES-GCM tag 验证成功。
2. 程序得到原始明文。
3. 弱协议只检查 provider，然后提前返回。
4. `session-b` 因而错误接受了属于 `session-a` 的工件。

上下文绑定协议会把会话等声明加入关联数据，并在解密后与可信运行上下文比较：

```text
authenticated claims: session-a
trusted replay context: session-b
result: session_mismatch
```

这里有一个重要细节：攻击者若保持声明不变，AES-GCM 认证仍会成功，真正拒绝跨会话使用的是后续授权比较。
如果攻击者把声明从 `session-a` 改成 `session-b`，关联数据随之改变，AES-GCM 会先报
`authentication_failed`。

## 第二步：理解 AAD 做了什么 {#aad}

AEAD 指“带关联数据的认证加密”，英文全称是 `authenticated encryption with associated data`。
本实验使用 AES-256-GCM：推理内容被加密，AAD 保持可见但参与认证。

两种实验协议的 AAD 如下：

```text
只认证内容的协议：
  version + binding mode + provider + key id

绑定上下文的协议：
  上述字段
  + artifact id
  + subject / tenant / session / branch
  + predecessor digest
  + allowed model ids
  + issued time / expiry time
```

把声明放进 AAD 能阻止攻击者悄悄改写它们。但“声明没有被改”仍不等于“声明属于当前请求”。因此解密成功
后，程序还要把已认证声明与 `ReasoningReplayContext` 逐项比较。

可信上下文应来自认证和控制面：例如服务端解析出的登录主体、租户和会话，以及运行时选择的模型。客户端
自报的 `user_id`、Prompt 文字或 envelope 外另附的 JSON 都不能替代它。

可以把三种安全属性记成：

- **保密性**：没有密钥的人读不到推理明文；
- **完整性/来源认证**：密文和 AAD 被改动后，tag 验证失败；
- **授权**：即使工件原封不动，当前上下文也必须与其允许范围一致。

AES-GCM 直接提供前两类密码学属性。第三类来自应用选择了哪些字段进入 AAD，以及解密后怎样比较可信上下文。

## 第三步：nonce 与消费记录解决不同问题 {#two-ledgers}

AES-GCM 要求同一密钥下不要重复使用 nonce。签发时，本实验用 `InMemoryNonceLedger` 记录
`(key_id, nonce)`；重复组合会得到 `nonce_reused`。

这仍无法发现某个合法 envelope 被第二次使用，因为第二次消费没有重新加密，也没有生成新 nonce。

一次性消费由另一份账本负责。`InMemoryConsumptionLedger` 记录 `(key_id, artifact_id)`：

```text
第一次消费 artifact-001 → claim 成功
第二次消费 artifact-001 → replay_detected
```

两份记录不能合成一个含糊的“重放集合”：

| 记录 | 何时写入 | 保护对象 | 重复时的含义 |
|---|---|---|---|
| nonce ledger | 加密签发时 | 同一密钥下的 nonce 唯一性 | 可能破坏 AEAD 安全前提 |
| consumption ledger | 工件消费时 | 业务工件的一次性使用 | 合法工件正在被再次消费 |

密钥停用又是第三项控制。若旧格式或密钥已经暴露，仅上线新代码并不能让公开副本失效；消费路径必须拒绝
retired key，必要时再为确认所有权的合法工作流重新签发新工件。

## 第四步：读懂 16 个案例的层次 {#matrix-layers}

命令行矩阵由两组案例组成。

**只认证内容的 5 个案例**：

- 正确上下文正常接受；
- 跨用户、跨租户、跨会话和跨模型都被错误接受。

**绑定上下文的 11 个案例**：

- 正确上下文第一次消费成功；
- 用户、租户、会话、分支、前一步摘要和模型漂移分别被拒绝；
- 过期工件和停用密钥被拒绝；
- 修改已认证声明会触发认证失败；
- 第二次消费会触发 `replay_detected`。

定向单元测试还让弱协议尝试跨分支、错误前一步摘要和过期上下文。它们同样说明：只认证 provider 和 key
不能建立完整的业务使用范围。

## 第五步：发布轨迹时只允许明确类型 {#release-gate}

推理工件的第二个风险出现在“把 Agent 轨迹发布给用户或作品集”时。内部轨迹可能同时包含可见文本、工具
调用、引用，以及不透明推理块或供应商扩展字段。

本仓库采用一个很窄的发布投影，只允许：

- `text`；
- `tool_call`；
- `tool_result`；
- `citation`。

运行：

```powershell
python -m about_llm.integrations.cloud_api_cli trajectory-release-gate `
  --input projects/cloud-api-contracts/trajectory-release.example.json `
  --output artifacts/cloud-api/trajectory-release-report.json
```

固定样例应得到：

```text
passed: true
opaque_reasoning_block_count: 0
unknown_block_count: 0
finding_count: 0
plaintext_values_emitted: false
```

如果把某个 block 的 `type` 改为 `thinking` 或一个未知值，命令应以状态码 1 退出。报告会指出数组位置和拒绝
类别，但不会回显该 block 的值；未知类型名也不会原样出现在报告中。

这条门禁检查的是**发布对象的结构**。`secret_pii_scan_performed: false` 明确表示它没有检查允许字段里的文本
是否含 secret、PII、版权或未经同意的数据。真实发布流程还需要独立的内容扫描、权限和人工审阅。

它也不负责清洗供应商原始响应。正确顺序是先从内部轨迹构造允许发布的窄投影，再把投影交给这条门禁；
不能把未知或加密 block 原样保留，指望前端选择不显示。

## 第六步：运行定向测试 {#tests}

```powershell
python -m pytest `
  tests/test_reasoning_artifact.py `
  tests/test_trajectory_release.py `
  tests/test_cloud_api_cli.py `
  -q
```

阅读失败用例时，把它们放回三道关卡：

- **认证失败**：声明或 ciphertext 任意 bit 漂移；同一密钥和 nonce 再次签发。
- **授权失败**：用户、租户、会话、分支、前一步摘要、模型或有效期不匹配；密钥已停用。
- **消费失败**：同一 `artifact_id` 在同一密钥下第二次使用。

发布门禁测试则覆盖 forbidden/unknown block、schema drift 和脱敏报告。它们检查拒绝原因能被运维定位，同时
不把原始名称和值复制进报告。

## 如果删掉一项控制，会发生什么 {#counterfactual-controls}

不用把全部源码都背下来。沿三个反事实理解每一层的作用：

1. **声明不进入 AAD**：攻击者可以改写明文声明，使它们匹配新上下文；密文不变时，tag 无法发现这次改写。
2. **授权阶段不检查用户或前一步摘要**：已认证工件仍可越过缺失的那条业务边界。
3. **消费时不使用 ledger**：第二次使用与第一次使用完全相同，无状态 AEAD 无法知道历史上已经消费过。

这些反例也说明“加密更强”不是完整修复。上下文绑定、可信身份解析、持久重放账本和密钥生命周期都属于
协议的一部分。

## 旧格式怎样迁移 {#migration}

假设线上已经存在只认证内容的旧 envelope，可以按阶段处理：

1. **发布修复**：停止签发旧格式；新工件使用上下文绑定格式，并记录 key id、owner 和格式版本。
2. **迁移窗口**：旧工件只允许原会话所有者申请重新签发；重新核对 archive、审批和有效期。
3. **窗口结束**：停用旧密钥，统计拒绝率和受影响会话；不能无限兼容旧格式。
4. **泄露事件**：按 artifact、key、session 和公开副本范围撤销、隔离并传播删除。

已经公开的旧工件需要密钥停用才能真正失效。暂停中的合法 Agent 工作流则需要在验证所有权后重新签发，
或者明确告知用户旧状态已经失效。

## 本实验说明了什么 {#evidence-boundary}

本实验调用 `cryptography` 提供的 AES-256-GCM，并使用仓库准备的固定 bytes。它说明“内容通过认证”和
“工件获准在当前上下文使用”是两种协议属性，也展示了一次性消费为什么需要有状态账本。

本实验只处理仓库准备的固定工件，运行过程完全离线。它能支持的结论仅限于本页展示的协议行为。真实供应商
API 的安全状态和修复方式需要对应产品的独立证据。

nonce 与消费账本都只存在于单个 Python 进程中。它们没有覆盖生产密钥托管、持久化、进程崩溃、并发节点或
多区域一致性。轨迹发布门禁只验证严格 schema，也没有完成 secret、PII、版权和 consent 审计。

## 自测

1. 为什么一份未被修改的密文仍可能被拒绝？
2. 把声明加入 AAD 后，为什么还要与可信运行上下文比较？
3. nonce ledger 和 consumption ledger 分别在哪个阶段写入？
4. 为什么顶层 `passed: true` 可以与 `unsafe_acceptance_count: 4` 同时出现？
5. 发布门禁为什么只允许固定 block 类型，而不是“保留未知字段但不显示”？

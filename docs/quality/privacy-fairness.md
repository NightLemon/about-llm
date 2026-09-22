# LLM 隐私与公平：从数据流到群体伤害

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：负责训练数据、产品风险、内容安全、隐私或公平评测的工程师与治理团队。
- **先修**：[系统安全](safety.md)中的资产、信任边界与证据边界。
- **首次阅读**：数据流与泄露面 → 模型隐私 → DP/联邦学习 → 内容安全 → 公平与现实伤害 → 监控。
- **完成信号**：能为一个用途写清攻击者、数据流、群体切片、误伤/漏放分母和持续监控动作。
- **卡住时**：先选一类数据和一个受影响群体，沿采集、使用、记录、删除的顺序追踪。

</div>

系统没有遭到入侵，也可能泄露训练样本、误拦正常请求，或让某些群体持续得到更差的服务。
本页把这些问题放回同一条数据与影响链：什么数据进入系统、谁能观察它、模型或控制怎样改变结果，
以及伤害是否集中在特定用途或群体。技术执行防线仍由[系统安全](safety.md)负责。

## 泄露面不只在最终答案

泄露不只来自训练记忆：

- prompt/context 中的其他租户数据；
- RAG index、embedding 和 metadata；
- conversation memory 与摘要；
- traces、analytics、exception 和 support ticket；
- tool output、screenshots 和临时文件；
- response cache 与 CDN；
- training/evaluation artifacts；
- model/provider telemetry。

做 data-flow inventory，按字段标注目的、访问者、位置、TTL、加密和删除路径。数据最小化比“收集后再脱敏”更可靠。

### Opaque reasoning 与共享轨迹

某些 API 会返回客户端不可读的 reasoning/thinking block，并要求后续请求原样带回。
客户端看不懂这段数据，不表示它不敏感。

这类 block 可能包含 Prompt、工具 observation、PII、secret 或隐藏 instruction，也可能影响下一次生成
与工具 proposal。

签名或 AEAD tag 只保护实际进入认证上下文的字段。认证数据还应绑定 subject、tenant、session、
predecessor 和 model audience。缺少其中任何一项，合法 ciphertext 都可能被搬到错误上下文中重放。

公开 Agent trajectory 应从字段 allowlist 重新生成，只保留明确允许公开的类型。Reasoning、signature 和
未知 opaque block 默认删除；发布结果要求 `opaque_reasoning_block_count == 0`。

从外部取得的 trajectory 是不可信序列化状态。继续调用模型或工具前，必须重新解析、授权和验证。
详见[不透明推理块与轨迹安全](reasoning-artifact-security.md)和
[实验 0D](../practice/labs/lab-0d-reasoning-artifact-security.md)。

## 模型隐私

### Memorization 与 extraction

重复出现、内容罕见且上下文容易预测的训练样本，可能更容易被逐字记忆。
Canary exposure、membership inference 和 extraction attack 分别测量特定攻击条件下的泄露能力。

报告应分别说明攻击者知识与预算。一次提取失败，只能支持“这次攻击在给定预算下没有成功”。
训练样本是否影响参数，需要其他实验回答。

### Differential Privacy

DP-SGD 通常按样本或用户裁剪梯度并加入噪声，再由 privacy accountant 组合多步隐私损失，得到
\((\epsilon,\delta)\) 保证。报告这两个数之前，必须同时说明：

- Adjacency 是相差一个样本，还是相差一个用户；
- Sampling scheme 与 clipping 方法；
- Noise multiplier 与训练 steps；
- \(\delta\) 的选择。

较小 \(\epsilon\) 通常代表更强的形式保证，但不同 adjacency、\(\delta\) 或 accountant 不能只比一个 epsilon。DP 保护其正式威胁模型中的训练贡献，不自动保护 prompt 日志、RAG、工具或输出中的主动泄密。

### Federated learning

数据留在设备上，上传的 gradient 或 update 仍可能泄露信息。Federated learning 还要处理 server 信任、
client poisoning、secure aggregation、DP 和设备身份。

Federated 描述的是训练拓扑。隐私强度要由具体协议、攻击模型和实验另行证明。

## 内容安全要同时看能力与使用场景

风险可能涉及诈骗、恶意软件、骚扰、自残、危险操作、隐私侵犯与大规模操纵。分类和缓解需要结合能力、意图、上下文、用户授权和现实可执行性；关键词黑名单会误伤安全研究、教育与求助。

### 分层控制

- account/age/region 与用途 policy；
- input/output classifier；
- 模型行为训练和安全 prompt；
- tool/capability 限制；
- rate limit、异常检测和 abuse monitoring；
- 高风险人工复核与申诉；
- 下游可执行验证。

Classifier 也会漂移、被规避并产生群体误差。所有拦截都要同时测 false negative 和 benign false positive。

### 拒答质量

拒答不应泄露隐藏政策或有害细节；对可安全帮助的请求提供降风险替代。测试：直接、多轮、角色扮演、翻译、编码、隐晦表达和 benign neighbor。全部拒绝可以得到很低 harmful-compliance，却不是可用系统。

## 公平与群体伤害

区分：

- **representational harm**：刻板、贬损、抹除或不当关联；
- **allocative harm**：资源、机会、价格或服务质量差异；
- **quality-of-service harm**：某语言/口音/设备持续更差；
- **interaction harm**：冒犯、操纵或不尊重用户自主。

公平指标回答的问题不同。例如 accuracy parity 比较总体正确率，equal opportunity 关注真实正例的召回差异，
calibration 则比较相同预测分数是否对应相近真实概率。Base rate 不同时，这些指标可能互相冲突。

因此，指标选择要从产品决策和潜在伤害出发，而不是寻找适用于所有场景的“唯一公平公式”。

敏感属性的收集本身有隐私和法律风险；群体分类也可能错误或文化不适配。与受影响者共同定义切片、阈值、人工复核和救济。

## 超时与重试也会造成现实伤害

在医疗、金融、招聘、法律和关键基础设施中，非恶意幻觉也会造成伤害。控制包括：

- 限定适用/禁用场景；
- evidence/citation 与 deterministic verification；
- 不确定性和不可回答机制；
- 人类复核与双重控制；
- 写清系统无法判断时是停止操作，还是进入预先设计的安全降级状态；
- SLO、监控、rollback 和 business continuity。

“Human in the loop”只有在人有信息、时间、权限和能力推翻系统时才是有效控制。

## 把一次评测变成持续监控

发布前为每项高影响结果写明总体指标、关键群体切片、false negative、benign false positive、拒答和
`unjudged` 分母。发布后继续监控数据与人群构成、模型/Prompt/策略版本、申诉和人工改判；分布变化时重新评测，
不能把上线前的一次通过永久外推。

指标发现差异后，要回到具体失败：数据代表性、标签、检索、语言/设备、阈值、人工流程或产品权限。
只有能够定位责任层、指定负责人并验证修复，监控才不只是仪表盘。

两个常见误判值得单列：较小的 DP `epsilon` 只有在 adjacency、`delta`、accountant 和机制口径一致时
才可比较；“human in the loop”也只有在人拥有足够信息、时间、权限并能推翻系统时才构成有效控制。

## 自测与实践

1. 为一类 prompt、RAG 文档和 trace 分别写出目的、访问者、位置、TTL 与删除路径。
2. 为什么只报告 `epsilon` 不能比较两个 differential privacy 方案？
3. 为内容安全控制同时设计 harmful request 与 benign neighbor。
4. 选择一个高影响用途，说明哪类群体伤害、指标冲突、申诉和上线后漂移最值得先监控。

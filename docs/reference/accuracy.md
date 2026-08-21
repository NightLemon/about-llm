# 怎样判断教材结论是否可靠

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：阅读技术资料、做实验或准备在项目中引用结论的开发者。
- **先修**：不要求统计学背景；只需愿意区分“看到什么”和“能推出什么”。
- **首次阅读**：陈述类型 → 贯穿示例 → 核验步骤 → 常见误区。
- **完成信号**：能为一个技术数字写出对象、条件、证据和不可外推边界。
- **卡住时**：先问“如果这句话是错的，最小反例会是什么？”

</div>

**准确性导航**：[证据台账](../evidence/accuracy-ledger.md) · [机器可读来源](official-sources.json) · [评测方法](../quality/evaluation-methodology.md) · [项目证据](../evidence/project-controls.md)
{ .doc-nav }

内容准确不是“引用越多越好”，而是结论、证据与边界彼此匹配。一个公式推导、一次 CPU 测试、官方产品页和生产压测都可以是好证据，但它们回答的是不同问题。

## 先判断这句话属于哪一类

| 陈述类型 | 示例 | 首选验证 | 必须补充 |
|---|---|---|---|
| 数学与算法 | attention 复杂度、KV payload 公式 | 推导、可手算的数值结果、边界测试 | 变量、单位、假设和省略项 |
| 代码与 API | request 字段、事件顺序、CLI 参数 | 固定版本文档、schema、契约测试 | provider、API/runtime 版本 |
| 产品事实 | 型号、价格、配额、区域、保留政策 | 对应产品的官方页面 | 平台、账号层级、核对日期 |
| 工程测量 | 显存、TTFT、吞吐、费用 | 目标环境 benchmark 与原始结果 | 硬件、workload、版本和分母 |
| 质量判断 | 准确率、引用、安全、用户收益 | 代表性数据、基线、切片和统计 | 标签、评测器、失败样例和限制 |

同一句话常混合多类陈述。例如“4-bit 模型更小、更快且质量不变”同时包含存储、runtime、性能和质量四个命题，不能靠一个数字一起证明。

## 贯穿示例：单矩阵压缩 7.5× 能说明什么

假设一个实验把目标模型中的单个权重矩阵从 FP32 量化，并测得该矩阵 payload 缩小 7.5×。先画出推理链：

~~~text
单矩阵 payload 缩小 7.5×
  ├─? 多矩阵 bundle 缩小多少
  ├─? 完整 checkpoint 缩小多少
  ├─? 加载后峰值显存降低多少
  ├─? 推理速度提高多少
  └─? 任务质量是否保持
~~~

第一项是已经测到的局部结果，后五项都需要新证据。

### 为什么不能直接外推到 checkpoint

完整 checkpoint 还包括未量化权重、embedding、normalization、scale/zero-point、metadata 和分片索引。要回答 checkpoint 大小，必须枚举全部文件和 tensor，再比较总字节数。

### 为什么不能直接外推到显存

runtime 还可能持有反量化 buffer、activation、KV Cache、allocator reserve 和 workspace。文件变小不等于加载后峰值显存按同一比例下降。

### 为什么不能直接外推到速度

低比特计算只有在目标硬件、shape 和 batch 下命中高效 kernel 才可能更快。反量化、scale 读取和 fallback 都可能抵消带宽收益。

### 为什么不能直接外推到质量

单矩阵误差不等于整模型任务误差。质量结论需要完整模型、固定数据、基线、切片和可接受门槛。

因此准确表述应是：

> 在指定 tensor、量化格式与计数口径下，payload 缩小 7.5×。该实验尚未测量完整 checkpoint、runtime 显存、端到端性能或任务质量。

这不是“保守措辞”，而是让下一项实验变得明确。

## 核验一条结论的六个步骤

### 1. 写出最小可证伪 claim

避免“效果很好”或“支持多模态”。改写成包含对象和结果的句子：

~~~text
在 workload W、runtime R 和硬件 H 上，
candidate 相对 baseline 将 p95 TTFT 降低 X%，
同时关键质量 gate 不退化。
~~~

如果无法写出什么结果会推翻它，这个 claim 还不能验证。

### 2. 记录实际验证的对象和版本

至少记录与语义有关的身份：

- model/checkpoint revision 或闭源 exact model ID；
- tokenizer、template、adapter 和 generation config；
- API surface、version、region 与账号 tier；
- data/case split、index、tool/schema 和 policy version；
- code commit、runtime、hardware 和 checked_at。

只写品牌名或 SDK 类名通常不够。闭源 alias 还要承认它可能漂移。

### 3. 选择回答同一问题的证据

| 你想知道什么 | 合适证据 | 不足的替代物 |
|---|---|---|
| 公式是否成立 | 推导加数值边界测试 | 一次 benchmark |
| API 字段是否存在 | 固定版本官方 reference | 博客截图 |
| 本地 parser 是否正确 | 固定的成功样例和失败样例 | 文档声称支持 |
| 目标 GPU 是否可运行 | 目标环境 smoke 与日志 | CPU 单测 |
| 性能是否改善 | 固定 workload 的重复测量 | 单次 latency |
| 用户任务是否改善 | 代表性 paired eval 与切片 | 训练 loss |

证据越接近真实环境，不一定越适合回答当前问题。例如线上日志来自真实流量，却可能没有可比较的 baseline；
手算小例子很适合检查公式，却不能说明生产性能。

### 4. 主动构造反例

每个成功实验至少配一个能让结论失败的改动：

- causal attention：只改变未来 token；
- RAG：移除或替换关键证据；
- Agent：审批后修改参数；
- streaming：terminal 前断开连接；
- quantization：换成没有低比特 kernel 的 shape；
- evaluation：加入一个关键 slice 的退化样例。

没有负例时，你可能只证明了程序执行到结束。

### 5. 保存原始结果和分母

汇总数字应能下钻到 per-case、per-request 或 per-tensor 结果。至少保留：

- 输入与配置 fingerprint；
- baseline/candidate 原始输出；
- 成功、失败、拒绝、超时和 unknown 的完整分母；
- metric 版本、aggregation unit 和 slice；
- 环境、时间、usage 与错误信息。

Hash 只能绑定被纳入计算的 bytes；它不是来源认证、保密、许可或业务正确性的证明。

### 6. 紧邻写出不能外推什么

边界不是附录里的免责句，而是结论的一部分。建议用固定句式：

> 该证据证明……；尚未覆盖……；下一项最可能推翻结论的实验是……。

读者应能立即区分已观察结果、合理推断和未来工作。

## 五个常见证据升级错误

### “官方文档写了，所以代码已经可用”

官方文档证明的是核对时的产品或接口承诺。它不证明当前账号、区域、网络、依赖版本或凭证已成功执行。

### “单元测试通过，所以生产安全”

单元测试只覆盖给定输入和实现路径。生产安全还需要身份、权限、秘密、并发、故障恢复、监控和目标环境验证。

### “JSON/schema 通过，所以结果正确”

Syntax 和 schema 只约束结构。字段可能事实错误、单位错误、引用不存在，工具动作也可能未授权。

### “文件 hash 一致，所以来源可信”

普通 hash 能检测已知 bytes 是否变化，不能认证发布者。来源认证还需要签名、可信分发、权限和审计链。

### “平均指标提升，所以可以发布”

平均值可能隐藏关键语言、租户、难度或安全 slice 的退化。发布判断还要看 effect size、区间、故障分母和预先定义的硬约束。

## 引用证据要分五层

一个答案带有 `[source-1]` 不等于 grounded。至少分开：

1. **语法**：引用能否解析，source ID 是否存在。
2. **授权**：当前主体是否有权读取该 source 与版本。
3. **证据 span**：quote/offset 是否真的对应原文。
4. **语义支持**：证据是否蕴含当前 atomic claim。
5. **发布 policy**：来源是否真实、时效是否满足，答案应发布、拒答还是升级。

前一层通过不能借给后一层。精确 span 也可能引用一段与 claim 无关的文字。

## 怎样处理会变化的事实

型号、价格、窗口、配额、区域、API 字段和数据政策都应带 `checked_at`。稳定教材优先解释选择方法，不维护“永久最新”的排行榜。

更新时：

1. 核对具体平台、API/version 和目标页面；
2. 更新机器可读来源与核对日期；
3. 判断变化影响的是文档声明、代码契约还是已录制实验；
4. 重新运行受影响的 capability probe 或评测；
5. 保留旧 artifact，不能用新依赖的 live 通过改写历史证据。

本仓库的逐来源日期、论文快照和官方链接保存在[内容准确性证据台账](../evidence/accuracy-ledger.md)。

## 怎样阅读本仓库的证据

看到“可运行”“通过”或“已验证”时，继续问四件事：

- 结果来自手算或参考实现、仓库准备的固定样例、真实 framework、目标权重，还是远端服务？
- 环境是 CPU、单卡 GPU、loopback 还是真实网络？
- 输入是本仓库准备的小样例，还是有代表性的 held-out data？
- 结论是机制、集成、性能、质量还是生产责任？

不同验证可以补足不同层面，但不能简单相加后升级结论。例如 CPU parser、官方 SDK in-memory 和
HTTP MockTransport 都通过，仍不等于真实远端认证、兼容性或生产安全已经完成。

固定样例的精确数字、runtime、hash 和未覆盖项统一放在[证据台账](../evidence/accuracy-ledger.md)，
避免它们淹没教材正文。

## 作者与维护者流程

修改公式、API、产品事实或实验数字时：

1. 确定陈述类型和最小 claim。
2. 找到同一层级的来源或可执行证据。
3. 添加成功路径、边界和故意失败用例。
4. 更新正文、测试、来源登记和证据台账。
5. 运行：

~~~powershell
python scripts/check_content_accuracy.py
python scripts/check_docs.py
python -m mkdocs build --strict
~~~

6. 目标环境结果保存 workload manifest、原始输出和软件/硬件版本。
7. 发现勘误时更新 `CHANGELOG.md`，不要静默改写历史 artifact。

## 读者快速自查

引用一个结论前，确认自己能回答：

- 它具体声称什么，怎样被推翻？
- 对象、版本、单位、环境和日期是什么？
- 证据直接回答了这个问题吗？
- 分母是否包含失败、拒绝、超时和 unknown？
- 哪个相邻结论还没有被证明？
- 下一项最值得做的反例是什么？

六个问题都能回答时，这条结论才适合进入设计、简历或发布判断。

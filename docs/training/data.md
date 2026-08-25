# 训练数据工程与治理

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：负责预训练、微调数据、数据治理或训练流水线的工程师。
- **先修**：知道 JSONL、hash、train/validation/test split 和 tokenizer 的基本作用。
- **首次阅读**：跟踪一个论坛帖子怎样进入 shard，又怎样响应删除请求。
- **完成信号**：能从训练 token 追到来源，并解释切分、去重和删除怎样重建。
- **卡住时**：先读[机器学习与深度学习](../foundations/ml-dl.md)中的数据划分。

</div>

设想我们从一个技术论坛采集到帖子 `thread-8841`。它包含一段很有价值的 GPU OOM 排查记录，
也混着代码缩进、用户签名和回复。三个月后，来源方要求删除这条帖子。

如果团队只保存了最终 `train-00031.bin`，就会立刻遇到一串无法回答的问题：帖子来自哪个快照？
解析器保留了哪些回复？它是否和另一站的镜像重复？进入 train 还是 validation？落在 shard 的哪个 token span？
哪些 checkpoint 或 adapter 消费过它？

训练数据工程的核心，就是在数据第一次进入系统时建立这些关系。模型最后看到的虽然只是 token IDs，
工程系统管理的却是一条从来源、快照、转换、决策到训练消费的供应链。

本章讨论技术设计与验证方法，不提供法律意见。版权、隐私、合同和地域要求应由具备相应职责与权限的人审查。

## 先看 `thread-8841` 怎样移动

它在一次构建中经历：

| 阶段 | 本例产生的记录 | 以后靠它回答什么 |
|---|---|---|
| Source registry | `forum-cn / terms@2026-07` | 为什么允许采集，用途与期限是什么 |
| Raw snapshot | 原始响应、时间、content hash | 当时到底取得了什么 |
| Parse | 主帖、回复、代码块与父子关系 | 解析器怎样理解页面结构 |
| Normalize | `forum-text@v3` | 哪些字符或空白发生过变化 |
| Policy / quality | keep、risk labels、reason codes | 为什么保留、隔离或删除 |
| Dedup cluster | canonical 与镜像成员 | 重复项之间是什么关系 |
| Split / mixture | `train`、domain、sampling weight | 它在哪个实验中怎样被抽样 |
| Tokenize / pack | tokenizer revision、shard、span | 哪些 objective token 真正进入训练 |
| Consumption | run、checkpoint、token ledger | 哪些产物受这条数据影响 |

```mermaid
flowchart LR
  A["Source registry"] --> B["Immutable raw snapshot"]
  B --> C["Parsed documents"]
  C --> D["Normalized candidates"]
  D --> E["Policy / quality / PII decisions"]
  E --> F["Exact + near-dedup clusters"]
  F --> G["Split and mixture manifests"]
  G --> H["Tokenized packed shards"]
  H --> I["Consumed-token ledger"]
```

每一层都保存父对象与版本。这样 parser 或 tokenizer 更新时会产生新 dataset version，
而不是静默覆盖旧数据。给定相同快照、配置和 seed，构建应得到相同 item 集合与稳定 ID；
不同硬件生成的压缩文件未必逐字节一致，但内容差异必须可解释。

## 先亲手追踪一次 { #training-data-lineage-run }

仓库已经把 `thread-8841` 的关系缩成一个可以在 CPU 上运行的小例子。它包含帖子 `r2`、`r3` 两个版本、
一份镜像、两个训练 token 区间、一次训练任务和两个 checkpoint。先从仓库根目录运行：

~~~powershell
python projects/training-data-lineage/thread_lineage.py verify
~~~

这个命令会重新读取关系图并从头计算报告。看到 `"verified": true`，说明仓库保存的报告与当前输入一致。
要查看完整路径，再运行：

~~~powershell
python projects/training-data-lineage/thread_lineage.py trace
~~~

先找 `trace.shard_spans`。主帖位于 `[4096, 4192)`，回复位于 `[4192, 4256)`；区间采用左闭右开写法，
所以二者相邻但不重叠。然后找 `trace.consumers`，就能从 `train-00031.bin` 追到
`pretrain-cn-run-42` 及它的两个 checkpoint。本章后面的设计，都是为了让这次反查在真实流水线中仍然成立。
命令、文件说明和故意失败练习见
[Training Data Lineage 项目](https://github.com/NightLemon/about-llm/tree/main/projects/training-data-lineage)。

## 下载之前，先登记来源和用途

“网页能打开”只说明可以访问，不回答是否允许训练、衍生权重、样本展示、再分发或商业使用。
Source registry 应记录：

- 获取方式、授权依据，以及 license / contract / terms 的快照；
- 允许的模型、产品和用途，是否允许再分发，何时到期；
- 个人信息、敏感类别、未成年人和数据驻留风险；
- opt-out、删除与争议处理入口；
- 来源中哪些部分属于第三方内容或不同许可证。

页面 footer 的许可证未必覆盖用户评论、图片、附件和引用。代码仓库也可能同时包含原创代码、vendored dependency、
生成文件和多个许可证。自动探测器适合发现待复核项，`unknown` 应进入隔离或人工决策，而不是自动变成允许。

来源白名单（source allowlist）应在下载前生效。若先抓取所有内容、再靠正则清理，敏感数据已经进入缓存、日志、
备份和中间文件。[SFT 数据流水线](sft-data-pipeline.md)展示了一个较小的工程样例：来源未登记时停止，并把用途
写进记录。它附带的扫描器只能发现少量已知模式，不能代替完整的个人信息检测或法律判断。

## 原始快照（raw snapshot）回答“当时拿到了什么”

原始数据层（raw layer）尽量不可变，访问权限也最严格。对 `thread-8841`，至少保存规范网址、采集时间、
HTTP 响应信息、原始字节的哈希和来源版本。后面的解析、规范化与分词结果都要指回这份快照。

只保存清洗后的正文会丢失两类证据：一是 parser 当时是否错删或错排，二是来源后来更新后，
我们无法重建旧版本。不可变不意味着无限保留；保留期限、访问和删除仍受来源政策约束。

### 不同格式会以不同方式解析错

- **网页**：导航、Cookie 提示、评论、隐藏文本和动态区域可能混入正文。论坛与文档站通常需要不同的解析配置。
- **PDF 与扫描件**：多栏、页眉、脚注、表格和 OCR 会破坏阅读顺序。抽样检查乱码、页码映射和 OCR 置信度。
- **代码**：保留仓库、版本、路径和许可证边界，把生成文件、压缩文件、锁文件与手写源码分开。
- **对话与日志**：保持完整会话边界，并单独处理用户同意、个人信息、工具结果和系统指令。

对 `thread-8841`，解析器要把主帖、回复和代码块分别保留，同时记录父子关系。若把同一讨论串拆成独立 JSONL 行，
后续再逐行随机切分，就可能让一部分回复进入训练集、另一部分进入测试集，制造明显泄漏。

## 规范化会改变信息

Unicode normalization、空白、HTML entity、大小写和数字格式处理都可能改变语义：

- NFKC 会合并一些原本需要区分的字符；
- 连续空格对 Python 缩进和 Markdown 表格有意义；
- 大小写影响专有名词与代码；
- 删除标点会改变版本号、数学式和否定语气；
- URL 参数有时是跟踪信息，有时决定页面内容。

因此 normalization profile 应按 format 或 domain 版本化。`thread-8841` 的正文可以折叠普通段落空白，
代码块必须保留缩进。用于 exact dedup 的规范化副本也不必等于最终训练文本。

Stable document ID 表示同一来源对象，content hash 表示某个 revision 的具体内容。正文更新后，
ID 可以保持关联、hash 应变化。只用正文 hash 做身份，会把一次更新误当成毫无关系的新文档。

## 过滤器每保留一次，也在改变数据分布

质量信号可以来自文本长度、字符分布、重复行、压缩率和语言识别，也可以来自困惑度、广告或安全分类器、
结构检查、代码测试和人工标签。无论来源是什么，系统最终都要给出一个可解释的决定：保留、隔离、降权或删除。

因此不能只报告“过滤了 18%”。至少还要看：

- 各语言与领域的 precision、recall 和 confusion matrix；
- 阈值附近样本的人工复核；
- 方言、口语、少数语言和辅助技术文本的 false positive；
- 多个规则串联后的累计选择偏差；
- 更新规则前后，哪些 item 改变了决定。

如果另一个 LLM 给出“高质量”标签，记录 model revision、Prompt、temperature、parser 与失败率。
Judge 可能偏好较长、较标准或与自身风格相似的文本，这种偏好会逐渐写进训练分布。

来源权限不明、含有明确密钥或被政策禁止的数据，通常需要直接排除。教育价值、写作风格和难度更适合分桶或调低权重，
因为一个不可靠的阈值可能永久删除稀有表达。无论是排除还是调权，决定与原因都要写进构建清单（manifest）。

个人信息扫描器本质上也是分类器。正则、校验和、命名实体识别、上下文分类器与人工审计可以组合使用，
但公开的企业电话与私人医疗记录不能仅因格式相同就得到同一处置。密钥还需要专门的扫描与吊销流程。

Redaction 会留下上下文线索，也会制造新的高频占位符。替换策略应在隐私威胁模型和下游质量上一起验证。

## 去重先找候选，再决定关系

另一网站完整转载了 `thread-8841`。Exact dedup 可以对规范化文本计算 cryptographic hash，快速发现逐字相同内容。
但系统仍需决定去重单位、canonical item，以及两个来源的许可冲突如何处理。

Cluster membership 必须保留。如果只留下 canonical hash，来源方要求删除 canonical 时，系统既无法证明镜像关系，
也不知道能否合法地切换到另一个成员。

### 近似去重到底近似什么

轻微改写、模板插入或代码格式变化不会得到相同 hash。将文档表示为 n-gram / shingle 集合 \(A,B\) 后，
可以用 Jaccard 衡量词面重合：

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
\]

MinHash 用签名相同率估计 Jaccard：

\[
\hat J(A,B)=\frac{1}{k}\sum_{i=1}^{k}\mathbf 1[h_i(A)=h_i(B)].
\]

将 \(k=br\) 个签名分成 \(b\) 个 band、每 band \(r\) 行。在理想独立模型中，
相似度为 \(s\) 的 pair 至少命中一个完整 band 的概率是：

\[
P(\text{candidate}\mid s)=1-(1-s^r)^b.
\]

这个式子用于理解参数方向。增加 \(b\) 往往带来更多候选和更高召回；增加 \(r\) 会收紧候选。
有限 hash family、短文本、公共模板和相关签名会让真实数据偏离理想曲线，所以最终仍要在目标切片上标注 pair。

MinHash 近似的是 shingle-set overlap，不是语义等价。SimHash 更接近加权特征的 Hamming neighborhood；
embedding 可以发现语义改写，同时也引入模型、语言和领域偏差。

### 动手观察一次 LSH 漏检和误报

运行：

~~~powershell
python projects/single-gpu-finetuning/minhash_lsh_toy.py
~~~

仓库实现先用 SHA-256 把 Unicode 文本片段稳定映射到 \(2^{61}-1\) 模域，再用随机种子派生哈希系数。
LSH 的分带命中只产生候选对，程序随后还会重算精确 Jaccard。

固定样例包含 5 个 item，一共形成 10 对比较。使用 64 个哈希并按 16 组、每组 4 行分带后，程序找到 3 个候选对；
其中只有 1 对达到 0.8，所以这次精确率是 `1/3`。穷举全部 10 对得到的召回率为 1。另一个只用 1 个哈希的反例中，
两集合的 Jaccard 是 `2/3`，却没有分带碰撞，实测召回率为 0。

这些数字只解释当前小样例的候选机制。换成新闻镜像、代码分支或数学证明后，需要重新标注文本对，
并报告精确率、召回率、簇大小、token 移除率，以及不同语言和来源的差异。canonical item 的选择也要单独说明。

去重顺序同样会改变结果：先切 chunk 容易删除共享局部段落；先对整文档去重会保留局部复制。
Pretraining、RAG 和 benchmark contamination 通常需要不同粒度。

## 切分（split）的单位要沿着真实依赖选

若同一讨论串、用户、代码仓库、模板家族或时间段内的样本彼此相关，就应该按这一组关系切分。
只有当每一行近似独立时，逐行随机切分才合理。

污染不只包括题干和答案原文，还包括翻译、格式转换、轻微改写、benchmark 上游网页，
以及 SFT example 或 RAG corpus 中可以直接取得的答案。一个稳妥流程是：

1. 构建训练集前，先固定评测集目录和保留集身份；
2. 在原始数据、解析结果和最终数据上分别查找精确重复、文本片段重合与语义近似候选；
3. 人工复核高风险文本对，保存匹配位置、来源与决定；
4. 后续发现污染时，标记受影响的模型版本，并在干净子集上重测。

检测报告要说明查过哪些来源、语言和变体。“没有命中”只表示当前方法没有发现候选。
测试集被反复用于过滤和调参后，也已经变成开发集，不再提供独立的最终评估。

## 混合比例（mixture）决定模型实际反复看什么

去重后的 item 不会自动按原始比例进入训练。若域 \(i\) 有 \(n_i\) 个 token，一种温度采样是：

\[
p_i=\frac{n_i^\alpha}{\sum_j n_j^\alpha},
\qquad 0<\alpha\le 1.
\]

当 \(\alpha=1\) 时按数据量采样；\(0<\alpha<1\) 会相对提高小域权重。
若计划消费 \(D\) 个 token，域 \(i\) 的期望消费量为 \(Dp_i\)，重复倍数约为 \(Dp_i/n_i\)。
训练前先算这两个数，可以发现一个小域是否会被重复几百次。

混合比例可以分阶段变化，例如先做广域预训练，再增加高质量领域数据。每次变化都记录训练步数或已消费 token 的边界、
采样器随机状态和每个领域的实际消费量。恢复位置、数据加载进程数或 shard 顺序都可能让实际取样偏离目标权重。

上采样、复制 item 和调整损失权重会产生不同的优化噪声，也会改变一轮训练的含义。数据报告应分别给出：

```text
unique raw / normalized tokens
unique tokens after dedup
consumed objective tokens
padding / masked tokens
per-domain repetition
```

## 分词与拼接（packing）决定训练目标

同一段文本更换 tokenizer 后，token 数、截断点、不同语言的成本和污染匹配都会变化。
因此，分词器的版本也属于数据集身份。

将短文档拼成长度 \(T\) 的 sequence 可以减少 padding，但必须明确：

- 文档间是否插入 BOS / EOS；
- label shift 是否跨过文档边界；
- attention 是否允许跨文档；
- position IDs 是否重置；
- 截断尾部是丢弃、续到下一条，还是先补 EOS；
- loss denominator 是否只包含有效 objective tokens。

例如 `[docA, EOS, docB]` 中，如果让 `EOS` 预测 `docB` 的首 token，就人为加入了跨文档目标。
可以 mask 该 label，也可以接受它，但训练定义必须明确。插入 EOS 本身不会改变 attention mask；
真正隔离文档需要 block-diagonal mask 或等价机制。

Packing efficiency 可以写成：

\[
\text{packing efficiency}
=
\frac{\text{valid objective tokens}}{\text{allocated sequence slots}}.
\]

这个比例只描述空间利用。错误的 loss mask 也可能得到 100% 利用率，同时把 padding 或错误边界纳入训练。
发布前应抽样打印 text、token IDs、labels、loss mask、position IDs 与 document boundaries，并用小例子逐位置核对。

## 合成数据也要进入同一条 lineage

指令扩写、难例、翻译、蒸馏、工具轨迹和拒答样本都可能来自生成模型。
每条合成数据至少保留：

- 生成模型、Prompt 模板、采样设置与生成时间；
- 原始响应和随机种子；只有目标接口承诺可重放时，种子才表示可复现；
- 验证器、执行测试、过滤原因与人工复核状态；
- 父数据，以及继承的许可证和隐私约束。

同一模型同时生成、评价和过滤时，盲点会相关。可以引入独立验证器、可执行测试、人工抽样、真实数据锚点和多样性约束。
拒绝采样提高的是通过特定验证器的比例，也可能只是在利用验证器漏洞。

合成数据不会自动导致或避免 model collapse。风险取决于占比、选择机制、真实锚点、模型和跨代迭代，
因此要观察覆盖、长尾和多代质量，而不只看平均 loss。完整讨论见[合成数据、蒸馏与反馈环](synthetic-data.md)。

## 删除请求怎样沿来源链返回

现在来源方要求删除 `thread-8841` 的全部历史版本。系统需要沿着这条关系查找：

```text
source item
  -> raw / parsed / normalized revisions
  -> dedup cluster and canonical decision
  -> split / mixture manifest
  -> tokenized shard and token spans
  -> training runs
  -> checkpoints / adapters
```

前面的 `trace` 命令会给出具体结果：`r2` 与 `r3` 都受影响；当前数据集需要移除主帖和回复；
`train-00031.bin` 需要重建；`pretrain-cn-run-42` 与两个 checkpoint 需要进入后续处置清单。

处理时先写删除标记（tombstone），让未来构建不再消费该来源；然后更新去重簇并重建受影响的 shard。
主帖是当前簇的 canonical item，删除后虽然还剩一份镜像，也不能直接拿镜像顶替。团队要重新判断镜像是否有独立授权，
以及它是否符合当前质量规则。

删除 final dataset 中一行不会让已经训练的权重恢复到未见过该数据的状态。后续处置可能包括未来训练排除、
从干净 checkpoint 重训、经过验证的近似 unlearning，或系统层过滤。采用哪种方式取决于政策和威胁模型，
报告中应区分“数据存储已删除”和“模型影响已处置”。

同一来源版本重跑应得到同一结果；解析器、规则或分词器更新后，则应产生新的数据集版本。
构建清单要保存父版本、新增、更新和删除数量，以及去重簇的变化。旧版本不能被悄悄覆盖。

## 构建清单（manifest）把一次构建固定下来

```yaml
dataset_id: pretrain-cn-2026-08-v3
parent: pretrain-cn-2026-08-v2
raw_snapshot: sha256:...
source_registry_revision: git:...
parser_profiles:
  web: readability-v4
  pdf: layout-ocr-v2
normalization_profile: multilingual-safe-v3
quality_policy: quality-gate-v7
dedup:
  exact_profile: exact-v2
  near_profile: minhash-5gram-v4
  canonical_policy: license-quality-time-v2
evaluation_registry: eval-holdouts-v5
pii_policy: pii-v6
tokenizer_revision: sha256:...
split_seed: 7319
mixture_revision: mix-v8
output_shards_manifest: sha256:...
```

版本字符串必须能找回当时的配置、代码版本和统计结果。只写 `quality-gate-v7`，
却找不到当时规则和输出，不足以重建数据集。

发布前用四组问题检查 manifest：

| 范围 | 必须能回答 |
|---|---|
| 身份与来源 | 每个最终 item 来自哪里，shard 中的 token 区间怎样回到来源，删除冲突是否阻止发布 |
| 数据质量 | 解码、长度、语言、来源、过滤和 cluster 分布是否符合预期 |
| 训练语义 | labels、损失掩码、文档边界、EOS、位置编号和实际混合比例是否正确 |
| 安全与评测 | 个人信息、密钥、保留集污染与恶意输入覆盖到什么范围 |

数据加载器还要限制路径、压缩炸弹、恶意 pickle 和超大对象。元数据来自不可信来源时，
不能让它直接决定训练脚本路径、命令或高权限配置。

## 从训练异常反查数据链

| 现象 | 优先沿哪些记录检查 |
|---|---|
| Loss 突然异常地好 | 训练与验证重叠、label 错位、padding 掩码、重复与答案污染 |
| 某语言 loss 变差 | 分词膨胀、语言识别误杀、混合比例、规范化、截断与去重变化 |
| 恢复训练后混合比例改变 | 数据游标、采样器随机状态、worker 数、shard 顺序、并行规模与 token 消费账本 |
| 数据量对不上 | 按原始、解析、保留、去重、分词、分配空间与训练目标逐层对账 |

这些层的计数不能合并成一个“总 token 数”。每层回答不同问题，差值本身就是定位 parser、filter、dedup 或 packing 的线索。

## 自测与实践

1. 为 `thread-8841` 设计稳定来源 ID、版本和内容哈希；三者分别在什么情况下变化？
2. 为什么精确去重使用的规范化规则不一定适合最终训练文本？
3. 两个代码文件共享许可证头时，shingle Jaccard 可能怎样误导去重？
4. 给三个领域计算温度采样后的预期消费 token 数与重复倍数。
5. 画出删除请求从来源 item 到 checkpoint 的路径，并标出能直接重建与需要额外处置的部分。
6. 构造 `[docA, EOS, docB]`，逐位置写出 input、label、loss mask 和 attention mask。

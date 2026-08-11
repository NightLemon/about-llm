# 训练数据工程与治理

训练数据不是一个 `.jsonl` 文件，而是一条从采集权限、原始快照、解析、过滤、去重、切分、混合到 token packing 的可追溯供应链。模型能力、记忆、偏见、污染和删除能力都受这条供应链约束。

本章讨论工程与验证方法，不提供法律意见。版权、隐私、合同和地域要求必须由具备相应权限的负责人审查。

## 1. 数据契约先于数据规模

每个进入训练候选池的对象至少需要：

- `source_id`：来源或授权集合；
- `document_id`：在规范化策略下稳定的文档身份；
- `snapshot_id` 与采集时间；
- 原始内容 hash、解析器版本和派生父对象；
- 语言、格式、域和时间元数据；
- 许可/使用范围、保留期限与删除标识；
- 过滤、去重、PII、污染处理决策及 reason code；
- train/validation/test 或 holdout assignment；
- tokenizer revision、token 数和最终 shard。

如果只保存最终 token IDs，就很难回答“某条数据从哪里来、为何被保留、进入了哪些模型、怎样删除和重建”。

## 2. 分层存储与不可变快照

推荐分层：

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

- **Raw 层**尽量不可变，便于重新解析和审计；访问权限应最严格。
- **Parsed 层**保留结构、页码、代码文件路径等可追溯信息。
- **Candidate 层**执行可版本化规范化，不覆盖 raw。
- **Decision 层**保存每个规则/模型的结果，而不只保存最终 `keep=true`。
- **Training 层**保存确定的 split、mixture、tokenizer 和 shard manifest。

“可重跑”要求给定相同原始快照和版本，能生成相同文档集合与稳定 ID；不要求不同硬件下压缩文件逐字节相同，但差异必须可解释。

## 3. 采集、许可与来源登记

“公开可访问”不等于“可用于训练、再分发或商业用途”。来源登记应记录：

- 获取方式与授权依据；
- license/contract/terms 的快照或引用；
- 是否允许训练、衍生权重、样本展示和数据再分发；
- 个人信息、敏感类别与未成年人风险；
- 国家/地区、数据驻留和跨境限制；
- opt-out、删除和争议处理入口；
- 允许的模型/产品范围与到期时间。

页面 footer 出现某个 license 字样不一定覆盖用户评论、图片、第三方引用或附件。代码仓库也可能包含多许可证文件和 vendored dependency。许可证探测器只能做初筛，不能把 `unknown` 自动当作许可。

### 3.1 Source allowlist 优于事后黑名单

高风险来源应在采集入口隔离。先抓取一切再靠正则删除，会让敏感内容进入日志、缓存、备份和中间对象。Source registry 应默认拒绝未知权限，而不是把“没有标记”解释为允许。

仓库的 SFT 项目给出一个可执行 reference：严格 registry 精确匹配 source/license，并绑定用途、evidence、review time、expiry 和风险标签；同时以不记录命中原文的有限 regex/checksum scanner 生成待复核候选。它展示的是 fail-closed 决策与 artifact identity，不是法律结论或完整 PII/secret 检测，详见[SFT 数据、模板与训练闭环](sft-data-pipeline.md)。

## 4. 解析：保留结构与失败证据

### 4.1 Web

需要处理导航、cookie banner、广告、评论、重复模板、隐藏文本与动态页面。正文提取器对论坛、文档站和表格可能采取不同策略。保留 canonical URL、抓取时间、HTTP 元数据和 DOM/正文 hash。

### 4.2 PDF 与扫描件

PDF 文本顺序不等于视觉阅读顺序。多栏、页眉页脚、公式、脚注和表格容易错位；扫描件还依赖 OCR。至少抽样检查：

- 字符缺失/乱码率；
- 阅读顺序；
- 重复页眉页脚；
- 页码与来源映射；
- OCR language/model 与 confidence；
- 表格是否被错误拼成陈述。

### 4.3 代码

保留 repository、revision、文件路径和 license boundary。不要把 minified、generated、binary-as-text、lockfile 和 vendored code 与手写源代码等权。Secret scanning 必须发生在进入训练集之前，但检测器不能保证无漏报。

### 4.4 对话与日志

会话必须按完整 session 保持边界，并处理 consent、PII、工具结果和系统提示。把同一会话轮次随机打散到 train/test 会严重泄漏。线上日志还受旧模型策略影响，不代表独立自然分布。

## 5. 规范化不是无损操作

可选操作包括 Unicode normalization、换行/空白处理、HTML entity 解码、大小写与数字格式统一。每一步都可能破坏信息：

- NFKC 可能合并视觉/语义上需要区分的字符；
- 连续空格对代码、Markdown 表格和 Python 缩进有意义；
- 大小写影响专有名词与代码；
- 删除标点会改变数学、版本号和否定语气；
- URL 参数可能是跟踪信息，也可能决定文档内容。

建议按 format/domain 选择 normalization profile，并保存 profile version。用于 exact dedup 的规范化文本不必等于最终训练文本。

## 6. 质量过滤是一个带偏差的分类系统

常见信号：

- 长度、字符类别、重复行、异常符号与压缩率；
- language ID 与 script consistency；
- 文本/代码 perplexity 或小模型 score；
- 广告、SEO、成人、仇恨、恶意代码等 classifier；
- 结构完整性、引用、可执行测试和人工质量标签。

### 6.1 不要只看过滤率

过滤器需要像分类器一样评估：

- 各语言/域的 precision、recall 和 confusion matrix；
- 阈值附近样本的人工复核；
- 少数语言、方言、口语和辅助技术文本的 false positive；
- 多个规则串联后的累计选择偏差；
- 模型/规则更新前后的保留集差异。

若“高质量”标注来自另一个 LLM，记录其 model revision、prompt、temperature、解析器和失败率。Judge 可能偏好更长、更标准化或与自身风格相似的文本。

### 6.2 Hard filter 与 soft weight

违法/无权/明确安全风险通常需要 hard exclusion。教育价值、风格或难度等连续属性可考虑分桶或采样权重，以免一个不可靠阈值永久删除多样性。权重仍会改变有效分布，必须进入 manifest。

## 7. 精确去重

Exact dedup 常对规范化内容计算 cryptographic hash。需要定义：

- 文档、段落还是固定窗口为单位；
- 是否忽略空白、大小写、时间戳和模板；
- cluster 中保留哪一个 canonical item；
- 来源许可冲突时如何处理；
- 删除 canonical 后是否可切换到其他成员。

只保留 hash 而不保留 cluster membership，会失去来源归因和删除能力。

## 8. 近似去重

### 8.1 Shingling 与 Jaccard

把文档表示为 n-gram/shingle 集合 \(A,B\)：

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
\]

MinHash 近似集合 Jaccard，LSH 用于减少候选对。它们不是“语义相似度”：短文本、模板和公共引用会显著影响结果。SimHash 更接近特征加权后的 Hamming neighborhood；embedding 又引入模型语义偏差。

对 (k) 个 MinHash 分量，可用签名相同率估计 Jaccard：

\[
\hat J(A,B)=\frac{1}{k}\sum_{i=1}^{k}\mathbf 1[h_i(A)=h_i(B)].
\]

将签名切成 (b) 个 band、每 band (r) 行且 (k=br)。在“分量独立且单分量相等概率恰为 (s=J(A,B))”的理想模型下，至少一个完整 band 相等、从而成为候选的概率是

\[
P(\text{candidate}\mid s)=1-(1-s^r)^b.
\]

这是调参启发式，不是单个 pair 的召回保证。增大 (b) 通常提高召回并增加候选；增大 (r) 通常减少候选并增加漏检。有限 hash family、shingle hash collision、短文本与相关分量会使实际行为偏离理想曲线，因此必须在目标数据切片上审计。

仓库 `minhash_lsh.py` 不使用进程随机化的 Python `hash()`：先用 SHA-256 将 Unicode shingle 稳定映射到 (2^{61}-1) 模域，再以 seed 派生 affine universal-hash 系数；band key 是完整 signature slice，候选必须重新计算精确 Jaccard 分子/分母。SHA-256 和 manifest 都没有密钥，不认证数据来源，也不把 lexical overlap 变成 semantic/translation duplicate。

运行：

~~~powershell
python projects/single-gpu-finetuning/minhash_lsh_toy.py
~~~

固定 5-item、10-pair authored fixture 用 64 hashes、16 bands×4 rows 得到 3 个候选；精确复核只有 1 个达到 0.8，另 2 个是 false-positive candidate，因此 snapshot precision=`1/3`。Exhaustive 10-pair ground truth 在这个快照上得到 recall=1；这不外推到其他数据。测试另固定一个 1-hash 反例：两个集合精确 Jaccard=`2/3` 却没有 band collision，observed recall=0，直接否定“LSH 不会漏”的说法。

### 8.2 阈值必须按域校准

同样 0.8 的相似度对新闻镜像、许可证文本、代码 fork 和数学证明含义不同。抽样标注 candidate pairs，至少报告：

- pair-level precision/recall；
- cluster size 分布与巨型 cluster；
- 文档/token 移除率；
- 各来源/语言的移除差异；
- canonical selection 对许可和质量的影响。

### 8.3 去重顺序改变结果

先切 chunk 再去重容易删除共享局部段落；先整文档去重会保留局部复制。Pretraining、RAG 和 benchmark contamination 需要不同粒度，不能共用一个阈值后宣称问题已解决。

## 9. Train/test 污染与独立单位

污染包括：

- 题干、答案或解析原文；
- 翻译、格式转换和轻微改写；
- benchmark 源网站的上游版本；
- SFT prompt、few-shot example 或 RAG corpus 中可直接访问答案；
- 同一作者、用户、模板或文档的近重复跨 split。

### 9.1 检测流程

1. 在训练构建前冻结 evaluation registry 和 holdout hashes。
2. 对 raw/parsed/final data 分别做 exact、shingle 和可审计的 semantic candidate search。
3. 人工复核高风险候选，保存 match type、span、source 和 decision。
4. 训练后发现污染时，不静默删除报告；标记受影响版本，重测 clean subset。

“未检测到”不等于“没有污染”，尤其在闭源数据或语义改写下。污染检测报告应给覆盖范围与漏检边界。

### 9.2 Split 的单位

按最小独立单位分组：document、thread、user、repository、template family 或 time period。随机逐行 split 只适用于行近似独立的情况。测试集被反复用于过滤/调参后，就不再是无偏 test。

## 10. PII、Secrets 与敏感内容

PII 检测可组合 regex、checksum、NER、context classifier 和人工审计。不同字段风险不同：公开企业电话与私人医疗记录不能只按“都是电话号码”处理。

最低要求：

- 在 raw、parsed 和 final 层记录检测版本与结果；
- 高风险 source 在采集层隔离；
- secret/API key 使用专门检测和吊销流程；
- 评估每类 precision/recall，而不只报命中数；
- 限制原文访问和日志复制；
- 数据删除能追踪到 shard、checkpoint 和衍生模型。

Redaction 会留下上下文线索，也可能破坏句子。用固定占位符会产生新的高频模式；删除/替换策略应在下游质量和隐私威胁模型中验证。

## 11. 数据混合与采样

若域 \(i\) 有 \(n_i\) 个 token，一种温度式采样为

\[
p_i=
\frac{n_i^\alpha}{\sum_j n_j^\alpha},
\qquad 0<\alpha\le1.
\]

- \(\alpha=1\)：按数据量比例；
- \(0<\alpha<1\)：相对上采样低资源域；
- 这不是唯一 mixing rule，也不证明低资源数据质量相同。

每个域的 expected consumed tokens 为 \(D p_i\)，重复倍数约为 \(Dp_i/n_i\)。在训练前计算这两个量，可以提前发现小域被重复数百次。

### 11.1 静态与阶段性 mixture

Mixture 可随阶段变化，例如先广域预训练，再加入更多高质量/领域数据。每次变化都可能导致 loss jump；必须记录 step/token 边界、随机采样器状态和每域实际消耗量，而不只保存目标权重。

### 11.2 Sample weight 不等于数据量

上采样、loss weighting 和重复复制会产生不同 optimizer noise、epoch 定义和数据顺序。报告“看过多少 token”时区分：

- unique raw/normalized tokens；
- final unique tokens after dedup；
- consumed objective tokens；
- padding/masked tokens；
- 各域重复轮数。

## 12. Tokenization 与 packing

Tokenizer revision 是训练数据版本的一部分。同一文本换 tokenizer 后 token 数、截断位置、语言成本和污染 match 都会变化。

### 12.1 因果 LM packing

把多个短文档拼入长度 \(T\) 的 sequence 可减少 padding，但要显式定义：

- 文档间是否插入 BOS/EOS；
- label shift 是否跨文档；
- attention 是否允许跨文档；
- position IDs 是否连续或重置；
- 被截断尾部是否丢弃、续到下一 sequence 或加 EOS；
- loss denominator 是否只包含有效 objective tokens。

若序列为 `[docA, EOS, docB]`，让 `EOS` 预测 `docB` 首 token 会给模型一个人为跨文档目标。常见方案是 mask 该边界 label，或接受这一目标但明确其含义。插入 EOS 本身并不会自动阻止 attention 跨文档；需要 block-diagonal mask 才能隔离。

### 12.2 Packing 效率

报告

\[
\text{packing efficiency}
=
\frac{\text{valid objective tokens}}
{\text{allocated sequence slots}}.
\]

高效率不代表语义正确。错误 mask 可以得到 100% “利用率”却训练了 padding 或跨文档泄漏。

## 13. 合成数据

完整的生成分布、rejection sampling、self-training、distillation、多代反馈、审计实现与工程实验见[合成数据、蒸馏与反馈环](synthetic-data.md)。本节保留数据供应链的最低要求。

常见用途：instruction expansion、难例、推理轨迹、翻译、纠错、蒸馏、工具轨迹和拒答样本。每条 synthetic item 应保留：

- generator model/revision、prompt/template 与采样配置；
- seed（若契约支持）、生成时间和 raw response；
- verifier、规则、执行测试和过滤 reason；
- 是否经人工复核；
- 父数据和许可证/隐私继承关系。

### 13.1 防止自我强化

同一模型生成、评价和过滤会共享盲点。可使用独立 verifier、可执行测试、人工抽样、真实数据锚点和多样性约束。拒绝采样提高通过某个 verifier 的比例，但也可能只优化 verifier 漏洞。

Model collapse 不是“用了合成数据就必然发生”的单一结果；风险依合成占比、选择机制、真实锚点、模型和代际迭代。应测分布覆盖、长尾与跨代质量，而不是只测平均 loss。

## 14. 增量更新与删除

### 14.1 Stable identity

文档 ID 应基于稳定 source identity 与版本，而不是只基于当前正文 hash；正文变化后仍需知道它是同一来源的新 revision。Content hash 用于完整性与 exact duplicate，不能独自承担 lineage。

### 14.2 Tombstone 与重建

删除请求需要映射：source item → parsed artifacts → dedup cluster → tokenized shards → training runs → checkpoints/adapters。删除 final dataset 中一行不会自动消除已经训练进权重的影响。处置可能是未来训练排除、checkpoint 重训、近似 unlearning 或系统层过滤，必须按政策和威胁模型声明。

### 14.3 可重复增量构建

对同一 source revision 重跑应幂等；parser/filter 更新则产生新 dataset version，而不是悄悄覆盖旧版本。Manifest 记录 parent version、added/updated/deleted counts 和 cluster changes。

## 15. Manifest 示例

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

真实 manifest 还需要每个组件的配置、代码 revision 和统计报告。只写版本字符串但无法取回对应 artifact 不算可复现。

## 16. 发布前不变量

### 身份与 lineage

- 每个 final item 有唯一 ID、来源和 parent chain；
- 所有 shard 可映射回 item/token span；
- 同一构建输入不会随机改变 split；
- 删除/许可状态冲突会阻止发布。

### 数据质量

- 解码失败、空文本、极端长度和重复率在阈值内；
- language/domain/source 分布与 manifest 一致；
- filter 在关键切片有人工校准；
- exact/near-dedup cluster 统计通过审查。

### 训练语义

- 抽样打印 text、token IDs、labels、loss mask 和 document boundaries；
- 每个 sequence 的 valid-token count 与 loss denominator 一致；
- packing boundary、EOS、position 和 attention mask 有单测；
- 目标 mixture 与小规模 dry-run 的实际 consumed tokens 相符。

### 安全与评测

- holdout registry 在所有可访问信息源执行污染检查；
- PII/secret 检测有覆盖与误差报告；
- adversarial documents 不会通过元数据控制训练脚本；
- 数据读取器限制路径、压缩炸弹、恶意 pickle 和超大对象。

## 17. 故障定位

### Loss 突然下降得异常好

检查 train/validation 重复、labels 是否等于输入当前位置、padding 是否大量计入、数据是否重复、评测答案是否混入。

### 某语言 loss 变差

检查 tokenizer 膨胀、language ID 误杀、mixture 实际消耗、Unicode normalization、截断与该语言 dedup removal。

### 恢复训练后 mixture 改变

检查 dataloader cursor、sampler RNG、worker 数、shard order、world size 和 consumed-token ledger，而不只检查 model/optimizer state。

### 数据量对不上

分别对 raw items、parsed items、kept items、unique items、tokenized items、allocated slots 和 objective tokens 对账；不要用一个“总 token 数”覆盖所有阶段。

## 18. 常见错误结论

- **“互联网可访问，所以可训练”**：访问权限、训练权利和再分发权利不同。
- **“MinHash 去重就是语义去重”**：它近似 shingle-set similarity，依规范化和 n-gram。
- **“插入 EOS 就隔离了文档”**：EOS 不会自动改变 attention mask。
- **“过滤更多就是质量更高”**：误杀会减少多样性并引入群体偏差。
- **“检测器没命中，所以没有 PII/污染”**：所有检测都有覆盖范围和漏报。
- **“最终数据删除就完成机器遗忘”**：已训练权重和衍生 artifact 需要单独处置。

## 自测与实践

1. 为网页、PDF、代码和对话分别定义 stable identity 与 parser acceptance test。
2. 为什么 exact-dedup normalization profile 不一定适合最终训练文本？
3. 给两个共享许可证头的代码文件计算 shingle Jaccard，并解释误删风险。
4. 对三个域计算温度采样后的 expected consumed tokens 与重复倍数。
5. 画出一个文档删除请求从 source 到 checkpoint 的 lineage 路径。
6. 构造两文档 packing 样本，逐位置写出 input、label、loss mask 和 attention mask。

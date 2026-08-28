# Synthetic Data Audit：四条候选为什么只剩一条新内容

**项目导航**：[返回项目索引](../project-index.md) · [合成数据](../../training/synthetic-data.md) ·
[SFT 数据流水线](../../training/sft-data-pipeline.md) · [生产检查表](../production-checklist.md)
{ .doc-nav }

输入文件里只有四条合成数据候选。`syn-001` 和 `syn-002` 通过了两项必要验证，但正文完全相同；`syn-003`
缺少一项验证，`syn-004` 则明确验证失败。因此，这次审计会得到三个不同的数字：

```text
4 条候选 → 2 条通过必要验证 → 1 份不重复的通过内容
```

本页先带你手算这三个数字，再运行程序逐条核对。最后，我们把原始输入、审计规则和计算结果放进一份可重新计算的
报告。这样再看到“通过”时，你知道它通过了哪条规则、分母是什么，也能回到原始记录检查原因。

项目生成的报告格式是 `about-llm.synthetic-data-audit.v2`。当前程序只在 CPU 上离线运行；它不会请求生成模型、
验证模型或 LLM judge，也不会启动训练。完整的适用范围集中放在本页末尾说明。

## 1. 先读懂 4 → 2 → 1

三个数字分别回答三个问题：

| 数字 | 程序实际问的问题 |
|---|---|
| 4 条候选（candidates） | 输入解析器成功读到了多少条结构明确的记录？ |
| 2 条通过（eligible） | 多少条记录包含全部必要验证，而且每项都标记为 `passed=true`？ |
| 1 份不重复内容（eligible unique） | 在已经通过的记录中，按选定的内容指纹规则去重后还剩几份？ |

数据来源关系（lineage）、人工复核标记和目标采样次数会分别记账。它们不会被压成一个看似方便、实际含义不清的
总分。这里的 `eligible` 只表示“通过本次指定的验证门槛”，不表示事实一定正确或已经适合发布。

## 2. 这份报告信任什么

```mermaid
flowchart LR
  A["真实来源或上代 candidate"] --> B["Generator + prompt revision"]
  B --> C["records JSONL exact bytes"]
  C --> D["无歧义 JSON loader"]
  D --> E["Lineage diagnostics"]
  D --> F["Required verifier gate"]
  D --> G["Exact identity groups"]
  H["mixture JSON exact bytes"] --> I["Target exposure planner"]
  P["Caller-supplied policy"] --> E
  P --> F
  P --> G
  E --> R["v2 audit artifact"]
  F --> R
  G --> R
  I --> R
  R --> V["Full local recomputation"]
  C --> V
  H --> V
  P --> V
  R -. "不是证明" .-> Q["来源认证、语义质量、训练收益"]
```

程序从调用者提供的三样东西开始计算：候选记录、采样配置和审计规则。报告中的 SHA-256 可以发现这些文件的字节
是否发生变化，但它不能回答“文件是谁发布的”。如果攻击者能同时替换输入、规则和报告，还需要签名、消息认证码
（MAC）或受控的发布系统来确认来源。

## 3. 先跑出第一份报告 { #run }

核心文件：

- `records.example.jsonl`：4 条固定候选；
- `mixture.example.json`：真实数据与合成数据的目标采样比例；
- `audit.example.json`：仓库保存的 v2 示例报告，可以从原始输入完整复算；
- `src/about_llm/synthetic_data.py`：来源关系、验证门槛、内容去重和采样计算；
- `src/about_llm/synthetic_data_cli.py`：JSON 解析、报告生成与命令行入口；
- 两个测试文件：正常路径和故意失败路径。

从仓库根目录运行：

~~~powershell
python -m about_llm.synthetic_data_cli `
  --records projects/synthetic-data-audit/records.example.jsonl `
  --required-verifier schema `
  --required-verifier grounding `
  --known-parent-id real-anchor-001 `
  --mixture projects/synthetic-data-audit/mixture.example.json `
  --output artifacts/synthetic-data/audit.json
~~~

默认情况下，`--output` 只会新建文件，不会覆盖同名报告。目标已经存在时命令会失败；确认旧报告可以被替换后，
再显式传入 `--overwrite`。这条规则可以防止误覆盖，但它还不是一个带历史版本的报告存储系统。

安装项目后，也可运行 `about-llm-synthetic-audit --help`。

## 4. 先手算，再相信程序

四条记录故意形成可辨认的反例：

| ID | 生成轮次 | 父记录 | 两项必要验证 | 预期结果 |
|---|---:|---|---|---|
| `syn-001` | 1 | `real-anchor-001` | Schema 通过；grounding 通过 | 通过；带有人工复核标记 |
| `syn-002` | 1 | `real-anchor-001` | Schema 通过；grounding 通过 | 通过；正文与 `syn-001` 完全相同；生成与验证版本字符串重合 |
| `syn-003` | 1 | `real-anchor-001` | Schema 通过；缺少 grounding | 缺少验证；不通过 |
| `syn-004` | 2 | `syn-001` | Schema 通过；grounding 失败 | 验证失败；不通过 |

固定审计账本是：

- `candidate_count=4`；
- `eligible_count=2`，所以 eligibility rate 为 0.5；
- `eligible_unique_content_count=1`；
- round 1：3 candidates / 2 eligible；
- round 2：1 candidate / 0 eligible；
- `self_verified_record_ids=["syn-002"]`；
- `missing_verifier_record_ids=["syn-003"]`；
- `failed_verifier_record_ids=["syn-004"]`；
- `unresolved_parent_pairs=[]`；
- `nonmonotonic_parent_pairs=[]`；
- `lineage_cycle_record_ids=[]`。

为了确认你运行的是同一份输入，仓库记录了文件大小和 SHA-256：

| 文件或报告 | 精确大小 | SHA-256 |
|---|---:|---|
| records JSONL | 1,457 bytes | `7b1fe32811b835aa20c579c0ed351311a617a0ec70cd582a5f5313b2b78d8530` |
| mixture JSON | 341 bytes | `4bef57e8ca23e35b8f3953bb442779f9d706ee2e9570ef5d4090a956c3669bc0` |
| v2 报告指纹 | 规范化后的报告内容 | `202d8db97b704c5542e8516c5bd0c945da1c1022100f6ecbfb828f2d2bb6f4cd` |

这些数值用于绑定原始文件的精确字节。即使只改换行、缩进或对象字段顺序，审计结果可能不变，文件的 SHA-256
仍会变化。此时旧报告应当验证失败，因为它记录的已经不是当前输入。

## 5. 为什么输入必须无歧义

一行 record 的最小结构是：

~~~json
{
  "record_id": "syn-001",
  "content": "候选文本",
  "parent_ids": ["real-anchor-001"],
  "generator_revision": "teacher@v1",
  "prompt_revision": "expand@v3",
  "generation_round": 1,
  "verifications": [
    {"verifier_id": "schema", "revision": "schema-rules@v2", "passed": true}
  ],
  "human_reviewed": false
}
~~~

除可以省略、默认值为 `false` 的 `human_reviewed` 外，其余字段都必须出现。以下输入会在计算开始前直接报错：

- 同一个 JSON 对象里出现重复字段。否则不同解析器可能分别采用第一个值或最后一个值；
- 出现 `NaN`、`Infinity` 或 `-Infinity`。它们不是标准 JSON 数字；
- 文件不是有效 UTF-8，或者包含不成对的 Unicode surrogate；
- Record、verification 或 mixture component 出现未知字段或缺少必要字段；
- `passed` 和 `human_reviewed` 不是 JSON 布尔值，例如用整数 `0/1` 代替；
- `generation_round` 或 token budget 不是整数，或者错误地使用了布尔值；
- ID、revision、content 必须非空；
- 同一作用域内的父记录 ID、验证器 ID 或记录 ID 重复；
- 一条记录把自己列为父记录；
- Records、mixture 或 report 超过实现规定的文件大小上限。

这些规则只保证每个对象有唯一、明确的解释。`generator_revision` 是否真实，或者 `human_reviewed=true` 背后是否
真的有人复核，仍要依赖来源系统的记录。

## 6. 父记录指向哪里，生成轮次有没有倒退

一条合成记录可以从另一条记录派生。这里把被依赖的记录称为父记录（parent）。父记录分成两类：

- **内部父记录**：ID 出现在当前 records 文件中，例如 `syn-004` 的父记录 `syn-001`；
- **外部父记录**：ID 不在当前文件中，但调用者通过 `--known-parent-id` 明确声明，例如 `real-anchor-001`。

其余父记录会进入 `unresolved_parent_pairs`。程序不会把未知 ID 自动解释成“外部真实数据”，因为一次拼写错误就可能
让本来断裂的来源链看起来完整。

对于内部边 \(p\rightarrow c\)，实现要求：

\[
round(c)>round(p).
\]

子记录必须来自更晚的生成轮次。若它与父记录处于同一轮，或者轮次反而更小，这一对 ID 会进入
`nonmonotonic_parent_pairs`。程序还会用深度优先搜索查找环，并把真正处于环中的记录写入
`lineage_cycle_record_ids`；仅仅指向某个环的后代不会被误标为环成员。

在当前示例中，来源问题和验证结果分别记账。未知父记录、轮次倒退或成环不会自动修改 `eligible_count`。这样你能看清
“内容验证通过”和“来源链异常”是两件事。真实发布策略通常会把严重的来源问题设成单独的拒绝条件。

真实流水线还要明确：外部父记录目录由谁维护，多份来源的许可如何合并，父记录被删除后怎样传播到派生数据和
checkpoint，以及“第几轮”究竟按单条记录、数据集版本还是模型代际定义。

## 7. Verifier 缺失、失败和共享盲点

设 required verifier 集合为 \(V\)，则当前 eligibility 定义为：

\[
eligible(x)=\bigwedge_{v\in V}
\left[present(v,x)\land passed(v,x)\right].
\]

必要验证缺失和“验证存在但失败”会进入两个不同列表。额外的验证结果仍保留在记录中，但不参与调用者为本次审计指定的
门槛。重复内容、来源问题和版本重合也不会把记录从候选分母中悄悄删除。

如果某个验证器的 `revision` 字符串与 `generator_revision` 完全相同，记录会进入 `self_verified_record_ids`。
这个字段只表示**版本字符串重合**，提醒你继续调查生成与验证是否真正独立：

- 字符串重合不能证明生成与验证真的使用了同一进程或同一份权重；
- 字符串不同也不能证明 judge 独立；
- 两个版本仍可能共享模型家族、训练数据和系统性偏差；
- 验证通过不能替代校准、事实核对和 reward hacking 检查。

同理，`human_reviewed_count` 只是对输入中布尔值的计数。它没有记录复核者身份、评分规则、是否盲评或复核者之间的
一致性。

## 8. 两条通过记录可能只有一份内容

默认 profile 是 `byte_exact`：

\[
id(x)=SHA256(UTF8(content_x)).
\]

任何空白或 Unicode 表示差异都会产生不同 digest。对明确允许文本归一化的 prose，可显式选择：

~~~powershell
python -m about_llm.synthetic_data_cli `
  --records projects/synthetic-data-audit/records.example.jsonl `
  --required-verifier schema `
  --required-verifier grounding `
  --known-parent-id real-anchor-001 `
  --fingerprint-profile nfc_whitespace
~~~

`nfc_whitespace` 先执行 Unicode NFC 规范化，再按 Unicode 空白字符切分，最后用一个 ASCII 空格重新连接。
这会折叠空白，因此不适合 Python、YAML、Markdown 表格或其他把排版也当作答案一部分的任务。

报告保留三种分母：

- candidate count：多少条进入 audit；
- eligible count：多少条通过 required verifier gate；
- eligible unique content count：eligible 集合中有多少 exact identity。

所以固定样例中的两条 eligible 仍计入 `eligible_count=2`，却只贡献
`eligible_unique_content_count=1`。选择哪条作为 canonical item，还要结合许可、parent、时间、review 与 split policy；
本项目不会擅自删除其中一条。

这里的内容指纹只能发现精确重复。两段含义相同但措辞不同的文本，需要 embedding、MinHash 或释义检测等近重复方法；
训练集与保留集之间是否存在语义泄漏，也要另外检查。

## 9. 计划看五遍，不代表真的看了五遍

设数据来源 \(i\) 的正相对权重为 \(w_i\)，计划消费的 token 总数为 \(D\)，该来源包含 \(n_i\) 个不重复 token。
第一步先把相对权重归一化：

\[
p_i=\frac{w_i}{\sum_j w_j}.
\]

再计算：

\[
E[C_i]=Dp_i,\qquad E[repeat_i]=\frac{Dp_i}{n_i}.
\]

本例预设的 total budget 是 2,000,000 tokens：

| Component | Weight | Fraction | Expected consumed | Unique | Expected repetition |
|---|---:|---:|---:|---:|---:|
| real anchor | 3 | 0.75 | 1,500,000 | 800,000 | 1.875 |
| synthetic round 1 | 1 | 0.25 | 500,000 | 100,000 | 5.0 |

因此 25% synthetic target 对应 unique synthetic tokens **预期消费 5 倍**。Weight 3:1 是相对权重，不可把 `3*D` 和 `1*D` 当实际 token 数。

这张表描述的是采样器的**目标值**，不是训练日志。Packing、动态过滤、不同 worker 的负载偏差、读取失败、
checkpoint 恢复和提前停止都可能改变实际消费量。

真实训练需要另建一份 token 消费账本，按数据来源、生成轮次和数据切分记录已经提交给训练的 token，再把实际值与
这张目标表对账。

测试 `test_mixture_plan_uses_normalized_not_pre_normalized_weights` 专门防止把归一化前的权重直接乘以总预算。

## 10. V2 报告究竟绑定了什么

`audit.example.json` 的顶层字段包括：

- `schema_version`：报告格式版本，本例为 `about-llm.synthetic-data-audit.v2`；
- `inputs`：Records 和 mixture 文件的精确大小与 SHA-256；
- `policy`：必要验证器、已知外部父记录、内容指纹规则，以及来源问题是否影响验证门槛；
- `audit`：来源关系、验证结果、内容去重和各轮次统计；
- `mixture`：各数据来源的目标采样比例和预计重复次数；
- `scope` 与 `evidence_boundary`：程序这次实际执行了什么，结论适用到哪里；
- `report_fingerprint`：移除本字段后，对规范化报告内容计算的 SHA-256。

换句话说，v2 同时绑定了**输入文件的精确字节和外部审计规则**。如果只复制一串报告哈希，却没有保存可信的输入和
规则，就无法复核这份报告究竟审计了什么。

## 11. 不信 self-hash，重新算一遍

用仓库固定输入验证 recorded artifact：

~~~powershell
python -m about_llm.synthetic_data_cli `
  --records projects/synthetic-data-audit/records.example.jsonl `
  --required-verifier schema `
  --required-verifier grounding `
  --known-parent-id real-anchor-001 `
  --mixture projects/synthetic-data-audit/mixture.example.json `
  --verify-report projects/synthetic-data-audit/audit.example.json
~~~

成功结果包含：

~~~json
{
  "report_fingerprint": "sha256:202d8db97b704c5542e8516c5bd0c945da1c1022100f6ecbfb828f2d2bb6f4cd",
  "schema_version": "about-llm.synthetic-data-audit.v2",
  "verification_scope": "full_local_recomputation",
  "verified": true
}
~~~

验证命令会执行四步：

1. 用同一套无歧义 JSON 规则解析现有报告；
2. 重新读取命令行提供的 records 和 mixture；
3. 按命令行提供的审计规则，从头计算 `inputs`、`audit`、`mixture`、`scope` 和报告指纹；
4. 把重新生成的规范化 JSON 与现有报告逐字段比较。

因此，它不是从报告中取出一串哈希，再让这份报告验证自己。

## 12. 故意破坏它

至少理解并演练以下失败路径：

1. **输入字节变化**：在 records 文件末尾增加空白，旧报告因大小和 SHA-256 不同而验证失败；
2. **审计规则变化**：修改必要验证器、已知父记录或内容指纹规则，重新计算的报告不再相同；
3. **篡改后重新算哈希**：修改 `eligible_count` 并为假报告重算无密钥哈希，仍无法通过基于可信输入的完整复算；
4. **JSON 解析失败**：重复字段、非法数值、未知字段或无效 UTF-8 会在审计开始前报错；
5. **输出文件冲突**：对同一路径再次使用 `--output`，默认的新建模式会拒绝覆盖；
6. **来源关系异常**：未知父记录、成环或轮次倒退会进入各自的错误账本。

当前实现写文件时会刷新缓冲区，并对文件执行 `fsync`。这只能提高文件内容落盘的可靠性，还没有覆盖目录项持久化、
原子重命名、进程崩溃或断电实验。验证完成到下游真正使用报告之间，也仍存在“验证后被替换”的时间窗口。

如果报告保存在对象存储中，还需要版本前置条件、租约或不可变对象键。若威胁模型包含恶意发布者，还要增加签名或
MAC、可信密钥分发和只追加的透明日志；继续增加无密钥 SHA-256 字段并不能认证发布者。

## 13. 测试与可复现实验

运行专项：

~~~powershell
python -m pytest `
  tests/test_synthetic_data.py `
  tests/test_synthetic_data_cli.py -q
~~~

写作时这两个文件共 **40 个测试**（数量会随仓库演进变化，以实际输出为准），主要分成三组：

- 审计计算：三种分母、版本字符串重合、来源关系图、成环与轮次倒退；
- 数据规则：两种内容指纹、采样公式和无歧义 JSON 解析；
- 失败路径：输入变化、篡改后重新算哈希、默认拒绝覆盖与显式覆盖。

优先运行的反例：

~~~powershell
python -m pytest `
  tests/test_synthetic_data.py::test_audit_reports_cycles_and_nonmonotonic_parent_rounds `
  tests/test_synthetic_data_cli.py::test_report_verifier_rejects_cooperative_rehash `
  tests/test_synthetic_data_cli.py::test_report_verifier_binds_exact_input_bytes `
  tests/test_synthetic_data_cli.py::test_output_is_exclusive_create_unless_overwrite_is_explicit `
  -q
~~~

若修改固定输入，应先生成一份候选报告，再用 diff 逐字段审阅，最后更新仓库保存的示例报告和准确性检查。
不要只复制一串新的指纹。

## 14. 验证失败时从哪里查

出现验证失败时，按以下顺序缩小范围：

1. 比较 records 和 mixture 的字节数与 SHA-256；
2. 比较必要验证器的集合与拼写；
3. 比较已知外部父记录和内容指纹规则；
4. 单独运行 JSON loader，定位 JSONL 行号或 schema 字段；
5. 比较 `audit` 中缺失、失败、来源关系和重复内容的子结构；
6. 比较 mixture 中归一化比例、预计消费量和预计重复次数；
7. 最后再比较规范化后的 `report_fingerprint`。

只看到指纹不匹配时，不要先猜“哈希算法坏了”。更常见的原因是输入换行改变、审计规则变化、固定输入字段变化，
或使用了错误的 mixture 文件。

## 15. 从离线参考实现走向真实流水线

先明确当前程序已经证明了什么：它能用无歧义的方式解析固定输入，计算来源关系、验证结果、精确内容重复和目标采样量；
它也能从调用者提供的输入与规则出发，完整复算 v2 报告。

当前程序的范围止于 CPU 上的离线审计。模型生成、模型验证、训练过程和真实 token 消费量需要其他系统提供证据。
因此：

- `eligible` 只表示通过指定的验证门槛，不等于事实正确或高质量；
- 精确内容不重复，不等于语义多样，也不能排除保留集泄漏；
- 生成与验证版本字符串不同，不等于两个 judge 真正独立；
- 目标采样量不等于训练实际消费量；
- CPU 离线结果不能证明模型收益或生产系统可靠性。

接入真实流水线时，至少还要补充：

- Teacher、student 和 verifier 的原始请求、响应、采样参数、随机种子、时间、费用与模型版本；
- Prompt、模板和工具 Schema 的不可变原文；
- 数据许可、用户同意、个人信息、密钥和删除策略；
- 验证器的输入输出、基础设施故障、校准切片和人工裁决记录；
- 语义近重复、聚类与按组切分，以及一份只含真实数据且从未参与训练的保留集；
- 目标采样账本、实际 token 消费账本和 checkpoint 恢复状态；
- 每一代数据集、模型和评测的 manifest，以及停止、回滚和删除传播规则；
- 报告签名、发布者身份、原子发布和“验证后立即使用”的绑定机制。

最终发布决策应显式组合多条规则，例如：

```text
无歧义解析 AND 来源规则通过 AND 验证规则通过 AND 无污染 AND 法务规则通过
```

其中“无歧义解析”指拒绝重复字段、非法数值和未知字段。不要把这些维度压成一个无法追责的“质量总分”。

## 16. 怎样把它讲成一个工程项目

简历不能只写“搭建合成数据流水线”。至少应展示：

- 一份绑定输入和审计规则、可以完整复算的报告；
- 候选、通过、通过且不重复这三种分母；
- 未知父记录、成环、轮次倒退、验证缺失和验证失败的独立账本；
- 目标采样量与实际 token 消费量的分账设计；
- 输入变化、篡改后重算哈希和输出文件冲突三个反例；
- 明确写出当前只是 CPU 上的离线参考实现。

常见追问：

1. 为什么 verifier pass 不是 ground truth？
2. 为什么 `self_verified_record_ids` 只能称 revision overlap？
3. 为什么精确重复与语义泄漏不等价？
4. 为何哈希可以绑定文件字节，却不能认证发布者？
5. file `fsync`、directory `fsync`、atomic rename 各解决什么？
6. 如何在 checkpoint 恢复后对账真实的合成数据 token 消费量？
7. 为什么来源关系异常不应被一个含义模糊的 `eligible` 布尔值吞掉？

完整实现见 [projects/synthetic-data-audit](https://github.com/NightLemon/about-llm/tree/main/projects/synthetic-data-audit)。

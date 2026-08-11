# RAG 上下文构造、引用与忠实度

检索结果不是答案。生成阶段要把授权证据压入有限上下文，告诉模型如何区分指令与资料，产出可解析引用，并在证据不足时拒绝猜测。本章关注“模型看到什么”和“我们能证明什么”。

## 上下文是一种协议

不要把若干文本直接拼进 prompt。每个证据单元应有不可歧义的 ID、来源、标题、时间和正文边界，例如：

~~~text
<source id="S1" document_id="policy-42" updated_at="2026-07-01">
……仅作为证据的原始文本……
</source>
~~~

system instruction 明确：source 内任何命令都属于不可信数据；回答只能引用给定 ID；每个可验证 claim 后放引用；证据不足时说明缺什么。短 ID 减少 token，也降低模型复制长 URL 的错误。最终 UI 再将 `S1` 映射为可点击来源。

## 指令层级与提示注入

RAG 文档可能包含“忽略之前规则”“调用这个链接”等恶意或无意文本。防护不是再加一句“不要被注入”：

1. 执行权限在模型外，检索内容永远不能授予工具能力。
2. system、developer policy 与 evidence 使用清晰分隔和固定 schema。
3. 只给生成器完成当前任务所需的最小工具与数据。
4. 高风险动作要求确定性校验和用户审批。
5. 对注入语料建立回归集，测数据外泄、越权和错误引用。

文档清洗可标记已知攻击，但不能依赖黑名单删除所有自然语言指令，因为正常技术文档也会讨论 prompt。

## Context packing

总预算可以写成：

\[
L_{system}+L_{history}+L_{query}+L_{evidence}+L_{output}\le L_{context}
\]

必须先为输出、system 和必要 history 预留，再分配 evidence。不能先塞满检索结果，再把输出上限压到无法完整回答。

上式是容量账本，不意味着各组件分别 tokenize 后的长度能精确相加。BPE/Unigram 边界、chat template control token 和 special token 会改变实际序列；插入文本后，边界 token merge 甚至可能让总 token 数小幅下降，因此也不能强制 `base_cost <= used_cost`。最可靠的做法是每次候选加入后重新渲染**完整 prospective prompt**，用目标 tokenizer/revision 计数，并把预留输出 token 一并计入 cost。

### 可执行的预算与决策账本

仓库的 `pack_citation_context` 接受 `cost_fn(rendered_context) -> int`。cost closure 可以捕获 system、history、query、chat template 和 reserved output，然后对完整 prompt 计数；packer 不假设单位一定是 token，也不把组件成本当作可加量。每个候选都在 budget/quota 判断前重新检查 tenant/principal ACL，因此不能用“最终会被预算丢掉”掩盖上游越权结果。

`make_rag_chat_prompt_cost` 把 system prompt、query、带唯一 `{query}`/`{context}` 槽位的用户模板、chat tokenizer 和输出预留组合成 cost closure。替换只解释原始模板槽位，不会把 query/context 中碰巧出现的 `{context}` 或 `{query}` 再次展开。

```python
from about_llm.rag import pack_citation_context

reserved_output_tokens = 512

def full_prompt_tokens(context: str) -> int:
    messages = [
        {"role": "system", "content": "只能依据授权证据回答。"},
        {"role": "user", "content": f"证据：\n{context}\n\n问题：{question}"},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(prompt_ids) + reserved_output_tokens

packed = pack_citation_context(
    ranked_results,
    tenant_id=tenant_id,
    principals=principals,
    budget_units=model_context_limit,
    cost_fn=full_prompt_tokens,
    cost_unit="tokens:model-and-tokenizer@revision",
    max_chunks_per_source=2,
)
```

算法按已给定 rank greedy 扫描：document id 去重，限制每个稳定 source 的 chunk 数；某个超预算候选被跳过后仍尝试更短的后续候选。每个候选记录 `selected / duplicate_document / source_quota / budget` 与加入后的 prospective cost。这样能够解释“为什么某段没进上下文”，也能验证最终 `used_cost_units <= budget_units`。

CLI 的 `pack`/`--budget-bytes` 只计算 UTF-8 serialized bytes，不是 token 数，也不能用于承诺模型窗口。`pack-tokenized` 则加载明确 tokenizer revision，运行 checkpoint 或本地 override chat template，对每个候选重算完整 prompt，并将 reserved output 一起与 `max-total-tokens` 比较；报告模板/prompt hash、最终 prompt token IDs 和逐候选决策。

~~~powershell
python -m about_llm.rag.cli pack-tokenized `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --query "ACL 为什么必须在检索前执行" `
  --tenant tenant-a `
  --principal engineering `
  --max-total-tokens 4096 `
  --reserved-output-tokens 512 `
  --tokenizer C:\path\to\target-tokenizer `
  --tokenizer-revision exact-revision `
  --local-files-only `
  --system-prompt-file projects/rag-foundations/system-prompt.example.txt `
  --user-prompt-template-file projects/rag-foundations/user-prompt-template.example.txt
~~~

该报告中的 `model_context_window_verified=false` 很重要：成功 tokenize 只证明“这组文件按这个模板产生这些 token IDs”，不证明 tokenizer 与部署权重匹配、用户给出的 4096 是 runtime 真正可用窗口、模型能生成预留长度，或回答会忠实使用证据。Tokenizer revision/hash 也不是来源认证。当前 packer 是透明 CPU reference：它不做相邻 chunk 合并、knapsack/集合覆盖、required-evidence 预留或语义压缩，并为准确计数反复渲染 prospective context，候选很大时不是高吞吐实现。生产优化必须与该 reference 在选择、授权和最终 tokenizer 计数上做 differential test。

### 去重、合并与压缩

- exact/near duplicate：优先保留更新、权威或位置更完整的版本。
- adjacent chunks：同来源相邻片段可合并，避免边界信息缺失。
- extractive compression：保留与 query 相关句子，容易审计。
- abstractive compression：更省 token，但可能引入二次幻觉；压缩结果仍需回链原文。

压缩器不能看到无权文档，也不能丢掉 negation、时间、单位和限定词。评测应比较压缩前后 evidence coverage，而不只看 token 节省。

## 对话历史与 query contextualization

“那它的限制呢？”需要从 history 恢复实体。常用做法是生成 standalone query，但改写可能过度补全。保留原 query、改写 query 和引用的 history turn，检索时可多路融合；涉及权限或动作时，不能从旧对话推断当前授权。

长对话可维护结构化状态和有来源的摘要。摘要是派生信息，必须能追溯 turn；关键用户约束不要只存在于滚动摘要中。

## 生成策略

事实型 RAG 通常使用低 temperature、明确答案格式和有限输出。更低 temperature 不保证忠实，只降低采样变化。可以要求先识别证据、再生成带引用答案，但不要把隐藏 chain-of-thought 当作可验证证据；可保存简短结构化 rationale 或 claim-source 映射。

复杂问题可先生成 answer plan：列出子问题和所需来源，再逐项回答。计划必须受最大步骤和 token 预算限制，失败时返回部分结果与缺口。

## 引用语法

一个实用规则是每个外部可验证段落至少一个 `[S1]`，多个来源写 `[S1][S3]`。引用应紧跟 claim，而不是文末堆一串。代码、纯建议或明确标记为推断的内容可有不同策略，但规则必须在评测中显式。

仓库的 `build_citation_context` 做两件事：拒绝跨 tenant 结果，按检索顺序去重并分配规范 ID。`audit_citations` 检查未知 ID 和漏引段落。

### 语法正确不等于忠实

“月球由奶酪构成。[S1]”即使 `S1` 存在，也可能完全不受支持。引用质量至少拆成：

- citation validity：ID 存在且用户有权查看；
- citation coverage：需要证据的 claim 是否有引用；
- citation correctness：引用内容是否支持 claim；
- source quality：证据是否权威、时效正确；
- completeness：是否遗漏相反或必要证据。

前两项可较可靠地程序化；correctness 需要 claim segmentation 加 entailment judge 或人工标注。judge 输入只包含一个 claim 与对应 evidence，允许 `supported / contradicted / insufficient`，并在人工集上校准。

## Claim 分解

一个段落可能包含多个事实：“A 在 2025 年上线，支持 X，但不支持 Y。”整段一个引用无法知道来源支持哪些部分。评测管道可把答案分成 atomic claims，保留原 span，然后为每个 claim 收集邻近引用。

自动 claim splitter 也会出错，因此报告：无法解析比例、每答案 claim 数、无引用 claim 比例、支持/矛盾/不足比例，并抽样人工检查。涉及数字、日期、否定和比较的 claim 应作为高风险切片。

## 证据不足

可靠系统不只会回答，还要知道何时不答。可使用：

- 检索零结果或分数/重排 margin 过低；
- 必要字段缺失、版本冲突或多跳证据不完整；
- 生成后所有 claim 都无法找到支持；
- query 超出知识库范围或需要实时数据。

阈值要用 coverage-risk 曲线选择：阈值越严格，错误回答减少但拒答增加。输出应说明缺少的证据和可行下一步，而不是泛化的“作为 AI 我不知道”。

### 把检索信号与拒答决策分开

检索零结果正确不证明最终拒答正确。系统可能在零结果时依靠参数记忆猜答，也可能在召回若干主题相关文档后仍正确识别“没有所问事实”。评测至少区分四格：

| Gold 可回答 | 最终系统回答 | 解释 |
|---|---|---|
| 是 | 有支持答案 | true answer；再检查 correctness/citation |
| 是 | 拒答 | false refusal；定位 corpus/retrieval/packing/generation |
| 否 | 拒答或澄清 | true refusal；检查缺口说明是否正确 |
| 否 | 给出事实答案 | unsupported answer；即使引用 ID 合法也算失败 |

再把 retrieval trace 叠加到四格上：`zero_results`、required evidence 是否齐全、gold 是否被 ACL 阻断、最终 context 是否保留证据。这样才能区分“没有证据所以拒答”和“有证据但模型没用”。阈值选择用独立校准集；不能在同一测试集上调到最好再报告准确率。

## 冲突证据

两个来源冲突时，不要让模型按相似度多数表决。将权威级别、发布日期、生效区间和文档状态作为显式 metadata。若规则不能确定，应呈现冲突、分别引用并请求用户选择适用范围。

旧政策可以对历史时间问题有效，所以“最新”不能简单删除所有旧版本；需要 effective_from/effective_to 和 query time。

## 输出结构

API 场景推荐结构化输出：

~~~json
{
  "answer": "... [S1]",
  "claims": [
    {"text": "...", "source_ids": ["S1"]}
  ],
  "insufficient_evidence": false,
  "missing_information": []
}
~~~

用 JSON Schema 校验字段、枚举和额外属性；失败可有限次数 repair，但 repair 模型不能新增上下文外来源。UI 渲染前仍需 HTML/Markdown 安全处理。

## 评测设计

一条 case 至少含 query、允许来源、gold evidence、reference answer/claims、时间与权限上下文、切片。组件指标包括 retrieval recall、context precision、引用 validity/coverage/correctness、answer correctness、拒答准确性和延迟。

无答案 case 不应伪造一个空字符串 reference answer 后与普通答案共同算字符串指标。它需要显式 `answerable`/expected action 标签，并分别统计 answerable-case quality、no-answer refusal、总体 coverage 和被接受答案的 risk。检索层的 zero-result accuracy 可以作为诊断列，但不能替代这些端到端分母。

### 可运行的 extractive oracle baseline

在接 LLM 前，先建立一个“只能复制已授权证据”的弱基线很有价值。仓库的 `answer-extractive` 依次执行 authorization-first BM25、复用 context packer、在 packed chunk 中按精确字符偏移切句/分句、greedy lexical set coverage 和 answer/abstain。答案只由 `原文 span [Sx]` 组成；artifact 会验证 offset slice、short/stable source、content hash、覆盖 token ledger 和 packing decision 一致。未进最终 context 的 chunk 无法被引用。

`evaluate-extractive` 先完成生成，再把 artifact 转成统一 `RecordedAnswer` 做离线评测。生成函数的参数没有 qrels、`answerable`、relevance 或 required source；测试把同一在线请求从 answerable 改标为 no-answer 时，生成 artifact 完全不变，只有事后 action accuracy 改变。这样可防止把 gold 答案泄漏进在线决策，但不能防止语料本身被评测集污染。

Exact copy 使“claim 是该 source 的逐字陈述”可由程序证明，所以机械 judgment provenance 是 `deterministic-exact-source-span-v1`。不要把它扩大成 semantic entailment judge：词面 overlap 可能选错句，来源本身可能错误，多个正确 span 也可能遗漏限制或冲突。默认 0.55 coverage 是 authored fixture threshold，必须在独立 calibration split 上画 coverage-risk 曲线后才能用于目标域。该路径使用 UTF-8 byte budget，仅用于无 tokenizer 的透明控制实验。

~~~powershell
python -m about_llm.rag.cli evaluate-extractive `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl
~~~

### 可执行的 recorded-answer gate

仓库的 `about-llm-rag evaluate-answers` 读取三份可版本化 artifact：带 answerability/权限上下文的 case、版本化 corpus，以及录制的 answer/abstain/error。recorded answer 把最终 context 的稳定 source id、atomic claims、claim citations、verdict 和 `judgment_source` 分开保存。真实系统应先保存模型原始输出，再由独立人工或经过校准的 judge 附加 verdict；不能把模型自述“supported”直接当标签。

每条 claim verdict 由工件显式提供，枚举为 `supported / contradicted / insufficient / unjudged`。本地评测器**不执行语义蕴含**，只做以下确定性工作：

1. case 与 output 必须 exact join，不能丢掉超时或解析失败；
2. error 是一等终态，进入 case 分母且不算 coverage；
3. 根据 corpus、tenant 和 principals 重新检查 recorded context；
4. claim 必须有引用，引用必须位于可见 context；
5. judged claim 必须记录 judgment provenance，unjudged 不得伪装成已判断；
6. answerable case 只有 action=answer 且所有 claim 都有合法引用、supplied verdict 都为 supported 才过 recorded gate；no-answer case 只有 abstain 且 context 未越权才过 gate。

报告的 action accuracy 只比较 answer/abstain/error 动作；citation validity 只检查 source id 与授权 context；supported-claim rate 只聚合已有判断；claim-judgment coverage 提示还有多少 claim 未判断。`grounded_answer_pass_rate` 的分母是实际回答的 case，`recorded_gate_pass_rate` 的分母是全部 case，二者不能互换。

这个 gate 刻意保守，但仍不证明 answer completeness、source quality 或 verdict 正确。一个 answerable case 只输出一条真实但不完整的 claim 仍可能通过；要验证 completeness，需为 case 建 reference claims/rubric 并逐项对齐。fixture 的手写标签只验证协议和聚合数学，不是独立人工评测，更不是某个 LLM 的实测质量。

~~~powershell
python -m about_llm.rag.cli evaluate-answers `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl `
  --answers projects/rag-foundations/sample_answers.jsonl
~~~

### 把 packing、原始输出与评测工件连成一条证据链

只保存 `context_source_ids` 无法回答“模型当时究竟看到了哪一版 chunk”。仓库的 `audit-traces` 要求每个 case 恰好对应一个 trace 和一个 recorded answer，并绑定 query SHA-256、tenant/principals、按 `S1..Sn` 排序的 document/stable-source/version/content SHA-256、规范化 rendered context、prompt token IDs 与 tokenizer/template/prompt identity、输出预留、generator revision、raw output bytes 和 recorded-answer canonical fingerprint。同一稳定来源可以贡献多个不同 chunk；answer 的稳定来源顺序按首次出现去重后核对。

~~~powershell
python -m about_llm.rag.cli audit-traces `
  --corpus projects/rag-foundations/sample_corpus.jsonl `
  --cases projects/rag-foundations/sample_eval.jsonl `
  --answers projects/rag-foundations/sample_answers.jsonl `
  --traces projects/rag-foundations/generation-traces.example.jsonl
~~~

审计从**当前** corpus 重建 chunk 和 ACL，所以能发现 query/security context、chunk id/version/bytes、source order、rendered context 或 answer fingerprint 不一致；它不能证明历史时刻也能取回相同 corpus。prompt fields 与 raw output 被纳入 trace fingerprint，但该命令不重新 tokenize、不向可信 registry 核验 revision，也不做 raw-output→claim 的解析或语义蕴含。fixture 是手写协议样例，不是模型执行证据；unsigned hash 若没有外部可信 manifest/签名，也不能阻止攻击者协同重写所有文件。

建立以下对抗集：

- 文档内注入和伪造 `[S999]`；
- 无权来源与同标题跨 tenant 文档；
- 旧版/新版冲突；
- 数字单位、否定和条件限定；
- query 无答案、部分答案和多跳答案；
- 检索证据正确但模型参数记忆冲突。

## 可运行实验

~~~python
from about_llm.rag import audit_citations, build_citation_context

context = build_citation_context(results, tenant_id="tenant-a")
answer = generator(question, context.rendered)
audit = audit_citations(answer, context.sources)
assert not audit.unknown_source_ids
~~~

该断言只形成语法门禁。下一步把答案分 claim，用人工 gold 或经过校准的 entailment judge 评 correctness；同时保存原始 evidence snapshot，避免索引更新后无法复现。

## 面试追问

**如何减少 hallucination？** 先区分语料缺失、检索失败和生成不忠实。提高 evidence recall、过滤/重排、结构化引用、证据不足拒答和 claim-evidence 评测共同作用；单纯改 prompt 不能修复不存在的证据。

**为何不让模型直接输出 URL？** URL 长、易拼错、可能泄露内部地址。模型输出受控短 ID，服务端在授权后映射显示信息，引用协议更稳定也更可审计。

**怎样证明某次回答可复现？** 保存 query/history 版本、权限上下文、索引/embedding/reranker/generator 版本、检索候选、最终 context、生成参数和原始输出；只保存最终答案不够。

# RAG Framework Adapters：比较接线，不比较品牌

**项目导航**：[项目索引](../project-index.md) · [RAG Foundations](rag-foundations.md) ·
[RAG 召回](../../applications/rag-retrieval.md) ·
[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/rag-framework-adapters/README.md) ·
[项目证据](../../evidence/project-controls.md)
{ .doc-nav }

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：已经理解 RAG 请求链，正在接入 LangChain、LlamaIndex 或其他编排框架的开发者。
- **先修**：[RAG Foundations](rag-foundations.md#run)中的授权、检索、Prompt 与引用顺序。
- **首次阅读**：统一对象 → 授权后的检索结果 → 两种框架映射 → round-trip 对账。
- **完成信号**：能指出框架接线必须保留的字段，并解释为什么最终答案相同仍不足以证明接入正确。
- **卡住时**：先只比较文档 ID、正文和顺序，再加入 metadata 与 Prompt。

</div>

项目固定一份 canonical `Document/SearchResult`，让同一批授权后的结果分别经过 LangChain 与 LlamaIndex，再转换回来。
它回答 adapter 是否丢字段、改顺序或污染 Prompt，不回答哪个框架的原生检索器更好。

## 一次完整对照 {#run}

```powershell
python projects/rag-framework-adapters/parity_control.py
```

安装、最小对象 demo 和测试命令见[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/rag-framework-adapters/README.md)。
运行前预测：Engineering 用户应看到 `acl-before-ranking → citation-binding`，Anonymous 只看到
`acl-before-ranking`；跨租户和 finance-only 文档不得进入 scorer。

## 比较方法

```text
trusted tenant/principals
→ canonical corpus
→ ACL filter
→ BM25 + top-k
→ SearchResult[]
   ├─ LangChain adapter
   └─ LlamaIndex adapter
→ round-trip field comparison
→ Prompt byte comparison
→ deterministic answer artifact
```

Canonical core 拥有文档 identity、租户、ACL、score 和 rank；框架对象只是运输容器。权限必须在转换以前执行，
metadata exclusion 也只减少字段意外进入模型文本，不能替代授权或日志脱敏。

## 三层验收

| 层 | 比较内容 | 能发现什么 |
|---|---|---|
| 结果集合 | 文档 ID、数量与 one-based rank | 丢项、重复、重排 |
| 每条结果 | 正文、tenant、ACL、有限 score 与 retriever identity | 字段丢失、覆盖与非法数值 |
| 下游输入 | Prompt bytes、答案来源与 artifact fingerprint | formatter 或模板漂移 |

Round trip 只能发现 adapter 相对 canonical 输入的变化。若攻击者同时替换原输入和框架对象，两边仍可能一致；
来源认证和受控存储需要独立解决。

## 什么时候才是在比较框架

要比较两套框架的 native index/query engine，必须固定 corpus/chunker、身份与权限、留出 queries/qrels、embedding、
reranker、Prompt/model、并发和冷热启动。此时 raw score 未必同尺度，应比较召回、排序、答案、延迟与失败切片。
那已经是新的检索与服务实验，不是本项目的 adapter parity。

## 扩展边界

接 learned retrieval 时继续输出 canonical `SearchResult`，并记录 index/encoder/reranker identity；接生成模型时绑定
tokenizer/template/model/sampling/source map；接异步服务时分别验证 batch、stream、callback、取消和重试。
这些系统机制分别由[RAG 召回](../../applications/rag-retrieval.md)与
[RAG 生产化](../../applications/rag-production.md)解释，本页不重复教程。

## 完成标准

- 两类用户在 canonical、LangChain 与 LlamaIndex 路径中的 IDs、顺序和字段一致。
- 两套框架渲染的 Prompt bytes 一致，保护 metadata 不会默认进入模型文本。
- 至少一个正文、rank、score、tenant 或 exclusion-key 漂移被拒绝。
- 报告保存真实加载的框架版本和未执行范围。
- 结论只限于固定语料与当前 adapter，不外推到 native index、生成质量、线上吞吐或框架默认安全。

# RAG Foundations

目标：先用透明组件构建可诊断 RAG，再接入 learned embedding、reranker、LangChain 和 LlamaIndex。

## 已实现的基线

- UTF-8/中英文透明 lexical tokenizer；
- BM25，包含长度归一化和稳定排序；
- 检索前 tenant ACL 过滤；
- Reciprocal Rank Fusion；
- 注入式 dense cosine index；
- sentence-transformers embedding 与 cross-encoder reranker adapter；
- Recall@k 与 MRR；
- Markdown 标题感知切分、超长段落兜底、稳定内容哈希与 chunk id；
- tenant/source/version/ACL 元数据和显式 upsert/delete 增量计划；
- 授权上下文的规范化 `[S1]` 来源编号、未知引用和漏引段落审计；
- 单元测试覆盖精确术语、租户隔离、稳定 ID、编辑/删除、重复结果和指标。

运行：

~~~powershell
pytest tests/test_rag.py tests/test_rag_ingestion.py tests/test_rag_citations.py
~~~

## 为什么先不用框架

RAG 的错误可能来自解析、chunk、召回、过滤、重排、上下文或生成。若第一版就把这些阶段封进链式框架，很难判断提升来自哪里。本项目的领域对象与指标保持框架无关，后续 adapter 只转换输入输出。

## 摄取与引用边界

`split_markdown` 的 chunk id 不包含顺序号，因此在同一标题下插入一个不同段落不会让后续 chunk 全部改名；修改内容、移动标题或相同内容的重复次数变化则会产生新 id。`plan_incremental_update` 明确返回写入和删除集合，调用方应在同一索引事务中应用，并拒绝把“空抓取结果”自动解释为删除全部来源。

`build_citation_context` 在渲染前再次检查 tenant，去重后分配短来源 id。`audit_citations` 只验证引用是否存在、id 是否已授权以及段落是否漏引；即使它返回成功，也不代表来源在语义上支持 claim。claim-evidence entailment 应使用人工标注集、NLI/LLM judge 和抽样审计，并报告误判率。

仍需在目标语料上完成 embedding/reranker 消融、真实向量库事务、可观测 trace 与语义忠实度评测；这些依赖部署环境，不能由离线单测代替。

## 安全不变量

ACL 在检索和缓存键中执行，不能先全局召回再让 LLM 忽略无权文档。生成器只收到已授权证据；日志、评测样本和 trace 同样按租户隔离。

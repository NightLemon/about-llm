# LangChain 与 LlamaIndex：让同一次 RAG 检索保持一致

这个项目演示怎样把已有 RAG 核心接入 LangChain 和 LlamaIndex，同时不改变原来的权限、排序、Prompt 与评测语义。
应用仍然拥有 canonical `Document` 和 `SearchResult`；框架对象只负责在各自 API 中传递结果。

第一次学习时直接运行 `parity_control.py`。先预测两类用户能看到哪些文档，再检查 canonical、LangChain 和
LlamaIndex 三条路径是否给出相同结果。完整推导见
[RAG Framework Adapters 教学页](../../docs/practice/projects/rag-framework-adapters.md)。

## 第一次运行

安装两套框架及开发依赖：

```powershell
python -m pip install -e ".[dev,torch,rag,langchain,llamaindex]"
```

执行完整对照：

```powershell
python projects/rag-framework-adapters/parity_control.py
```

脚本会输出 JSON。先找到 `cases.engineering` 与 `cases.anonymous`，逐项比较三组 document IDs：

- Engineering 用户应得到 `acl-before-ranking → citation-binding`；
- Anonymous 用户只应得到 `acl-before-ranking`；
- `finance-secret` 与 `other-tenant` 不应进入这两类用户的结果。

然后检查 `assertions` 是否全部为 `true`，并阅读 `scope`。其中的 `false` 表示实验没有运行 learned embedding、
向量索引、生成模型或生产负载，并不是程序失败。

## 一次请求怎样经过两个框架

```text
可信 tenant / principals
    → canonical corpus
    → ACL filter
    → BM25 score + top-k
    → SearchResult[]
       ├─→ LangChain BaseRetriever.invoke()
       └─→ LlamaIndex BaseRetriever.retrieve()
    → round-trip 字段检查
    → 相同 Prompt
    → 相同抽取式答案与评测结果
```

权限过滤发生在评分和 top-k 之前。这样，无权文档不会进入 scorer、候选列表、Prompt 或后续缓存。
先对全库排序再删除无权结果，不仅会让无权文档占用候选位，也已经跨过了不该进入的处理边界。

LangChain 和 LlamaIndex 收到同一份 canonical index、查询和安全上下文。这里观察的是 adapter 与 orchestration
能否保持结果一致。若要比较两个框架各自的检索质量，需要另行固定它们的 index、retriever 与评测数据。

## 两条运行路径

### 只看对象怎样转换

```powershell
python projects/rag-framework-adapters/demo.py
```

`demo.py` 把同一组 `SearchResult` 转成 LangChain `Document` 和 LlamaIndex `TextNode + NodeWithScore`，再打印三组 ID。
它适合用来熟悉对象结构，但没有调用 Retriever、渲染 Prompt 或计算答案与指标。

### 检查完整的 framework parity

```powershell
python projects/rag-framework-adapters/parity_control.py
```

`parity_control.py` 会真实调用两个框架的 Retriever 和 PromptTemplate API，并完成以下步骤：

1. 用四份固定文档建立 canonical BM25 index；
2. 分别以 Engineering 与 Anonymous 身份执行授权优先的检索；
3. 将结果送入 LangChain 与 LlamaIndex Retriever；
4. 把框架对象转换回来，逐字段与 canonical 结果比较；
5. 渲染两份 Prompt，并比较字节与 SHA-256；
6. 生成确定性的抽取式答案，再计算 Recall@4 与 nDCG@4。

报告中的 `framework_versions` 是当前环境真实加载的版本。依赖升级后如果 Prompt 或对象行为变化，应先判断原因，
不要直接把新输出当成新的正确答案。

## Adapter 必须保留哪些字段

| Canonical 信息 | LangChain 中的位置 | LlamaIndex 中的位置 |
|---|---|---|
| Document ID | `Document.id` 与 metadata | `TextNode.node_id` 与 metadata |
| 正文 | `page_content` | `TextNode.text` |
| Tenant 与 ACL | metadata | metadata |
| Retrieval score | metadata | `NodeWithScore.score` 与 metadata |
| One-based rank | metadata | metadata |
| Retriever 来源 | metadata | metadata |

业务 metadata 不能覆盖 `document_id`、`tenant_id`、`acl`、`retrieval_score`、`retrieval_rank` 和 `retriever`。
输入还必须满足 ID 不重复、rank 从 1 连续递增、score 是有限实数。Adapter 遇到含糊或冲突的数据会停止，而不是猜测修复。

LlamaIndex 在为 embedding 或生成模型构造文本时，可能把 metadata 一起带入。项目因此把控制字段放进
`excluded_embed_metadata_keys` 和 `excluded_llm_metadata_keys`，从默认内容中排除它们。

这项设置只影响当前 node 的默认构造方式。自定义 formatter、callback、日志或 Prompt 仍然可以读取 metadata，
所以权限检查和脱敏需要单独实现。

## 怎样阅读输出

| 输出位置 | 你要确认什么 |
|---|---|
| `framework_versions` | 实际运行的 LangChain/LlamaIndex core 版本 |
| `cases.*.*_document_ids` | 三条路径中的 ID 与顺序完全相同 |
| `retrieval_scores` | Canonical 排序得到的分数，没有经过框架静默改写 |
| `prompt_sha256` / `prompt_utf8_bytes` | 两个框架渲染的是同一份 Prompt |
| `answer_artifact_fingerprint` | 抽取式答案及其来源绑定保持一致 |
| `metrics` | 固定 qrels 上的 Recall@4 与 nDCG@4 |
| `assertions` | 本次运行真正检查并通过的条件 |
| `scope` | 哪些组件真实执行，哪些仍未进入实验 |

固定四文档样例中的满分指标衡量的是 adapter 一致性。若要评估 LangChain 或 LlamaIndex 的 native index/query
engine，需要使用独立检索配置和 held-out 数据集重新实验。

## 主要文件

| 文件 | 用途 |
|---|---|
| `demo.py` | 展示 canonical 结果到两种框架对象的最小映射 |
| `parity_control.py` | 运行 Retriever、Prompt、答案和指标的完整离线对照 |
| [`rag_frameworks.py`](../../src/about_llm/integrations/rag_frameworks.py) | Adapter、round-trip validator 与 Retriever wrapper |
| [教学页](../../docs/practice/projects/rag-framework-adapters.md) | 解释 ACL 顺序、字段契约、公平比较与生产扩展 |
| [项目证据页](../../docs/evidence/project-controls.md) | 保存录制版本、结果与验证范围 |
| `tests/test_rag_framework_adapters.py` | 覆盖字段漂移、安全上下文和异常输入 |

## 常见故障

| 现象 | 先检查什么 |
|---|---|
| 三条路径的 ID 不同 | Tenant/principals、top-k、adapter 是否重排或丢失结果 |
| ID 相同但 Prompt hash 不同 | 正文、metadata formatter、模板和换行是否变化 |
| LlamaIndex 内容多出控制字段 | 两组 metadata exclusion keys 与自定义 formatter |
| Rank 或 score 校验失败 | Rank 是否从 1 连续递增；score 是否为有限实数而非 bool |
| Anonymous 看见私有文档 | 身份来源与 ACL 是否在评分前执行 |
| 升级依赖后报告变化 | 实际框架版本、对象字段与 PromptTemplate 渲染行为 |
| 指标仍满分但安全测试失败 | 无权文档可能进入过 scorer；最终 ID 一样不能证明过程安全 |

排错时先找到最早发生漂移的层：canonical retrieval、对象转换、round trip、Prompt，还是答案 artifact。
只比较最终答案会把前面的字段和权限问题隐藏起来。

## 运行专项测试

```powershell
python -m pytest tests/test_rag_framework_adapters.py -q
python scripts/check_docs.py
python scripts/check_content_accuracy.py
```

专项测试会故意制造以下错误：

- 保护字段冲突、rank gap 与重复 ID；
- NaN/Inf score、错误 tenant 与重复 principal；
- 正文或 metadata exclusion key 漂移。

测试还会执行两个真实 Retriever API，确认框架适配层能识别这些错误。

## 从这个实验继续扩展

继续扩展时，每增加一层都要补充相应的运行身份：

- Learned retrieval：保留 canonical `SearchResult` 出口，并记录 index、embedding、reranker、候选集和 score 语义；
- 生成模型：固定 tokenizer、chat template、model revision、sampling、stop 和 source map；
- 异步服务：分别测试 async、batch、stream、callback、取消与重试；
- Trace 与 cache：绑定 tenant、policy、index 和模型版本，并为 Prompt、正文与 metadata 配置访问控制和脱敏。

当前项目证明的是：在本地固定语料上，两个框架的 Retriever/Prompt API 能承载同一次 canonical 检索，而没有改变
项目检查的字段、Prompt 和抽取式答案。它没有验证框架默认 ACL、向量数据库、LLM 生成质量、线上吞吐或生产安全。

# 编码轮：六道题的完整题面与解法

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：准备 LLM 应用、算法、推理或平台岗位现场编码环节的开发者。
- **先修**：能用 Python 写出带测试的函数；读过 attention、采样、BM25 或 LoRA 中的至少一条主线。
- **首次阅读**：先看时间分配和"能跑之外说什么"，再挑与目标岗位最相关的两题动手写。
- **完成信号**：能在 45 分钟内澄清契约、写出最小实现、主动补三个边界测试并说明复杂度。
- **卡住时**：先写最朴素的正确版本并说明它慢在哪里，不要一上来追求最优解。

</div>

**求职导航**：[面试题与回答方法](interview-questions.md) · [岗位路线](roadmap.md) ·
[应用与治理题](applied-questions.md) · [系统设计](system-design.md) · [行为面试](behavioral.md) · [简历项目](resume-projects.md)
{ .doc-nav }

LLM 岗位的编码轮很少考纯算法竞赛题。更常见的是给一个本领域的小组件，观察你是否理解它的输入输出契约、
数值边界和验证方法。同一道题，能跑通只是及格线，区分度在于你主动说出了哪些它会挂的场景。

本页六道题都来自本仓库已经实现并测试过的组件，每题末尾给出参考实现位置，可以对照阅读。

## 45 分钟怎样分配

编码轮的时间压力通常来自没有澄清就动手。建议的节奏：

| 时段 | 应完成什么 | 常见失误 |
|---|---|---|
| 0–5 分钟 | 澄清输入输出、规模、错误契约，写下两三个例子 | 直接开始敲代码 |
| 5–25 分钟 | 写出最朴素的正确实现，边写边说思路 | 一次性追求最优解，写到一半推翻 |
| 25–35 分钟 | 主动补边界测试并跑通 | 等面试官问"有什么边界" |
| 35–45 分钟 | 说明复杂度、数值稳定性与优化方向 | 沉默地微调代码 |

与[系统设计的 40 分钟分配](system-design.md#forty-minute-plan)一样，前五分钟的澄清收益最高。

## "能跑"之外要主动说什么

写完后不要停在"跑通了"。用四句话收口：

1. **复杂度**：时间和空间分别是多少，瓶颈在哪一步。
2. **数值边界**：溢出、下溢、除零、空输入、全部非法输入分别会怎样。
3. **错误契约**：非法输入应该抛异常、返回空还是返回哨兵值，为什么选这个。
4. **怎样验证**：给一个能证伪当前实现的测试，而不是只测正常路径。

第三点最容易被忽略。"空查询返回空列表"和"空查询抛异常"都可以，但你必须解释为什么，
以及调用方要怎样区分"没有结果"和"参数错了"。

## 题目一：数值稳定 softmax 与 causal attention { #q-softmax-attention }

**题面**：实现 `scaled_dot_product_attention(query, key, value, mask=None)`，
返回 `(output, probabilities)`。query 形状 `(..., query_length, head_dim)`，
key/value 形状 `(..., key_length, head_dim)`，前导维度需要广播。再实现一个
`causal_mask(query_length, key_length)`，`True` 表示该 key 对该 query 可见。

**必须先问清楚的三件事**：

- mask 的语义是 `True 可见` 还是 `True 屏蔽`？两种约定都常见，写反了结果完全错。
- `key_length` 可以大于 `query_length` 吗？（带 KV Cache 的 decode 就是这种情况。）
- 需要返回注意力概率吗？返回它才能做后面的不变性测试。

**最小实现**：

```python
import numpy as np


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def causal_mask(query_length, key_length=None):
    key_length = query_length if key_length is None else key_length
    past_length = key_length - query_length
    if past_length < 0:
        raise ValueError("key_length cannot be smaller than query_length")
    query_positions = np.arange(query_length)[:, None] + past_length
    key_positions = np.arange(key_length)[None, :]
    return key_positions <= query_positions


def scaled_dot_product_attention(query, key, value, *, mask=None):
    scale = float(query.shape[-1]) ** -0.5
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale
    if mask is not None:
        mask = np.broadcast_to(mask, scores.shape)
        if np.any(np.all(~mask, axis=-1)):
            raise ValueError("every query row must have at least one visible key")
        scores = np.where(mask, scores, -np.inf)
    probabilities = softmax(scores, axis=-1)
    return np.matmul(probabilities, value), probabilities
```

`causal_mask` 里的 `past_length` 偏移是这题真正的考点：decode 时只送入一个新 query，
但 key 包含全部历史，`query_positions` 必须右移 `key_length - query_length` 才对齐。

**必须自己补的边界测试**：

| 边界 | 不处理会怎样 | 怎样测 |
|---|---|---|
| 极大 logit | 不减最大值时 `exp` 溢出成 `inf`，再相除得 `nan` | 输入 `[1000., 1001.]`，断言结果有限且和为 1 |
| 整行被 mask | 全 `-inf` 行 softmax 得 `0/0 = nan`，且会静默传播 | 构造全 `False` 的一行，断言抛出异常而不是返回 `nan` |
| 未来 token 不变性 | mask 广播方向写反时仍能跑，但泄漏未来信息 | 只改 \(t\) 之后的 token，断言位置 \(0..t\) 的输出不变 |
| `key_length > query_length` | 偏移漏掉时 decode 路径全错 | 令 `query_length=1`、`key_length=5`，断言该行全部可见 |

**为什么全 mask 行应该抛异常而不是返回 0**：返回零向量会让上游把"无可见 key"当成一次正常注意力，
错误一直传到输出层才暴露。抛异常把失败点固定在构造 mask 的地方。

**面试官的下一个追问**：

- *为什么除以 \(\sqrt{d}\) 而不是 \(d\)？* —— 见[核心题第 1 题](interview-questions.md#attention-scaling)。
- *这个实现要保存完整的 `(query_length, key_length)` 概率矩阵，长序列怎么办？*
  这引向 online softmax 与分块计算，仓库的 `blockwise_online_attention` 是可对照的实现。

**仓库参考实现**：`src/about_llm/from_scratch/attention_numpy.py`，测试见
`tests/test_attention_numpy.py`。

```bash
python -m pytest tests/test_attention_numpy.py -q
```

## 题目二：top-k 与 top-p 采样 { #q-sampling }

**题面**：给定 logits 向量和 `temperature`、`top_k`、`top_p`，返回下一个 token id。
为了可测试，把随机数作为参数 `uniform ∈ [0, 1)` 传入，不要在函数内部调用全局随机数。

**必须先问清楚的三件事**：

- **顺序**：temperature、top-k、top-p 按什么次序作用？归一化发生在截断之前还是之后？
  不同框架的实现并不一致，这是本题最大的歧义来源。
- **并列**：两个 token 分数完全相同且卡在 top-k 边界上，保留哪个？
- **top-p 的边界**：累计概率恰好等于阈值的那个 token，包含还是排除？

**最小实现**：

```python
import numpy as np


def sample_next_token(logits, *, temperature=1.0, top_k=None, top_p=None, uniform):
    values = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= uniform < 1:
        raise ValueError("uniform must be in [0, 1)")

    scaled = values / temperature
    # 并列时按 token id 升序打破，保证可复现。
    token_ids = np.arange(scaled.size)
    order = np.lexsort((token_ids, -scaled))

    keep = order[: top_k] if top_k is not None else order
    probabilities = _masked_softmax(scaled, keep)

    if top_p is not None and top_p < 1:
        cumulative = np.cumsum(probabilities[keep])
        # side="left" 保留第一个达到或越过阈值的 token。
        cutoff = min(int(np.searchsorted(cumulative, top_p, side="left")), len(keep) - 1)
        keep = keep[: cutoff + 1]
        probabilities = _masked_softmax(scaled, keep)

    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0  # 消除浮点累加误差导致的末尾小于 1
    return int(np.searchsorted(cumulative, uniform, side="right"))


def _masked_softmax(scaled, keep):
    probabilities = np.zeros_like(scaled)
    subset = scaled[keep]
    exponentials = np.exp(subset - subset.max())
    probabilities[keep] = exponentials / exponentials.sum()
    return probabilities
```

注意 top-p 之后**重新归一化**了一次。如果沿用 top-k 阶段的概率直接采样，
累计和小于 1，落在尾部的 `uniform` 会取不到任何 token。

**必须自己补的边界测试**：

| 边界 | 不处理会怎样 | 怎样测 |
|---|---|---|
| `top_k` 大于词表 | 切片静默成功，但与预期语义不符 | 断言抛异常或明确定义为"全部保留" |
| 分数并列 | 依赖排序稳定性，跨版本结果漂移 | 两个相同 logit，断言固定返回较小的 token id |
| `top_p` 卡在边界 | `side` 参数写反时少保留一个 token | 概率 `[0.6, 0.3, 0.1]`、`top_p=0.6`，断言只保留第一个 |
| `uniform` 接近 1 | 浮点累加使末尾小于 1，`searchsorted` 越界 | 传 `uniform=0.999999`，断言返回合法 id |
| `temperature=0` | 直接除零得 `inf`/`nan` | 断言抛异常，并说明 greedy 应走单独分支 |

**面试官的下一个追问**：

- *`temperature=0` 为什么不能靠"除以一个很小的数"实现？* 会先溢出成 `inf`，再在 softmax 里变成 `nan`；
  greedy 是独立的 `argmax` 分支，不是极限情形。
- *相同参数、相同 seed，两次请求一定输出相同 token 吗？* 不一定，见
  [核心题第 30 题](interview-questions.md#replay-identity)。

**仓库参考实现**：`src/about_llm/inference/sampling.py`。该实现还加入了 repetition penalty，
并把每一步的候选集合与概率完整记录下来便于复核。测试见 `tests/test_sampling.py`。

```bash
python -m pytest tests/test_sampling.py -q
```

## 题目三：带权限的 BM25 检索 { #q-bm25 }

**题面**：实现一个内存 BM25 索引，`search(query, tenant_id, principals, top_k)`
只返回该租户下调用方有权访问的文档。文档带 `tenant_id` 和 `acl` 字段，
空 `acl` 表示租户内公开。

**必须先问清楚的三件事**：

- 权限过滤发生在打分**之前**还是之后？（这是本题真正的考点，见下。）
- 空查询、查询词全部未登录，返回空列表还是抛异常？
- 需要支持文档更新和删除吗？会影响索引结构的选择。

**最小实现**（BM25 打分部分）：

```python
import math
from collections import Counter


def bm25_scores(query_terms, term_frequencies, lengths, *, k1=1.5, b=0.75):
    number_of_documents = len(term_frequencies)
    average_length = sum(lengths) / number_of_documents

    document_frequency = Counter()
    for frequencies in term_frequencies:
        document_frequency.update(frequencies.keys())

    idf = {
        term: math.log(1 + (number_of_documents - count + 0.5) / (count + 0.5))
        for term, count in document_frequency.items()
    }

    scores = []
    for frequencies, length in zip(term_frequencies, lengths):
        score = 0.0
        for term in query_terms:
            if term not in frequencies:
                continue
            frequency = frequencies[term]
            denominator = frequency + k1 * (1 - b + b * length / average_length)
            score += idf[term] * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores
```

**权限必须先于打分**，这是最容易答错的一层。很多人先算完分数再过滤结果，
但 BM25 的 IDF 和平均文档长度是**语料级统计量**：如果它们由全部文档算出，
分数本身就编码了不可见文档的信息。攻击者可以通过观察分数变化、结果数量或响应时间
推断某份机密文档是否存在。正确做法是先算出可见文档集合，再在这个子集上计算 IDF、
平均长度和分数。

**必须自己补的边界测试**：

| 边界 | 不处理会怎样 | 怎样测 |
|---|---|---|
| 跨租户 | 过滤写在打分之后，分数已泄漏存在性 | 同一查询在两个租户下断言结果集合不相交 |
| 可见集合为空 | 平均长度除零 | 断言返回空列表而不是崩溃 |
| 空查询 | 返回全部文档或崩溃 | 断言返回空列表 |
| 未登录词 | `idf[term]` 抛 `KeyError` | 断言跳过该词而不是报错 |
| 长文档 | `b=0` 时长文档因词频高而霸榜 | 对比 `b=0` 与 `b=0.75` 下长短文档的排序 |
| 分数并列 | 排序不稳定，翻页结果抖动 | 断言按 `document_id` 打破并列 |

**面试官的下一个追问**：

- *为什么不直接用向量检索替代 BM25？* 见[深挖题 16](../evidence/interview-controls.md)。
- *重排序阶段还需要再检查一次 ACL 吗？* 需要。重排器可能引入索引之外的候选，
  权限必须在每一层重新成立，而不是依赖上游。

**仓库参考实现**：`src/about_llm/rag/bm25.py` 的 `BM25Index`。它把"先授权再统计"
做成了默认行为，并保留一个 `_legacy_global_statistics` 开关专门用于复现旧工件——
这本身就是一个可以讲的工程决策。测试见 `tests/test_rag.py`。

```bash
python -m pytest tests/test_rag.py -q
```

## 题目四：Reciprocal Rank Fusion { #q-rrf }

**题面**：给定多个检索器各自的排序结果，融合成一个排序。不同检索器的分数不可比
（BM25 是无界正数，余弦相似度在 \([-1, 1]\)），因此只能使用排名。

**必须先问清楚的三件事**：

- 同一文档在多个列表中出现，怎样合并？（按 id 累加。）
- 某个文档在同一个列表里出现两次怎么办？
- 排名从 0 还是从 1 开始？直接影响 \(k=0\) 时是否除零。

**最小实现**：

```python
from collections import defaultdict


def reciprocal_rank_fusion(rankings, *, rank_constant=60, top_k=10):
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for result in ranking:
            document_id = result.document_id
            if document_id in seen:      # 同一列表内去重，防止自我加权
                continue
            seen.add(document_id)
            scores[document_id] += 1.0 / (rank_constant + result.rank)
    # 并列时按 document_id 升序，保证结果可复现
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return ordered[:top_k]
```

公式是 \(\mathrm{RRF}(d)=\sum_{r} \frac{1}{k + \mathrm{rank}_r(d)}\)。
常数 \(k\)（通常取 60）压低了头部名次之间的差距。排名从 1 开始时，第 1 名与第 2 名的贡献比是
\(\frac{k+2}{k+1}\)：\(k=60\) 时约为 \(1.016\)，两者几乎等价；而 \(k=0\) 时是 \(2\)，
第一名的权重是第二名的两倍。
所以 \(k\) 越大越接近"多个检索器投票"，越小越接近"只信第一名"。

**必须自己补的边界测试**：

| 边界 | 不处理会怎样 | 怎样测 |
|---|---|---|
| 同列表重复文档 | 该文档被重复加分，等于自我投票 | 构造重复项，断言只计一次 |
| 只有一个列表 | 应退化为原排序 | 断言输出顺序与输入一致 |
| 融合分并列 | 排序不稳定 | 断言按 `document_id` 打破并列 |
| `rank_constant=0` 且排名从 0 开始 | 除零 | 断言抛异常或要求排名从 1 开始 |
| 空输入 | 崩溃 | 断言返回空列表 |

**面试官的下一个追问**：

- *为什么不把两个分数归一化后加权求和？* 归一化依赖当前批次的极值，
  一个离群分数就会改变全部结果；RRF 只用排名，对分数尺度免疫。代价是丢掉了
  "第一名比第二名好很多"这类信息。
- *融合之后还要重新检查权限吗？* 要，理由同上一题。

**仓库参考实现**：`src/about_llm/rag/rank_fusion.py` 的 `reciprocal_rank_fusion`，
测试见 `tests/test_rag.py`。

## 题目五：LoRA Linear { #q-lora }

**题面**：包装一个 `nn.Linear`，冻结其权重，只训练低秩增量。
有效权重为 \(W + \frac{\alpha}{r} BA\)，其中 \(A \in \mathbb{R}^{r \times d_{in}}\)、
\(B \in \mathbb{R}^{d_{out} \times r}\)。再实现把 adapter 合并回普通 `Linear` 的方法。

**必须先问清楚的三件事**：

- \(A\) 和 \(B\) 分别怎样初始化？（这决定了包装后模型行为是否改变。）
- 缩放是 \(\alpha/r\) 还是 \(\alpha\)？换 rank 做实验时这会影响可比性。
- 需要保存完整模型还是只保存 adapter？

**最小实现**：

```python
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / rank
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        update = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return self.base(x) + update * self.scaling
```

\(B\) 初始化为**零**，\(A\) 用随机初始化。于是包装瞬间 \(BA = 0\)，模型输出与原来逐位相同，
训练是从原模型出发的连续微调，而不是一次随机扰动。两个都置零则梯度恒为零，永远学不动；
两个都随机则包装本身就破坏了预训练权重。

参数量从 \(d_{in} \times d_{out}\) 降到 \(r(d_{in} + d_{out})\)。
例如 \(d_{in}=d_{out}=4096\)、\(r=8\) 时，从约 1678 万降到约 6.6 万，约为原来的 0.4%。

**必须自己补的边界测试**：

| 边界 | 不处理会怎样 | 怎样测 |
|---|---|---|
| 包装后行为不变 | \(B\) 初始化非零时训练起点已偏移 | 断言包装前后输出逐位相等 |
| base 确实冻结 | 忘记设 `requires_grad=False`，实际是全量微调 | 反向后断言 `base.weight.grad is None` |
| 合并等价 | 合并时漏乘 `scaling` | 断言 `merged(x)` 与 `forward(x)` 在容差内相等 |
| 保存重载 | 只存权重不存 rank/alpha，重载后形状或缩放错 | 存盘再读回，断言输出一致 |
| `rank` 非正 | 创建出空参数，静默变成恒等层 | 断言抛异常 |

第二条最值得主动提。"loss 在下降"并不能证明你在做 LoRA——全量微调的 loss 同样会降。
真正的证据是 base 梯度为 `None` 且可训练参数量符合预期。

**面试官的下一个追问**：

- *QLoRA 是不是"用 4-bit 训练"？* 不是，见[核心题第 9 题](interview-questions.md#qlora-scope)。
- *合并后还能切换回原模型吗？* 能，但要保留原始权重；`merged()` 应返回独立对象而不是原地修改。

**仓库参考实现**：`src/about_llm/finetuning/lora.py` 的 `LoRALinear`，
包含 `merged()` 与只导出 adapter 的 `adapter_state_dict()`。测试见 `tests/test_lora.py`。

```bash
python -m pytest tests/test_lora.py -q
```

## 题目六：幂等的工具执行 { #q-idempotent-effect }

**题面**：Agent 要调用外部 API 执行退款。同一笔退款可能因为重试、崩溃恢复或用户重复点击
被触发多次，但真实副作用只能发生一次。设计并实现这个执行层。

**必须先问清楚的三件事**：

- 幂等键由谁生成？基于什么内容？（不能用随机 UUID，否则重试会产生新键。）
- 外部 API 是否支持幂等键？如果不支持，能否查询"这笔退款是否已完成"？
- 请求超时后，远端结果是**未知**而不是失败——业务上允许悬挂多久？

**最小实现**（状态机部分）：

```python
class EffectLedger:
    """把 effect_id 到终态的映射持久化，保证同一 effect 只执行一次。"""

    def __init__(self, storage):
        self.storage = storage

    def execute_once(self, effect_id, action):
        record = self.storage.get(effect_id)
        if record is not None:
            if record.state == "succeeded":
                return record.result          # 重放已知结果，不再调用外部
            if record.state == "in_flight":
                raise PendingEffect(effect_id)  # 结果未知，必须先对账
            # failed 是"已确认未发生"，才可以安全重试

        self.storage.put(effect_id, state="in_flight")
        try:
            result = action()
        except TimeoutError:
            # 关键：超时不等于失败，状态保持 in_flight 等待对账
            raise PendingEffect(effect_id) from None
        except PermanentError:
            self.storage.put(effect_id, state="failed")
            raise
        self.storage.put(effect_id, state="succeeded", result=result)
        return result
```

核心是三态而不是两态：`succeeded`、`failed` 和 **`in_flight`（结果未知）**。
把超时当成失败直接重试，是这类系统最常见的重复扣款来源。`in_flight` 必须由对账流程
（查询远端状态或人工介入）推进，不能由重试自动清除。

**必须自己补的边界测试**：

| 边界 | 不处理会怎样 | 怎样测 |
|---|---|---|
| 重复调用 | 副作用发生两次 | 同一 `effect_id` 调两次，断言外部只被调用一次 |
| 超时后重试 | 把 unknown 当 failed，重复扣款 | 模拟超时，断言第二次抛 `PendingEffect` 而非重新执行 |
| 写账本后崩溃 | 恢复时状态丢失 | 在 `put` 与 `action` 之间注入崩溃，断言恢复后进入对账 |
| 审批参数被改 | 用旧审批执行了新参数 | 改变参数后断言指纹不匹配，拒绝执行 |
| 并发同键 | 两个 worker 同时执行 | 断言只有一个成功获取执行权 |

**面试官的下一个追问**：

- *事务发件箱能保证 exactly-once 吗？* 不能。它只保证本地状态与待发送记录原子提交，
  投递仍是 at-least-once；远端成功但本地确认丢失时仍会重发。
  见[核心题第 17 题](interview-questions.md#effect-idempotency)。
- *参数做成 SHA-256 指纹后是否就安全了？* 无密钥哈希只能发现漂移，
  不认证执行者、时间或来源。

**仓库参考实现**：`src/about_llm/agents/outbox.py` 的 `SQLiteTransactionalOutbox`，
包含租约（lease）、过期回收和 attempt 计数。测试见 `tests/test_agent_outbox.py`。

```bash
python -m pytest tests/test_agent_outbox.py -q
```

## 练习顺序建议

不要六题平均用力。按目标岗位挑两题写到能默写：

| 目标岗位 | 优先两题 |
|---|---|
| LLM 应用 / RAG | 题目三（带权限 BM25）、题目四（RRF） |
| Agent / 平台 | 题目六（幂等执行）、题目三 |
| 模型 / 训练 | 题目五（LoRA）、题目一（attention） |
| 推理 / 系统 | 题目二（采样）、题目一 |
| 评测 / 数据 | 题目四、题目二 |

每题写完后做一次自检：把你的实现和仓库参考实现对照，找出**你没想到的那个边界**。
那个边界通常就是面试官准备的追问。

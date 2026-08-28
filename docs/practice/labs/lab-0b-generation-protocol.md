# 实验 0B：生成、停止与流式协议

**定位**：推荐进阶，预计 60–90 分钟；先完成[实验 0A](lab-0a-sampling.md)。

**实验导航**：[返回总览](../labs.md#lab-0) · [生成与解码](../../core/generation.md) · [进入 0C](lab-0c-cloud-budget.md)
{ .doc-nav }

## 开始前

**先修知识**：能够解释 greedy、概率采样、EOS 和序列 log probability。

**本页完成后**：你应该能区分搜索剪枝、合法 token mask、接受状态、长度截断、流式 stop 和服务端取消，不把它们合并成一个 `finished=true`。

## 第一部分：Beam search

~~~powershell
python projects/inference-serving/beam_search_toy.py
~~~

脚本内的两张概率表就是全部输入，不涉及模型：

```text
剪枝反例（vocab = A B EOS，EOS id = 2，max_new_tokens = 2）
  前缀 ()   → A 0.6, B 0.4, EOS 0.0
  前缀 (A)  → A 0.49, B 0.0, EOS 0.51
  前缀 (B)  → A 0.0,  B 0.0, EOS 1.0

长度惩罚反例（vocab = A B C EOS，EOS id = 3，max_new_tokens = 3）
  前缀 ()     → A 0.6, B 0.4
  前缀 (A)    → EOS 1.0
  前缀 (B)    → C 1.0
  前缀 (B,C)  → EOS 1.0
```

先手算再运行：beam 1 在第一步只留 A（0.6 > 0.4），于是最终概率 `0.4 × 1.0 = 0.4` 的 `B EOS`
根本没机会展开，而它优于 beam 1 保留的 `A EOS`（`0.6 × 0.51 = 0.306`）；beam 2 两条都留，因此能找到它。

长度惩罚这组要对照 `normalized_score = cumulative_log_probability / generated_length**alpha`：

| 候选 | `generated_length` | `cumulative_log_probability` | alpha=0 | alpha=2 |
|---|---:|---:|---:|---:|
| `A EOS` | 2 | ln 0.6 ≈ −0.5108 | −0.5108 | −0.1277 |
| `B C EOS` | 3 | ln 0.4 ≈ −0.9163 | −0.9163 | −0.1018 |

alpha=0 时短的那条赢；alpha=2 时排名翻转，长的那条赢。

还要注意报告里的 `length_definition` 字段：长度只数**生成的** token，其中包含发出的 EOS，不包含 prompt。
换一种长度定义（比如把 EOS 排除掉），上面的排名就可能又翻回去——所以长度惩罚的结论离不开长度的定义。

**最低通过**：修改 EOS 是否计入长度、early stopping 或 candidate cap，并明确把修改后的行为记录为新算法契约。

## 第二部分：约束解码

~~~powershell
python projects/inference-serving/constrained_decoding_toy.py
~~~

合法最终文本只有 `{"x":1}` 和 `{"x":2}`。token 表与关键第三步的概率是：

```text
token_texts = ('{"x"', ':', '1}', '1]', '2}', <EOS>, 'garbage')
前缀 ()        → '{"x"' 0.8, 'garbage' 0.2
前缀 (0,)      → ':' 0.9,   'garbage' 0.1
前缀 (0,1)     → '1}' 0.25, '1]' 0.65, '2}' 0.10   ← 关键步
前缀 (0,1,2)   → <EOS> 1.0
```

第三步概率最高的是非法的 `1]`（0.65）。逐 token 检查的是**整个 token 的字符串转移**，不是首字符：
`1]` 会把前缀带成 `{"x":1]`，两个合法字符串都不再可能完成，因此它在采样前就被 mask 掉。
剩下 `1}` 0.25 与 `2}` 0.10，合法质量 0.35，重新归一化得 `0.25/0.35 = 5/7` 与 `0.10/0.35 = 2/7`。

运行后重点看 `critical_step` 这个字段，它单独输出了第三步的允许集合与归一化结果。

**最低通过**：分别构造“语法已接受但 length 截断”“EOS 在非接受状态概率最高”“所有合法 token 概率为零”，证明三者不能合并为成功状态。

## 第三部分：流式 stop

~~~powershell
python projects/inference-serving/stop_matching_toy.py
~~~

两组[固定样例](../../reference/glossary.md#term-fixture)是：

```text
UTF-8 分块组：文本 "甲🙂乙<END>尾"，stop = ("<END>", "STOP")
              按字节切成 4 块，边界故意落在 emoji 内部和 <END> 内部

重叠组：      输入 "ABCZ"，stop = ("BC", "ABC")
```

先预测再运行，重点核对这几个字段：`updates[].emitted_text`、`updates[].held_characters`、
`utf8_split_fixture.returned_text`，以及重叠组的 `matched_stop`。

`<EN` 不能立刻显示，因为它还可能长成 `<END>`；一旦显示就撤不回来了。emoji 被切开时，
matcher 按**字符**而不是字节匹配，所以分块方式不改变结果——这正是 `held_characters` 存在的意义。

重叠组的答案可能与直觉相反：`matched_stop` 是 **`BC`** 而不是更长的 `ABC`。规则是**按配置顺序取第一个命中**，
`"BC"` 声明在前。这是一条需要写进文档的契约，不是"更长优先"这类最优性结论。

**最低通过**：画出 byte fragment、可见文本、partial prefix 和 terminal event 的状态变化，并指出客户端隐藏文本不等于服务端停止生成或计费。

## 常见失败

- 只比较最后文本，不保存 beam 的逐步候选与剪枝原因。
- 只校验字符前缀，不按 tokenizer 的完整 token transition 枚举合法候选。
- 把这五件事当成同一个状态：长度上限、模型发出 EOS、命中 stop string、连接断开、供应商返回的 finish reason。

## 交付与结论边界

最低交付物：三份原始输出、每部分一个手算或状态图、至少一个失败案例，以及一张“停止原因 → 是否已接受 → 是否继续计费”的对照表。

这些 CPU 小实验由本仓库准备输入，用来核对有限状态和确定性搜索契约。真实 tokenizer、JSON Schema、
目标 runtime、provider finish reason、网络取消和账单语义需要在各自环境继续验证。

下一步：使用云 API 的工程师继续[实验 0C](lab-0c-cloud-budget.md)；其他读者可回到[生成与解码](../../core/generation.md)完成自测。

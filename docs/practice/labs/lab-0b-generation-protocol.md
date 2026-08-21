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

逐步复算 beam 1 为什么剪掉最终概率 0.4 的 B 路径、beam 2 为什么能保留它；再复算 `log_probability / generated_length**alpha` 在 alpha 0/2 下为何翻转长短候选。

**最低通过**：修改 EOS 是否计入长度、early stopping 或 candidate cap，并明确把修改后的行为记录为新算法契约。

## 第二部分：约束解码

~~~powershell
python projects/inference-serving/constrained_decoding_toy.py
~~~

解释为什么 `1]` 不能因首字符 `1` 合法而进入候选，手算合法质量 0.35 如何变成 `1}`/`2}` 的 `5/7`/`2/7`。

**最低通过**：分别构造“语法已接受但 length 截断”“EOS 在非接受状态概率最高”“所有合法 token 概率为零”，证明三者不能合并为成功状态。

## 第三部分：流式 stop

~~~powershell
python projects/inference-serving/stop_matching_toy.py
~~~

解释 partial `<EN` 为什么不能立刻显示、emoji 的 UTF-8 byte split 为什么不应改变结果，以及 overlap 的优先级如何影响匹配。

**最低通过**：画出 byte fragment、可见文本、partial prefix 和 terminal event 的状态变化，并指出客户端隐藏文本不等于服务端停止生成或计费。

## 常见失败

- 只比较最后文本，不保存 beam 的逐步候选与剪枝原因。
- 只校验字符前缀，不按 tokenizer 的完整 token transition 枚举合法候选。
- 把 length cap、EOS、stop string、连接断开和 provider finish reason 当作同一状态。

## 交付与结论边界

最低交付物：三份原始输出、每部分一个手算或状态图、至少一个失败案例，以及一张“停止原因 → 是否已接受 → 是否继续计费”的对照表。

这些 CPU 小实验由本仓库准备输入，用来核对有限状态和确定性搜索契约。真实 tokenizer、JSON Schema、
目标 runtime、provider finish reason、网络取消和账单语义需要在各自环境继续验证。

下一步：使用云 API 的工程师继续[实验 0C](lab-0c-cloud-budget.md)；其他读者可回到[生成与解码](../../core/generation.md)完成自测。

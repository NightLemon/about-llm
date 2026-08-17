# 生成与解码入门

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：第一次理解模型如何从 logits 逐 token 生成文本的工程师。
- **先修**：token、条件概率、softmax 和 Transformer 输出 logits。
- **首次阅读**：最小循环 → 采样链 → 停止原因 → 最小实验。
- **完成信号**：能复算一步采样，并区分 EOS、长度和 stop string。
- **卡住时**：回到[新手知识地图](../guide/beginner-map.md)或[实验 0A](../practice/labs/lab-0a-sampling.md)。

</div>

一句话心智模型：生成不是模型一次写出整段文字，而是重复执行“前向得到 logits → 处理候选分布 → 选择一个 token → 判断是否停止”。

## 最小生成循环

```text
tokens = prompt_tokens
while budget remains:
    logits = model(tokens)[last_position]
    scores = apply_processors(logits)
    probabilities = softmax(apply_sampling_controls(scores))
    next_token = select(probabilities)
    tokens.append(next_token)
    if terminal_condition(tokens, next_token):
        break
```

先抓住五个对象：已有 token、下一步 logits、处理后的候选、选中的 token、停止原因。不要一开始同时学习 beam、grammar、SSE 和 provider 重试。

## 从 logits 到一个 token

常见处理链为：

1. 应用禁止项、重复惩罚或合法 token mask。
2. 用 temperature 改变分布尖锐程度。
3. 用 top-k、top-p 等规则限制候选 support。
4. 对保留分数重新归一化。
5. greedy 取最大值，或按概率采样。

processor 的顺序属于算法契约。即使参数名相同，不同 runtime 的默认顺序、tie-break、最小保留数和数值精度也可能产生不同结果。

## 停止不是一个布尔值

| 原因 | 说明 | 可以自动视为任务成功吗 |
|---|---|---|
| EOS | 模型或调用配置接受终止 token | 不一定，仍需检查任务输出 |
| 长度上限 | 预算耗尽 | 通常不能，内容可能被截断 |
| stop string | 客户端或服务端匹配文本 | 取决于匹配位置和服务端协议 |
| 约束无合法候选 | 当前状态死路 | 不能，应明确失败 |
| 连接断开/超时 | 客户端未得到完整结果 | 不能推断服务端已停止或未计费 |

JSON 能解析只说明语法通过；字段、事实、权限和业务约束仍需独立验证。

## 30 分钟最小实验

~~~powershell
python projects/inference-serving/sampling_toy.py
~~~

最低通过条件：

1. 复算 top-k 后的概率和 top-p crossing token。
2. 用固定 uniform 找出对应 CDF 区间。
3. 交换 top-k/top-p 顺序，解释 support 为什么变化。
4. 写明该 CPU fixture 不能证明目标模型质量或 runtime 默认行为。

完整步骤和反馈卡见[实验 0A](../practice/labs/lab-0a-sampling.md)。

## 常见误判

- temperature 是概率尺度参数，不是“创造力按钮”。
- greedy 可复现不等于答案正确；随机采样多样不等于质量更高。
- `max_new_tokens`、EOS 和 stop string 不应合并成同一种 finish reason。
- 客户端停止显示文本不证明服务端停止计算或计费。

## 进入完整章节

当你能手算一步采样后，再读[生成与解码完整协议](generation.md)：beam search、logits processors、约束解码、KV Cache 成本、流式 UTF-8 边界和生产验收都保留在那里。

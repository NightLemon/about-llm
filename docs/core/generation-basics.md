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

假设模型已经读到“今天天气”，屏幕上还没有后半句。它下一次前向只需要回答一个问题：下一个 token 选什么？
选中后，新 token 会接回输入，模型再回答同一个问题。我们看到的一整段文字，就是这个小循环反复运行的结果。

## 最小生成循环

```text
tokens = prompt_tokens
while budget remains:
    logits = model(tokens)[last_position]
    scores = apply_processors(logits)
    if mode is greedy:
        next_token = argmax(scores)
    else:
        require temperature > 0
        scaled_scores = scores / temperature
        top_k_support = keep_top_k(scaled_scores)
        probabilities_for_top_p = softmax(scaled_scores on top_k_support)
        support = keep_top_p_crossing(probabilities_for_top_p)
        probabilities = softmax(scaled_scores on support)
        next_token = sample(probabilities)
    tokens.append(next_token)
    if terminal_condition(tokens, next_token):
        break
```

第一次读这段循环，只跟住五样东西：已有 token、最后位置的 logits、处理后的候选分布、本轮选中的 token，
以及最终停止的原因。Beam search、grammar、SSE 和 provider 重试会在后续章节加入；它们都没有改变这个基本循环。

## 从 logits 到一个 token

模型给出的 logits 还不是最终抽样表。贪心生成从处理后的分数中取最大值；下面继续跟随机采样分支，要求 temperature 大于 0。
为便于手算，这里采用本仓库的参考顺序；真实 runtime 必须以其版本契约为准：

1. 应用禁止项、重复惩罚或合法 token mask。
2. 用 temperature 改变分布尖锐程度。
3. 先按分数保留 top-k 候选；未设置 top-k 时保留全部候选。
4. 先在 top-k support 内归一化，累计概率并保留让 top-p 首次 crossing 的 token。
5. 对最终 support 重新归一化，再按概率采样。

顺序会改变结果。比如先做 top-k 再做 top-p，与先做 top-p 再做 top-k，留下的 token 集合可能不同。
所以 `temperature=0.7, top_p=0.9` 还不足以复现实验；runtime 的处理顺序、平分规则、最小保留数和数值精度也要固定。

例如 logits 对应原始概率 `[0.4, 0.3, 0.2, 0.1]`。先取 top-k=3，分母从 1 变为
`0.4+0.3+0.2=0.9`，所以临时概率是 `[4/9,3/9,2/9,0]`。top-p=0.7 的累计值依次为
`4/9,7/9,...`，第二个 token 是 crossing token，必须保留。最终分母变为 `0.4+0.3=0.7`，
抽样分布才是 `[4/7,3/7,0,0]`。两次归一化回答不同问题：第一次决定 nucleus，第二次才是实际抽样。

## 停下来也要说明原因

循环结束只说明“不会继续生成”，并没有说明任务完成。下面几种终态需要分别保存：

| 原因 | 说明 | 可以自动视为任务成功吗 |
|---|---|---|
| EOS | 模型或调用配置接受终止 token | 不一定，仍需检查任务输出 |
| 长度上限 | 预算耗尽 | 通常不能，内容可能被截断 |
| stop string | 客户端或服务端匹配文本 | 取决于匹配位置和服务端协议 |
| 约束无合法候选 | 当前状态死路 | 不能，应明确失败 |
| 连接断开/超时 | 客户端未得到完整结果 | 不能推断服务端已停止或未计费 |

终态之后还可能有一层任务验收。例如 JSON 已经闭合，业务字段却缺失；模型输出了 EOS，回答却没有覆盖问题。
生成协议负责说明文本怎样结束，应用 verifier 再判断结果能否使用。

## 30 分钟最小实验

~~~powershell
python projects/inference-serving/sampling_toy.py
~~~

运行前先猜 top-k 和 top-p 各会删掉哪些候选。运行后完成四个观察：

1. 复算 top-k 后的概率和 top-p crossing token。
2. 用固定 uniform 找出对应 CDF 区间。
3. 交换 top-k/top-p 顺序，解释 support 为什么变化。
4. 把结论限定在当前 CPU 样例；目标模型和目标 runtime 需要另行实测。

完整步骤和反馈卡见[实验 0A](../practice/labs/lab-0a-sampling.md)。

## 怎样解释常见现象

- Temperature 调整分布的尖锐程度。“更有创造力”只是某些任务上的主观表现，不是它的数学定义。
- Greedy 的优势是路径确定，正确性仍由任务评价；随机采样带来不同候选，质量可能升也可能降。
- `max_new_tokens`、EOS 和 stop string 都能结束循环，但恢复、计费和用户提示方式不同，因此要保留各自的 finish reason。
- 客户端停止显示文本，只能说明客户端不再接收。服务端是否停算、KV 是否释放、费用是否继续，需要服务端证据。

## 进入完整章节

当你能手算一步采样后，再读[生成与解码完整协议](generation.md)：beam search、logits processors、约束解码、KV Cache 成本、流式 UTF-8 边界和生产验收都保留在那里。

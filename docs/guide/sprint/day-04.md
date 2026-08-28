# 第 4 天：真实 tokenizer 与 chat template

**今日目标**：从玩具分词器切到真实 tokenizer，理解 special token、chat template
和"同一段文本在不同模型下 token 数不同"这件事对成本与截断的实际影响。

**导航**：[上一天](day-03.md) · [返回速成总览](index.md) · [下一天](day-05.md)
{ .doc-nav }

## 时间盒

| 时段 | 内容 | 产出 |
|---|---|---|
| 上午 2h | [分词与词表](../../core/tokenization.md) | 中英文 token 效率差异的具体数字 |
| 上午 2h | [Transformers 基础](../../practice/projects/transformers-basics.md) 报告部分 | 一份自己跑出来的 tokenizer 报告 |
| 下午 2h | chat template 与 special token | 一次手工渲染的对话 prompt |
| 下午 2h | 复盘 + 日志 | 必答题答案 |

## 动手

跑 transformers-basics 的报告脚本。注意两个容易看错的点：

- `model.tokenizer_class` 只在加了 `--json` 时出现；
- `all_special_ids` **不是**报告字段，不要在输出里找它。

跑完后做一个对照实验：同一段中文，分别用两个不同家族的 tokenizer 编码，
数一数 token 数差多少。这个数字直接决定你的 API 成本和上下文预算。

## 为什么今天要认真对待 chat template

应用工程里最常见的静默 bug 之一：手工拼接 `"User: ...\nAssistant: "` 而不用模型自带的
chat template。模型训练时见到的是特定的 special token 序列，手拼的字符串看起来一样、
token 序列却不一样，结果是模型行为轻微退化——不报错，只是变差。

今天要做的是：把同一段对话分别用 `apply_chat_template` 和手工拼接编码，
把两个 token id 序列打印出来对比。看到差异之后，你就不会再手拼了。

## 必答题

1. 为什么中文的 token 数通常比同等信息量的英文多？
2. special token 的 id 在不同模型之间可以复用吗？
3. 截断 prompt 时，为什么不能简单地截断字符串？

## 今日交付

```text
两个 tokenizer 对同一段中文的 token 数对比
apply_chat_template 与手工拼接的 token id 序列差异
transformers-basics 报告的 scope 字段摘录
```

## 可以跳过

- 教学版 BPE 训练实现
- tokenizer 的正则预分词细节

## 明天接什么

[第 5 天](day-05.md) 是第一周的收口：注意力机制，以及把它和前四天的生成协议连起来。

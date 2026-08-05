# Llama 家族

## 学习价值

Llama 是公开权重生态的重要基线，适合研究 tokenizer、decoder 架构、LoRA/QLoRA、量化和本地服务。不同代际与尺寸配置不完全相同，所有具体结论以 checkpoint 的 config、tokenizer、generation config、model card 和许可为准。

## 常见架构元素

公开 Llama 系模型常见：

- decoder-only causal Transformer；
- pre-norm 与 RMSNorm；
- SwiGLU 类门控 MLP；
- RoPE 位置编码；
- 部分代际/尺寸使用 GQA；
- 输入 embedding 与输出 head 是否共享依配置；
- 多语言与更长上下文随代际调整。

不能把某一代的 head 数、词表、RoPE scaling 或上下文直接套到另一代。加载后检查 config，而不是凭模型名猜。

## Base 与 Instruct

Base 模型学习续写分布，适合继续预训练、研究和自定义后训练；Instruct 模型经过对话/偏好训练，依赖对应 chat template。给 Base 套 Instruct 模板不会自动获得指令能力；给 Instruct 用错误角色 token 也会显著退化。

## 单卡路线

1. 固定 revision 和许可；
2. 先检查 tokenizer/chat template；
3. 选择能容纳的尺寸与量化；
4. 测 BF16/FP16 或 8/4-bit 的质量与显存；
5. LoRA/QLoRA 从短上下文、小 batch dry-run；
6. 合并 adapter 前后做 logits/任务回归；
7. 用 Transformers 和 vLLM 在相同 workload 下比较。

权重 4-bit 不等于所有内存 4-bit。KV、激活、adapter 和工作区必须计入。

## 许可与分发

“可下载权重”不自动等于 OSI 开源。不同代际可能有独立社区许可、可接受用途和归因要求。发布派生 adapter、合并权重或服务前核对具体版本许可。

## 面试追问

1. RMSNorm 与 LayerNorm 差异？
2. GQA 如何减少 KV Cache？
3. RoPE 扩展配置为何不等于有效长上下文？
4. Base/Instruct 的 loss 和 chat template 有何关系？
5. 怎样证明量化后业务质量没有不可接受回归？

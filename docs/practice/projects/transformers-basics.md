# Transformers Basics：从 bytes 到真实 checkpoint

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：会写 Python，想把 token、attention、训练和生成连接到真实 checkpoint 的开发者。
- **先修**：高中数学与基础 Python；框架 API 可以边做边学。
- **首次阅读**：四级证据 → Phase 0 → 90 分钟 CPU 路线 → 静态 checkpoint → 按需机制实验。
- **完成信号**：能为每个结论指出输入、实现层、运行产物和不能外推的边界。
- **卡住时**：回到[新手知识地图](../../guide/beginner-map.md)，一次只运行一个脚本并先写预测。

</div>

**项目导航**：[项目索引](../project-index.md) ·
[Tokenization](../../core/tokenization.md) ·
[Transformer](../../core/transformer.md) ·
[生成与解码](../../core/generation.md) ·
[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/transformers-basics/README.md) ·
[证据页](../../evidence/transformers-controls.md)
{ .doc-nav }

这个项目不是第二套基础教材。它负责把已经学过的概念排成一条可运行路线：先用透明小输入观察状态变化，
再接 Transformers 接口与固定 checkpoint。公式回到主题正文，完整命令和故障处理只在 README 维护，
精确数值与历史运行只在证据页维护。

## 先认识四级证据

| 层级 | 本项目怎样观察 | 可以支持 | 不能自动支持 |
|---|---|---|---|
| 参考实现 | Byte BPE、NumPy attention、显式 loss | 公式和状态在固定输入上的变化 | 框架、目标权重或性能 |
| Tiny framework | MiniGPT、tiny GPT-2、JAX/PyTorch | 训练与生成接口已经接通 | 发布模型的质量 |
| 静态 checkpoint | 固定 config、tokenizer、模板和文件清单 | 发布对象与配置关系 | 权重真实执行 |
| 目标权重 | 固定 Qwen 的 forward/cache/generate | 该环境与输入的执行路径 | 总体质量、GPU 性能或生产安全 |

每个观察只使用自己所在层的结论。不要把多个独立实验拼成“已经完整验证某模型”。

## Phase 0：先把一次预测和更新算明白 {#phase-0}

第一条命令只使用 Python 标准库：

~~~powershell
python projects/transformers-basics/math_learning_walkthrough.py
~~~

运行前预测最大 logit、正确 token 的梯度方向和一次更新后的概率变化。运行后应能沿
`hidden state → logits → softmax → NLL → gradient → update` 解释每个字段。
逐步手算由[数学基础](../../foundations/math.md)负责，本页只把它作为后续脚本的共同起点。

## 推荐运行顺序 {#run}

完整可复制命令见
[项目 README](https://github.com/NightLemon/about-llm/blob/main/projects/transformers-basics/README.md#第一次运行)。
第一次会话只走 CPU 主线，并在每一步前写预测、运行后留下一条失败或边界解释。

| 阶段 | 只观察什么 | 成功后能回答 | 深入正文 |
|---|---|---|---|
| 1. Byte BPE | Bytes、pair count、merge rank、round trip | 字符、byte 与 token 为何不同 | [Tokenization](../../core/tokenization.md) |
| 1.5. LM sample | Input IDs、labels、causal/loss mask | 同一序列怎样成为训练目标 | [NLP](../../foundations/nlp.md) |
| 2. Attention | Shape、causal mask、cached/full 对照 | Query 怎样读取允许的 key/value | [Transformer](../../core/transformer.md) |
| 3. Online softmax | Running max 与归一化 | 为何能分块计算而不保存完整 score | [Attention 数值](../../foundations/attention-numerics.md) |
| 3.5. RMSNorm | Layout、FX、ATen、backend event | 数学、框架图与 kernel 为何不是一层 | [算子栈](../../systems/operator-stack.md) |
| 4. MiniGPT | Logits、masked loss、gradient、update | 一个样本怎样真正更新参数 | [训练数学](../../foundations/math-training.md) |
| 4.5. Tiny Transformers | Model API、Trainer 接线、generate | 自写机制怎样映射到框架接口 | [Transformer](../../core/transformer.md) |
| 5. Generation | EOS、长度和调用参数覆盖 | 生成为何停止，哪一层决定默认值 | [生成](../../core/generation.md) |

完成 CPU 路线的信号不是“八条命令都成功”，而是你能为其中任一步制造一个预期失败，并解释错误出现在哪层。

## 第二遍：从 tokenizer 走到 checkpoint

本地已有固定 snapshot 时，再进入三步：

1. 用 `trace_qwen3_tokenizer.py --local-files-only` 查看真实中文对话模板与 token IDs；
2. 用 `verify_release_evidence.py` 和 `inspect_checkpoint.py` 核对 revision、文件、config 与 processor；
3. 只有身份与 runtime 能力匹配时，才用 `run_target_checkpoint.py --local-files-only` 执行权重。

静态检查通过不等于权重能运行；一次 forward/generate 通过也不等于长上下文、GPU、质量或服务容量已经验证。
下载条件、完整参数和离线失败处理见 README，固定 Qwen 数值见证据页。

## 第三遍：只选一个机制实验

| 当前问题 | 选修入口 | 先读 |
|---|---|---|
| 哪个内部表示影响输出 | Activation patching | [机制可解释性](../../core/mechanistic-interpretability.md) |
| MoE 怎样选择、限流并通信 | Routing/training/all-to-all controls | [架构谱系](../../core/architectures-interpretability.md) |
| 张量怎样落到 backend | RMSNorm operator trace | [算子栈](../../systems/operator-stack.md) |
| 局部 INT4 误差是什么 | Qwen weight quantization control | [推理优化](../../systems/inference-optimization.md) |
| KV cache 与生成是否一致 | Target checkpoint control | [推理请求](../../systems/inference-request-lifecycle.md) |

这些选修互相独立，不应成为完成 CPU 主线的前置。CPU/Gloo、单矩阵量化或单事实 patching 都不能外推为
目标 GPU/NCCL 性能或模型总体能力。

## 你应保存哪些实验记录 {#artifact}

~~~text
experiment manifest
├── code revision and command
├── environment and hardware
├── input identity
├── prediction
├── raw output
├── one negative case
├── interpretation
└── claims not supported
~~~

截图可以辅助展示，但机器可读输出、固定输入和失败样例更便于复核。项目页只规定交付形状；
字段、输出目录和验证命令见 README，已录制的数值与 hash 见证据页。

## 怎样定位失败

| 症状 | 首先回到哪一层 |
|---|---|
| BPE merge 顺序变化 | 文档边界、同频 tie-break 与 merge rank |
| 模型复制当前 token | Input/labels shift 是否重复或遗漏 |
| Attention 看到未来 | Causal mask 方向、广播 shape 和 softmax 轴 |
| Cached/full logits 不一致 | Position IDs、past length、mask 与比较位置 |
| Loss 不下降 | Labels、padding mask、optimizer 参数和 train/eval mode |
| Generation 提前或不停止 | EOS/PAD/BOS、长度上限、template 与参数覆盖 |
| Config 容量数字异常 | 是否把 MLA 或自定义布局套入标准 GQA 公式 |

### 保存的报告通过，但新运行结果不同 {#recorded-report}

录制报告是历史 artifact。先比较代码、依赖、输入、snapshot、dtype、设备与容差，再运行当前环境的目标脚本。
旧报告内部 hash 一致，不表示本轮重新执行，也不能认证远端模型或执行者。

## 项目完成标准

你应能提交：

- BPE 合并与 round-trip 记录；
- Causal attention 的 mask 和 cache 正负对照；
- Tiny 模型的 labels、gradient 和 update 记录；
- Generation 的终止与参数覆盖实验；
- 一份配置字段“可以推导 / 信息不足”的表；
- 可选 checkpoint 的文件清单或冒烟结果；
- 每项结论的证据等级和不可外推边界。

完整运行手册位于
[projects/transformers-basics/README.md](https://github.com/NightLemon/about-llm/blob/main/projects/transformers-basics/README.md)；
精确证据位于[Transformers 证据台账](../../evidence/transformers-controls.md)。

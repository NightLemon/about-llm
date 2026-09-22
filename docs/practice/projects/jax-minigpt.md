# JAX MiniGPT：纯函数训练、跨框架对账与精确恢复

**项目导航**：[项目索引](../project-index.md) ·
[JAX 与 Optax](../../training/jax-optax.md) ·
[Transformer](../../core/transformer.md) ·
[分布式正确性](../../systems/distributed-training-correctness.md) ·
[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/jax-minigpt/README.md) ·
[项目证据](../../evidence/project-controls.md)
{ .doc-nav }

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：想从 PyTorch 迁移到 JAX，或需要验证训练恢复正确性的开发者。
- **先修**：Transformer 前向、反向传播、AdamW 和基本 checkpoint 概念。
- **首次阅读**：Tiny overfit → 同权重 SGD → 三步 AdamW → 跨进程 resume。
- **完成信号**：能解释 params、Optax state、PRNG key 和 data cursor 怎样共同决定下一步更新。
- **卡住时**：只运行 `train_tiny.py`，先画一次 train step 的输入和输出。

</div>

设想 MiniGPT 训练 3 步后进程退出，新进程加载 checkpoint 再训练 3 步。若恢复正确，第 4–6 步的样本、
dropout mask、loss、gradient 和参数应与不中断运行一致。这个项目用四个递进实验建立这条证据链，
而不是同时讲一套 JAX 教材、命令手册和结果台账。

## 四个实验分别回答什么

| 顺序 | 实验 | 唯一问题 | 通过后仍不能推出 |
|---:|---|---|---|
| 1 | Tiny-batch overfit | JAX forward、gradient、Optax 与 JIT 是否接通 | 泛化、生成质量或性能 |
| 2 | Plain-SGD parity | 同权重同公式下，两框架一步更新是否一致 | AdamW、随机层与长期轨迹 |
| 3 | Shared-mask AdamW parity | Clipping、moments、schedule 与三步轨迹是否一致 | 原生 RNG 等价 |
| 4 | Cross-process resume | 完整状态能否从第 3 步续到同一轨迹 | 多设备、对象存储或断电原子性 |

前一个实验通过不能替后一个回答问题。每次只新增一个状态维度，差异出现时从最早分叉处排查。

## 第一次运行 {#run}

完整安装和命令由
[项目 README](https://github.com/NightLemon/about-llm/blob/main/projects/jax-minigpt/README.md)维护。
第一次只运行：

~~~powershell
python -m pip install -e ".[dev,torch,jax]"
python projects/jax-minigpt/train_tiny.py --steps 60 --learning-rate 0.02 --seed 11
~~~

先确认 backend/device，再比较初末 loss，并把编译加首次 step 与后续稳定 step 的时间分开。
这个脚本输出训练指标，不生成文本；tiny batch 被记住只说明训练接线能工作。

## 先跟一个 batch 走完一步 {#one-train-step}

一次 JAX train step 的状态流是：

```text
params + optimizer state + batch + optional PRNG
→ forward
→ masked token-mean loss
→ value_and_grad
→ clip + Optax update
→ new params + new optimizer state
```

更新前返回的 loss 与使用新参数再次前向得到的 loss 属于两个时间点。JAX 计算还可能异步提交，
因此计时需要在明确结果上 `block_until_ready()`。公式、PyTree 与 JIT 语义见
[JAX/Optax 教材](../../training/jax-optax.md)。

固定 batch overfit 没有 dropout，也不推进数据游标。进入恢复实验后，完整状态至少包含：

| 状态 | 为什么必须恢复 |
|---|---|
| Params | 定义当前模型 |
| Optax state | 保存 moments、count 与更新历史 |
| PRNG key | 决定下一次随机操作 |
| Data permutation/cursor | 决定下一批样本 |
| Global step | 对齐 schedule、日志和 checkpoint generation |

能反序列化权重只是最低要求。若 key 或 cursor 被重置，文件仍可读取，但后续训练已经不是同一条轨迹。

## 第二步：先对齐数学身份

跨框架比较前要固定同一组解析权重、LayerNorm/GELU、Linear 布局、causal mask、weight tying、
输入、loss reduction 与容差。不要比较两个随机初始化的“相似 GPT”。

Plain-SGD 实验依次比较 logits、loss、每个参数梯度、一步更新和更新后输出；RMSNorm 反例故意改变一个数学组件。
成功信号是误差在事先声明的容差内，失败时能指出第一个分叉张量，而不是把容差调大。

## 第三步：增加 optimizer 与随机性

三步 AdamW 对账使用相同 dropout masks，把原生随机数差异暂时隔离，集中检查 global-norm clipping、
一阶/二阶 moments、schedule count 和连续参数轨迹。错位 mask 必须让结果分叉。

Shared mask 只能证明给定随机输入下的 optimizer parity，不能证明 PyTorch 与 JAX 的 native RNG 相同。
这条边界要和结果一起保存。

## 第四步：跨进程恢复

恢复实验先完整运行 6 步作为基线，再让另一条路径在第 3 步保存，由新进程完成第 4–6 步。
比较对象不是最终 loss 一个数，而是后半程的样本顺序、dropout mask、loss/gradient trace、参数和完整状态 fingerprint。

仓库的 `ALLMJAX1` 是为教学状态面准备的单文件格式。它不替代 Orbax/TensorStore，也没有覆盖多设备分片、
对象存储一致性或断电原子性。精确版本、误差和负例结果只在[项目证据](../../evidence/project-controls.md)维护。

## 失败时从最早分叉处排查

| 症状 | 先检查 |
|---|---|
| Loss 不下降 | Target shift、mask 分母、gradient tree、更新后的 Optax state |
| 首步很慢 | 编译与执行是否分开、shape/dtype 是否变化 |
| 初始 logits 不同 | 参数映射、norm、GELU、mask、weight tying |
| Logits 相同但 gradient 不同 | `-100` mask、reduction、共享参数 |
| AdamW 后续漂移 | Dropout mask、clipping、moments、schedule count |
| Resume 后漂移 | PRNG、permutation/cursor、step、Optax state、dataset identity |
| 计时异常短 | 是否在正确结果上等待设备完成 |

README 保存完整命令和测试集合；本页只决定应运行哪一个实验以及如何解释它。

## 三个入口怎样分工

| 入口 | 负责什么 |
|---|---|
| 本页 | 四步学习路线、每步观察和升级条件 |
| [项目 README](https://github.com/NightLemon/about-llm/blob/main/projects/jax-minigpt/README.md) | 安装、命令、文件、故障处理和测试 |
| [JAX 教材](../../training/jax-optax.md) | 纯函数、PyTree、PRNG、JIT 与工程机制 |
| [项目证据](../../evidence/project-controls.md) | 固定版本、精确数值、容差、负例与不可外推边界 |

## 完成与下一步

完成本项目时，应能：

- 画出 params/Optax/PRNG/data cursor 的状态流；
- 区分首次编译和稳定 step 计时；
- 说明跨框架 parity 必须对齐哪些数学身份；
- 用 RMSNorm、wrong-mask、wrong-PRNG、wrong-cursor 解释因果；
- 说明 bit-exact resume 为何需要 trace 与 full state，而不只是最终 loss；
- 把 CPU、单 accelerator、多设备和目标模型证据分栏。

之后再选择 Flax/Orbax、mesh/sharding、混合精度或目标设备性能中的一条路线。当前 CPU tiny 结果不证明
JAX 比 PyTorch 更快，也不证明已经完成生产级分布式大模型训练。

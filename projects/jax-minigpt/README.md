# JAX/Optax MiniGPT 训练实验

这个项目用一个很小的 decoder-only Transformer 回答四个逐步深入的问题：JAX 训练链路能不能跑通，
PyTorch 与 JAX 是否真的实现了同一个函数，AdamW 的连续更新是否一致，以及训练中断后能否沿原来的轨迹继续。

第一次学习时，只运行 `train_tiny.py`。确认 tiny batch 能被模型记住后，再按顺序执行其余三个实验。
完整原理、公式和结果解释见 [JAX MiniGPT 教学页](../../docs/practice/projects/jax-minigpt.md)；本页只负责帮你把项目跑起来。

## 第一次运行

安装 JAX、Optax、PyTorch 和测试依赖：

```powershell
python -m pip install -e ".[dev,torch,jax]"
python scripts/doctor.py
```

然后运行最小训练：

```powershell
python projects/jax-minigpt/train_tiny.py `
  --steps 60 `
  --learning-rate 0.02 `
  --seed 11
```

脚本会让一个 632 参数的模型反复学习两条相同的 token 序列。当前录制的 CPU 结果中，loss 从约 `2.1086`
下降到 `0.0030`。你首先要检查三件事：

- `backend` 和 `device` 是否符合预期；
- `final_loss` 是否明显低于 `initial_loss`；
- `compile_plus_first_step_seconds` 是否与后续 step 的平均时间分开记录。

这个脚本只训练，不生成文本。Tiny-batch overfit 的意义是尽早发现 target shift、梯度断开或 optimizer 没有更新；
它不能衡量泛化能力，也不能说明 GPU、TPU 或大模型的训练性能。

## 四个实验分别回答什么

| 顺序 | 运行入口 | 要回答的问题 | 重点查看的结果 |
|---:|---|---|---|
| 1 | `train_tiny.py` | JAX 的前向、反向、Optax 更新与 JIT 是否接通？ | 初末 loss、梯度范数、实际 device |
| 2 | `cross_framework_parity.py` | 对齐权重和公式后，PyTorch 与 JAX 的前向、梯度和一步 SGD 是否一致？ | 各阶段最大绝对误差与容差 |
| 3 | `cross_framework_training_parity.py` | 两边使用相同 dropout mask 时，三步 AdamW 轨迹是否一致？ | 裁剪后梯度、moments、step count 和参数误差 |
| 4 | `checkpoint_resume_control.py` | 第 3 步退出并换进程恢复后，第 4—6 步是否与不中断训练相同？ | 样本顺序、loss/gradient trace 和完整状态 fingerprint |

它们是递进实验，不是四种训练方案的性能排名。前一个实验通过，也不能替后一个实验回答问题。

## 1. 跑通原生 JAX 训练

```powershell
python projects/jax-minigpt/train_tiny.py --steps 60 --learning-rate 0.02 --seed 11
```

核心实现位于 [`src/about_llm/from_scratch/gpt_jax.py`](../../src/about_llm/from_scratch/gpt_jax.py)。它没有使用
Flax module，而是显式传递参数 PyTree 和 Optax state：

```text
params + optimizer state + batch
    → forward
    → masked cross entropy
    → value_and_grad
    → clip + AdamW
    → new params + new optimizer state
```

JAX 可能异步提交计算，所以脚本在计时前等待 `loss.block_until_ready()`。首次计时包含编译，后续计时才接近
已经编译后的 step 成本。即便完成同步，这个 tiny CPU 结果也只适合检查计时方法，不适合外推目标硬件吞吐。

## 2. 对齐 PyTorch 与 JAX 的数学实现

```powershell
python projects/jax-minigpt/cross_framework_parity.py
```

这个实验不比较两个各自随机初始化的模型。它先生成一份确定性权重，再把 Linear 的布局、LayerNorm、GELU、
causal mask、共享 embedding/head 和 loss reduction 全部对齐，然后比较：

1. 初始 logits 与 loss；
2. 20 组独立参数的 gradients；
3. 一步 plain SGD 后的参数；
4. 更新后的 logits 与 loss。

结果中的每个误差都应低于报告给出的容差。实验还保留一条 RMSNorm 负对照：如果只看到“两边都是 GPT”就直接
比较，而没有对齐归一化方法，logits 会明显分叉。这说明 parity 的前提是同一个数学函数，而不是相似的模型名称。

## 3. 连续检查三步 AdamW

```powershell
python projects/jax-minigpt/cross_framework_training_parity.py
```

一步 SGD 没有 moments，也没有随 step 改变的学习率。这个实验因此连续更新三步，并提前生成相同的 dropout masks
交给两个框架。这样可以把“随机输入不同”这个变量暂时排除，集中检查：

- Global-norm clipping 是否按相同顺序执行；
- AdamW 的一阶、二阶 moments 是否一致；
- Optimizer 与 schedule 的 step count 是否同步；
- 每一步更新后的参数和 logits 是否仍在容差内。

报告还会把 JAX mask 故意错开，展示随机输入改变后参数如何分叉。共享 mask 只服务于这次数学对账，不能证明
PyTorch 与 JAX 的原生随机数算法相同。

## 4. 在新进程中继续训练

```powershell
python projects/jax-minigpt/checkpoint_resume_control.py
```

实验先完整训练 6 步作为基线。另一条路径训练 3 步后写出 checkpoint，再由新进程加载并完成第 4—6 步。
如果恢复正确，两条路径后半程使用的样本、dropout mask、loss、gradient 和最终状态都应相同。

本项目保存的不只是模型参数，还包括 Optax state、PRNG keys、数据排列、读取位置和 global step。报告中的两个
负例会分别重置 dropout key 和 data cursor；文件仍能读取，但后续训练轨迹会发生变化。

`ALLMJAX1` 是仓库为了讲清状态恢复而实现的单文件格式，不是 Orbax 或 TensorStore 的替代品。它会检查字段、
数组布局和哈希，但没有覆盖多设备分片、对象存储一致性或断电原子性。

## 主要文件

| 文件 | 用途 |
|---|---|
| `train_tiny.py` | 运行原生 JAX tiny-batch overfit，并区分首次编译与后续 step 计时 |
| `cross_framework_parity.py` | 比较 PyTorch/JAX 的 forward、backward 与一步 SGD |
| `cross_framework_training_parity.py` | 比较 shared-mask 条件下的三步 AdamW 轨迹 |
| `checkpoint_resume_control.py` | 验证跨进程 checkpoint 与 bit-exact resume |
| [`gpt_jax.py`](../../src/about_llm/from_scratch/gpt_jax.py) | 从零实现的 JAX decoder、loss 与 train step |
| [教学页](../../docs/practice/projects/jax-minigpt.md) | 解释纯函数状态、PRNG、parity 和 checkpoint 设计 |
| [项目证据页](../../docs/evidence/project-controls.md) | 保存录制结果、版本与验证范围 |

## 常见故障

| 现象 | 先检查什么 |
|---|---|
| `train_tiny.py` 的 loss 不下降 | Targets 是否右移、梯度树是否回传、optimizer state 是否使用了新值 |
| 首步远慢于后续 step | 首次 JIT 编译是否被单独记录；输入 shape 或 dtype 是否持续变化 |
| 初始 logits 已经不同 | 参数映射、Linear transpose、norm、GELU、mask 和 weight tying |
| Logits 相同但 gradient 不同 | `-100` mask、loss reduction 和共享参数是否只计算一次 |
| AdamW 前一步一致、后续漂移 | Dropout mask、clipping 顺序、moments 和 schedule count |
| Checkpoint 能打开但续训结果不同 | PRNG key、数据 permutation/cursor、global step 和 Optax state |
| 计时异常地短 | 是否缺少 `block_until_ready()`，测到的可能只是 Python 提交时间 |
| JAX 没有使用预期设备 | 查看脚本输出的 `backend/device`，再检查本机 JAX 安装与驱动 |

定位 parity 问题时，从最早出现差异的位置向后查，不要先调大容差，也不要只比较最终 loss。

## 运行专项测试

```powershell
python -m pytest tests/test_gpt_jax.py -q
python scripts/check_docs.py
python scripts/check_content_accuracy.py
```

专项测试覆盖因果 mask、有限 loss、参数更新、跨框架误差、AdamW 状态、checkpoint 完整性和两个错误恢复负例。
它们运行在 CPU tiny 配置上，不会替你验证 CUDA、TPU、多设备 sharding、混合精度或目标模型的收敛情况。

## 下一步怎么扩展

完成四个实验后，再选择一个方向继续：

- 想研究真实 JAX 训练栈：把状态模型迁移到 Flax/Orbax，并验证异步保存和拓扑变化后的 reshard；
- 想研究多设备：为 mesh、`PartitionSpec`、全局/本地 shape 和 collective 建立独立测试；
- 想研究混合精度：分别记录参数、计算和 optimizer state dtype，以及 loss scaling 与溢出处理；
- 想研究性能：固定 shape、dtype、同步点、warmup 和 compile cache，再在目标设备保留原始样本。

当前项目能支持的结论是：在 CPU tiny 固定输入上，JAX 训练链路、指定的 PyTorch/JAX 数学对账，以及包含完整
训练状态的跨进程恢复都经过了可执行检查。它没有证明 JAX 比 PyTorch 更快，也没有完成生产级分布式大模型训练。

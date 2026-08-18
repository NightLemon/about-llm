# JAX、Optax 与函数式训练闭环

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：用 JAX/Optax 实现可复现训练步的工程师。
- **先修**：Python、线性代数、自动微分和优化器状态基础。
- **首次阅读**：纯函数与 pytree → 显式随机 key → loss/grad → Optax/JIT。
- **完成信号**：能写一个可 JIT、显式传递参数和 RNG 的训练步。
- **卡住时**：先读[机器学习与深度学习](../foundations/ml-dl.md)，再跑[JAX MiniGPT](../practice/projects/jax-minigpt.md#run)。

</div>

## 学习目标与证据边界

读完本章应能把 decoder-only Transformer 写成参数 PyTree 上的纯函数，解释 `value_and_grad`、Optax transformation 与 optimizer state，构造一个 `jax.jit` 训练步，并正确测量异步执行时间。还应能判断“单设备可训练”与“多设备分片可扩展”之间缺少哪些证据。

本仓库的 `gpt_jax.py` 与 `projects/jax-minigpt` 已在当前 **CPU JAX device** 完成 tiny-batch overfit；独立 PyTorch↔JAX 同权重 parity control 又在强制 CPU 的 Float32 contract 下对账 forward、masked loss、全部 unique parameter gradients 与 plain-SGD 单步。这证明两条明确路径可运行，不证明 CUDA/TPU、混合精度、多机网络或大模型吞吐。JAX、jaxlib、Optax 与 accelerator plugin 的兼容矩阵随平台变化，安装时应使用官方说明并记录实际版本。

## 为什么 JAX 看起来不像传统 Module

JAX 的核心心智模型是“数组程序变换”。模型可以写成：

\[
\text{logits}=f(\theta, x; c)
\]

其中参数 \(\theta\) 是 PyTree，输入 \(x\) 是数组，配置 \(c\) 在 trace 时保持静态。训练状态不会隐式藏在 Module 对象里，而是显式流过函数：

\[
(\theta_{t+1}, s_{t+1}, L_t, \lVert g_t\rVert)
=\operatorname{step}(\theta_t,s_t,x_t,y_t)
\]

这里 \(s_t\) 是 optimizer state，可能包含 Adam 的一阶/二阶矩与计数器。显式状态更适合编译、复制、分片和 checkpoint，也要求调用者认真管理每一项状态。

### PyTree

PyTree 是嵌套的 dict/list/tuple/dataclass 等容器，叶子通常是 JAX arrays。`jax.tree.map`、自动微分和 Optax 会以相同树结构遍历参数与梯度。

本仓库的参数树包含：

```text
params
├── token_embedding
├── position_embedding
├── blocks[]
│   ├── qkv / output
│   ├── up / down
│   └── attention_norm / mlp_norm
└── final_norm
```

树结构是 optimizer/checkpoint 契约的一部分。改参数名称、容器类型或层数后，旧 optimizer state 通常不能直接套用；恢复时应验证 treedef、shape、dtype 和 sharding。

## 显式随机数：key 不是全局种子

JAX PRNG key 是数据。初始化、dropout、数据增强和采样都应得到独立子 key：

```python
key = jax.random.key(seed)
init_key, dropout_key, data_key = jax.random.split(key, 3)
params = init_params(init_key, config)
```

同一个 key 重复使用会产生相关或完全相同的随机结果。多设备时还要用 step、process/device index 等折叠出不同流，并把 RNG 状态写入 checkpoint。只保存整数 seed 而不保存消费位置，不能精确恢复一个已运行数千步的任务。

本仓库 tiny GPT 没有 dropout，因此训练步不接收 RNG；一旦加入随机层，key 必须作为输入和输出状态显式传递，不能在 jitted 函数里依赖 Python 全局随机数。

## 从 loss 到参数更新

### Masked cross entropy

对 logits \(z_{b,t,v}\) 和 target \(y_{b,t}\)，可见 token 的平均负对数似然为：

\[
L=-\frac{1}{\sum_{b,t}m_{b,t}}
\sum_{b,t}m_{b,t}\log\operatorname{softmax}(z_{b,t})_{y_{b,t}}
\]

`ignore_index` 位置先替换成合法的 safe target，再乘 mask；不能先用负 id gather。分母是有效 token 数，不是固定的 batch × sequence，否则不同 padding 比例会改变 loss 尺度。

全 `ignore_index` 或可见 target 越界不是“loss 为 0”的正常 batch。低层 JIT-compatible primitive 对这两种输入返回非有限 sentinel；`make_train_step` 的 host wrapper 在进入 compiled update 前直接拒绝，避免 AdamW 在零监督下仍因 weight decay 修改参数。生产数据管线仍应更早统计并拒绝零监督样本和 batch。

### `value_and_grad`

训练需要 loss 和梯度：

```python
def loss_function(current_params):
    logits = forward(current_params, input_ids, config)
    return cross_entropy_loss(logits, targets)

loss, gradients = jax.value_and_grad(loss_function)(params)
```

被求导函数应返回标量 loss。若同时需要统计量，可使用带 auxiliary output 的形式；统计数组的生命周期和跨设备归约仍需显式设计。`stop_gradient`、离散索引、错误 mask 或把数组转成 Python number 都可能截断梯度。

## Optax 是梯度变换，不是持有参数的对象

Optax transformation 通常暴露三步：

1. `optimizer.init(params)` 创建与参数树相关的 state；
2. `optimizer.update(grads, state, params)` 产生 updates 与新 state；
3. `optax.apply_updates(params, updates)` 得到新参数。

本仓库组合全局 norm 裁剪与 AdamW：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(max_grad_norm),
    optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
)
```

顺序有语义：这里先裁剪梯度，再交给 AdamW。日志中的 gradient norm 在裁剪前测量，便于发现尖峰。若只记录裁剪后的 norm，所有超阈值尖峰看起来都一样。

### AdamW 与参数 mask

AdamW 将 decoupled weight decay 作为参数更新的一部分，不等同于把 L2 项随意加到任意 loss 实现。真实 Transformer 通常不给 norm scale 和 bias-like 参数做 weight decay；JAX/Optax 可用与参数 PyTree 对齐的 mask 指定集合。

本教学实验为透明起见对所有参数使用同一 decay，并将其设为 0 做 overfit。不能把这个配置直接称为生产预训练配方。

## JIT：先 trace，再编译，再异步执行

`jax.jit` 首次看到一组函数、PyTree 结构、shape、dtype 和静态参数时，会 trace Python 函数并编译。后续兼容输入复用 executable。以下变化可能触发新编译：

- batch/sequence shape 改变；
- dtype 改变；
- 参数 PyTree 结构改变；
- 被当作 static 的配置值改变；
- Python 控制流依赖动态数组值。

频繁变长输入若不分 bucket，会产生 compile cache 膨胀。生产数据管线通常把长度分桶、padding 到有限 shape 集，并监控 compile 次数和 cache 命中。

本仓库训练步把 config 和 optimizer 捕获在闭包中，再对完整 step 做 JIT：

```python
def step(params, optimizer_state, input_ids, targets):
    loss, gradients = jax.value_and_grad(loss_function)(params)
    gradient_norm = optax.tree.norm(gradients)
    updates, new_state = optimizer.update(gradients, optimizer_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, loss, gradient_norm

train_step = jax.jit(step)
```

### Python side effect 为什么危险

`print`、list append、文件写入等 Python side effect 通常发生在 trace 阶段，而不是每个设备 step。动态分支应使用 `jax.lax.cond`/循环原语或在 JIT 外控制。日志可在同步边界采集返回的统计量；调试输出工具也不应当作生产日志系统。

## 计时必须同步

JAX 会异步 dispatch。以下计时通常只测到 enqueue：

```python
started = time.perf_counter()
loss = train_step(...)[2]
elapsed = time.perf_counter() - started  # 可能尚未完成设备计算
```

正确的 wall-clock 边界需要等待结果：

```python
started = time.perf_counter()
params, state, loss, grad_norm = train_step(...)
loss.block_until_ready()
elapsed = time.perf_counter() - started
```

首次时间包含 trace/compile/执行，不应与 steady-state 混为一项。benchmark 至少报告 warmup/compile、同步方式、step 数、shape、dtype、device、数据传输和统计分位数。

## 数据放置与 host-device 边界

Python/NumPy 数据在 host，JAX arrays 可能在 accelerator。若每步临时构造数组或做 Python tokenization，device 会等待 host。数据管线需要：

- 固定或分桶 shape；
- 预取与 device transfer 的时间边界；
- 避免无意的 device→host 标量转换；
- 记录有效 token 数，而不是只报 examples/s；
- 在多 process 下保证 shard 不重不漏，并保存 iterator 位置。

`float(loss)`、`np.asarray(array)` 和打印数组都会触发同步/传输。调试时有用，训练热路径中过度使用会破坏吞吐。

## 从单设备到 Sharding

单设备 JIT 并不会自动成为多设备高效训练。JAX `Array` 可以携带 sharding；mesh 把物理设备组织成命名轴，参数/激活/数据用分区规则映射到这些轴。概念上仍需选择：

- data parallel：复制参数，分 batch，梯度做 collective；
- tensor/model parallel：分矩阵维，层内需要 collective；
- sequence/context parallel：分序列/激活，attention 通信更复杂；
- expert parallel：按专家分权重与 token，依赖 all-to-all。

分片是否正确要检查实际 array sharding 与 addressable shards，不能只看配置字符串。多 process 还需验证 process topology、global/local batch、collective 顺序、checkpoint 聚合和故障恢复。

本仓库没有多设备测试，因此不声称 `NamedSharding`、mesh 或任何分片策略已在目标 GPU/TPU 验证。项目 README 中的这些项目仍是下一步，不是已完成功能。

## 数值精度与等价性

JAX 默认 dtype、是否启用 x64、accelerator kernel 和 matmul precision 都会影响结果。BF16/FP16 混合精度训练还要明确：

- 参数、计算、梯度归约和 optimizer state dtype；
- FP16 是否使用 loss scaling；
- softmax、norm、loss 等敏感算子是否升精度；
- 溢出/NaN 检测和跳步策略；
- checkpoint 恢复后的 dtype/sharding 是否一致。

与 PyTorch 对齐时，先固定相同权重、输入、mask、GELU 近似、norm epsilon 和 tied embeddings，比较 logits/loss，再比较单步梯度与更新。只比较最终生成文本无法定位数值差异。

### 已执行的 PyTorch↔JAX 同权重 parity control

“同为 decoder-only Transformer”不是充分契约。仓库原生 PyTorch MiniGPT 使用 affine LayerNorm/epsilon=`1e-5`，原生 JAX MiniGPT 使用无 bias RMSNorm/epsilon=`1e-6`；LayerNorm 的 mean subtraction 本身就会改变函数。因此 control 不修改这两份默认身份，而是把 PyTorch 参数显式映射到一条 JAX LayerNorm parity forward，并统一 tanh-GELU、causal mask、tied embedding、一个 `-100` masked target 和 plain SGD。

固定 2-layer/8-dim Float32 fixture 的初始 logits/loss 最大差为 `7.636845111846924e-08/0`，20 个 parameter gradients 的全局最大差为 `2.384185791015625e-07`。一步 `lr=0.025` 后参数/logits/loss 最大差为 `7.450580596923828e-09`、`7.450580596923828e-08`、`2.384185791015625e-07`，都通过 `2e-6` authored tolerance。若把同一主干权重直接送入原生 RMSNorm 路径，logits 最大差为 `0.37747739627957344`；这个反事实防止把“参数 shape 可映射”误写成“架构等价”。

此处没有比较框架 RNG、AdamW moments/state、weight-decay mask、schedule、dropout、JIT executable、异步 timing 或 accelerator kernel。数值容差是当前 CPU Float32 fixture 的回归门槛，不是所有设备/dtype 的通用精度保证。

### 已执行的三步 AdamW trajectory parity

`cross_framework_training_parity.py` 在上述架构契约之上对账三步真实更新。为了隔离框架 RNG 算法，NumPy PCG64 先生成三张相同的 embedding inverted-dropout mask，再把同一 Float32 mask bytes 分别传给 PyTorch/JAX；这叫共享随机输入，不叫原生 RNG 等价。每步都比较 masked loss、raw gradients、pre-clip norm、global-norm clipped gradients、AdamW **first/second moments** 与 count、参数和 post-step forward。学习率固定为 `0.02→0.01→0.005`，beta 为 `0.9/0.95`、epsilon=`1e-8`、全参数 weight decay=`0.03`、clip threshold=`0.08`。

固定三步的 raw/clipped gradient 最大差为 `3.129243850708008e-07/1.862645149230957e-08`，moments 为 `2.561137080192566e-09/8.003553375601768e-11`，参数和 post-step logits 为 `2.5480985641479492e-06/1.564621925354004e-07`；Adam/schedule count 两边逐步都是 `[1,2,3]`。所有 pre-clip norm 都大于 1.3，证明 clip 路径实际生效。循环移位 JAX mask 的反事实产生 `0.06900620367377996` 最终参数差；两次独立进程报告 fingerprint 同为 `sha256:68ffa8093a1f2b98…`。

这个 control 的 weight decay 故意作用于所有参数，因此不覆盖生产中常见的 norm/bias mask；它也不对账 PyTorch/JAX native PRNG state、JIT/异步执行、checkpoint、混合精度、CUDA/TPU、sharding、长训练收敛或性能。`5e-6` 是当前 pinned CPU Float32 contract 的回归门槛，不是跨硬件普适误差保证。

## Checkpoint 必须包含完整训练状态

可恢复训练至少保存：

```text
params
optimizer state
global step / consumed tokens
PRNG keys
data iterator/shuffle position
config and schedule
PyTree structure, dtype and sharding metadata
code/dependency/data manifest versions
```

保存后要在独立进程加载，运行相同 batch 并检查 loss/下一步更新。跨设备拓扑恢复可能需要重新分片；“文件能打开”不等于训练语义连续。

### 已执行的独立进程 bit-exact resume control

`projects/jax-minigpt/checkpoint_resume_control.py` 把上述清单变成一条可重放实验。`ALLMJAX1` 单文件工件用 canonical manifest 固定模型、optimizer 和数据身份，并逐叶绑定 name、shape、dtype、offset、size 与 SHA-256；连续 little-endian payload 保存参数 PyTree、Optax state、typed PRNG key data、dropout/data key、shuffle permutation/cursor 和 global step。loader 会在创建 JAX array 前拒绝非 canonical/重复字段、叶子顺序、shape/dtype、内外 digest、截断和多余 bytes；writer 采用 exclusive create 与 file `fsync`。

固定 7 条样本、batch 2、embedding dropout 0.2、clip + AdamW 的 6 步 CPU fixture 在 step 3 交接：第一个 spawn process 写出 **13,476 bytes** 的工件（`sha256:e9252e5dddfa4aa5…`）后退出，第二个独立进程恢复。split-run 与 uninterrupted 的 sample IDs、逐步 loss/pre-clip gradient norm、最终参数/Optax/PRNG/data state 均 bit-exact，终态 fingerprint 为 `sha256:720817cca4c067cf…`。只把 dropout key 重置到初始 seed，或只把 cursor 从 6 改回 0，会让最终参数最大差分别变成 `0.037261832505464554` 与 `0.03700308472616598`；因此只恢复 params/optimizer 不能推出训练连续。

这个 control 不是 Orbax、Flax 或 TensorStore，也没有 directory `fsync`/断电原子性、来源认证、Python/NumPy/worker/accelerator RNG、CUDA/TPU、多设备 sharding、目标模型、收敛或性能证据。跨拓扑恢复还必须另外验证 treedef、sharding、global array 语义和重新分片。

## 可运行实验与验收

运行：

```powershell
python -m pip install -e ".[jax]"
python projects/jax-minigpt/train_tiny.py --steps 60 --learning-rate 0.02 --seed 11
python projects/jax-minigpt/cross_framework_parity.py
python projects/jax-minigpt/checkpoint_resume_control.py
python -m pytest tests/test_gpt_jax.py
```

验收项：

1. 改未来 token 不影响过去位置 logits，证明 causal mask；
2. 初始 loss 有限；
3. JIT train step 返回有限 pre-clip gradient norm；
4. token embedding 确实变化；
5. 固定 tiny batch 的 final loss 显著低于 initial loss；
6. 输出 backend/device 和同步计时边界；
7. 未安装 Optax 时测试不能被宣称为通过。
8. parity control 的 LayerNorm 路径通过，而原生 RMSNorm 反事实保持显著非零。

tiny-batch overfit 只验训练闭环，不验泛化；跨框架 parity 只验当前显式相同契约，不验两个默认模型或训练栈整体等价。三步 control 已对账 AdamW state、schedule 与共享的 embedding dropout masks，但仍应增加独立 validation batch、多 seed、全部 dropout site、norm/bias decay mask，并分别验证两个框架的 native PRNG state 与恢复语义。

## 常见错误

- 在 jitted 函数中依赖 Python 全局状态、随机数或 side effect；
- 重复使用同一 PRNG key；
- 把首次 compile 时间混入 steady throughput；
- 计时不调用 `block_until_ready`；
- 每个 batch 使用不同 shape 导致反复编译；
- 忘记保存 optimizer/RNG/data iterator state；
- 对 norm/bias 无差别 weight decay，却称为标准配方；
- 用单 CPU/GPU overfit 结果声称多设备 sharding 已验证；
- 只比较 PyTorch/JAX 最终文本，不做 logits/gradient 对照。

## 面试追问

1. PyTree 和显式 optimizer state 为什么适合程序变换与分片？
2. `jax.jit` 在什么条件下重编译，动态长度数据怎样控制 shape 集？
3. JAX 异步 dispatch 为什么让朴素 wall-clock benchmark 失真？
4. AdamW、gradient clipping 与参数 mask 的顺序分别影响什么？
5. dropout key 在 step、device 和 process 维度应怎样派生？
6. 单设备代码迁移到 mesh/sharding 时要新增哪些正确性验证？
7. 怎样设计 PyTorch/JAX 单步等价实验？

## 一手资料

- JAX 官方文档，[Just-in-time compilation](https://docs.jax.dev/en/latest/jit-compilation.html)。
- JAX 官方文档，[Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)。
- JAX 官方文档，[Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)。
- Optax 官方文档，[Getting started](https://optax.readthedocs.io/en/latest/getting_started.html)。
- 本仓库 `src/about_llm/from_scratch/gpt_jax.py`、`projects/jax-minigpt/train_tiny.py` 与 `tests/test_gpt_jax.py`；当前可执行证据的最高优先级来源。

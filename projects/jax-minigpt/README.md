# JAX/Optax MiniGPT 训练实验

目标：在核心 JAX 前向实现上补齐自动微分、Optax 状态、全局梯度裁剪、AdamW 更新和 `jax.jit` 训练步，并用一个确定性 tiny batch 证明参数确实更新且 loss 可下降。

## 运行

~~~powershell
python -m pip install -e ".[jax]"
python projects/jax-minigpt/train_tiny.py --steps 60 --learning-rate 0.02 --seed 11
python projects/jax-minigpt/cross_framework_parity.py
python projects/jax-minigpt/cross_framework_training_parity.py
python projects/jax-minigpt/checkpoint_resume_control.py
python -m pytest tests/test_gpt_jax.py tests/test_gpt_cross_framework_parity.py `
  tests/test_gpt_cross_framework_training_parity.py `
  tests/test_jax_training_resume.py -q
~~~

输出分开记录首次 trace/compile + step 时间与后续同步 step 均值。JAX 默认异步 dispatch，因此计时前调用 `loss.block_until_ready()`；不做同步的 wall time 只量到 Python enqueue，不能当作训练吞吐。

报告还包含 backend、实际 device、参数量、初末 loss、pre-clip gradient norm 和验证范围。当前实验只证明输出中列出的 JAX device 上 tiny float32 训练可运行，不证明 CUDA、TPU、多设备 sharding、混合精度或大模型性能。

## 代码路径

- `init_params`：显式 PRNG key 初始化 PyTree；
- `forward`：无 Flax 封装的 decoder-only Transformer；
- `adamw_optimizer`：`clip_by_global_norm` + AdamW 的 Optax transformation；
- `make_train_step`：闭包捕获静态 config/optimizer，内部执行 `value_and_grad`、optimizer update 与 apply updates，再整体 `jax.jit`；
- `tests/test_gpt_jax.py`：因果性、有限 loss、参数更新、梯度 norm 与 tiny-batch overfit。

## PyTorch↔JAX 同权重对账

`cross_framework_parity.py` 先钉死一个 11-vocab、2-layer、8-dim、2-head 的 CPU Float32 contract。它不用任一框架 RNG，而是用同一解析式生成 PyTorch 参数，再把 embedding、转置后的 Linear、LayerNorm scale/bias 显式映射到 JAX。两边统一 LayerNorm mean subtraction/epsilon=`1e-5`、tanh-GELU、causal mask、tied token embedding/LM head、masked mean cross entropy 与无 momentum/weight decay 的 plain SGD。

固定 fixture 的初始 logits/loss 最大差为 `7.636845111846924e-08/0`，20 个 unique parameters 的 gradient 全局最大差为 `2.384185791015625e-07`；一步 `lr=0.025` SGD 后，参数/logits/loss 最大差分别为 `7.450580596923828e-09`、`7.450580596923828e-08`、`2.384185791015625e-07`，均低于 authored `2e-6` 容差。两次当前 pinned CPU 环境报告完全一致。

原有两份 MiniGPT 不能直接互称等价：PyTorch 使用 affine LayerNorm，JAX 原生路径使用无 bias RMSNorm。把同一主干权重直接送入原生 RMSNorm 路径的 logits 最大差为 `0.37747739627957344`，作为架构未对齐的反事实。该 control 不比较 AdamW state、schedule、dropout/RNG、JIT/异步计时、CUDA/TPU、多设备 sharding、大模型训练、收敛或性能。

### 三步 stochastic AdamW trajectory

独立的 `cross_framework_training_parity.py` 继续使用同一 LayerNorm/tanh-GELU/tied-embedding contract，但不假设 PyTorch/JAX 原生 RNG 算法相同。NumPy PCG64 seed `20260814` 先物化三张完全相同的 **materialized dropout masks**，两边只在 embedding sum 上执行 rate=0.25 的 inverted dropout；三步学习率为 `0.02→0.01→0.005`，AdamW 固定 beta=`0.9/0.95`、epsilon=`1e-8`、weight decay=`0.03`，并在每步用 global norm `0.08` 裁剪。

三步 pre-clip norm 均为 `1.32–1.90`，所以 clipping 不是空操作。raw/clipped gradient 最大差为 `3.129243850708008e-07/1.862645149230957e-08`，first/second moments 最大差为 `2.561137080192566e-09/8.003553375601768e-11`，Adam/count/schedule count 均为 `[1,2,3]`；参数与 post-step logits 最大差为 `2.5480985641479492e-06/1.564621925354004e-07`，通过 `5e-6` 门槛。把 JAX mask 的最后一维循环移位后，最终参数差为 `0.06900620367377996`。两次独立进程报告均为 `sha256:68ffa8093a1f2b98…`。

这证明 shared-mask、CPU Float32、三步 authored trajectory 的 optimizer plumbing 对齐；不证明 native RNG equivalence、PRNG state advance、norm/bias decay mask、JIT、checkpoint、CUDA/TPU、sharding、长训练收敛或性能。旧 plain-SGD control 的 RMSNorm 反事实仍承担架构身份边界，不能被这条训练轨迹替代。

## Strict checkpoint 与跨进程恢复

`checkpoint_resume_control.py` 使用 `ALLMJAX1` 单文件格式：canonical JSON manifest 绑定模型/optimizer/dataset identity 与每个 array 的 name、shape、dtype、offset、size、SHA-256；连续 little-endian payload 保存全部参数叶子、Optax count/moments、dropout/data typed-key data 和 permutation。文件以 outer SHA-256 收尾，loader 在构造 JAX arrays 前拒绝 duplicate/non-canonical JSON、字段、顺序、shape/dtype、digest、截断和多余 bytes；writer 用 exclusive create 与 file `fsync`。

固定 7-example/batch-2、embedding dropout=0.2、clip+AdamW 的六步 fixture 在 step 3 由第一个 spawn process 写出 13,476-byte artifact（SHA-256 `e9252e5dddfa4aa5…`），第二个独立 process 加载并完成后三步。Uninterrupted/resumed sample IDs 均为 `[[0,4],[3,2],[5,1],[6,3],[2,1],[6,4]]`，loss/gradient trace 与最终 full-state fingerprint `sha256:720817cca4c067cf…` bit-exact。

两个因果负例分别只把 checkpoint 中的 dropout PRNG 重置到初始 seed，或把 cursor 从 6 重置为 0：最终参数最大差为 `0.037261832505464554`/`0.03700308472616598`。这证明在当前 authored CPU fixture 中，params+Optax state 并不足以恢复训练语义。它不使用 Orbax/Flax/TensorStore，不保存 Python/NumPy/worker/accelerator RNG，不证明 directory `fsync`、断电原子性、来源认证、加密、CUDA/TPU、sharding、目标模型、收敛或性能。

## 证据边界

tiny-batch overfit 是训练闭环的单元验收：它能发现 target shift、梯度断开、optimizer state 或冻结错误，但不能证明泛化。固定样例上的最终 loss 也不是与 PyTorch/其他模型的性能排名。

教学 optimizer 对所有参数使用同一 weight decay。真实 LLM 训练通常通过 PyTree mask 排除 norm scale、bias-like 参数，并加入 schedule、mixed precision、数据迭代与 checkpoint；这些策略必须分别测试。

## 四层证据不能合并

四个入口回答不同问题：

| 层 | 入口 | 已证明 | 未证明 |
|---|---|---|---|
| 原生 JAX 闭环 | `train_tiny.py` | PyTree/autodiff/Optax/JIT 接通 | 泛化、生成、性能 |
| 架构/梯度 parity | `cross_framework_parity.py` | 同函数前向/反向/plain SGD | AdamW、RNG/JIT |
| Optimizer trajectory | `cross_framework_training_parity.py` | shared-mask 三步 AdamW 对账 | native RNG、checkpoint |
| 跨进程 resume | `checkpoint_resume_control.py` | full state bit-exact | Orbax、sharding、durability |

前一层通过不能替后一层作证；四层都通过也不表示已完成 CUDA/TPU、多设备或目标 LLM 训练。

## 从零实现的 JAX decoder

核心实现位于 `src/about_llm/from_scratch/gpt_jax.py`，不依赖 Flax module：

```text
PRNG key
  → init_params
  → params PyTree
  → forward(params, input_ids, config)
  → logits
  → masked cross entropy
  → value_and_grad
  → Optax update
  → apply_updates
```

### 参数树

每个 block 包含：

- attention RMSNorm scale；
- fused QKV matrix；
- attention output matrix；
- MLP RMSNorm scale；
- MLP up/down matrices。

顶层还包含：

- token embedding；
- position embedding；
- final RMSNorm scale。

LM head 与 token embedding 绑定。当前原生实现没有 Linear bias 或 norm bias，这正是它不能与 PyTorch MiniGPT 默认 affine LayerNorm 路径直接互称等价的原因之一。

### 因果 attention

`input_ids [B,T]` 经 embedding 后进入 pre-norm attention：

1. fused QKV projection；
2. reshape 为 `[B,H,T,D_h]`；
3. scaled dot-product score；
4. causal mask；
5. softmax；
6. value aggregation；
7. output projection 与 residual。

测试会改写后续 token，验证更早位置 logits 不变。当前实现是教学 dense attention，不是 FlashAttention、paged attention 或目标 accelerator kernel。

### Masked token mean

对非 `ignore_index=-100` token 集合 $M$：

$$
\mathcal L=-\frac{1}{|M|}\sum_{(b,t)\in M}
\log p_\theta(y_{b,t}\mid x_{b,\le t}).
$$

实现先把 ignored label 暂替换为安全 gather index，再应用 mask。分母下限为 1，所以**全 ignored batch 返回 0**，不会产生 NaN。

有限 0 loss 只是数值定义；训练数据门禁仍应拒绝没有监督 token 的 batch，否则会制造看似正常的空更新。

## 原生 JAX tiny overfit 的固定证据

运行：

~~~powershell
python projects/jax-minigpt/train_tiny.py `
  --steps 60 `
  --learning-rate 0.02 `
  --seed 11
~~~

固定两条 `[0,1,2,3]→[1,2,3,4]` 样本与 tiny config 共有 **632 个参数**。

当前已记录 CPU float32 结果：

- initial loss：`2.108591318130493`；
- final loss：`0.0030041998252272606`；
- final/initial ratio 约 `0.00142474`；
- final pre-clip norm：`0.011140906251966953`。

该脚本**不生成文本**。它只输出参数量、loss、梯度 norm、版本、backend/device 与计时。

Tiny overfit 能发现：

- target shift；
- 梯度断开；
- optimizer 没更新；
- 参数意外冻结；
- loss reduction 错误。

它不能证明：

- held-out 泛化；
- 语言生成质量；
- 长训练收敛；
- LLM scaling；
- 目标硬件性能。

## JIT 与异步计时

JAX dispatch 可能异步。脚本在每步对 loss 调用 `loss.block_until_ready()`，再记录时间。

报告分开：

- first trace/compile + step；
- warm synchronized step mean。

若不等待 device result，wall time 主要是 Python **enqueue latency**，不能当训练吞吐。

即使已经同步，tiny CPU shape 的计时也不是严谨 benchmark。必须固定：

- backend/device；
- warmup；
- shape/dtype；
- compile cache；
- 同步边界；
- 输入生成与 host-device transfer；
- 采样窗口与原始样本。

CPU tiny 结果不能预测设备峰值、GPU/TPU throughput 或多设备 scaling。

## Plain-SGD parity 的身份契约

`cross_framework_parity.py` 不比较两个随机初始化的模型，而是固定：

- 同一解析参数；
- affine LayerNorm 与 epsilon=`1e-5`；
- PyTorch/JAX Linear layout transpose；
- tanh approximate GELU；
- 相同 causal mask；
- tied embedding/head；
- masked mean loss；
- plain SGD lr=`0.025`；
- CPU float32；
- 不使用 framework RNG。

它逐项比较：

1. initial logits；
2. initial loss；
3. 20 个 unique parameters 的 gradients；
4. post-SGD parameters；
5. post-step logits；
6. post-step loss。

最大梯度差 `2.384185791015625e-07`，低于 `2e-6` authored tolerance。

### RMSNorm 反事实

把同一主干送入原生 JAX RMSNorm 路径，logits 最大差为 `0.37747739627957344`。

这说明 architecture identity 必须包含 normalization、epsilon、bias、activation、mask、weight tying 和 loss reduction；“都是 GPT”不构成 parity。

## 三步 AdamW trajectory parity 的解释

这一 control 使用外部 materialized masks，而不是声称 PyTorch/JAX native RNG 相同。

固定变量：

- NumPy PCG64 seed `20260814`；
- embedding inverted dropout rate `0.25`；
- global-norm clip `0.08`；
- Adam beta `0.9/0.95`；
- epsilon `1e-8`；
- weight decay `0.03`；
- learning rate `0.02→0.01→0.005`。

三步 AdamW trajectory parity 对账：

- raw gradients；
- clipped gradients；
- first moments；
- second moments；
- Adam count；
- schedule count；
- parameters；
- post-step logits/loss。

三个 pre-clip norm 均显著大于 `0.08`，所以 clipping 不是空操作。

错误 mask 反例最终参数差 `0.06900620367377996`，说明随机输入 identity 是训练轨迹的一部分。

Shared mask 只隔离了随机变量，不证明 **native RNG equivalence**、PRNG state advance 或框架默认 dropout 完全一致。

## Strict checkpoint：可解析不等于可恢复

`ALLMJAX1` artifact 保存：

| 状态 | 当前 control |
|---|---|
| params | 全部 PyTree leaves/treedef identity |
| optimizer | Optax count、first/second moments |
| dropout | typed key data |
| data shuffle | typed key data、permutation、cursor |
| progress | global step |
| identity | model/optimizer/dataset contract |

可反序列化不等于 exact resume。只有继续训练的 sample IDs、loss/gradient trace 与最终完整状态都相同，才能支持 bit-exact 结论。

### 文件格式

```text
ALLMJAX1 magic
canonical JSON manifest
little-endian array payloads
outer SHA-256
```

Manifest 绑定：

- array name；
- shape；
- dtype；
- offset；
- byte size；
- per-array SHA-256；
- tree/训练身份。

Loader 在构造 JAX arrays 前拒绝：

- duplicate/non-canonical JSON；
- unknown/missing field；
- array order 漂移；
- shape/dtype 漂移；
- digest mismatch；
- truncation；
- trailing bytes。

Writer 使用 exclusive create 和 file `fsync`。File `fsync` 不等于目录项已 durable，也不证明断电原子性。

无密钥 SHA-256 可以检测意外漂移，不能认证发布者；能同时修改文件和 hash 的攻击者仍可制造自洽 artifact。

### 跨进程结果

固定六步训练在 step 3 交接：

- first spawn process 写 checkpoint；
- second spawn process 加载并继续；
- artifact 大小 `13,476 bytes`；
- SHA-256 前缀 `e9252e5dddfa4aa5`；
- uninterrupted/resumed sample ID trace exact；
- loss/gradient trace exact；
- final full-state fingerprint bit-exact。

### 因果负例

**wrong PRNG**：只把 dropout key 恢复为初始 seed，最终参数差 `0.037261832505464554`。

**wrong cursor**：只把 data cursor 从 6 恢复到 0，最终参数差 `0.03700308472616598`。

两个反例证明只保存 params+optimizer state 不足以恢复未来训练语义。

## 运行与验证矩阵

~~~powershell
python projects/jax-minigpt/train_tiny.py --steps 60 --learning-rate 0.02 --seed 11
python projects/jax-minigpt/cross_framework_parity.py
python projects/jax-minigpt/cross_framework_training_parity.py
python projects/jax-minigpt/checkpoint_resume_control.py
~~~

专项测试：

~~~powershell
python -m pytest `
  tests/test_gpt_jax.py `
  tests/test_gpt_cross_framework_parity.py `
  tests/test_gpt_cross_framework_training_parity.py `
  tests/test_jax_training_resume.py -q
~~~

测试覆盖包括：

- config/shape/context validation；
- causal future-token invariant；
- finite loss/gradients；
- all-ignored loss；
- parameter update 与 overfit；
- forward/backward/SGD parity；
- architecture negative control；
- AdamW gradient/moment/count/schedule parity；
- wrong-mask counterfactual；
- canonical checkpoint serialization；
- cross-process exact resume；
- wrong PRNG/cursor；
- duplicate/noncanonical/truncated/tampered artifact；
- exclusive create/no-overwrite。

## 故障定位

| 现象 | 优先检查 | 不应直接归因 |
|---|---|---|
| 初始 logits 不同 | param mapping、norm、GELU、mask | “JAX 数值差” |
| logits 同、gradient 不同 | loss mask/reduction、tied weight | optimizer |
| gradient 同、一步参数不同 | LR、layout、update sign | convergence |
| 前几步同、后续漂移 | mask/RNG、schedule、moments | framework quality |
| checkpoint 可读但续训漂移 | PRNG、cursor、optimizer count | serialization 成功即可 |
| 时间异常快 | 缺 `block_until_ready()` | JAX 性能更好 |
| 重新编译 | shape/static config/cache | device throughput |

先检查 identity 与最早分叉点；不要只比较最终 loss。

## 扩展到生产 JAX 训练

### PyTree decay mask

真实 LLM 常排除 norm scale、bias-like 参数的 weight decay。需要按参数 path 建 mask，并用显式测试证明分类，不应靠字符串 contains 的脆弱约定。

### Mixed precision

必须分别定义：

- parameter dtype；
- compute dtype；
- optimizer state dtype；
- loss scaling；
- overflow consensus；
- checkpoint cast/reload policy。

CPU float32 control 不借给 BF16/FP16/FP8 作证。

### Orbax/TensorStore

接入 Orbax/TensorStore 后要验证：

- async save completion；
- manifest/commit point；
- partial shard；
- topology change；
- reshard；
- retention/GC；
- storage consistency；
- rollback 与 no-overwrite。

当前 strict file 可作为逻辑 oracle，但不能冒充生产分片 checkpoint。

### 多设备 sharding

需要保存 mesh、PartitionSpec、global/local shape、replication 与 reshard policy，并分别验证 data/parameter/optimizer sharding。

目标证据应来自真实 `NamedSharding`/mesh 与设备，而不是 CPU 单设备的 shape 推导。

## 项目验收清单

- [ ] 参数树与架构 identity 明确；
- [ ] masked mean/all-ignored 语义有测试；
- [ ] JIT 计时同步且 compile/warm 分开；
- [ ] parity 使用同解析权重和同函数约定；
- [ ] unique/shared parameter 只计一次；
- [ ] gradient、optimizer state 和 post-step forward 都对账；
- [ ] RNG equivalence 与 shared mask 结论分开；
- [ ] checkpoint 保存所有影响未来轨迹的状态；
- [ ] resume 由跨进程 continuation 证明；
- [ ] wrong PRNG/cursor 负例存在；
- [ ] artifact 完整性与来源认证分开；
- [ ] CPU、GPU、TPU、多设备证据不混写；
- [ ] 性能数字包含同步、shape、backend 和原始样本；
- [ ] 简历结论可链接到 report/test/artifact。

## 求职与简历边界

可如实写：

> 用 core JAX/Optax 实现纯函数 decoder、PyTree state、JIT update 与显式 PRNG；以同解析权重对账 PyTorch/JAX 的 20 个 unique parameter gradients 和 plain-SGD step，再以 shared-mask 三步 AdamW 对账 clipping/moments/schedule，并由两个 spawn processes 验证 params/Optax/typed PRNG/permutation/cursor 的 bit-exact resume。

必须紧邻说明：

- CPU tiny fixture；
- shared mask 不证明 native RNG equivalence；
- `ALLMJAX1` 不是 Orbax/TensorStore；
- 没有 GPU/TPU、多设备 sharding；
- 没有目标模型、泛化、收敛或性能对比。

不能写：

- “JAX 性能优于 PyTorch”；
- “完成 TPU/GPU 训练”；
- “实现生产级分片 checkpoint”；
- “PyTorch/JAX 天然数值等价”；
- “训练模型质量得到提升”。

## 下一步

1. 将 strict artifact 对接 Orbax/TensorStore，并验证拓扑变化后的 reshard；
2. 将 shared-mask trajectory 扩展到 attention/residual/MLP 全部 dropout site，并对齐生产式 norm/bias weight-decay mask；
3. 在实际多设备环境验证 `NamedSharding`/mesh、数据分片和参数分片；
4. 分别在 PyTorch/JAX 内验证 native RNG state advance、checkpoint/resume 与多 seed 轨迹，不把 shared mask 当 RNG 等价；
5. 记录 compile cache、shape recompilation、HLO/Profiler 与通信时间；
6. 分开验证 CPU、单 GPU 和多设备，不从一个 backend 外推另一个。

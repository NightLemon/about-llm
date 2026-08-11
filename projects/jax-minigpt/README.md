# JAX/Optax MiniGPT 训练实验

目标：在核心 JAX 前向实现上补齐自动微分、Optax 状态、全局梯度裁剪、AdamW 更新和 `jax.jit` 训练步，并用一个确定性 tiny batch 证明参数确实更新且 loss 可下降。

## 运行

~~~powershell
python -m pip install -e ".[jax]"
python projects/jax-minigpt/train_tiny.py --steps 60 --learning-rate 0.02 --seed 11
~~~

输出分开记录首次 trace/compile + step 时间与后续同步 step 均值。JAX 默认异步 dispatch，因此计时前调用 `loss.block_until_ready()`；不做同步的 wall time 只量到 Python enqueue，不能当作训练吞吐。

报告还包含 backend、实际 device、参数量、初末 loss、pre-clip gradient norm 和验证范围。当前实验只证明输出中列出的 JAX device 上 tiny float32 训练可运行，不证明 CUDA、TPU、多设备 sharding、混合精度或大模型性能。

## 代码路径

- `init_params`：显式 PRNG key 初始化 PyTree；
- `forward`：无 Flax 封装的 decoder-only Transformer；
- `adamw_optimizer`：`clip_by_global_norm` + AdamW 的 Optax transformation；
- `make_train_step`：闭包捕获静态 config/optimizer，内部执行 `value_and_grad`、optimizer update 与 apply updates，再整体 `jax.jit`；
- `tests/test_gpt_jax.py`：因果性、有限 loss、参数更新、梯度 norm 与 tiny-batch overfit。

## 证据边界

tiny-batch overfit 是训练闭环的单元验收：它能发现 target shift、梯度断开、optimizer state 或冻结错误，但不能证明泛化。固定样例上的最终 loss 也不是与 PyTorch/其他模型的性能排名。

教学 optimizer 对所有参数使用同一 weight decay。真实 LLM 训练通常通过 PyTree mask 排除 norm scale、bias-like 参数，并加入 schedule、mixed precision、数据迭代与 checkpoint；这些策略必须分别测试。

## 下一步

1. 将数据 iterator/RNG state 与 optimizer state 一起保存和恢复；
2. 做 PyTorch/JAX 同权重小模型的 logits、loss 和单步更新对照；
3. 在实际多设备环境验证 `NamedSharding`/mesh、数据分片和参数分片；
4. 记录 compile cache、shape recompilation、HLO/Profiler 与通信时间；
5. 分开验证 CPU、单 GPU 和多设备，不从一个 backend 外推另一个。

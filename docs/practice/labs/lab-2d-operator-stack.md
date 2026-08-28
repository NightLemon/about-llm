# 实验 2D：跟一次 RMSNorm 走过 PyTorch 算子栈

**定位**：进阶选做，预计 60–90 分钟；第一到第三步只需 CPU，第四步需要一块 NVIDIA GPU。

**实验导航**：[返回总览](../labs.md) · [LLM 算子与计算栈](../../systems/operator-stack.md)
{ .doc-nav }

这个实验回答一个很具体的问题：模型代码中的一次 `RMSNorm`，怎样一路变成张量布局、FX 节点、ATen 算子和
运行时事件？我们会逐层保存证据，让你能分清当前看到的是框架算子，还是已经下钻到设备 kernel。

## 完成标准

完成实验后，你应该能够：

1. 解释 shape、stride、view 和 contiguous copy 的关系；
2. 手写 RMSNorm，并与 PyTorch 框架实现对账；
3. 区分 `nn.Module`、FX graph、ATen graph、profiler event 和 GPU kernel；
4. 为一个 backend 写出至少包含 dtype、shape、layout、forward/backward 和 fallback 的支持卡；
5. 不把一次成功运行或一次 profiler 截图写成“完整算子支持”。

原理入口见[LLM 算子与计算栈](../../systems/operator-stack.md)。

## 准备环境

安装带 PyTorch 的本地环境：

```powershell
python -m pip install -e ".[torch]"
```

确认版本：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

这个实验使用 PyTorch 2.4 或更高版本，因为框架的 RMSNorm 接口从 2.4 开始提供。实验默认只用 CPU；GPU 阶段
还要求当前 PyTorch 能看到 CUDA。整个过程不下载模型，也不依赖 nano-vLLM。

## 第一步：运行前先画三层预期

先不要运行程序，写下这三个预测：

1. `[2,3,4]` 连续张量的 stride 是什么？交换前两个维度后，shape 和 stride 怎样变化？
2. RMSNorm 参考实现至少包含哪些 elementwise 和 reduction 运算？
3. 你认为一个 RMSNorm Module 会对应一个还是多个 ATen operator？这能否推出 GPU kernel 数？

预测的价值不在于一次答对，而在于运行后能指出哪一层理解发生了变化。

## 第二步：让同一份输入经过数学和两层图

运行：

```powershell
python projects/transformers-basics/trace_rmsnorm_operator_stack.py
```

先看布局部分。`transpose` 得到的 view 与 base 共用 storage，但通过不同 stride 解释同一块内存；
`contiguous()` 生成适合连续访问的新副本。由此可以得到两个结论：

- reshape/view/transpose 不一定搬运数据；
- 模型公式相同，不代表 backend 看到的 layout 和复制成本相同。

再看数学部分。脚本用下面的分解与 `torch.nn.functional.rms_norm` 对账：

```text
x * rsqrt(mean(x², dim=-1) + eps) * weight
```

最大绝对误差只验证当前 dtype、输入和 epsilon 下的 forward parity。脚本另外执行 backward 并检查梯度有限，
但没有做完整的 gradient check 或高阶梯度验证。

最后对比两层图：

```text
FX:           operator.mul → mean → operator.add → torch.rsqrt → operator.mul → operator.mul
torch.export: aten.mul.Tensor → aten.mean.dim → aten.add.Tensor → aten.rsqrt.default
              → aten.mul.Tensor → aten.mul.Tensor
```

这里可以回答第一步的第三个预测：一个 RMSNorm Module 展开成 **6 个**节点，两张图在本例中一一对应。
但节点数不等于 kernel 数——报告里的 `scope.kernel_count_inferred_from_fx_or_export` 就是 `false`。

FX 更接近捕获到的 Python 运算；export graph 把它表达成带 overload 的 ATen operator。两张图都还不是
目标设备的 kernel 列表。

## 第三步：观察当前 PyTorch 的算子事件

加入 profiler：

```powershell
python projects/transformers-basics/trace_rmsnorm_operator_stack.py --profile
```

程序分别 profile 手写分解和框架 RMSNorm。记录：

- 是否出现 `aten::rms_norm`；
- 是否仍能看到 `aten::mean`、`aten::rsqrt`、`aten::mul`；
- 非连续输入是否导致 `contiguous`、`clone` 或 `copy_`；
- 两条路径的事件为何不能直接当作跨版本稳定接口。

Profiler 展示的是当前 PyTorch build 的运行事件。编译器可能融合图节点，库实现可能在事件内部启动 kernel；
GPU 时间线还需要按目标设备进一步观察。

## 第四步：在 3070 Laptop 上确认 CUDA 路径

没有 NVIDIA GPU 也不必跳过这一步的**结论**：直接读下面的命令与支持卡模板，把「这些证据我拿不到」
写进支持卡的空缺项即可——承认覆盖面缺口，本身就是这个实验要练的能力。有 GPU 再往下执行。

先记录环境：

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())"
```

再运行：

```powershell
python projects/transformers-basics/trace_rmsnorm_operator_stack.py `
  --device cuda --dtype float32 --profile --json
```

保存 JSON，然后只改变 dtype：

```powershell
python projects/transformers-basics/trace_rmsnorm_operator_stack.py `
  --device cuda --dtype float16 --profile --json
```

接着只改变 shape：

```powershell
python projects/transformers-basics/trace_rmsnorm_operator_stack.py `
  --device cuda --dtype float16 `
  --batch-size 4 --sequence-length 128 --hidden-size 1024 `
  --profile --json
```

这些命令用于检查路径和契约，不用于比较速度：每种配置只运行一次，没有独立 warm-up、重复采样或计时同步协议。
某个 dtype 或 shape 失败时保留错误；它正是当前版本支持面的证据。

## 第五步：填写一张支持卡

不要只写“3070 支持 RMSNorm”。填写下面这张表：

| 项目 | 本次观察 | 证据 | 尚未验证 |
|---|---|---|---|
| 版本 | `torch.__version__` 与 `torch.version.cuda` | JSON 的 runtime 字段与环境记录 | 其他 PyTorch 版本上的图与事件 |
| 语义 | reference 与 framework 的误差 | JSON 中的最大绝对误差 | 更大数值范围和极端 epsilon |
| Layout | 非连续输入能否执行 | stride、copy/profile 事件 | 更多 stride/alignment |
| Dtype | FP32/FP16/BF16 各自结果 | 每个 run 的成功或错误 | 混合累加精度 |
| Shape | 已运行的 B/T/D | 完整命令和 JSON | 动态 shape 与边界维度 |
| Backward | 梯度是否有限 | 当前输入的 backward | 数值 gradient check |
| Device | CPU 或 CUDA | 环境 identity 和 profiler | 其他 GPU/NPU backend |
| Compile | 本实验没有运行 | 明确记录为未验证 | `torch.compile`、graph break、recompile |
| Performance | 本实验没有 benchmark | 明确记录为未验证 | warm-up、同步、分位数和 Nsight |

“未验证”不是缺点，而是防止一条局部证据被外推成平台承诺。

## 第六步：把算子放回完整 Transformer

RMSNorm 通常不是单独出现。沿一次 decoder layer 标出：

```text
RMSNorm
→ Q/K/V projection
→ RoPE
→ Attention 与 KV Cache
→ output projection + residual
→ RMSNorm
→ SwiGLU MLP
→ residual
```

为每段写出主要模式：reduction、elementwise、GEMM、indexing/layout、复合 Attention 或 collective。
然后选择一个你真正要分析的瓶颈：

- Attention/online softmax：进入[实验 2](../labs.md#lab-2)；
- Paged KV 与布局：进入[实验 7A](lab-7a-paged-kv.md)；
- Qwen3 + nano-vLLM 的调度与 CUDA Graph：进入[实验 7B](lab-7b-nano-vllm-qwen3.md)；
- Roofline 与服务瓶颈：进入[推理优化](../../systems/inference-optimization.md)。

## 当前实验覆盖到哪里

本实验完成“数学 → 张量布局 → FX 图 → ATen 图 → 当前设备运行事件”这段学习链。到这里，你已经可以根据实际
记录区分模型表达、框架算子和运行路径。

下一阶段进入 kernel 工程时，可以加入 Triton/CUDA 实现，再用 PTX/SASS 和 Nsight 分析生成的代码及运行表现。
一份完整的优化报告至少还需要：

1. 独立正确性基线与 backward 范围；
2. shape/dtype/layout 测试矩阵；
3. warm-up、同步和重复计时；
4. Nsight Systems 的时间线；
5. 对单个热点使用 Nsight Compute；
6. 优化前后端到端回归，而不只比较 microbenchmark。

# 实验 7A：亲手追踪 Paged KV 与 copy-on-write

这个实验不要求 GPU，也不要求下载模型。你会跟踪两条序列如何共享前缀、为什么未满尾块不能直接写，
以及一次 copy-on-write（COW）怎样改变 block table 和真实 K/V tensor。

**相关教材**：[一次请求如何穿过推理引擎](../../systems/inference-request-lifecycle.md) ·
[推理优化](../../systems/inference-optimization.md) ·
[Inference Serving 项目](../projects/inference-serving.md)
{ .doc-nav }

## 完成标准

完成后，你应该能在不运行脚本的情况下预测：

1. Prefix 长度为 5、block size 为 3 时需要几个逻辑块。
2. Fork 后两条序列的 block table 和 refcount 如何变化。
3. 父序列 append 一个 token 时，哪个物理块被复制到哪里。
4. 为什么 logical tokens、physical token values 和 allocated slots 是三个不同数字。
5. 这个 CPU 实验为什么不能证明 CUDA PagedAttention 的显存或性能收益。

预计时间为 45–90 分钟。

## 准备环境

从仓库根目录安装项目和 PyTorch 依赖：

~~~powershell
python -m pip install -e ".[torch]"
~~~

先不要运行脚本。打开 `projects/inference-serving/paged_kv_tensor_toy.py`，只读配置部分：

```python
store = PagedKVTensorStore(
    num_layers=1,
    total_blocks=4,
    block_size_tokens=3,
    num_kv_heads=2,
    head_dim=2,
    dtype=torch.float64,
)
```

预分配 K/V arena 的逻辑形状是：

```text
[layers, physical_blocks, kv_heads, tokens_per_block, head_dim]
= [1, 4, 2, 3, 2]
```

K 和 V 各有一份。Float64 每元素 8 bytes，所以固定 tensor payload 是：

\[
2\times1\times4\times2\times3\times2\times8=768\text{ bytes}.
\]

这个数字不包含 Python、PyTorch runtime、allocator metadata 或进程 RSS。

## 第一步：先画状态，不要先看 JSON

脚本依次执行四个动作：

```text
1. request-a append 5-token prefix
2. 从 request-a fork request-b
3. request-a append 1 token
4. release request-a
```

在纸上画四个物理块：`0 1 2 3`。每块有三个 token slot。

### 预测 append prefix 后的状态

Allocator 总是先使用最小可用 block id。五个 token 应得到：

```text
request-a block table: [0, 1]
block 0: 3/3
block 1: 2/3
```

此时 block 1 是 partial tail，仍有一个空位。

### 预测 fork 后的状态

Fork 不复制 K/V。Request B 先共享 A 的两个物理块：

```text
request-a: [0, 1]
request-b: [0, 1]
block 0 refcount: 2
block 1 refcount: 2
```

逻辑上有四个 block reference，物理上仍只有两个 block。

### 预测 A 再 append 一个 token

Block 1 还有空间，但它被 A/B 共享。如果 A 原地写入，B 的前缀会被污染。

因此 A 需要把 block 1 的两个已有 token 复制到新 block 2，再写入第六个 token：

```text
request-a: [0, 2]
request-b: [0, 1]
copied partial block: 1 -> 2
```

写完后：

- block 0 有 3 个物理 token，仍被两条序列共享；
- block 1 有 2 个物理 token，只属于 B；
- block 2 有 3 个物理 token，只属于 A；
- block 3 仍空闲。

先填写预测表，再运行命令：

| 指标 | 你的预测 |
|---|---:|
| `allocated_blocks` |  |
| `free_blocks` |  |
| `logical_block_references` |  |
| `sharing_saved_blocks` |  |
| `logical_tokens` |  |
| `physical_token_values` |  |
| `allocated_token_slots` |  |
| `internal_fragmentation_slots` |  |

## 第二步：运行并解释输出

~~~powershell
python projects/inference-serving/paged_kv_tensor_toy.py
~~~

关键结果应为：

```text
storage_shape                         [1, 4, 2, 3, 2]
resident_bytes                        768
physical_block_ids after A append     [0, 2]
copied_partial_block                  [1, 2]
child_prefix_unchanged                true
parent_append_materialized            true
attention_matches_dense_reference     true
```

COW 后、释放 A 之前，账本应为：

| 指标 | 值 | 为什么 |
|---|---:|---|
| allocated blocks | 3 | 物理块 0、1、2 |
| free blocks | 1 | 物理块 3 未用 |
| logical block references | 4 | A/B 各引用两个逻辑块 |
| sharing saved blocks | 1 | 四个引用只需三个物理块 |
| logical tokens | 11 | A 长度 6，加 B 长度 5 |
| physical token values | 8 | `3 + 2 + 3`，共享 block 0 只算一次 |
| allocated token slots | 9 | 三个物理块，每块三个 slot |
| internal fragmentation | 1 | block 1 还空一个 slot |

如果把 `logical_tokens=11` 当作物理存储的 token 数，就会重复计算共享前缀。
如果用 `allocated_slots=9` 当作有效 K/V 数，又会忽略尾块碎片。

### 释放 A 后发生什么

释放 A 会回收它独占的 block 2，并把共享 block 0 的 refcount 从 2 降到 1。
Request B 仍需要 block 0 和 1，所以它们不能被回收。

最终应看到：

```text
allocated_blocks = 2
free_blocks = 2
logical_tokens = 5
physical_token_values = 5
internal_fragmentation_slots = 1
```

## 第三步：理解 attention parity 在证明什么

这个固定样例使用 2 个 KV heads 和 4 个 query heads。每两个 query heads 共享一个 K/V head，形成一个最小 GQA 场景。

`PagedKVTensorStore.attention()` 先按 block table 收集 A 的完整 K/V，再计算 causal attention。
测试中的独立 dense reference 则显式扩展 K/V heads、构造 causal mask 并计算 softmax。

两者在 float64 容差内一致，说明：

- block table materialize 保持逻辑 token 顺序；
- COW 后 A 的新值和 B 的旧值没有串线；
- 当前 GQA head mapping 与 causal mask 能和独立 dense 公式对账。

它没有说明：

- 执行了 CUDA page-table 或 PagedAttention kernel；
- 避免了 dense score/probability 的物化；
- GPU VRAM、延迟或吞吐得到改善；
- 该状态机等价于某个 vLLM release。

## 第四步：故意制造容量不足

把脚本中的 `total_blocks=4` 临时改为 `total_blocks=2`，先预测异常发生在哪一步。

五 token prefix 已经占满两个可分配 block。Fork 不需要新块，但 A append 时必须为共享 partial tail 执行 COW，
因此需要第三个物理块。正确行为是抛出容量错误，而不是先改坏 block 1。

仓库中的原子失败测试会比较异常前后的 allocator 和 tensor：

~~~powershell
python -m pytest tests/test_paged_kv_torch.py::test_capacity_failure_preserves_allocator_and_tensor_state -q
~~~

运行后把脚本恢复为 `total_blocks=4`。这个负例比“正常输出全是 true”更重要：
它证明失败发生在 mutation 之前，而不是只证明 happy path 能运行。

## 第五步：沿实现读一次 COW

按下面顺序阅读，不必逐行通读整个文件：

1. `PagedKVAllocator.append()` 先计算 `copy_tail`、`tail_free_slots` 和 `required_blocks`。
2. 它在修改 sequence 之前一次性预留所需 block。
3. Shared partial tail 被复制后，只有当前 sequence 的最后一个 block id 被替换。
4. `PagedKVTensorStore.append()` 再把相同状态变化应用到真实 K/V arena。
5. Tensor backend 更新失败后 store 会进入 poisoned state，拒绝继续返回可能失配的数据。

最后运行本实验的相关测试：

~~~powershell
python -m pytest tests/test_paged_kv_torch.py -q
~~~

## 实验记录模板

不要只保存终端截图。提交以下内容即可：

```text
配置：block size、总 block、dtype、K/V 与 query head 数
运行前预测：block table、refcount、logical/physical/slot 三份账本
观察结果：哪些预测正确，哪些错误
负例：容量不足发生在哪一步，状态是否保持
解释：为什么 partial tail 需要 COW
边界：这个实验没有证明哪些 GPU/runtime 结论
```

下一步回到[一次请求如何穿过推理引擎](../../systems/inference-request-lifecycle.md)，
把本实验的 block table 放回 prefill、decode、取消和调度时间线中。

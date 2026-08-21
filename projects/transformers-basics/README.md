# Transformers：离线机制与真实 checkpoint

第一次学习只运行 BPE、online softmax、tiny model 和 generation protocol 四条 CPU 命令。每条命令前先写预测，
运行后保留一个失败例。MoE、分布式通信、activation patching 和真实 Qwen checkpoint 都是独立选修；
它们出现在同一个目录中，不表示必须一次做完。完整的引导顺序见
[Transformers Basics 项目页](../../docs/practice/projects/transformers-basics.md)。

~~~powershell
python projects/transformers-basics/train_byte_bpe.py --vocab-size 280
python projects/transformers-basics/online_softmax_demo.py
python projects/transformers-basics/smoke_tiny.py
python projects/transformers-basics/generation_runtime_control.py
~~~

四条命令依次输出 BPE merge、dense/online attention 对账、tiny model 的 loss/gradient，以及 EOS/长度终止轨迹。
如果某一步失败，先查看对应章节的输入和中间状态，不要直接跳到真实 checkpoint。

## 从零训练 byte-level BPE

`train_byte_bpe.py` 不下载语料或模型，使用 `src/about_llm/from_scratch/tokenizer.py` 中的确定性 reference：基础词表为 256 个 raw byte；pair 频次只在每篇 document 内统计；同频时按 id pair 字典序打破平局；编码按已学习的 merge rank 执行。

~~~powershell
python projects/transformers-basics/train_byte_bpe.py --vocab-size 280
python projects/transformers-basics/train_byte_bpe.py `
  --text "banana bandana" --text "banana" --sample "bandana"
~~~

输出包含实际词表大小、每条 merge 的 byte expansion、样本 token 数与 UTF-8 round-trip。它用于理解 BPE 机制，
不包含 normalization、pre-tokenizer、special token、offset map 或 checkpoint chat template，不能替代真实模型
tokenizer，也不能由这份小型自编语料推断多语言压缩率。

## 用 NumPy 参考实现核对现代 attention

NumPy reference 不依赖 PyTorch kernel，提供 stable softmax、past-aware causal mask、RMSNorm、interleaved RoPE、显式 K/V repeat 的 GQA，以及不物化完整 score/probability 的 blockwise online-softmax recurrence。测试验证：fully masked row 会失败；RoPE 保持向量范数及共同 position shift 下的 Q/K dot product；GQA 等于对应的显式 MHA 展开；逐 token cache attention 等于完整 causal attention；online path 在多种 block size、causal prefill、decode、稀疏 mask 与大 logits 下对齐 dense reference。

~~~powershell
python projects/transformers-basics/online_softmax_demo.py
python -m pytest tests/test_attention_numpy.py -q
python -m pytest tests/test_gpt_torch.py tests/test_gpt_jax.py -q
~~~

固定 demo 的 query/key 长度为 5/7、block size 为 3：online path 分三块处理，最大 logical score tile 为 15 个元素，而 dense score 为 35 个元素；当前运行和 dense reference 的最大绝对误差应在 `1e-12` 内。Demo 为了比较会另外执行 dense reference，所以这不是整个进程的峰值内存实验。显式 K/V repeat 和 float64 累积用于解释数学，不是生产 kernel；这些小数组测试不证明 FlashAttention/CUDA/vLLM backend 已执行，也不证明目标 dtype、cache allocator、HBM traffic、GPU 性能或三套完整模型逐层等价。

## 用可手算输入理解 MoE top-k、capacity 与 sparse combine

~~~powershell
python projects/transformers-basics/moe_routing.py
python -m pytest tests/test_moe_routing.py -q
~~~

NumPy reference 对 `[tokens, experts]` logits 做稳定 softmax 与 deterministic top-k，padding token 不计 capacity；per-expert capacity 为 `ceil(factor * active_tokens * top_k / experts)`。同分 expert id 小者优先，expert 内按 probability 降序、token/rank 升序保留。输出同时保存 pre/post capacity counts、dropped assignment、整 token drop、gate weights、router entropy、z-loss 和本仓库明确命名的 generalized balance diagnostic。

固定 4-token/3-expert/top-2 fixture 的 capacity=2，count 从 `(3,4,1)` 变为 `(2,2,1)`，8 个 assignment 丢 3 个但没有整 token 全丢。脚本还真实执行 kept assignment 的 bias-free linear expert 与 weighted combine。它没有训练 router/MLP，不做 backward、all-to-all 或 GPU kernel，也不复现 DeepSeek/Qwen checkpoint；auxiliary loss、capacity group、drop/reroute 和归一化语义必须按目标实现重建。

## 让 MoE router 与 MLP 真正产生梯度

~~~powershell
python projects/transformers-basics/moe_training_control.py
python -m pytest tests/test_moe_training.py -q
~~~

第二条 CPU Float64 control 使用 5 tokens、3 个 bias-free `Linear(3,4)→Tanh→Linear(4,2)` experts 与稳定 top-2 router。Sparse path 只对被选 token 执行 expert，dense oracle 则计算所有 token×expert 后用同一 gate mask 合并；`task + 0.05 × balance + 0.001 × z` 的输出最大差为 0，所有参数梯度最大差约 `6.94e-18`。一次真实 SGD step 同时改变 router 与三个 experts，authored MSE 从约 `0.0886473` 降至 `0.0875580`；这只是单步 plumbing，不是收敛或质量证据。

同一路径还能显式启用 `capacity_factor=0.5`。每 expert capacity 为 2，pre/post counts 从 `[4,3,3]` 变为 `[2,2,2]`，10 个 top-2 assignments 丢 4 个但没有整 token 全丢；score-priority 按 probability 降序、token/rank 升序保留。Capacity 开启后 sparse/dense 的输出与全参数梯度最大差均为 0。Post-drop 重归一化让五个 token 的 kept weight sum 都回到 1；保留丢失 mixture mass 时 sums 为 `[0.805338,0.668188,1,0.638763,0.549834]`，两种策略的输出最大差约 `0.125542`。另一个同分 fixture 的 capacity 为 1，后两个 token 的两个 assignments 全丢，routed expert 输出精确为零。

第三个 control 把 padding 与 routing group 也放进该训练路径。Mask `[T,T,T,T,F]` 排除最后一个 token；active group labels 为 10/20，各含 2 tokens，所以各组 capacity 都是 1。两组 pre/post counts 分别为 `[2,1,1]→[1,1,1]` 与 `[1,1,2]→[1,1,1]`，全局为 `[3,2,3]→[2,2,2]`。逐组 balance/z diagnostics 按 active-token 数加权；grouped sparse/dense 的输出与全参数梯度最大差均为 0。若把四个 active tokens 改成单一 group，scalar capacity 变为 2，assignment competition 也改变，当前输出最大差约 `0.329387`。Padding routed output、padding hidden gradient 以及修改 padding value/group id 后的 active-output/aux 差均为 0。

两个因果控制把梯度来源拆开。只 detach selected combine weights、仍执行相同 hard top-k 与 expert task loss 时，三个 experts 都得到非零 finite gradient，router 的主任务 gradient 却缺失；这说明离散 expert index 本身不能替代 selected probability 的可微路径。另一个 collapsed top-1 fixture 使用本仓库明确写出的 `E × Σ stop_gradient(f_e) × mean(p_e)`：一次只更新 router 的 balance step 在 assignments 仍全为 expert 0 时，把该诊断从约 `2.567724` 降至 `2.552751`。这只证明局部优化信号，不证明最终负载会均衡、expert 会专门化或 task loss 会改善。

v3 再加入三个显式 overflow policy。固定 4-token/top-1 fixture 的完整稳定排名均为 `[0,2,1]`、nominal capacity=2：`drop` 将 counts `[4,0,0]→[2,0,0]` 并丢 2 个 assignments；deterministic full-ranking `reroute` 得到 dispatched experts `[[0],[0],[2],[2]]`、counts `[2,0,2]`、rerouted=2、dropped=0 且 post-policy excess 为 0；`dropless` 保留四个 expert-0 assignments，并如实报告 nominal-capacity excess `[[2,0,0]]`。Reroute 禁止同一 token 重复 expert，按原 selected score/token/rank 处理溢出，再扫描完整 ranking；按 routing group 独立计容。Rerouted gate 可选择重新归一化为 `[1,1,1,1]`，或以原 selected top-k probability sum 为分母保留 `[1,1,0.4493289641,0.4493289641]` 的 mass，两者输出最大差约 `0.06399997`。Reroute/dropless 的 sparse—dense 输出差和将缺失 sparse gradient 物化为零后的全参数梯度差均为 0。

这些是本仓库 authored deterministic contracts，不是 PyTorch、DeepSeek、Qwen 或任意训练框架的默认 overflow 语义。该 control 仍无跨设备 capacity-group collective、shared/fine-grained experts、expert parallel、all-to-all、grouped GEMM、GPU、目标 checkpoint、收敛、质量或性能证据；整数 group label 不等于真实 device/process group，几条 fixtures 也不能拼成生产 MoE 已复现。

## Two-process Gloo capacity-group control

`moe_distributed_capacity_control.py` 用两个真实 spawn worker、CPU/Gloo 与 temporary FileStore 把“本地 label”和“collective group”分开。Rank 0 的 hidden states 为 `[[2],[1]]`，rank 1 为 `[[3],[0.5]]`；两边的 replicated router 都使用 weights `[[1],[0]]`、top-1、2 experts 与 `capacity_factor=0.5`。一次 `all_gather` 真正形成按 rank 排序的 4-token global routing batch，两个 `all_reduce(SUM)` 分别让两 rank 观察 active tokens=4 和 pre-capacity selected counts `[4,0]`。

若两 rank 各自独立按 2 个 local tokens 计算 capacity=1，它们都会保留本 rank 的最高分 token，合计 kept assignments=2。Collective global batch 的 capacity 仍为 1，但统一 score-priority competition 只保留全局最高分的 rank-1/token-0：global kept mask 为 `[F,F,T,F]`，counts `[4,0]→[1,0]`，drop 3 个，rank 0 相对 local-only 输出的最大差为 `tanh(2)=0.9640275800758169`，rank 1 为 0。两 rank 得到相同 global-route fingerprint `sha256:71a66eeb…`，去除原始 PID 后 strict report fingerprint 稳定为 `sha256:9e342b0b…`。

这个固定输入真实执行了 same-host Gloo `all_gather`/`all_reduce` 和 replicated global capacity competition；
它不是 scalable MoE 实现。Router 与 experts 在两 rank 完全复制，没有 token-to-expert `all_to_all`/`reduce_scatter`、
distributed autograd、DDP backward、shared/fine-grained experts、CUDA/NCCL、多节点、目标 checkpoint、性能、收敛或质量证据。
Collective call count 是源码中当前三类调用的审计账本，不是网络抓包或框架级通信 profiler。

## Two-process Gloo token-to-owner all-to-all control

~~~powershell
python projects/transformers-basics/moe_all_to_all_control.py
~~~

`moe_all_to_all_control.py` 隔离 expert dispatch/return 语义：rank 0 只驻留 expert 0（`2x+0.5`），rank 1 只驻留 expert 1（`-3x+1`），router 在两 rank 复制。Rank 0 的三个 tokens `[-1,2,-2]` 路由到 `[1,0,1]`，rank 1 的 `[1]` 路由到 `[0]`，所以 source→owner counts matrix 为 `[[1,2],[1,0]]`，owner←source 为 `[[1,1],[2,0]]`。每 rank 真实调用五次 `all_to_all_single`：交换 counts、dispatch float payload、dispatch metadata、return output/gate、return metadata；`[1,0]` 的 split 还覆盖零长度 destination chunk。

Owner 计算后按 source rank 返回，但 rank 0 的 return arrival global token 顺序 `[1,0,2]` 与 source-local 顺序 `[0,1,2]` 不同。只有根据 `(source_rank, source_local_index, global_token_id, expert_id)` scatter，才能得到与单进程 oracle 完全相同的 `[3.5231883119115293,4.419062055170588,6.874096530265359,2.201992694944706]`；若把 arrival row 当原顺序，最大差为 `0.8958737432590591`。该 authored top-1 combine 明确保留 selected softmax probability，并非所有框架的默认 top-1 归一化策略。

逻辑 tensor-payload 账本按源码张量的 dtype/numel 计 rank 0/1 各 256/160 bytes、合计 416 bytes；它不等于 wire bytes，不含 Gloo/TCP/FileStore 协议、分包、对齐或 allocator overhead。此 control 没有 capacity/drop/reroute/dropless、backward/optimizer、DDP/FSDP/ZeRO、shared/fine-grained expert、CUDA/NCCL、多节点、目标 checkpoint、吞吐、延迟、显存、收敛或质量证据；也不能与上一条 replicated-capacity fixture 拼接后宣称生产 EP 已复现。

## Two-process Gloo all-to-all forward/backward + SGD control

~~~powershell
python projects/transformers-basics/moe_all_to_all_training_control.py
~~~

这条独立训练 fixture 复用同一组 routes、targets 和 owner-only experts，但把可微 float payload 的 variable-split 通信封装为 authored `torch.autograd.Function`。Forward 以原 splits dispatch/return；backward 把 `input_splits`/`output_splits` 互换，再执行 reverse `all_to_all_single`，把 raw-output/gate 梯度送回 owner、把 hidden/gate 梯度送回 source。Metadata 与 count 仍走不可微 collectives，不能期待 autograd 自动恢复 token identity。

每个 source rank 先按 `local squared-error sum / global_token_count=4` backward。Owner expert 已经接收所有 source 发给该 expert 的 tokens，所以 owner expert gradients 直接是 global-mean 目标的完整局部参数梯度，不应再按 data-parallel 语义重复 all-reduce；replicated router 则只看到本 source 的 gate 路径，必须做 router gradient SUM all-reduce。两个 local router gradients 相加得到 `[[2.2904292655042227],[-2.290429265504225]]`，expert-0/1 weight gradients 为 `6.460938946431114/-7.209951147135929`。一步 `lr=0.01` 无 momentum SGD 后，两 rank router 与各 owner expert 参数都和单进程 global-batch oracle 精确一致。

训练前/后再各执行一次完整分布式 forward，global-mean MSE 从 `20.78017329703821` 降至 `19.41091750734501`。每 rank 的 authored call ledger 为：可微 payload forward 4 次、其 autograd backward 2 次、count/metadata 6 次、router gradient SUM all-reduce 1 次；这是源码包装器内计数，不是 backend profiler 或 wire trace。该控制不使用 `torch.distributed.nn.functional` wrapper、`torch.distributed.autograd` RPC context 或 DDP；没有 capacity/drop、momentum/weight decay/state resume、CUDA/NCCL、多节点、目标模型、收敛、质量或性能证据。

## Two-process Gloo capacity + all-to-all backward + SGD control

~~~powershell
python projects/transformers-basics/moe_all_to_all_capacity_training_control.py
~~~

这条独立 control 才把 global capacity、owner-only dispatch、reverse-split backward 与一步 SGD 接入同一计算图。四个 active tokens 的初选 counts 为 `[2,2]`；`capacity_factor=0.5` 令每 expert capacity=1，按 selected probability、再按 global token id 稳定竞争后，global keep mask `[F,T,T,F]`、kept counts `[1,1]`、drop 2 个。只有幸存 assignments 进入 source→owner all-to-all：两 rank 的 splits 为 `[[1,1],[0,0]]`，owner←source 为 `[[1,0],[1,0]]`。

Rank 1 因本地 token 被丢而成为 zero-assignment source rank；它仍通过 zero-size collective graph edge 参加两次 reverse collective，避免其他 rank 在 backward 等待。Dropped tokens 0/3 的 routed output 与 task hidden gradient 都严格为 0；幸存 hidden gradients、router SUM gradient、owner expert gradients、一步参数和 post-step outputs 全部与单进程 capacity oracle 相同。Global-mean MSE 为 `15.253670387373656→14.530264380025987`，strict report fingerprint 为 `sha256:33f11f199b9668c…`。

每 rank authored ledger 为 payload forward/backward `4/2`、count+metadata `6`、capacity-route `all_gather` `4`、router all-reduce `1`。它没有执行 reroute/dropless、shared/fine-grained experts、DDP/FSDP/ZeRO/TP/PP、mixed precision、optimizer state、CUDA/NCCL、多节点、目标 checkpoint 或 backend profiling；一步 toy loss 下降也不证明收敛、质量、扩展性或生产性能。

## Activation patching 因果控制实验

`activation_patching.py` 在固定 seed 的随机两层 MiniGPT 上运行真实 forward hooks：缓存第 0 层 post-residual tensor，把 clean activation 按 batch/position patch 到 corrupted forward，并报告 raw logit difference 与不裁剪的 normalized recovery。

~~~powershell
python projects/transformers-basics/activation_patching.py
python -m pytest tests/test_activation_patching.py -q
~~~

fixture 同时包含正干预、联合 causal-prefix patch 和未来位置负对照；测试还检查 hook 移除、detach/clone、shape/device/token 边界与小分母拒绝。联合 patch 的 recovery=1、未来位置 control=0，证明这个固定计算图中的干预管线和 causal visibility 按定义工作。

模型没有训练，token 27/19 是根据当前 fixture 的 clean-corrupt 差异事后选择，batch 也只有 1；因此结果不能解释成语言机制、目标 checkpoint circuit、跨 prompt 稳定性或安全解释。完整 prefix activation 被替换后恢复 clean metric 也不惊人：它一次带入该位置的全部纠缠特征。真正研究必须先固定行为和 metric，再扩展多样本、模板、seed、随机 source、无关 site、component/path patch 与 held-out replication。

### 固定 Qwen 权重的 activation-patching control

在 toy hook 之上，下面的入口会复用本项目已审阅的 Qwen2.5-0.5B-Instruct revision 与 7-file snapshot，先重哈希约 999.6 MB selected bytes，再加载 494,032,768 个 FP32 参数。已有完整 cache 时不会联网：

~~~powershell
python projects/transformers-basics/run_qwen_activation_patching_control.py `
  --local-files-only
~~~

固定 chat-template pair 只有位置 19 的 ` France`/` Germany` 不同；metric 是 position 25 上无前导空格单 token `Paris−Berlin`。协议在 patch 前固定 first/lower-middle/final layer 0/11/23，而不是扫完热图再挑层。录制结果的 clean/corrupt metric 为 9.210311/-7.700302；source patch recovery 为 1.000024/0.992244/0。完整 layer-0 prefix 与 final-layer readout 两个构造性正对照均为 1，future-position 负对照为 0，全部 hooks 在结束后移除。机器报告位于 `target-checkpoints/qwen2.5-0.5b-instruct.activation-patching.recorded-report.json`，self-fingerprint 是 `sha256:3f8410f5c31666b1be4f83e343a5b849a0545b2f635f7d415da85a195eebb18c`。

城市 token 是在核对 templated baseline 后、任何 patch condition 被接受前修正的 tokenizer 边界，因此这是可复查的 authored fixed protocol，不是外部可信时间戳 preregistration。高 recovery 只适用于这个 batch-1 pair 和“替换整个 896-d post-layer source residual”的干预；它不定位 attention head/MLP/feature，不证明事实存储层、唯一自然 circuit、总体事实性或安全性。final-layer source recovery=0 是因为该 hook 之后不再跨 position 混合；不能推成“最后层没作用”。CPU FP32 eager 也不证明 CUDA、量化、vLLM 或其他 hook layout。

## 离线 smoke test

smoke_tiny.py 不下载任何模型。它从 GPT2Config 建立微型模型，在固定 batch 上训练 12 步并执行 greedy generation，用于验证：

- config 到模型的映射；
- labels 触发 causal LM shift/loss；
- optimizer 与 train/eval 状态；
- generate 的输入输出 shape；
- 参数量和纯权重存储量。

~~~powershell
python -m pip install "transformers>=4.48,<5"
python projects/transformers-basics/smoke_tiny.py
~~~

纯参数存储不是实际训练或推理显存；后者还包含梯度、optimizer、激活、KV cache 和工作区。

## 检查真实模型

`inspect_checkpoint.py` 只下载 config、tokenizer 与可用的 generation config，不加载权重。命令行强制填写 revision；为了真正固定内容，应传完整 immutable commit hash，而不是仍可能移动的 branch/tag：

~~~powershell
python projects/transformers-basics/inspect_checkpoint.py Qwen/Qwen2.5-0.5B-Instruct --revision <commit-hash>
~~~

脚本输出 requested revision、Transformers 在 config 上记录的 resolved commit metadata、`AutoConfig.to_dict()` 规范化快照的 canonical fingerprint、保守归一化的 attention/MoE/MLA contract，以及 tokenizer 模板的示例文本与模板直接产生的 token IDs。它还尝试读取 `GenerationConfig`，逐项比较 tokenizer、model config、generation config 的 BOS/EOS/PAD/decoder-start ID，并标出集合 disjoint、tokenizer/model vocab 越界和同时存在的 `max_length`/`max_new_tokens`。config 或 generation snapshot 可能包含库补出的默认值和运行时 metadata，不是远端原始 JSON bytes 的哈希。

Base tokenizer 没有 chat template、checkpoint 没有独立 generation config 都可能是合法状态；前者输出模板字段 null，后者输出 `unavailable_or_load_error`，这个状态也可能表示认证/网络/缓存错误，不能武断写成“文件不存在”。脚本不会猜模板或有效 generation defaults，也不会把渲染后的字符串再次 `encode`，从而避免重复添加 special tokens。

默认 `trust_remote_code=False`。若某 checkpoint 必须执行远程代码，先审查对应 revision，再在隔离环境显式开启。resolved commit 是库对象暴露的 metadata，不是签名或来源认证；三方 special-token 完全一致也不证明 `generate()` kwargs、模型类 fallback、vLLM/provider defaults、stop-string tokenization 或实际停止行为一致。config/tokenizer/generation config 成功读取仍不证明权重匹配、许可、有效上下文、质量或 runtime 支持。

## 离线 config contract 与 KV 账本

`inspect_config.py` 对本地 JSON 做 strict load，拒绝 duplicate key、`NaN`/`Infinity` 和非 object 根节点。它只在 `num_hidden_layers`、`num_attention_heads`、`num_key_value_heads` 与 head dimension 足够明确，且没有已知 MLA marker 时，计算标准 dense K/V 的理想 tensor payload：

~~~powershell
python projects/transformers-basics/inspect_config.py `
  projects/transformers-basics/configs/standard-gqa.example.json `
  --tokens 4096 --batch-size 1 --element-bytes 2
python projects/transformers-basics/inspect_config.py `
  projects/transformers-basics/configs/mla-moe.example.json --tokens 4096
~~~

三个 `configs/*.example.json` 的 `model_type` 和 `architectures` 都以 `authored` 开头，表示它们是本仓库自编的
公式回归样例，不是 Llama、Qwen、DeepSeek 或任何发布 checkpoint 的配置快照。标准 GQA 样例在 4096-token、
batch 1、2-byte element 下得到 536,870,912 bytes；MoE-GQA 样例在相同 token 数和 batch 2 下得到
402,653,184 bytes。MoE marker 不改变标准 attention 的 K/V 公式，但也不能据此推断 total/active parameters。
出现 MLA marker 时，程序会停止套用标准公式；这个分支不能反推出某个 MLA runtime 的真实 cache layout。

这些数字不含 allocator metadata、block 对齐、量化 scale、workspace、临时张量和权重，也不证明显存峰值或吞吐。`max_position_embeddings` 只是被记录的 config 字段，不是有效上下文或质量证据。

## 不可变发布证据：Llama、Qwen 与 DeepSeek

固定配置样例之外，`release-evidence/manifest.json` 把三种不同证据绑定到同一个逐字段验证程序：

- Meta Llama 3.2 text-only：固定官方 GitHub model card commit、原始 byte hash 与六段 exact fragments；输出只称 vendor-reported claims；
- Qwen2.5-0.5B-Instruct：固定 Hugging Face 官方组织的 immutable config URL、原始 byte hash、本地完整 semantic snapshot 与标准 GQA 账本；
- DeepSeek-V3：固定 immutable config URL、原始 byte hash、完整 semantic snapshot，并要求 MLA+MoE markers 触发标准 KV 公式拒绝。

~~~powershell
# 完全离线：检查 manifest、local semantic snapshot、字段投影和数值
python projects/transformers-basics/verify_release_evidence.py

# 联网：额外重新下载三个 immutable artifacts，核对 byte length/SHA-256；
# config 还要与本地 snapshot 语义完全相同，model card 要包含全部固定 fragments
python projects/transformers-basics/verify_release_evidence.py --verify-upstream
~~~

离线报告 fingerprint 为 `sha256:40b3fe7b…e4638`；2026-08-13 联网报告 fingerprint 为
`sha256:6fffd665…587b`。后者说明当次下载 bytes 与 manifest 一致，但不认证 DNS/HTTPS 之外的发布者身份、签名
或未来可用性；上游仓库/组织被接管仍不由无密钥 hash 解决。Qwen 的 402,653,184-byte 结果只含 32,768 tokens、
batch 1、2-byte element 的理想 K/V tensor payload。DeepSeek-V3 即使显式含 `num_key_value_heads`，
程序也会因 MLA markers 而拒绝套用标准 GQA 公式。

Control 不下载模型权重/tokenizer，不执行 remote code、forward、MLA kernel 或长上下文任务；不证明 config 与权重匹配、有效上下文、参数量、质量、许可、runtime 支持、显存峰值、性能或生产安全。Meta 记录来自 model card 而不是 gated config，因此不能与 Qwen/DeepSeek 的 config-level deduction 混为一类证据。

## 让固定 Qwen 权重真实执行一次

发布证据 control 明确不加载权重；`run_target_checkpoint.py` 是下一层、范围更窄但确实执行目标 checkpoint 的控制。Manifest 固定 `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`、CPU/FP32/eager、固定 messages 和 7 个必需文件。7 个文件合计 999,586,347 bytes；其中 [immutable model.safetensors](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors) 为 988,097,824 bytes，SHA-256 是 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`。

~~~powershell
# 首次运行允许匿名下载固定 revision 的选定文件，约 1 GB
python projects/transformers-basics/run_target_checkpoint.py

# 已有完整 Hugging Face cache 时，强制不联网
python projects/transformers-basics/run_target_checkpoint.py --local-files-only
~~~

控制先对实际打开的 7 个文件逐 byte 重算长度和 SHA-256，再从已验证的本地 snapshot 以 `trust_remote_code=False` 加载。它要求 `Qwen2ForCausalLM`/`qwen2` 身份、tokenizer chat template、CPU placement 和 FP32 参数，随后冻结全部参数、进入 eval/inference mode，执行一次完整 prefill、一次带 `past_key_values` 的 cached step、同位置 full recompute 和 greedy `generate()`。

2026-08-13 的录制环境为 CPython 3.12.10、PyTorch 2.13.0+cpu、Transformers 4.57.6、Windows CPU；报告 self-fingerprint 为 `sha256:56528a3e02ed6ef9d205dcf83ba456658d639d7681ab6a1ad9eb110211edba62`。固定观察值为：494,032,768 个参数、参数存储 1,976,131,072 bytes、trainable 0；31 个 prompt tokens；prefill logits `[1,31,151936]`；continuation token IDs `[17,151645]`，保留 special token 解码为 `2<|im_end|>`；manual prefill/cache argmax 均与 `generate()` 一致，cached/full 第二步 argmax 一致，全部 logits 的 max absolute error 为 `3.719329833984375e-05`，低于 `1e-4` 门槛。v1 只接受恰好两个生成 token；录制文件会由 nested closed-schema、内部一致性、自指纹和内容准确性检查重新校验，不会在普通 CI 中重新下载 1 GB 权重。

这次运行只绑定选定文件，不证明仓库中所有文件或 config-weight 语义都匹配；HTTPS + 无密钥 SHA-256 不是发布者签名。Verifier 先从 open handle 哈希，Transformers 随后仍按路径重新打开文件，因此没有消除两者之间的并发替换 TOCTOU；生产消费需要不可变目录、ACL/lease、内容寻址句柄或等价控制。一个英文算术 prompt 的 CPU FP32 greedy 结果不证明训练复现、模型总体/中文质量、32k 有效上下文、许可适用性、CUDA/vLLM、峰值内存、吞吐、延迟或生产安全。参数存储也不是进程峰值 RSS。

## 固定 Qwen 单矩阵 packed INT4 control

通用 quantization toy 与完整 tiny MiniGPT checkpoint 都没有碰到发布 Qwen 权重。下面的 control 复用已验证的同一 7-file snapshot，加载 CPU FP32 模型，并只选择第一层 attention 的 bias-free `model.layers.0.self_attn.o_proj.weight`：

~~~powershell
# 真实重哈希、加载约 1 GB snapshot、捕获激活并执行两次 forward
python projects/transformers-basics/run_qwen_weight_quantization_control.py `
  --local-files-only

# 普通 CI 只验证冻结的 closed report，不加载权重
python projects/transformers-basics/run_qwen_weight_quantization_control.py `
  --verify projects/transformers-basics/target-checkpoints/qwen2.5-0.5b-instruct.weight-int4.recorded-report.json
~~~

所选矩阵 shape 为 `[896,896]`、共 802,816 个参数，只占 494,032,768 参数的 `0.0016250258120530175`。Reference 使用 contiguous-row group size 128、每行 7 groups、FP32 absmax scale 和对称 code range `[-7,7]`；`-8` 故意不用。它不是 NF4、GPTQ、AWQ、SmoothQuant 或任一 runtime 的专有 layout。

2026-08-15 的真实运行结果：

| 项目 | 观察值 |
|---|---:|
| 该矩阵 FP32 bytes | 3,211,264 |
| ideal packed code bytes | 401,408 |
| FP32 scale metadata bytes | 25,088 |
| strict one-tensor bundle bytes | 427,328 |
| selected-matrix serialized ratio | 7.514752134192002× |
| weight relative-L2 / max-abs | 0.1323337087062499 / 0.0400739386677742 |
| captured `o_proj` output relative-L2 / max-abs | 0.07000153078579582 / 0.010923892259597778 |
| last-position logits relative-L2 / max-abs | 0.08513807180570929 / 1.6255179643630981 |
| baseline / partial-quantized argmax | 17 / 17 |

Control 在真实 31-token forward 中捕获 `[1,31,896]` 的 `o_proj` 输入/输出，证明 hook output 与直接 FP32 linear 的 max error 为 0；随后把 packed artifact 严格重载，in-memory 与 reloaded dequantized layer output exact，并暂时替换这一矩阵执行新的完整模型 forward。结束时原始 source weight byte-exact 恢复。Artifact SHA-256 为 `sha256:006cc9a2…0bf7`，closed report 为 `sha256:df9ee045…f5cb`；翻转 artifact digest byte 会在 decode 前拒绝。

这些数字最重要的结论不是“INT4 无损”，而是相反：只量化 0.1625% 参数，末位 logits 仍出现 8.51% relative-L2 与 1.6255 max-abs 变化。当前 prompt 的 argmax 恰好没变，不能推断其他 token、序列、生成或任务质量不变。427,328 bytes 只是这一个 weight bundle；完整模型其余参数仍为 FP32，forward 使用反量化 FP32 weight，没有完整 low-bit checkpoint、量化 runtime、fused kernel、GPU/CUDA/vLLM、resident/peak memory、latency、throughput、GPTQ/AWQ calibration 或代表性质量证据。

## 离线 generation protocol 对账

`inspect_generation_protocol.py` 先拒绝重复字段和非法数值，再对一个明确的三方快照做 token-ID 对账：

~~~powershell
python projects/transformers-basics/inspect_generation_protocol.py `
  projects/transformers-basics/protocols/aligned-superset-eos.example.json
python projects/transformers-basics/inspect_generation_protocol.py `
  projects/transformers-basics/protocols/drift-out-of-range.example.json
~~~

第一份固定样例中 tokenizer/model EOS 为 `{2}`，generation EOS 为 `{2,3}`，因此只报告 generation EOS 是
tokenizer/model EOS 的严格超集，不判错：额外停止 token 可能是 checkpoint 的有意协议。第二份把 generation
BOS/EOS/PAD 改成与另外两方 disjoint 的 `{4}/{5}/{9}`，其中 9 超出 tokenizer/model 的 8-token 空间，
检查器必须逐项暴露。两份文件都不是任何发布模型快照，也没有执行 `generate()`；`PAD=EOS` 只会得到“可能有意”的 observation。

检查器不裁决谁正确，不推断 `max_new_tokens` 与 `max_length` 在目标版本的优先级，也不把 `do_sample`、beam/contrastive search 或 stop strings 拼成所谓“有效配置”。最终部署仍应显式传参，并对 Transformers/vLLM/provider 分别保存请求、token trace、finish reason 与版本。

## Transformers `generate()` 停止控制实验

静态对账之后运行真实框架控制：

~~~powershell
python projects/transformers-basics/generation_runtime_control.py
~~~

脚本在随机、未训练的 3,824-parameter tiny GPT-2 上真实执行 forward 与 `GenerationMixin.generate()`，但一个 authored `LogitsProcessor` 会把每步全部 next-token scores 覆盖成单个确定 token。因此权重不决定输出，三条路径可以精确审计：

1. GenerationConfig EOS `{2,3}`，计划 `[4,3]`，token 3 后停止；
2. 同一 config 在调用时传 `eos_token_id=5`，计划 `[3,5]`，token 3 不停止、token 5 停止；
3. Config 的 `max_new_tokens=5`，调用时传 2，计划 `[4,6]`，未遇 EOS 也恰好生成两个 token。

实验还验证 caller-owned `GenerationConfig` 没被 mutation。报告中的 finish reason 是根据受控 token plan 与 EOS/length 条件**推断**的；Transformers 的该返回对象没有 provider 风格 finish reason。它证明当前安装 Transformers 版本的这三条控制流，不使用真实 tokenizer/chat template，不加载公开 checkpoint，也不证明正常模型 logits、vLLM/provider precedence、质量、性能或 GPU 行为。

## 加载权重时的检查顺序

1. 阅读 model card、许可和用途限制；
2. 固定 model revision 与 tokenizer revision；
3. 核对 chat template、BOS/EOS/PAD 和 generation config；
4. 选择 torch_dtype，不用字符串“auto”掩盖实际 dtype；
5. 单张 GPU 先用 device_map 明确 placement；
6. 打印参数量、权重存储和实际峰值显存；
7. 用小型质量集确认量化/模板没有破坏输出；
8. 再做 batch、上下文和并发扫描。

## 家族差异

AutoModel 统一接口不代表模型协议相同。Llama、Qwen 和 DeepSeek checkpoint 可能在 GQA/MoE、RoPE、special tokens、tool template 和 generation config 上不同。GPT、Claude、Gemini 等云 API 也不能假设与 Transformers messages 一一等价。

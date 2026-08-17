# LLM 面试题与回答方法

## 怎样使用

每题先用 30 秒给结论，再用机制、权衡、失败模式和验证实验展开。优秀回答不只复述定义，还能说清 shape、指标、约束以及如何证明。

## Transformer 与生成

### 1. 为什么 attention 要除以根号 head dimension？

**回答骨架**：若 Q/K 分量近似独立且方差稳定，点积方差随维度增长；不缩放会让 softmax 饱和、梯度变小。缩放稳定 score 尺度。进一步说明实际分布不严格独立，但初始化与训练中该归一化仍有用。

**追问**：LayerNorm 已归一化，为什么还需要？温度与这个缩放有何区别？

### 2. causal mask 怎样防止信息泄漏？

训练可并行计算所有位置，但把未来 key 的 score 设为负无穷，softmax 后权重为零。要区分 causal mask、padding mask 和 loss mask。验证实验：只改未来 token，过去位置 logits 必须不变。

### 3. MHA、MQA、GQA 如何取舍？

MHA 每个 query 头有独立 K/V；MQA 共享一组；GQA 多个 query 头共享一组。减少 K/V 头可显著降低 KV Cache 与 decode 带宽，质量可能下降。给出 KV 元素公式并说明 query 计算没有按相同比例消失。

### 3.1 拿到 `config.json`，什么时候可以估 KV Cache，什么时候必须拒绝？

先限定为标准 dense MHA/GQA/MQA layout，并要求层数、query/KV head 数和 head dim 显式且自洽；若 head dim 由 hidden size 推导，要检查整除。理想 payload 是 `2 × L × H_kv × d_head × tokens × batch × bytes(element)`，不含 block/allocator、对齐、scale、workspace 或临时张量。出现 MLA/latent attention 或未知 remote-code architecture 时不能套公式；MoE 本身不改变 attention cache 公式，但总/激活参数也不能从几个 marker 猜出。`max_position_embeddings` 不证明有效上下文，config hash/commit metadata 不证明权重匹配、来源、许可、质量或 runtime layout。高质量回答还应提出固定 immutable revision、检查实际实现/权重 shape，并用目标 runtime 实测峰值。

### 3.2 Online softmax 为什么能避免保存完整 attention matrix？它把复杂度变成线性了吗？

对每个 query row 和当前 key block，维护 running maximum \(m\)、相对该 maximum 的分母 \(\ell\) 与 value 加权分子 \(o\)。新 block 的 maximum 为 \(m'\) 时，先用 \(e^{m-m'}\) 同时缩放旧 \(\ell\) 和旧 \(o\)，再加入 \(e^{s_j-m'}\) 及其与 \(v_j\) 的加权和；最终返回 \(o/\ell\)。因此只需当前 score tile 和每行状态，不必保存完整 score/probability；fully masked row 的 \(\ell=0\) 必须拒绝。

这没有把精确 dense attention 的 QK/AV 算术从 \(O(T_qT_kD)\) 普遍变成线性，主要改变中间存储和数据搬运。实数算术下等价，浮点归约顺序可产生微差。高质量回答还会区分三层证据：NumPy recurrence 对齐 dense reference只证明数学实现；logical tile shape 不是进程峰值或 HBM 测量；只有确认目标 dtype/mask/head dim/dropout/GQA/hardware 实际选择相应 GPU backend 并做 profile，才能讨论 FlashAttention kernel 与性能，配置开关本身不能证明没有 fallback。

### 4. Prefill 与 decode 的瓶颈为何不同？

Prefill 对多个 token 做大矩阵乘，较易 compute-bound；decode 每步 batch/序列维小，却反复读权重和 KV，常 memory-bound。用 TTFT、TPOT、GPU 利用率和 memory bandwidth 验证，不能只看总延迟。

### 5. temperature、top-k、top-p 分别做什么？

temperature 缩放整个 logit 分布；top-k 按排名截断，是否“恰好 k 个”取决于并列 tie-break；top-p 保留排序概率累计和首次达到阈值的最小前缀，必须保留 crossing token。说明 repetition/temperature/top-k/top-p 的处理顺序和每次重新归一化时点：若先 top-k 再 top-p，nucleus 用的是 top-k support 上归一化后的概率。`temperature=0` 通常是 API 的 greedy 特例，不能代入除法。即使概率相同，seed、RNG、CDF traversal、并列规则与 GPU 非确定性仍会影响重放；仓库 fixed-uniform CPU oracle 只证明一份显式单步契约，不证明 runtime 默认等价或生成质量。

### 5.1 tokenizer、model config 与 generation config 的 EOS 不同，谁错了？

不能先选赢家。把 BOS/EOS/PAD/decoder-start 归一成 ID 集合，逐对报告 exact/subset/overlap/disjoint，并检查所有 ID 是否落在 tokenizer size 与 model vocab 内。Generation EOS 是 tokenizer EOS 的 superset 可能有意加入 turn-end token；PAD=EOS 也常是明确选择。未解释的 disjoint 或越界是 review signal，但静态一致仍不证明运行时正确。

有效停止行为还取决于模型类 fallback、`generate()` kwargs、`max_new_tokens`/`max_length` precedence、sampling/beam/contrastive mode、stop-string tokenization、Transformers/vLLM/provider 版本和 server request override。normalized `to_dict()` snapshot 可能含库默认值，不是原始 JSON byte hash。高质量回答会保存三方 snapshot、最终请求、模板 token IDs、输出 token trace 与 finish reason，并在目标 runtime 做对照，而不是改一个 JSON 后宣布修复。

可进一步给受控实验：用 logits processor 强制 `[4,3]`，证明 config EOS `{2,3}` 在 3 停；call override EOS=5 后强制 `[3,5]`，证明 3 不停而 5 停；强制 `[4,6]` 配 `max_new_tokens=2`，证明无 EOS 的 length stop。该实验真实走 Transformers generation loop，却不让模型权重选择 token；它不能证明真实 checkpoint、vLLM/provider 或 finish-reason 字段，只能定位当前版本控制流。

### 5.2 Activation patching 能证明什么？怎样避免自欺？

先在 clean/corrupt pair 上固定连续 metric，缓存明确 hook site 的 clean activation，再把它 patch 进 corrupted forward。若 metric 改变，能支持“在这个干预定义、样本和 site 下存在因果作用”，不能自动证明这是唯一、自然或完整回路。报告 (m_{clean},m_{corrupt},m_{patched}) 和未裁剪 recovery；分母接近零时拒绝归一化。负对照至少包括未来/无关位置、随机 clean source、同分布无关样本和多个模板/seed。还要确认 hook 是 pre/post norm/projection 哪个 tensor，shape、batch 对齐、dropout 和缓存路径都正确。仓库随机 MiniGPT fixture 的 joint recovery=1、future control=0 只证明 hook 与 causal mask 控制流；token pair 是 post-hoc 选择，不能当目标模型语言机制证据。

### 5.3 流式 stop string 为什么不能对每个 chunk 单独调用 `endswith`？

因为 stop 可跨 token、event 和 UTF-8 byte chunk；一个 chunk 也可能同时含 stop 前文本、stop 与其后文本。先 strict incremental UTF-8 decode，再暂存“当前 suffix 中也是任一 stop prefix”的最长部分，只 emit 已不可能参与匹配的前缀。必须固定 include/exclude、大小写/normalization 和 overlap 语义；仓库选择逐 character first-completion，同一 character 多个完成按配置顺序。

客户端 matcher 只处理已收到文本。它命中后截断 UI 不等于 provider 返回 stop finish reason，也不证明服务端停止 decode、usage 不再增长、KV 已释放或停止计费。SSE framing、provider event state、应用 stop matching 和 server cancellation 是四层不同证据。

### 5.3.1 客户端断开 SSE 后，怎样证明服务端真的停止工作？

先把命题拆开：连接断开被 ASGI 观察、generation task 收到取消、backend cooperative iterator 停止产生 token、阻塞 thread/process/kernel 停止、scheduler sequence 移除、KV/GPU block 释放、provider 停止计费，是至少七层证据。只看 client `aclose()`、HTTP EOF 或 access log 499 最多证明连接侧状态；即使 coroutine 收到 `CancelledError`，`asyncio.to_thread()` 中的 Transformers `generate()` 也可能继续运行。

可控验收应在首个 content 时证明 backend 尚未完成，再关闭 client，并以同一 request id 关联 stream task、backend token trace、scheduler/allocator trace 和 usage/billing reconciliation。仓库 authored async control 精确证明 ASGI task/iterator cancellation 与后续 scripted token suppression，但未执行 tokenizer/model、blocking generation thread、vLLM/CUDA 或 KV；高质量回答会主动限定这个边界，并为目标 runtime 分别测 disconnect-to-work-stop 与 disconnect-to-resource-release latency。

仓库另有 tiny Transformers 对照：随机 1,272 参数 GPT-2 在 Python thread 中真实执行一次 forward 与 `GenerationMixin.generate()`；streamer 在首 token 后人为暂停，disconnect 设置 event，authored `StoppingCriteria` 观察后让 thread 返回并 join。它证明“显式 cooperative check 可工作”，不证明 Python 可强杀 thread、未修改/已进入不可中断 kernel 的调用会退出，也不证明 KV/CPU/GPU memory release。面试时若把这个 control 简写成“实现 Transformers 断连取消”，应继续追问 cooperative hook、检查频率、join timeout、kernel 边界和 allocator 证据。

### 5.4 云 API token/费用 hard cap 为什么需要 reservation？

响应后才累加 usage 会让并发请求重复花同一份剩余额度。发送前应以目标 tokenizer 估计 input，从实际 request body 提取唯一 maximum output，并将 request identity/cap/计费 scope 与 receipt 绑定后原子预留 token 和估算费用；成功按 provider-reported usage 结算、释放未用容量。只有确定 transport 没发送才能 cancel；timeout/partial stream 后 usage 未知时保守占用 reservation，并异步对账。若 actual 超过 reservation，调用已经发生，必须先记录超额再 fail closed。

这仍不等于“费用绝不会超”。Estimate 可能不含 provider 隐藏、cache 或 reasoning token，费率会按 model/tier/time 漂移，重试/取消也可能计费。价格快照要绑定 provider/model/revision/checked_at；多 worker 还需要 durable atomic quota，最终用 provider billing export reconciliation。Request hash 只绑定选中的 bytes，不证明 caller、真实发送或保密；micro-USD policy estimate 不能证明发票。

若用 SQLite，`BEGIN IMMEDIATE` 可让同一文件的多 writer 争抢容量时串行化，并让 reservation/event 在进程退出后保留；但不能和远程 HTTP/provider billing 做原子提交。Crash 后 active 记录不能靠 TTL 自动 cancel，因为进程死亡不证明请求未发送；要用 stable call id、attempt/request id 与 billing export 对账，能证明未发送才释放，否则 conservative uncertain。无密钥 fingerprint 只能发现非协同漂移，不能抵抗能改库并重算 hash 的攻击者；单机 SQLite 也不是分布式 quota 或 exactly-once billing。

### 5.5 为什么一次逻辑调用不能只 reserve 一次再自动重试三次？

因为每次 replay 都可能是独立计费调用。只预留一次 maximum output，却允许三次 attempt，最坏 token/费用暴露接近三份；最终成功响应的 usage 只描述它自己，不能证明前两次 timeout/5xx 免费。准确实现要在每个 attempt 发送前用唯一 attempt id 独立 reserve，并分别 settle、确定未发送才 cancel、否则 uncertain；logical call id 只用于聚合，不替代 attempt ledger。单-attempt helper 应强制 `max_attempts=1`；retry orchestrator 则需要 attempt-start/attempt-finish hook，而且必须先 terminalize 旧 attempt，再 sleep 或发送下一次，不能在外层另写循环而丢掉 `Retry-After` 与 logical deadline。

收到 HTTP 500 虽然 outcome known，也只说明“服务端返回了 500”，不说明零 usage/零费用；2xx 缺 usage、response parse failure 和 client cancellation 同样不能释放。只有 Pool/Connect 前失败等结构化证据能证明 request 未发送。可用数字反例回答：每 attempt cap 为 80 micro-USD，第一次 500 uncertain 后已占 80；hard limit=140 时第二次 80 reservation 会把 projected total 推到 160，所以应在 transport 前挡住，网络只有一次。若 limit 足够，第二次成功 usage 为 66，则 logical call 合计 146，而不是 66。SQLite 解决本地并发与重开，不消除 provider effect—local commit 的 crash window，也不证明 provider invoice 或 exactly-once billing。

## 训练与微调

### 6. 预训练 loss 下降为何不保证产品质量？

next-token loss 是代理目标，可能改善流畅度却不改善事实、工具、拒答或业务成功。需要域外验证、任务评测、安全与成本共同判断。

### 7. LoRA 的参数量和前向公式是什么？

冻结 W，学习 B 与 A，更新为 W 加 alpha/r 乘 BA。参数量约为 r 乘输入输出维度之和。B 零初始化保证初始函数不变。追问 target modules、rank、合并和多 adapter 服务。工程上 adapter 必须绑定 exact base、target module/fan-in-out/scaling 约定和 tokenizer/template；`merge_and_unload` 成功只说明当前内存路径能合并。可信导出还应从独立 base 重载 adapter，比较 logits，保存 merged full weights 后再次重载比较，并重跑任务/安全评测。

目录 manifest 必须覆盖 config、weights、tokenizer/template 等完整 file set，并在 load 前强制验证 extra/missing/path/size/hash 和关键语义；框架不会因旁边存在 JSON 就自动执行它。路径或 identity string 不是 base 内容认证，unkeyed hash 也不认证来源；FP32 tiny merge 不证明量化基座、目标 checkpoint、跨版本兼容或原子发布。

### 8. QLoRA 为什么不等于“4-bit 训练”？

通常只有冻结基座权重低位存储；反量化计算、LoRA、梯度、optimizer 和激活仍用更高精度。显存还受序列长度和 checkpointing 影响。

### 9. 如何构造 assistant-only loss？

先用真实 chat template 序列化消息，再通过 token 边界生成 label mask；user/system/tool/padding 设 ignore index。用 token 级可视化和特殊 token 单测，不能按字符串长度猜边界。

### 9.0 模板返回 assistant mask，如何证明 Trainer 最终监督正确？

至少分三层核对：模板渲染后的 input IDs 与 assistant mask、padding/truncation 后的 batch、collator 最终 labels。监督位置必须保留对应 input ID，system/user/tool、非 assistant 与 padding 必须是 `-100`；还要确认终止 token 是否按目标纳入。tool 数据还有第四层：异构 arguments 进入 Arrow 后是否被 struct widening 注入 `null`。仓库固定 Qwen 反例中，原生模板对三条多轮/并行 tool fixture 返回全零 mask；审核 `{% generation %}` 模板保持 47 / 301 / 200 个 input IDs 相同，在 Arrow 前标出 8 / 51 / 31 个 assistant tokens，真实 TRL collator 得到 `[3, 301]`、90 个监督 labels。no-grad loss `1.251716` 只证明 batch 被消费，不等于 backward、收敛、数据合法或任意 provider tool schema 都正确。

### 9.0.1 目标 Qwen 上 LoRA backward、B 非零且 adapter 重载 exact，能否证明微调成功？

只能证明被观察的训练/导出链路成功。还要分别问：加载前是否绑定 exact base revision/bytes，assistant label boundary 是否按 token 核对，冻结基座是否真的不变，optimizer 是否只接收预期参数，保存的 A/B 是否完整，以及独立基座重载是否等价。即使这些都通过，单样本/单步也没有代表性、验证集或收敛证据。本仓库控制中 270,336 个 adapter 参数真实更新、重载 logits max error=0，但 loss 从约 0.003864 升到 0.584557；这正说明 plumbing success 不等于 objective/quality success。CUDA、QLoRA、峰值显存和部署 runtime 也需独立验证。

### 9.0.2 可变长度 SFT 做 gradient accumulation，为什么 `loss/accumulation_steps` 可能是错的？

因为许多 loss 已是“当前 micro-batch 有效 token 的 mean”。若第 \(i\) 批有 \(n_i\) 个监督 token，把 \(M\) 个 local mean 等权平均会给每个 token 系数 \(1/(Mn_i)\)；全窗口 token mean 要的是 \(1/N\)，等价于按 \(n_i/N\) 加权每批 mean。Padding、prompt 和 `-100` 位置不进 \(N\)。推荐累积 loss sum 与有效 token count，再按整个 optimizer-update window 的 count 缩放；DDP 还要按 reducer 的 sum/mean 处理 world-size 因子，clip 必须发生在完整归一化后的累计梯度上。

“sum/count 与 full batch 相同”也不是无条件的：dropout/RNG、BatchNorm 等跨样本运算、AMP overflow/unscale、逐 micro-batch clipping、提前 optimizer/scheduler step、`no_sync`/collective 边界和浮点求和顺序都会影响等价口径。仓库 `[1,3]` token 的 `Fraction`/PyTorch Float64 toy 得到 full/count-scaled `(23/40,-23/40)`、naive `(7/20,-7/20)`，但没有 optimizer、DDP、CUDA 或目标 LLM，只证明局部 reduction 反例。

### 9.0.3 默认 DDP 对 gradient 取 mean 时，global token mean 为什么要乘 `D/N`？

设 rank `r` 的 local loss sum 为 `S_r`，`D` 个 rank 的全局有效 token 数为 `N`。默认 DDP 同步结果是各 rank gradient 的 `1/D` mean；若每个 rank backward `(D/N)S_r`，最终正好是 `(1/N)Σ_r ∇S_r`。若误用 `(1/N)S_r`，同步后变成目标 gradient 的 `1/D`；若各 rank backward `S_r/n_r`，每个 token 的权重则是 `1/(D n_r)`，高 padding/短 shard 会被过度加权。

仓库的双进程 CPU/Gloo control 固定 `D=2,N=4,n_r=[1,3]`：`D/N=1/2` 得到 full-batch `(23/40,-23/40)`，漏 world size 的 `1/N=1/4` 得到 `(23/80,-23/80)`，rank-local mean 得到 `(7/20,-7/20)`；两个 rank 都观察到 all-reduced count 4 和相同梯度。这证明当前 PyTorch/Gloo/default reducer 固定路径，不证明 accumulation + `no_sync`、AMP、optimizer、FSDP/ZeRO、GPU、多节点、目标 Trainer 或模型质量。

### 9.0.4 DDP 的 `no_sync` 为什么要同时包住 forward 与 backward？

DDP 在 forward 阶段按当时的 `require_backward_grad_sync` 状态准备 reducer；等 forward 已在同步模式运行，再只给 backward 套 `no_sync` 已经太晚。正确模式是把非末尾 micro-batch 的 forward、loss 和 backward 全部放在 context 中，最后一个 micro-batch 离开 context 触发同步；zero-grad、clip、optimizer 和 scheduler 只在完整 update window 的正确边界执行。

仓库固定 `D=2`、两批/rank、counts `[[1,2],[3,1]]`、`N=7`。built-in DDP 正确 `no_sync` 的 pre-clip gradient 为 `(+19/35,-19/35)`，同步后 clip 和 plain SGD update 与 full batch 一致。独立 PyTorch reference all-reduce hook 对照中，正确 scope 为 1 次 hook，只包 backward 为 2 次。后者在这个线性单参数 fixture 上数值仍正确，因此负对照证明的是“通信没有省掉”，不是“梯度必然错误”。built-in reducer 本身没有直接计数；多 bucket、随机层、AMP、AdamW、FSDP/ZeRO、GPU、目标 Trainer 与性能仍需独立验证。

### 9.0.5 AMP accumulation 中为什么必须先 unscale 再 clip，overflow 后 scheduler 能否照常 step？

GradScaler 在 backward 前把 loss 乘 scale，因此 `.grad` 暂时也是 scaled gradient。若对它先做 norm clipping，再 `unscale_`，clip threshold 会再除一次 scale；仓库 CPU FP16 fixture 中正确路径是 `24→unscale 3→clip 约 0.5`，错误路径是 `24→clip 约 0.5→unscale 约 0.0625`。一个 update window 内任一 micro-batch non-finite 都会污染累计梯度；`scaler.step(optimizer)` 应跳过整个 optimizer update，依赖 update 计数的 scheduler 也不能无条件前进。不要用 `optimizer.step()` 的返回值判断成功，因为许多 optimizer 成功时也返回 `None`；应按目标框架公开状态/协议检测 skip，并做参数、optimizer step 与 scale 的集成断言。

GradScaler 自身也是 checkpoint state。仓库先建立非空 AdamW moments，再用三个 overflow window 观察 scale `8→4→2→1` 且 step 保持 1；进程内恢复 scale=1 时 gradient 10000 执行 step=2，漏恢复而回到 scale=8 时 scaled gradient overflow、scale 降到 4、step 仍为 1。这只证明当前 CPU 单参数 state replay 的因果差异；没有文件 checkpoint/进程重启、scheduler/RNG/DDP overflow 共识、CUDA 或目标 Trainer 证据。

### 9.0.6 DDP 下一个 rank overflow，所有 rank 一定都会跳过 optimizer 吗？

不能脱离 overflow 发生位置回答。若 non-finite 在默认 DDP gradient reduction **之前**产生，并进入相同的 sum/mean collective，Inf 通常会传播到同步结果；仓库双进程 CPU/Gloo 单参数 fixture 中，rank 0 在 `no_sync` 首批产生 Inf，末批 reduction 后两个 rank 都检测到 non-finite，AdamW/StepLR 共同 skip。但这只是当前 reducer 路径，不覆盖 sanitizing/custom hook、条件/未使用参数、per-rank gradient transform 或 reduction 后故障。

反例先让两个 rank 完成 finite reduction，再在 rank 0 的 `unscale_` 前人为把 gradient 改成 Inf。各 rank 独立 GradScaler 会让 rank 0 skip、rank 1 update，于是参数、optimizer moments/step、scheduler、LR、scale 与 growth tracker 同时分叉。稳健设计应在所有可能产生 non-finite 的 transform 之后、任何 optimizer mutation 之前形成 global decision；step 后才发现 checksum 不同已经太晚。仓库用 unscaled local flag 的 `all_reduce(MAX)` 让两边共同 skip，但 `update(new_scale=...)` 只是示例 scale policy，growth tracker 与 native overflow transition 不同，不是通用 distributed scaler。面试答案还应说明目标框架的公开 overflow 协议、clip 顺序、多个 optimizer/parameter group、checkpoint 与故障注入验证；不要用 step 返回值当通用成功 receipt。

### 9.1 如何证明 SFT 切分没有泄漏？

不能绝对证明“没有”。先按用户、thread、source document 或 problem family 分组再切分，门禁检查 id、exact content 和 group 跨 split；再用规范化字符 n-gram/MinHash/LSH、embedding 或任务特定规则检查 near duplicate、答案独特片段与时间穿越。Jaccard 回答的是 shingle-set 重叠，不是语义等价；必须报告 normalization、n、阈值、比较分母、人工复核和漏检边界，并隔离 test 权限。exact/lexical gate 通过不代表语义无重复，manifest hash 也不证明许可或隐私合规。

### 9.2 MinHash/LSH 为什么不能直接作为“无泄漏”门禁？

MinHash 签名相等率只是集合 Jaccard 的随机近似；将 \(k=br\) 个分量切成 \(b\) bands×\(r\) rows 后，理想候选概率为 \(1-(1-s^r)^b\)，不是确定召回。增大 bands 提高召回也增加候选，增大 rows 则相反。LSH 命中后仍要 exact Jaccard recheck；未命中可能是假阴性。仓库 64/16×4 authored snapshot 的 10 个 pair 产生 3 candidates、1 true positive、2 false positives，snapshot recall=1/precision=1/3；但 1-hash 反例会漏掉 Jaccard=2/3 的 pair。生产要在目标语言/长度/来源切片抽样 exact ground truth，报告 precision/recall/漏检和区间；exhaustive recall audit 本身仍是 \(O(N^2)\)。这些都是 lexical shingle 证据，不覆盖语义改写、翻译、答案片段或许可隐私。

### 10. 何时用 RAG，何时微调？

易变、私有、需引用事实优先 RAG；行为、格式、风格和稳定领域模式可微调。先做 Prompt 基线，按错误 taxonomy 决策；二者可组合。

### 11. DPO 与 PPO/RLHF 的训练信号有何不同？

DPO 用 chosen/rejected 对和 reference policy，把偏好优化写成分类式目标；PPO 通常先训练 reward model，再对在线采样 response 估 advantage，并用 clipped ratio、KL 等约束更新 policy。DPO 实现简单不等于天然无偏：它仍依赖偏好覆盖、reference、采样分布与超参数。回答时写清 sequence log-prob 是 response token log-prob 的和，prompt/padding 要由 completion mask 排除，length normalization 若使用就是另一个目标口径。工程追问可展示：A/B presentation order、tie 与逐标注者 raw judgment 必须保留，先 gate 未知/train case、重复 annotator-case、rubric、固定 rater 数和双顺序覆盖，再报告 raw agreement 与 Fleiss’ κ；顺序覆盖本身不证明随机化，case 内 position effect 也不自动是因果效应。trainer 只能接收经 combined audit 绑定的 binary train subset；prompt token IDs 必须是两侧完整对话 token IDs 的精确前缀，否则基于长度切片会错位；超过 `max_length` 应显式处理而不是静默截断；policy/reference 初始相同时标准 DPO loss 理论为 \(\log2\)；reference 必须冻结；tiny-pair loss 下降只证明控制流能优化，不证明人类偏好质量或安全对齐。

### 11.0.1 目标 Qwen DPO 同 batch loss 下降，为什么仍不能说“对齐成功”？

先把机制证据与质量证据拆开。仓库固定 Qwen control 的确验证了 exact pair/token/mask、真实 TRL/PEFT backward、96 个 finite gradients、冻结 parameter/state/config 指纹，以及两条 positive reference-relative margin；但只有两条 authored `good/bad` pair 和一次 step，没有人类标注、held-out prompt、置信区间、通用/安全回归或线上结果。还要注意 reference 的“身份冻结”不等于每次浮点 replay bitwise：adapter forward 内实测 disabled 且 state/config exact，log-prob replay 仍有 `0.547077` drift。正确回答应披露 drift 并以同一 reduction/mask 重算 loss，不能把它解释成 reference 权重更新，也不能只挑同 batch loss 下降来替代泛化评测。

### 11.1 Reward model 训练准确率很高，为什么仍不可信？

Pair accuracy 只回答 RM 能否排序当前 pair，不能证明它依据了正确理由。若 chosen 在训练集中总是更长、格式更整齐或语气更自信，模型可利用 length/style shortcut 获得满分；同分 margin 还必须明确按 tie、半正确还是错误处理。先保留 train/held-out 分离与 pair binding，再构造事实质量不变但长度、格式、身份线索反转的 counterfactual pairs，报告 loss、margin、strict accuracy、tie count 和按属性切片的结果，并用 held-out human preference 与真实 task verifier 校验。Reward scale 只在当前 checkpoint、normalization 和数据分布内有意义；policy 会产生 OOD response，甚至主动放大 RM 漏洞，所以还要做 adversarial/OOD slice、不同 RM 或人工复核、KL/score 分布监控和 reward-hacking red team。工程上还要防止 `RewardTrainer` 静默过滤超长 pair：在模型加载前用目标 tokenizer 渲染两侧完整序列并拒绝超限，trainer 准备后再次核对 pair 数。仓库的线性 NumPy control 演示 confounding；随机 tiny GPT-2 control 展示 lexical shortcut；本地 checkpoint 又实际贯通正式 RewardTrainer/LoRA optimizer 和非零 adapter 保存。这些证明数学、捷径和框架链路，仍不证明目标 RM 或真实人类偏好质量。

### 11.2 GAE 中 terminated 与 truncated 为什么不能共用一个 done？

GAE 要区分 value bootstrap 与 advantage continuation：

\[
\delta_t=r_t+\gamma b_tV_{t+1}-V_t,\qquad
A_t=\delta_t+\gamma\lambda c_tA_{t+1}.
\]

真正 `terminated` 的吸收状态令 \(b_t=c_t=0\)。time-limit/collector `truncated` 若保留了有效 next state/value，可令 \(b_t=1\)，但它仍是 episode boundary，所以 \(c_t=0\)；否则下一条轨迹的 reward 会泄漏回来。Padding 同样不能进递推或均值。进一步追问 PPO：clipped ratio 只改变已采样 action 对 surrogate 的贡献，`clip_fraction` 或 sampled KL proxy 都不是完整 token/sequence 分布 KL 的硬上界；正 advantage 主要裁 ratio 上界，负 advantage 主要裁下界。

### 11.3 一个可信的 PPO smoke 至少要冻结什么、验证什么？

Rollout 必须绑定生成它的 behavior-policy revision、逐动作 old log-prob、value、reward、terminated/truncated/EOS/padding 与采样配置；多 epoch 更新期间 old log-prob、advantage 和 return 不能随当前 policy 重新计算。最小控制要证明：stored log-prob 可由 rollout snapshot 重算，policy/value 参数确实更新，ratio/clip fraction/KL proxy 分母明确，loss 有限，精确或独立评测 objective 改善。只看训练 reward 不够，因为它可能来自采样噪声、RM 漏洞或数据变化。仓库两状态 control 有可枚举 expected return，适合证明 optimizer 链路；它没有语言模型、RM、reference KL、截断环境或多 seed，因此不能支持“完成 RLHF”或“训练稳定”。

### 11.4 Sampled log-ratio penalty 与完整 KL 有什么区别？

对从 behavior policy 采到的动作，\(\log\pi_{old}(a\mid s)-\log\pi_{ref}(a\mid s)\) 是单样本量，可以为负；在固定 state 下对 \(a\sim\pi_{old}\) 取期望，才等于 \(D_{KL}(\pi_{old}(\cdot\mid s)\|\pi_{ref}(\cdot\mid s))\ge0\)。有限 rollout 的 mean 仍有采样误差，state 分布也由 policy 决定。面试中应分别说明 sampled estimator、对全部 action 求和的 categorical KL、逐 token 聚合和整条 sequence-distribution KL，不能只说“算了 KL”。仓库 tiny Transformer PPO 同时输出 sampled log-ratio mean 与 sampled states 上的 exact categorical KL，但词表仅 6、horizon 仅 2，不证明大词表长序列的估计质量。

### 11.5 `max_new_tokens` 截断后 GAE 一定要 bootstrap 吗？

不一定。`length`/truncated 是 collector 或生成 API 的停止原因，\(b_t\) 则由训练 objective 的 continuation semantics 决定。若任务 return 明确定义为“只评价 generation cap 内的 response”，cap 后没有纳入目标的 reward，默认不应借一个 learned next value 改写这个有限时域目标；若 cap 只是观察窗口，底层 MDP 继续且保存了与同一 objective 匹配的 next state/value，才可以 bootstrap。两种情况都不能让 advantage 跨到下一条 episode。回答时应同时给出 stop reason、reward horizon、next-state 构造、value revision 和 bootstrap/continuation mask，而不是把一个 `done` 布尔值直接取反。

仓库 text PPO control 会计算截断后的 post-action value，但有限两 token oracle 默认不使用它；显式打开 bootstrap 时，报告会标记 optimizer 与该 oracle 不一致。这个反例也说明：训练曲线改善与独立精确 objective 改善必须分开检查。

### 11.6 RM 训练准确率 100%，PPO reward 又上升，能否证明对齐改善？

不能。RM train accuracy 只约束已标注 pair 的相对顺序；pairwise loss 甚至不识别全局 score offset。Policy optimization 会改变 response distribution，主动搜索训练 support 外的高分区域，因此还要报告 held-out/counterfactual preference、完整或抽样的 policy-support coverage、独立 task verifier、score/长度/格式切片、KL 与人工复核。减去 chosen/rejected midpoint 只能固定一个 offset 约定，不改变排序，更不能校准 OOD utility。

仓库 learned-RM PPO control 给出可穷举反例：唯一训练 pair 达到 accuracy=1、margin=5.57，但 chosen 在 57 条 allowed response 中只排第 38，最高分是 `good., good`；PPO 把精确 RM proxy 从 2.739 提到 4.652，同时把严格 `good, EOS` 成功率从 \(1/64\) 降至 \(4.99\times10^{-4}\)。dense partial credit 反而从 \(15/64\) 升至 0.566，所以回答时还必须说明“哪个外部 objective 恶化”，不能宣称所有任务指标都下降。因为 support 完整枚举且 RM/reference 冻结，这能证明该 authored tiny control 中相对严格目标发生 proxy exploitation；它仍不是人类效用、目标模型或生产 reward hacking 的实证。

### 11.6.1 单样本准确率大于 50%，self-consistency 多数票为何仍可能变差？

“\(p>0.5\) 时多数票改善”要求候选 correctness 在相应层级近似 i.i.d.。若每题先有共享 latent difficulty R，候选只在给定 R 后条件独立，那么跨题平均 \(\bar p>0.5\) 不足以确定多数票；hard items 的 \(p_R<0.5\) 会随 N 更稳定地投错。还要区分二元 majority 与开放文本 plurality：多个错误答案如何 canonicalize 会改变票数。

精确反例中，independent 场景 \(p=0.6\)；correlated 场景等权混合 easy \(p=0.9\) 与 hard \(p=0.3\)，边缘 \(\bar p\) 同样是 0.6，但候选 pairwise correlation 为 3/8。N=11 时多数票分别为 0.75349813248 和 0.53896454244。面试回答应进一步提出逐 item 保存 N 个候选、固定 canonicalizer、按 item cluster 估计 paired effect/区间、报告无效/拆票/平票，并把 model calls、tokens、latency 与 cost 一起计入；这个 authored binary toy 没有运行模型或证明真实系统退化。

### 11.7 oracle@N 与 verifier-selected@N 为什么必须分开？增加 N 一定更好吗？

oracle@N 问 N 个候选中是否至少存在一个 target-success candidate；selected@N 问实际 selection policy 最终返回的候选是否成功。若单次成功概率为 \(p_s\) 且候选 i.i.d.，前者是 \(1-(1-p_s)^N\)；后者还取决于 verifier 的排序误差、tie-break 和候选联合分布，不能从 oracle 推出。增加 N 给 oracle 更多机会，也给高分漏洞更多机会，因此 selected quality 不保证单调。

可给精确反例：`wrong/correct/verifier_hack` 的概率为 `0.5/0.4/0.1`、verifier score 为 `20/80/99`，只有 `correct` 真成功。N=1/4/16 时 oracle@N 为 `0.4/0.8704/0.9997178890`，selected@N 为 `0.4/0.5936/0.1852867601`，期望 verifier score 却为 `51.9/82.7841/95.4783461`。回答还应说明这是 authored finite distribution、i.i.d.、deterministic score 的闭式数学控制；没有 model/tokenizer/PRM/GPU/provider、calibration、语义或 wall-clock/cost 证据。生产实验还需固定 sampling policy 与 verifier revision，联合报告 oracle@N、selected@N、校准/OOD slice、model/verifier calls、tokens、费用、wall-clock 和 tail latency。

### 12. Scaling law 能告诉你该训练多大模型吗？

只能在数据、架构、损失和算力口径相近且已验证的区间内做经验外推。`6ND` 是 dense Transformer 训练 FLOPs 的常用一阶口径，不包含所有 attention、embedding、optimizer、通信和失败重跑成本。MoE 要区分 total/active parameters，数据要区分 unique/consumed tokens；最终决策还受推理、数据质量和产品约束影响。

### 13. MoE 为什么可能省计算却不省显存或通信？

每个 token 只激活部分 expert，active compute 可低于同总参数 dense 模型；但所有 expert 权重仍需放在设备群上，路由会引入 dispatch、all-to-all、负载不均、capacity drop 和小 batch 效率问题。比较时同时报告 total/active parameters、每 token FLOPs、显存、通信和端到端吞吐。

### 13.1 MoE capacity 与 token drop 怎样算才不会混淆？

先写 routing group：若有 \(N\) 个非 padding token、\(E\) 个 expert、top-k 为 \(k\)，一种常见教学约定是每 expert \(C=\lceil\phi Nk/E\rceil\)，但真实框架可能有不同 group/minimum/dropless/reroute 语义。再写 expert 内谁优先占 capacity，以及 gate 在 top-k 后、drop 后是否重归一化。

分母必须分开：top-2 的一个 assignment overflow 不等于整个 token 被丢；报告 `dropped assignments / (Nk)` 和 `all-assignments-dropped tokens / N`。Balance auxiliary 也不是跨实现通用公式，需说明使用 pre/post-capacity count、top-1/top-k、probability、stop-gradient 和 reduction。CPU routing/control-flow 通过不证明 all-to-all、GPU grouped GEMM、目标模型质量或吞吐。

仓库的 trainable CPU control 给出一个同路径例子：`factor=0.5` 时 capacity=2，counts `[4,3,3]→[2,2,2]`，4/10 assignments 被丢；capacity-enabled sparse/dense 输出和全参数梯度仍对齐。它还分别执行 drop 后重归一化、保留丢失 mixture mass 与全丢 token，后者的 routed expert 输出为零。

第二个 fixture 显式排除一个 padding token，并把四个 active tokens 分到两个 2-token groups；每组 capacity=1，grouped sparse/dense 前后向仍精确对齐。改成单一 group 后 capacity=2、kept assignments 改变，输出最大差约 `0.329387`；这说明 capacity 必须写清 group 边界。逐组 balance/z diagnostics 按 active-token 数加权，padding 不进 output、aux 或 hidden gradient。

第三个 fixture 把 overflow policy 也拆开：四个 top-1 assignments 都先选 expert 0、capacity=2。Drop 得到 `[2,0,0]` 并丢 2 个；本仓库的 deterministic full-ranking reroute 按原 gate score/token/rank 扫描备选且禁止 token 内重复，得到 `[2,0,2]`、无 drop/excess；dropless 保持 `[4,0,0]`、不 drop，但报告 expert 0 超额 2。回答时必须把“实际 dispatch”与“原始 selected top-k”分账，也要说明 rerouted gate 是重归一化，还是保留相对原 selected top-k mass。CPU int64 group IDs 不证明真实 distributed collective；authored reroute/dropless 也不证明任意框架或目标 MoE 实现。

第四个 fixture 回答“有 collective 是否就等于 expert parallel”：两个 Gloo ranks 用 `all_gather` 形成 4-token replicated global routing batch，`all_reduce` 观察 global count=4/selected `[4,0]`。Local-only capacity=1 会跨 ranks 共保留 2 个；global competition 只保留全局最高分的 1 个，mask `[F,F,T,F]`。这证明 capacity group 的通信边界会改变 dispatch decision，但 router/experts 在两边复制，没有 token-to-owner `all_to_all`、distributed backward 或 optimizer。面试中应分别回答 capacity-group collective、expert dispatch collective、gradient collective与它们的进程组/分母，不能用一个 `all_reduce` 概括全部 EP。

第五个 fixture 回答“为什么 all-to-all 返回后仍要 metadata scatter”：source 按 owner 打包后，return collective 只保证按 split/source chunk 到达，不保证恢复 source-local token order。当前 rank 0 的 arrival global IDs 为 `[1,0,2]`；必须携带 source rank、source local index、global token id 与 expert id，按 local index scatter 后再 combine。否则即使每个 expert output 都正确，最大输出差仍为 `0.8958737432590591`。同时要区分 logical tensor payload 不等于 wire bytes：当前 416-byte 账本只算张量 numel×dtype，不含协议、分包、对齐和 allocator，也不能据此声称 NCCL/GPU 性能。

### 13.2 owner-only MoE backward 中哪些梯度需要 collective？

先按参数/激活的 ownership 回答。Forward 的 token 与 gate 到 owner，output/gate 回 source；backward 要反向交换 splits，把 output/gate gradient 送回 owner，再把 hidden/gate gradient 送回 source。某个 owner expert 已看见全 expert-group 发来的本 expert tokens；若 loss contribution 已按同一 global-token 分母缩放，该 expert 参数梯度不应再全局 all-reduce，否则会重复计数。Replicated router 的每个副本只从本 source tokens 获得 gate gradient，所以 router 要跨 source ranks 求和，之后副本才能做一致 optimizer step。

仓库控制用两个 ranks 得到 local router gradients `[[1.8045724],[-1.8045724]]` 与 `[[0.4858569],[-0.4858569]]`，SUM 后和单进程 global oracle 相同。高质量回答还要声明 process groups、loss reduction、expert replication/sharding、top-k/capacity 与参数是否共享；当前 authored CPU/Gloo fixture 没有 DDP、capacity、混合精度、CUDA/NCCL 或目标模型，不能把这条归约规则无条件套到另一种 EP/DP 拓扑。

#### 13.2.1 capacity drop 与 expert-parallel backward 放在同一图时，零 token rank 能跳过 collective 吗？

不能按本 rank kept count 条件跳过。Collective 要求 process group 中各 rank 以兼容顺序参与；某个 source 的全部 assignments 被 drop，只说明它发送/接收零行，不代表其他 owner 没有来自别处的工作。Autograd 图还必须保留从最终 loss 到 empty returned payload 的 zero-size collective graph edge，否则该 rank 不会触发 reverse all-to-all，peer 可能永久等待。生产实现还要核对 backend 对 zero splits 的支持、collective ordering、异常传播与 watchdog，不能只看 tensor shape。

仓库固定 control 的 global keep mask 是 `[F,T,T,F]`，source→owner splits 为 `[[1,1],[0,0]]`；rank 1 source 零 dispatch/return，却仍参加两次 backward payload collective。Dropped token 的 routed output 与 task hidden gradient为 0，rank-1 router local gradient为零，SUM 后梯度与单进程 capacity oracle一致。这只证明 CPU/Gloo authored fixture；不证明 NCCL、DDP、目标 MoE、容错、收敛或性能。

### 13.3 Hard top-k 不可导，router 为什么还能训练？

Top-k expert indices 是离散选择，普通反向传播不会穿过“换成另一个 expert”的决定；但被选 gate 的 softmax probability 可继续作为 combine weight，task loss 因而能对当前选中集合内的 router logits 求梯度。Balance/z-loss 又可直接对完整 router probabilities/logits 提供训练信号。具体实现必须说明 top-k 后是否归一化、哪里 stop-gradient、是否有 noise/straight-through 或其他 estimator，不能只说“softmax 可导”。

仓库 CPU control 给出因果反例：保持相同 hard indices 与 expert task loss，只 detach selected combine weights，三个 experts 仍得到非零 gradient，router 的 task gradient却消失。另一个 collapsed top-1 control 对 hard count 使用 stop-gradient、对平均 probability 保留梯度；一次 balance step 在 assignments 仍全选 expert 0 时已经降低诊断值。这证明的是当前梯度路径，不证明最终负载均衡、expert specialization、目标模型公式、分布式通信、收敛或质量。

### 14. “支持 1M context”是否等于有效使用 1M token？

不等于。API 接受长度、训练长度、位置外推稳定性、检索/推理能力和成本是不同问题。需要位置分层的 needle、multi-hop、干扰、顺序与长输出测试，并与 RAG 或分段方案在相同预算下比较。

### 14.1 怎样评测持续学习中的遗忘？Replay 实验最容易误读什么？

固定 \(R_{i,j}\) 的含义：完成顺序任务 \(i\) 后，在任务 \(j\) 上的同方向指标。报告最终 ACC、旧任务 BWT、逐任务 peak-to-final forgetting，以及相对独立 pretraining baseline 的 FWT；同时给完整矩阵、多个 seed 和置信区间。必须声明 forgetting 是否把最终阶段纳入历史最大值，因为这决定改善时记 0 还是负值；FWT 还要求在学习每个未来任务前就评测它。

Replay 后旧任务不掉，不等于方法普遍有效。先排除任务本身不可联合求解、显式 task-id 泄漏、全量旧数据冒充有限 buffer、更新预算不等、只挑单 seed，以及隐私/删除/计算成本未计入。B 阶段的 replay 不能改变训练 B 之前已经确定的 FWT；把二任务 synthetic toy 外推到真实 LLM、安全行为或长期序列也是证据越界。

多 seed 也不自动解决外推问题：如果每个 seed 只改变初始化和 reservoir 选择，而 task/data 固定，paired bootstrap interval 只描述这些随机源。还要区分“相同 optimizer steps”与“相同样本/FLOPs/时间”；每步把 256 个新样本扩成 320 或 512 个总样本，本身就是额外训练预算。

## RAG

### 15. chunk 越大还是越小？

小块匹配精确但上下文不足；大块语义完整但噪声与 token 成本高。按文档结构、答案跨度和检索模型实验，比较 Recall@k、上下文覆盖、冗余和最终忠实度。

### 16. 为什么 dense retrieval 不能完全替代 BM25？

Embedding 擅长语义相似，BM25 擅长型号、错误码、姓名等精确稀有词。混合召回后用 RRF 或 reranker。向量分数跨 query/模型不可直接用统一阈值。

### 17. RAG 返回正确文档但答案仍错，怎样排查？

依次检查 chunk 是否含完整证据、重排是否截断、上下文是否重复/冲突、Prompt 是否要求引用、模型是否正确使用证据、引用是否真正蕴含主张。分层指标比只看最终答案有效。

### 18. 多租户 RAG 怎样防泄漏？

身份传到检索层，ACL 在召回查询前执行；索引、cache key、trace 和评测也隔离。用交叉租户 canary 做负向测试。不能先全局召回再让模型忽略。

### 19. Recall@k、MRR、nDCG 各测什么？

Recall@k 看相关文档覆盖；MRR 看第一个相关结果位置；nDCG 支持分级相关性并惩罚位置。指标选择取决于生成需要一个证据还是多个证据。

### 19.1 为什么不能把每段分别 tokenize 的长度相加作为 context 上限？

Chat template、special token 和片段边界会改变最终 tokenization，BPE merge 还可能让插入前后的总数非单调。每加入一个候选后渲染完整 prospective prompt，用固定 tokenizer/revision 重新计数，并预留输出；记录 budget、selected/dropped reason 和最终 token ids/数量。字符或 UTF-8 byte 可做透明 fixture，但不能冒充目标模型 token。

### 19.2 怎样证明 RAG 评测的是模型当时看到的证据？

用 request/case identity exact join，绑定 query hash、tenant/principals、按序 chunk id + stable source + version + content hash、实际 rendered context、tokenizer/chat/system/user prompt identity、最终 prompt token IDs、decoding/model/runtime revision、raw output 和 parsed answer fingerprint；从版本化 snapshot 重建 ACL/context，并让运行时把 manifest 写入签名或 append-only 存储。仅有自洽 JSONL 与 SHA-256 不证明真实模型调用：审计器若不重新 tokenize、验证签名或判断 raw-output→claim entailment，就必须逐项声明这些边界。

### 19.3 Reranker 为什么必须再次检查 ACL？怎样重放一次排序？

上游 candidate list 可能来自错误 cache、旧 policy 或错误 tenant；直接交给 cross-encoder 已经泄露正文，即使最终 top-k 再过滤也来不及。Scorer 前重查 tenant/principals，拒绝重复 id 与 rank 漂移，只把 visible candidates 交给模型。

重放时保存 query hash、ordered candidate id/rank/source/score、每段 content hash、scorer/model revision、tokenizer/max length/truncation、逐候选 rerank score 与最终 rank。Recorded score 能验证 identity 和排序代码，但 unsigned `scorer_identity` 不认证模型执行；没有 held-out qrels、延迟和成本对照，也不能声称相关性提升。

### 19.4 为什么先做 extractive RAG baseline？它证明什么？

它把检索、ACL、packing、引用、拒答和评测串成可执行控制组，且逐字 span 能机械验证没有从 packed source 外新增 claim。它不等于生成质量或 entailment：lexical overlap 会选错句，来源可能错，答案可能不完整。阈值要在独立校准集上选；qrels 只在生成后用于评测，不能传进在线 answer API。真实生成还需目标 tokenizer packing、raw-output trace 和独立 claim judge。

### 19.4.1 Citation syntax、source ID、evidence span 与 entailment 有什么区别？

Syntax 只问引用标记是否可解析、该引用的段落是否漏引；source-ID validity 再问 ID 是否存在于本次授权 context；span identity 进一步问 `source_text[start:end]` 是否逐字等于记录的 quote。三者都不能回答“quote 是否支持 atomic claim”。Entailment 必须另有 `supported/contradicted/insufficient` 判断、明确 judge/人工来源与代表性校准集，source truth/currentness 和 answer completeness 还要再分账。

仓库 `citation_evidence_span` 的故意反例让“The moon is cheese.”精确绑定 `Earth is round.` 中的 `Earth`，仍得 1。优秀回答应解释这是 construct boundary：strict JSON、授权 source membership、end-exclusive offset 与 quote equality 只是可重放 identity gate；case metadata 若不是从真实 ACL/corpus snapshot 产生，也不能反过来证明授权或 provenance。

### 19.5 怎样公平比较原生 RAG、LangChain 与 LlamaIndex？

先固定 canonical corpus/chunk/query/security context/top-k/qrels，让同一个 authorization-first retriever 作为排名权威；框架只承接 Retriever、Prompt 或 orchestration adapter。逐项比较 ordered document ID、正文、score/rank、metadata、最终 Prompt、模型输入/输出和同一评测，不允许三套实现各自换 embedding、切分和 Prompt 后再把差异归因给框架。

框架对象转换通过只证明字段没有相对 supplied canonical results 漂移，不证明框架默认 ACL，也不认证 supplied results 来源。LlamaIndex node 的控制面 metadata 应明确排除出默认 embedding/LLM content，但自定义 formatter 仍可能读取；LangChain prompt 也只有显式选择 `page_content` 才不会自动引入 metadata。完整结论还要加入相同 provider/checkpoint、tokenizer/template、decoding、重试、费用、延迟和并发。仓库 control 使用 deterministic extractive non-LLM answer，因此只能证明离线接口、ACL 和 identity parity，不能证明 learned retrieval/generation 质量或生产性能。

### 19.6 FastAPI 返回 504 后，为什么 RAG work 可能仍在运行？

`asyncio.wait_for(asyncio.to_thread(sync_query), timeout)` 只能取消等待它的 coroutine，不能强制终止已经运行的 Python thread、SQLite 调用或模型 kernel。如果 504 时立刻释放 semaphore，后续请求会进入，而旧 work 仍占 CPU/连接/GPU，实际并发就超过声明上限。Reference 用 `shield` 保留 task，并把 permit 延迟到后台 work 真正结束后释放；client cancellation 也遵循同一规则。

这只是诚实记账，不是强制终止：卡死 thread 会长期占位。生产上要给下游传 cooperative deadline/cancel token，使用支持取消的 driver，或在可回收进程/worker 中隔离；多 Uvicorn worker 和多副本还需要全局 admission。还要区分 queue timeout、execution timeout、下游 timeout 与 client disconnect，分别记录 all-attempt denominator，不能把 504 当作服务端 work 已停止或费用为零的证据。

### 19.7 检索为空时怎样保证拒答？只写 Prompt 是否够？

不够。模型仍可用参数记忆猜答；greedy/低温只减少采样变化，不建立 evidence constraint。先把 zero/insufficient/conflicting evidence 变成模型外的显式状态：确定性 short-circuit、受限输出 schema 或只允许从 source span 生成；模型回答后再做 citation、claim-evidence 和 policy gate，失败时拒绝发布。阈值必须用独立 no-answer/partial-answer 集校准。

证据也要分层保存：retrieval/context 为空、模型 raw output、stop reason、最终 action 和 verifier 结论。仓库固定 Qwen control 就出现“ACL 与零检索正确，但模型编造步骤”的反例；它说明组件正确不能替代端到端拒答率，也不能为了得到成功截图而丢弃首次失败。

### 19.8 为什么要区分 abstain、reject 和 error？

`abstain` 表示在调用模型前或经过证据判断后，系统确定当前授权证据不足；`reject` 表示已经观察到模型输出，但它未通过 citation/claim/policy 发布门；`error` 表示 timeout、provider failure、解析失败或结果不确定，系统没有足够证据把它叫正常拒答。三者的成本、告警、重试和产品文案不同，必须用 typed action/stage 记录，不能从某句固定文本反推状态。raw output 属于审计面，只有 `publish` action 才能授权它进入用户面。实现上还要分开 audit/public projection：前者可含 raw/finding，后者必须 allowlist；否则“拒绝发布”会被一次 `return decision.to_dict()` 绕过。

仓库对真实 Qwen attempt-1 的 replay 正好展示差异：有证据漏引是一次 generator call 后 `post_generation/reject`；空证据是 call count 0 的 `pre_generation/abstain`。但这是 counterfactual policy replay，不是 guard 与原模型运行同时执行的记录；优秀回答会主动指出它不能证明真实调用节省、语义蕴含或生产集成。

后续的**真实 guarded control** 才让 policy 与 Qwen runtime 同时执行：有证据 case 进入 `GenerationMixin.generate()` 1 次后因漏引 reject，空证据 case 的 callback/framework method 都是 0 次并 abstain。面试中仍不能把这个计数夸成内部 forward、kernel、远端请求或计费次数；它只证明固定 CPU callback 路径的 API invocation 边界。两个共享 authored corpus/checkpoint 的 query 也不是总体拒答率或质量证据，offline verifier 更不会重放模型、token IDs 与 decode。

## Agent

### 20. Agent 和 workflow 的边界？

分支确定、风险高的流程用代码状态机；模型只处理开放语义决策。开放性越高，越需要预算、审批、幂等、恢复和审计。

### 21. 如何避免重复转账或重复发消息？

模型生成稳定 call_id 不够；执行层用幂等键、参数指纹、数据库唯一约束和业务事务。崩溃恢复先查 ledger/外部状态，不能盲目重放。

### 22. 提示注入为什么不能靠 system prompt 解决？

外部内容与高优先级指令共享模型上下文，模型可能错误服从。真正边界在最小权限、秘密隔离、参数/ACL 校验、域名限制和人工审批。

### 23. Agent 怎样判断停止？

模型输出 `finish` 只是 proposal；确定性 verifier 通过才是完成。最大 decision step、模型 token、wall time、费用和外部资源预算是硬边界；连续相同 action fingerprint、`A/B/A/B`、同类错误与状态无进展是不同停止信号。审批暂停、升级、预算耗尽和安全停止不算成功。还要说明 token/费用通常在模型响应后才能记账，同步调用越过 deadline 也不能保证抢占。

### 23.1 为什么 handler 返回 completed 仍不能证明邮件已发一次？

本地 completed 只证明 runtime 记录了 handler 返回；远端可能已成功但本地落账失败，也可能 provider 接收请求后异步拒绝。反过来，handler timeout 也不证明动作没发生。要用 provider idempotency key、outbox/业务状态或 provider audit 做 effect verifier，并把 `handler_attempted`、`effect_applied` 与 unresolved pending 分开记录。

### 23.1.1 Transactional outbox 能否保证 exactly-once？

不能。它只让本地业务状态与待投递 row 原子提交。worker 用 lease 并发 claim，但 provider 成功、本地 ack 前崩溃仍会重投；lease 不是远端 exactly-once。若 provider honor 稳定 `effect_id` idempotency key，at-least-once request 才可能折叠成一个 effect；否则需查询 provider/业务状态、reconciliation 或补偿。receipt 也只是 supplied artifact，dead letter 需要 operator runbook，不能无限自动重试。

### 23.2 Agent 评测为什么不能只报一个平均成功率？

任务完成不能抵消一次越权或重复付款。task success 只在有 final-state verifier judgment 的 case 上计算；policy violation、policy over-refusal、未审批副作用 attempt、重复 applied effect 与 pending 分别保留 numerator/denominator，并作为独立发布 guardrail。分母为零是 N/A，不是 0% 风险。

### 23.3 参数 fingerprint 做成 SHA-256 后是否已经保护秘密并证明审批安全？

没有。hash 只绑定所选 canonical bytes；它不证明业务语义、权限或 payload 无秘密，低熵值仍可能被枚举。审批还要绑定主体、task、call id、scope、资源版本和过期时间，原始敏感 payload 应最小化、隔离加密并受 retention 控制。

### 23.4 为什么 cache hit 也要重新授权？proposal fingerprint 与 execution fingerprint 有何区别？

用户权限可能在第一次执行后被撤销；先查 cache 会把旧敏感结果返回给当前已无权主体。proposal fingerprint 只标识模型提出的 tool + arguments；execution fingerprint 还绑定可信 task/subject/tenant、tool contract、server-resolved resource revision 与 policy decision。每次 replay 先重新授权，同 call id 但 execution identity 改变应冲突，不能静默复用。

### 23.5 为什么 handler 返回的 dict 仍要 JSON snapshot？

handler 可能返回仍被它或调用方持有的嵌套可变对象；直接放入 cache 会产生“审批/执行后结果漂移”。先做严格 JSON 编码与 round-trip，再递归冻结；NaN、set 和自定义对象失败并保持 pending。这样只证明结果值域稳定，不证明内容真实、业务正确或远端 effect 已发生。

### 23.6 approval pause 后怎样安全恢复？

checkpoint 绑定可信 task/subject/tenant、原 loop/handler cap、累计 token/cost/active time 与 handler counter、事件/action 历史、pending decision 和 execution fingerprint。恢复先重新授权并执行同一 decision，不重新调用 planner 或重复计 usage；旧身份、清零 counter、扩大 cap、过期/漂移 grant 都拒绝。还要指出 checkpoint SHA-256 没有密钥，不防攻击者重算；文件与 ledger 需事务/一致性协议，等待审批要另设绝对 deadline，并用签名/MAC、ACL、加密、lease 和一次性 approval store 补齐生产控制面。

### 23.7 怎样把模型自由文本安全地接到 typed Agent loop？

先构造 canonical request identity，绑定 prompt/prompt revision、task state、剩余预算、tool catalog、输出 cap 与预期 model revision；transport 返回时要求 exact model revision、provider request id、input/output usage、cost 和 finish reason。然后用拒绝 duplicate key、non-finite number、Markdown fence、未知字段/工具的 closed JSON parser，只生成 `tool/finish/escalate` typed proposal。最后仍由模型外 runtime 重新执行 schema validator、server-owned resource resolution、policy、approval、预算与幂等，`finish` 也必须过 verifier。

Request/response/decision fingerprint 只能证明所列 canonical bytes 一致，不认证 provider，也不证明安全或语义正确。Tool observation 即使在 system prompt 中标为 untrusted，仍与指令共享上下文；真正安全边界是最小权限和外部控制面。Provider usage 缺失不能猜，output usage 超 request cap 要拒绝；整次 input+output 仍可能事后越过总预算，此时记账但不执行 action。Recorded response control 只证明 parser/runtime 控制流，不能说成目标模型、线上 API、账单或生产 IAM 实测。

若 Prompt 展示 JSON Schema，还要回答怎样防“文档 schema 与执行 callback 漂移”：用一个版本化 contract 同时派生 Planner catalog 与 runtime Tool，并把 schema/validator revision 纳入 request identity。显式固定 draft；限制 schema/instance bytes；不允许运行时从模型 URL 取 remote `$ref`；说明 `format` 是 annotation 还是 enforced。标准 validator 不做 coercion、不应用 `default`，schema violation 错误不能回显 secret value。即使 schema 通过，resource resolver、tenant/policy、approval 和业务 cross-field check 仍必须独立执行。

### 23.8 MCP stdio 接通后，为什么仍不能说“工具互操作和安全已经完成”？

先说明具体实现证据：固定 MCP version；client 启动 server subprocess；UTF-8 JSON-RPC 经 stdio 传输；initialize response 完成版本/capability 协商，initialized 后才 list/call tools。仓库的 authored strict control 把 schema 错误作为 `isError: true` tool result、unknown tool 作为 JSON-RPC error；official-SDK stdio control 中 unknown name 则进入应用 handler 并返回 tool error。错误分层是实现/版本/请求路径的一部分，不能在没有具体 trace 时用一句“模型重试”抹平。

再说明缺口：discovery 和 schema-valid 不建立 authenticated subject、tenant ACL、资源归属、approval、幂等或 effect verifier；tool result/prompt/resource 仍是不可信输入。仓库已有 official-SDK Streamable HTTP control 把官方 client/session manager 与真实 loopback TCP/HTTP 放进同次运行，但它没有 malformed body、Host/Origin、resumption、TLS/OAuth、远程或 conformance 负例，私有 shutdown token 也不是 MCP auth。另一个 authored HTTP control 覆盖 Origin/Bearer/session/cancel，却只是自写固定子集。完整回答应分别给 framing/lifecycle、transport、官方 conformance/SDK matrix、真实远程网络、安全控制面和业务 verifier 证据，而不是只展示一次 tools/list 成功。

### 23.8.1 官方 MCP SDK 已拒绝 schema-invalid 参数，为什么 unknown tool 仍需应用层 gate？

先区分“已发现工具的参数 schema”与“工具名注册表”。仓库用 `mcp==1.29.0` low-level client/server 的 memory control 中，带额外字段的 `fixture.add` 参数由 SDK validation 在应用 handler 前拒绝；但对未在当前 discovery 中列出的名字，client 没有 cached schema 可校验，server SDK 会进入注册的 `call_tool` handler，因此应用仍须按 allowlist 拒绝 unknown name。这个观测只适用于固定 SDK/version/control，不能泛化成所有 SDK 或 transport 的规范承诺。

即使两类请求都被拒绝，也只证明结构和名称 gate 的局部控制流。它不证明 authenticated subject、tenant、resource ownership、capability grant、approval、idempotency 或 effect verifier。memory stream 没有执行 OS pipe 或 HTTP；仓库另有 official-SDK stdio/HTTP controls 分别执行真实 pipe 与 loopback TCP/HTTP，但仍不包含 TLS、OAuth、远程 server 或 conformance suite。回答时必须把 SDK validation、transport interoperability 和业务 authorization 分开。

### 23.8.2 官方 MCP SDK client/server 已通过真实 stdio，为什么仍不是 conformance？

先列实际执行：固定 `mcp==1.29.0` 与协议 2025-11-25；官方 `stdio_client` 启动使用官方 `stdio_server` 的独立 subprocess；真实 OS stdin/stdout pipe 上执行 initialize、ping、tools/list 和三类 call；临时最小 receipt 证明 handler 只见到成功与 unknown-tool 两个名字，server 收到 EOF 后退出，stderr 为空。这比 memory control 多了 SDK+transport 的同次集成证据。

再列未执行：control 没有向官方 parser 注入 missing LF、duplicate key、invalid UTF-8、超 byte cap、stdout 污染，没有触发 forced terminate/kill、取消、deadline、并发或 server-to-client request，也没有跑官方 conformance suite。它只连接同一 Python SDK 的 client/server，不是独立实现或跨厂商矩阵；更没有 TLS/OAuth、远程主机、tenant/resource 授权、approval、effect verifier 或生产 supervisor。源码里存在某个分支不等于该分支已被本次 control 观测。

### 23.8.3 官方 MCP SDK 已通过真实 Streamable HTTP，为什么仍不是认证或 conformance？

先列实际执行：固定 `mcp==1.29.0`/2025-11-25；官方 `streamable_http_client`/`ClientSession` 连接独立 subprocess 中的 low-level `Server`、`StreamableHTTPSessionManager` 与 SDK ASGI adapter；真实 loopback TCP/HTTP 上观察 stateful session、7 POST、1 GET SSE、1 DELETE、schema-invalid 未进 handler、unknown tool 进入应用 gate，以及 manager graceful shutdown。这建立的是“同一 SDK 两端 + 真实本地 HTTP transport”的局部集成证据。

再解释两条分账。第一，随机 token 只保护测试编排的 readiness/shutdown endpoint，它在 MCP endpoint 之外，不是 MCP Authorization/OAuth、subject、tenant、scope 或资源授权。第二，control 没注入 malformed/duplicate/invalid-UTF-8/oversize body、Host/Origin failure、断网、cancel/deadline、reconnect/resumption，没有 TLS、代理、远程/跨厂商 server、multi-worker 或官方 conformance suite。method/status/media-type 计数与无密钥 receipt/hash 也不认证对端或执行来源；源码具备这些分支仍不等于本次运行覆盖。

### 23.9 A2A Agent Card、官方 SDK 与 completed task 都通过后，还缺什么？

先列实际证据：固定 A2A 1.0 与 SDK 版本；通过 well-known URI 解析 Agent Card；按 `supportedInterfaces` 选择 JSON-RPC/HTTP binding；发送 `A2A-Version`；执行 `SendMessage`/`GetTask`；用官方生成类型或冻结 schema 拒绝旧 `kind`/非法参数；把远端 completed 与本地 artifact verifier 分开。还要说明 JSON-RPC、HTTP+JSON/REST 和 gRPC 是不同 binding，loopback TCP 不等于 TLS/代理/公网网络。

再列缺口：Agent Card 是 capability 声明，不是身份或授权证明；unsigned card、endpoint DNS/TLS、credential audience、tenant、ACL、approval、幂等和 effect verifier 都要由可信控制面处理。单一 Python SDK 的两个方法没有覆盖 TCK、SSE、取消、push notification、extended card、错误/重试/超时未知结果，也不证明跨语言/厂商兼容。无密钥 schema/report hash 只能检测已选 canonical 内容漂移；不能认证发布者、进程或业务真实性。

## 评测与实验

### 24. LLM-as-judge 有哪些偏差？

位置、长度、风格、自我偏好、知识和提示敏感。交换顺序、匿名、结构化 rubric、多 judge，并用专家集校准一致性。不能只保存 adjudicated winner：逐判断绑定 case、annotator/batch、presentation order、tie/invalid、rubric revision 和盲化/独立声明。Raw agreement 的分母是同 case 内 annotator 无序对；Fleiss’ κ 还要求每 case 固定 rater 数。位置诊断应逐 case 比较 \(P(A\mid A\ first)-P(A\mid A\ second)\)，再以 case bootstrap；它不因保存顺序或区间就成为随机实验的因果估计。

### 24.1 为什么 literal exact、normalized exact 与 token F1 会给出不同结论？

它们比较的对象不同。Literal exact 直接比较 decoded string；normalized exact 先执行声明的 Unicode/case/whitespace policy；token F1 再把文本切成 token multiset，通常会丢顺序或标点信息。仓库固定 Qwen 七例中，`LLM-2026 → llm-2026` 得到 `0/1/1`，说明 `casefold()` 对大小写复制任务吞掉了真实错误；`{"answer":42} → {"answer": 42}` 得到 `0/0/1`，说明当前 whitespace normalization 没消除 JSON 内部空格，而 token regex 忽略标点/空白。七例汇总 `4/7、5/7、6/7` 不是三种可互换准确率。

回答时先定义 construct：大小写敏感 ID 用 literal/typed equality，JSON parse 后做 schema 与字段语义，抽取任务才考虑明确 normalization 的 exact/F1。再说明 metric revision、raw output、逐例失败和分母都必须保存。即使七例是真实目标权重生成，它仍是 authored、非 held-out、非代表性小集，不能外推总体模型质量。

### 24.2 JSON schema-valid、JSON value exact 与业务语义有什么区别？

先 strict parse：普通 `json.loads` 风格实现可能接受 `NaN/Infinity` 或用最后一个 duplicate object key 覆盖前值，不能直接当协议门禁。Schema 再检查类型、required、枚举和 additional properties；`{"answer":43}` 可以完全符合“answer 是 integer”的 schema，却不等于 gold 42。Value exact 比较 parsed value，需要声明 object key order/whitespace 是否忽略、array order 是否保留、integer/float 是否区分；它适合唯一 gold，不适合多个等价开放答案。

最后才是 domain semantics：金额/单位、资源归属、数据库当前状态、证据蕴含和授权无法由通用 schema/value equality证明。高质量回答还应提 metric revision、invalid schema 是 case error 而不是模型失败、local `$ref/$dynamicRef` 与 external resolution 边界，以及 `format` 是 annotation 还是 enforced。仓库 v2 schema metric 严格拒绝 duplicate/nonfinite、`$id` 与 external ref；v1 value exact 也不等于业务语义。

### 25. 为什么用 paired bootstrap？

基线和候选回答同一批 case，差值天然配对。对 case 级差值重采样得到均值差置信区间，比独立均值更贴近问题。但用户/文档内多个 case 相关时应按 cluster 重采样；逐行 bootstrap 会虚增有效样本量。Bootstrap 改善概率是重采样统计量的比例，不是“候选真实更好的后验概率”。Percentile interval 也不自动具有小样本、偏斜分布或重尾下的标称 coverage。

### 25.1 Cluster bootstrap 的 case-weighted 与 equal-cluster 路径怎样算？

每次有放回抽 \(G\) 个完整 cluster。Case-weighted target 用抽中 cluster 的 difference sums 作分子、抽中 cluster sizes 总和作分母；分母必须随 resample 改变。Equal-cluster target 则平均抽中 cluster 的 mean differences。前者回答平均 case，后者回答平均 cluster，不能看 outcome 后切换。

小 \(G\) 时可枚举 \(G^G\) 个 ordered resample 来消除 Monte Carlo 误差，但这不会制造更多独立 cluster，也不保证 percentile coverage。报告 cluster 数与 size 分布、weighting、quantile method、resample 数/seed、最大 cluster sensitivity 和 interval；cluster 定义、独立/代表性、interference 与抽样设计仍需外部证据。

工程门禁还必须把这些选择写进不可静默漂移的 artifact。仓库 comparison v2 在 root 绑定 `unit`、cluster metadata key、weighting、exact threshold、requested Monte Carlo samples/seed；每个 overall/slice result 单独保存 cluster sizes、estimand、`exact|monte_carlo`、实际 resample 数和有效 seed。因为不同 slice 的 cluster 数不同，method 不能只在 root 写一次。Artifact-only 验证不重开 cases，也不会证明 metadata key 真的是正确 sampling unit。

### 25.2 Paired randomization/sign-flip test 与 bootstrap 有什么区别？

Sign-flip 在 sharp null 与 pair-label exchangeability 下翻转每个非零差值符号，计算 observed mean difference 的尾部概率；\(m\) 个非零 pair 可 exact 枚举 \(2^m\) 项。零差值留在 mean denominator，但无需把相同 assignment 加倍。大样本 Monte Carlo p-value 用 `(extreme+1)/(draws+1)`，不能报伪精确 0。

它给 p-value，不给 effect-size 置信区间；p-value 不是 null 为真概率，也不保证业务重要。单/双侧需预注册，cluster correlation、多个 metric/slice、反复改 Prompt 和数据选择分别要求正确随机化单位与 multiplicity/selection 处理。仓库 authored CPU fixture 没有证明真实 case 抽样、交换性、因果或模型改善。

### 25.3 同一用户有多条 case，怎样做 cluster sign flip？

若用户才是可交换单位，同一用户内所有 case difference 必须共享一个正负符号，不能逐行独立翻转。这样不要求用户内 case 独立，但仍要求 cluster-level label exchangeability、cluster 间可独立组合，并需要足够多且有代表性的 cluster。

还要先选 estimand：case-weighted 用每个 cluster 的 difference sum、总 case 数作分母，大用户权重更高；equal-cluster 用每个 cluster 的 mean difference、cluster 数作分母，每个用户等权。它们不是两种“标准误算法”，而是两个问题。仓库反例中 5 个 `+1` 属于用户 A、1 个 `-1` 属于用户 B：逐 case greater p=7/64，case-weighted cluster-joint p=2/4，而 equal-cluster observed difference=0。不能看完三项后挑最好看的一项；报告 cluster 定义、size 分布、weighting、干扰/抽样假设与 effect size。

### 25.4 同时评测多个 metric/slice 时，Holm correction 怎样算、不能解决什么？

先在看结果前定义 family，把 \(m\) 个有效 p-value 升序排列。第 \(i\) 项先乘 \(m-i+1\)，再对这些 scaled value 做前缀最大值并 cap 到 1；最后映回原 hypothesis。即

\[
\tilde p_{(i)}=\min(1,\max_{j\le i}(m-j+1)p_{(j)}).
\]

在每个 component p-value 有效的前提下，Holm 对任意依赖结构控制 FWER。不能漏掉 running maximum，也不能把 adjusted p-value 解释为 null posterior。它不修复事后挑指标/family、可选停止、测试集调参、错误的 cluster unit 或本来就无效的 p-value，也不回答效果大小和业务重要性。面试中还应说明为何这些检验属于同一 family、报告全部原始/adjusted p-value，而不是只展示显著项。

### 25.5 为什么每周看一次 `p < 0.05`、显著就停止会出错？怎样控制？

固定样本 p-value 只对预定的单次分析保证其 null 口径。重复 looks 的事件是“任一次越界”，各次结果相关但 union probability 仍会累积；不能把最后一次的 0.05 当成整段实验的错误率。仓库的无 tie、i.i.d. fair-sign exact oracle 在 `n=[10,20,30,40,50]` 五次 look 上得到 naive `p<=0.05` 首次拒绝概率约 0.1010，而只看最终 n=50 的实际离散拒绝概率约 0.03284。

最简单的事前控制是确认最多五次 look，并用 Bonferroni 把 familywise 0.05 分成每次 0.01；同一离散 fixture 的实际总体错误约 0.01522，union bound 保证不超过 0.05。更高效的设计可用 group-sequential boundary、alpha spending、always-valid p-value/e-process 或 confidence sequence。必须预先记录最大样本/时长、look schedule、随机化单位、主指标、停止/异常中止规则和全部 looks。Bonferroni 很保守，不允许临时增加 look，也不修复 effect estimate 的停止选择偏差；fair-sign toy 更不证明真实 case 独立、cluster 正确、抽样代表性、因果或业务收益。

### 26. 测试集反复调 Prompt 有什么问题？

测试集变成事实上的开发集，结果乐观。保留隐藏集、滚动新鲜集和时间切片，记录试验次数与版本。

### 27. 线上总体提升但中文用户下降怎么办？

分层结果必须显式报告；关键群体设 guardrail，不能用总体均值抵消。检查数据、路由、tokenization、Prompt 和 judge 语言偏差，再决定阻断或局部发布。

### 28. pass@k 测的是什么？为什么不是线上单次成功率？

当同一任务有 \(n\) 个采样、其中 \(c\) 个通过测试时，无放回选择 \(k\) 个至少一个成功的估计是：

\[
\operatorname{pass@k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},\quad 1\le k\le n
\]

它描述给同一任务最多看 \(k\) 个候选的能力；线上只生成一次时更接近 pass@1，还受采样、测试覆盖、任务分布和选择策略影响。测试通过也不证明补丁安全或满足未编码需求。

### 29. 怎样校准 LLM judge，而不是直接相信它？

先定义 rubric 和专家 gold subset；测 pairwise/分类的一致性、各 slice 混淆、位置交换与重复运行稳定性。judge model、prompt/template 和解析器都要版本化。高一致性仍只说明与该专家协议接近，不证明真实用户价值。

### 29.1 baseline 与 candidate 的 case id 完全一致，是否足以做可信配对比较？

不足。同一 ID 下的 input、gold/rubric、slice、metadata 可能已变，metric 同名实现也可能升级。run manifest 应绑定 ordered case 全语义、ordered result、recorded output、metric revision、scorer 与 system revision；compare 在统计前重算并 fail closed。canonical SHA-256 只能检测已列内容相对可信 manifest 的变化，unsigned manifest 和自报 `system_id` 不证明输出来自该模型、评测集代表真实分布或指标有效。

最终 gate report 还要绑定 bootstrap seed/sample/confidence、质量/安全/延迟与 slice 阈值、两侧 manifest、统计结果和全部失败原因。否则同一组分数可在事后换阈值或改 `passed`。comparison fingerprint 能发现相对可信锚点的局部漂移；若 JSON 与 hash 一起被重写，仍需签名或 append-only 发布记录发现。

### 29.2 HMAC 链通过，是否已经证明评测发布历史没有被回滚？

没有。先用每条记录声明的 `key_id` 从可信 resolver 取 key，按固定 canonical/domain-separated 规则重算 MAC，并验证连续 sequence 与前序 MAC；这只证明相对这些共享密钥的链。再用 artifact-id→path 的精确映射重读 size/hash，才证明当前引用 bytes 一致。最后必须从 ledger 外取得 trusted head（至少 sequence+MAC）并匹配，才能发现删除尾部或换回旧的合法 snapshot；否则任何有效前缀都能通过。

还要说明 HMAC 不提供公钥签名的不可否认性，key custody/轮换/吊销要靠 KMS/HSM 与权限协议；被 MAC 的 timestamp 只是 caller 字符串，不是可信时间。Exclusive-create/file-fsync 也不证明 parent directory durability、目录原子发布或 verify 后文件不变。面试中的完整方案应包含外部 transparency/object-lock anchor、消费端强制验证和 TOCTOU 边界，而不是只说“用了 hash chain”。

### 29.3 Artifact-only 校验与完整证据图复算有什么区别？

Artifact-only 只检查最终 comparison 的 schema、内部算术/判定和无密钥 fingerprint，不知道引用的 manifests/results/cases 是否存在或匹配。完整本地复算还要重开 cases、两侧 answers/results/manifests；按 case 顺序重算 answer/case identity，要求 metric revision 可由当前实现精确执行，重新评分，并从记录的 resampling/gate 配置重跑统计与最终 decision。这样能发现单文件自洽但跨文件错误的 score 或 summary。

边界仍要说完整：复算没有重新请求模型/provider，不认证 `system_id` 或本地文件来源；能改全套 bytes 的攻击者仍可协同重写。它也不验证 sampling/cluster 假设、指标 construct validity 或线上效果。可信链路需要把“计算可复现”“artifact 被认证”“外部 head 未回滚”“真实 runner/model 执行”和“统计/业务有效”作为五个不同命题。

### 29.4 评测 HTML 报告怎样避免成为安全漏洞或“证据升级”？

只从严格加载的 artifact 派生；system id、slice、reason 等所有动态文本统一 HTML escape，不拼接可信标签。页面不需要 JavaScript 或外部资源，使用 restrictive CSP，并以文字而非只靠红绿颜色表达判定；cluster/case 结果按 schema 分支展示，不能把字段缺失默认为 0。对闭合 table cell、`script/img/onerror`、超长文本和 Unicode 做回归。

更重要的是 scope：报告 receipt 应写 `artifact_only_render`、`statistics_recomputed=false`、`artifact_authentication_verified=false`。HTML 是可覆盖的展示物，不是 canonical decision artifact；消费方必须回到 JSON verifier、完整 evidence recomputation 和独立认证链。XSS-safe 只证明渲染边界，不证明数据、统计或发布决定正确。

## 系统与故障

### 30. LLM 服务 p95 TTFT 突然升高如何排查？

先确认计时起点：负载生成器 `offered_at`、HTTP dispatch、网关进入、engine 入队、prefill 开始不能混成一个 timestamp。若请求在 client concurrency semaphore 前等待、取得槽位后才开始 TTFT，测试工具本身会隐藏排队。再分解 client/gateway/engine queue、prefill、外部检索与网络；看输入长度/并发分布、batch 调度、长 prompt、GPU 利用率、KV 容量和 preemption。p50 正常而 p95 高通常提示排队或长请求干扰，但只有 client trace 不能定位服务端根因。

### 30.1 Closed-loop 与 open-loop 压测有什么差别？

Closed-loop worker 通常等上一个请求完成再发下一个，服务变慢时 offered load 会下降，容易漏掉饱和后的排队。open-loop 按外生 constant/Poisson schedule 到达，completion 不改变后续时刻，更适合扫容量 knee；但还要证明 generator 跟得上。将 scheduled timestamp 记为 `offered_at` 可把 event-loop 迟到计入 client queue，不代表请求真的准时 dispatch，也不能把 client queue 当 server queue。有限 seeded schedule 只证明该轮到达工件可复现，不证明生产流量分布。

### 30.1.1 怎样设计 LLM serving 的 admission 与 backpressure？

不要只放一个 request-count semaphore。至少同时限制 active sequences、预计 token/KV capacity、queue 条数/总工作量和最大 queue age；普通 causal generation 可用 `estimated_prompt + output_cap - 1` 做一阶 token-position reservation，但它不是 GPU 秒或显存，beam、prefix hit、speculative、preemption/recompute 和 padding 都会改变实际工作。状态应显式区分 offered、queued、admitted、running、terminal；只有 queue 中确定未 dispatch，或 backend/scheduler/allocator 已证明释放，才能归还对应 capacity。504/client disconnect 本身不够。

再回答公平性与范围：FCFS 会 head-of-line blocking，长度分 lane 可能饿死长请求，priority 需要 trusted policy、aging/tenant quota；请求体不能自报 critical。Replica-local semaphore 不等于服务级全局配额，多 worker/副本要分账或使用可验证的全局 admission。超载时快速 429/降级通常优于无限排队，但 429 仍进入合格 offered 流量的失败分母；retry hint/backoff 还要防 retry storm。最后用固定长度/租户/cache 分布的 open-loop sweep 联合观察 success rate、queue、TTFT/TPOT、KV/preemption 和 GPU，不能只取吞吐峰值。

### 30.2 Paged KV 为什么仍会碎片化？Prefix sharing 怎样安全追加？

固定 block 避免每请求预留最大连续区，但每条序列 tail 仍可能未填满，block size 是 metadata/调度开销与内部碎片的权衡。Prefix fork 让多张 logical block table 引用同一 physical block；full shared tail 后追加只分配新块，partial shared tail 必须 copy-on-write，否则父子序列互相污染。

Append 应先算出并预留“COW replacement + 填满 tail 后仍需的新块”，容量不足则在 occupancy/length 改变前整体失败。指标分开 logical tokens、physical materialized token positions、logical references、allocated blocks 和 internal fragmentation；logical tokens 会重复共享 prefix，不能直接当物理使用量。CPU metadata 模拟通过不等于真实 KV copy、GPU page table、preemption 或性能通过。

### 30.3 Continuous batching 中 prompt+output 为什么不等于模型 forward token work？

标准 causal LM 的 prefill 最后一个 prompt position 已产生首个输出 token 的 logits；若请求 \(i\) 有 \(P_i\) 个 prompt token、实际发出 \(O_i\ge1\) 个输出 token，且没有 prefix reuse、speculative verification 或 beam，一阶 forward positions 是 \(\sum_i(P_i+O_i-1)\)。API usage 仍可报告 prompt+completion token，不能因计算口径减一就修改计费/输出分母；padding、prefix cache、speculative、kernel 和 scheduler 还会让物理工作不同。

回答调度题时继续固定 arrival/admission boundary、sequence cap、token budget、prefill chunk、decode/prefill priority、preemption 与首 token boundary。离散 CPU state machine 可证明某份 policy 的 queue/TTFT/TPOT step 和 work conservation，但 step 不是秒、token-slot utilization 不是 GPU utilization，也不证明某版 vLLM scheduler 或吞吐。

### 30.4 KV 抢占为什么会让实际 forward work 大于 `prompt+output-request_count`？

无 prefix reuse/speculation/beam 时，后者是 logical causal positions。Recompute preemption 释放某条 sequence 的 KV；恢复时要重跑已经处理的 prompt/历史输出 positions，所以 `executed = logical + recomputed`。重建 KV 不应重复向用户发 token。仓库反例把 logical 9、recomputed 2、executed 11 分账，B 的输出仍只有 boundary 2/6。回答还要说明 victim/priority、当轮保护、完成释放、单请求能否独占放下、swap 与 recompute 的区别；CPU block trace 不证明目标 vLLM、VRAM 或时延。

### 30.4 Prefix cache 为什么不能只用 prompt 的 SHA-256？

因为“同 hash”既不是授权，也不表达完整执行 identity。候选必须绑定可信 tenant/visibility domain、authorization/policy revision、model/tokenizer/template/adapter revision、RoPE/position config 与 KV dtype；cached token ids 还必须逐项等于请求 token ids 的某个前缀，多个候选取最长。Fingerprint 只能缩小候选桶，命中前仍做 full identity/token comparison，因此碰撞不能导致复用。

并发系统还需 lease/refcount pin 使用中的 entry，LRU 只能淘汰未 leased 项；满容量且全 leased 应在 mutation 前失败。Unkeyed hash 不隐藏低熵 prompt，cache timing 也可能泄露前缀是否存在。仓库 collision fixture 只证明 metadata 状态机，不证明真实 K/V、vLLM policy、VRAM、prefill savings 或 timing-channel mitigation。

### 30.5 Beam search 为什么不保证找到全局最高概率序列？length penalty 又怎样改变答案？

有限 beam 只按当前 prefix 累计 log probability 保留 (B) 条，较差 prefix 以后可能接上高概率后缀，但已经无法恢复。反例：root 的 `A=0.6,B=0.4`；下一步 `A→EOS=0.51`、`B→EOS=1`。beam 1 返回 `A,EOS`，概率 0.306；beam 2 才保留并返回 `B,EOS`，概率 0.4。因此增加宽度只扩大搜索，不构成任意生成树上的全局最优证明。

若最终分数约定为 (s=\log p/T^\alpha)，由于 log probability 为负，正的 \(\alpha\) 会让较长候选的分数更接近 0，可能翻转 raw-probability 排名。答题时必须先定义 (T) 是否含 prompt/EOS，再说明 active pruning 用 raw 分数还是 normalized score、EOS 何时进入 finished set、finished-candidate cap、tie-break 和 early stopping。仓库 oracle 的 (T) 只计生成 token、含 EOS、不含 prompt，保留所有从 active prefix 产生的 EOS且无 heuristic early stopping；不能把这份教学契约说成 Transformers、vLLM 或云 API 默认。

### 30.6 约束解码为什么不能只检查 token 的第一个字符？

Tokenizer token 可能一次解码成多个字符或 bytes。若 grammar 在当前状态允许 `1`，token `1]` 的首字符检查会通过，但完整转移可能在 `]` 处失败；必须计算 token 完整片段的 (delta^*) 并据此屏蔽。屏蔽后对合法 token 的原始概率质量重新归一化；合法集合质量为零要显式 constraint error，不能解除约束。EOS 只在 accepting state 合法，而“状态已接受”与“请求已 EOS 完成”也要分开，length 截断仍是不同 finish reason。

工程上每条 beam/sequence 都携带 parser state，缓存 allowed-token mask 时绑定 grammar 与 tokenizer revision。JSON Schema/CFG 约束最多保证被编码的语法属性，不保证字段事实、数据库 ID、权限或副作用安全。仓库 toy 只对 supplied Unicode text fragment 和有限 literal trie 做 CPU 验证，不执行真实 tokenizer bytes、完整 JSON Schema、模型、GPU 或 provider runtime。

### 30.7 怎样证明一个 OpenAI-compatible demo 真的调用了目标权重？

把证据拆成 artifact、load、network、request 和 server execution 五层：模型绑定 immutable revision 与逐文件 size/hash；loader 只从已验证 snapshot 读取并记录 class/dtype/parameter count；client 与独立 server process 走真实 socket；正负请求覆盖 auth、model、closed schema、non-stream/SSE、usage/finish；server-side audit 记录 framework generation 次数和精确 token projection。只展示 curl 200、mock transport、model-id 字符串或 client response 都不能证明目标 checkpoint 被加载执行。

还要主动缩窄结论。仓库 fixed-Qwen control 证明的是 Transformers CPU FP32 eager、IPv4 loopback、单 worker 和两次 `GenerationMixin.generate()`；SSE 在完整生成后才发送，所以不证明 incremental decode/cancel。它没有 vLLM/CUDA、TLS/OAuth、远程、多 worker、性能、质量、完整 OpenAI compatibility 或来源签名，recorded self-hash 也只用于一致性检查。

### 31. 4-bit 模型为何不一定更快？

取决于硬件内核、反量化、group size、未量化层和 batch。若计算不是权重带宽瓶颈或 kernel 不成熟，文件更小不代表端到端更快。还应区分三个数字：裸 code 的理论 bit 数、含 scale/zero point/alignment 的 artifact 或 resident bytes，以及运行时峰值显存。

对 \(R\times C\) 权重、bit width \(b\)、每个 contiguous row group \(G\) 保存一个 FP32 scale，理想下界是 `ceil(bRC/8) + 4R*ceil(C/G)` bytes；它仍没含容器、未量化层和 workspace。真正达到 code 下界还需固定 signed-code 映射、bit/endian 顺序、padding 与 alignment；完成 CPU dense bit packing 也不代表 runtime 采用同一 layout。若单矩阵格式再加 32-byte header 和 32-byte digest，文件大小就是 raw payload + 64 bytes；小 fixture 的相对开销会很大。小 group 通常降低局部量化误差，却增加 metadata。CPU 上 strict reload/unpack/反量化后做 FP32 matmul 只能验证 artifact 和数值 plumbing，不能声称执行了 int4 kernel 或得到加速；unkeyed hash 也不是签名。

### 31.1 多矩阵量化 bundle 为什么仍不等于完整模型 checkpoint？

“能按 tensor name 重载多个量化矩阵”只解决 state serialization 的一个子集。完整 checkpoint/runtime artifact 还要定义并验证 architecture/config 语义、tokenizer vocab/merges/chat template、embedding/bias/norm 等未量化 state、tied weights、dtype、shard index、device/runtime layout 与 model forward compatibility。只有 tokenizer id/revision 不能恢复 tokenizer；只有 architecture JSON identity 也不会自动提供可执行 forward。

还要区分三层完整性：每个 tensor digest 检测 tensor 漂移，outer digest 检测 bundle body 漂移，签名或受控发布链才可能认证来源。攻击者能协同改内容并重算 unkeyed SHA-256。Exclusive create 防覆盖，但写入中崩溃可留下 partial target；file `fsync` 也不等于 parent-directory durability 或断电原子发布。仓库 two-layer NumPy fixture 只证明严格 manifest、多 tensor round trip 和 control flow，不证明 GGUF/safetensors 兼容、LLM 质量、resident VRAM、fused kernel 或加速。

仓库进一步给出的 repo-native MiniGPT checkpoint 展示了怎样跨过这条边界：保存 Byte-BPE merges、严格 config、全部唯一二维/一维参数和 tied-weight contract，并由固定 architecture revision loader 恢复 causal forward。但“完整”仍是相对契约：base-byte mapping 与 forward code 来自 trusted repo；normalizer/special/chat template、训练状态、sharding/device layout 和外部 runtime compatibility 均不在格式内。它只对当前 tiny inference model self-contained，不能据此宣称支持任意 Llama/Qwen、训练 resume 或低位 GPU 执行。

### 31.2 目标 Qwen 单矩阵 7.5×，为何仍不能说“模型压缩 7.5×”？

分母不同。仓库固定 Qwen control 的 7.514752× 只比较一个 `[896,896]` `o_proj.weight` 的 3,211,264-byte FP32 payload 与 427,328-byte strict packed bundle；该矩阵 802,816 参数只占 494,032,768 的 0.1625%。其余权重仍是 FP32，artifact 不含完整 config/tokenizer/未量化 state，模型执行时又先反量化成 FP32。因此它没有给出 whole-checkpoint bytes、resident/peak memory 或 low-bit kernel throughput。

数值层也不能用 argmax 一致掩盖。真实单提示中 last argmax 恰好 17→17，但 selected output relative-L2 为 0.0700，last logits relative-L2/max-abs 为 0.0851/1.6255。正确回答应要求完整 tensor policy、校准/算法、目标 runtime loader/kernel、代表性任务/安全/长上下文集，以及固定硬件 workload 下的内存和性能记录；否则只能写 selected-weight artifact/forward control。

### 31.2 INT8 KV cache 为什么通常达不到 FP32 的 4× 压缩？

若 K/V head dim 都是 (D)，per-token/per-KV-head 独立保存 K/V 两个 FP32 scale，则 FP32 是 (8BH_{kv}TD) bytes，INT8 payload 是 (2BH_{kv}T(D+4))，比率为 (4D/(D+4))，只在大 (D) 时接近 4×。实际还要计 paged block、alignment、allocator、workspace 和临时 dequant buffer；GQA 公式使用 (H_{kv}) 而不是 (H_q)。容量变小也不代表更快：若先 dequantize 再用浮点 attention，只证明数值路径，不能声称执行了融合 KV kernel。质量需分 K→logits/softmax 与 V→output 误差，并测长位置、检索、多跳和安全切片。

### 31.3 Speculative decoding 为什么能保持 target sampling distribution？

proposal \(x\sim q\) 以 `min(1,p(x)/q(x))` 接受；接受路径给 token \(i\) 的质量是 `min(p_i,q_i)`。总拒绝概率是 `1-sum(min(p,q)) = TV(p,q) = sum((p-q)_+)`，拒绝后从 normalized positive `(p-q)` residual 采样，所以两条路径相加恰为 \(p_i\)。一步接受率因此是 `1-TV(p,q)`。

block 顺序验证，只保留首个拒绝之前的 draft；拒绝位置发 residual token并丢弃后续 proposal，全部接受才发一个 bonus target token。必须强调相同 vocabulary/prefix 和采样变换后的真实 \(p,q\)。Greedy prefix verification 是另一算法；概率恒等式不等于速度保证，后者还取决于 draft 成本、verification kernel、接受长度、KV 管理和 batch。

### 31.4 为什么只保存 model、optimizer 和 global step 仍未必能精确恢复训练？

下一批样本和下一次随机算子还依赖 sampler/permutation/cursor、数据 loader worker/prefetch 与 Python/NumPy/Torch CPU/CUDA RNG；AMP 依赖 scaler，gradient accumulation 依赖窗口位置和未提交梯度，scheduler 可能按 update、token 或 metric 前进，FSDP/ZeRO 还要绑定 shard/world topology。数据本身若漂移，即使路径和 shape 相同也不是同一 run。正确做法是先定义只允许保存的一致性边界，再把所有实际消费的状态纳入 snapshot，并用 uninterrupted-vs-kill/reload split run 比较逐步 sample/LR/loss 与最终参数/optimizer/stream/RNG。近似 loss 或能继续跑只说明 warm restart，不是 bit-exact resume。

仓库 CPU FP32 MiniGPT control 在 zero-grad AdamW boundary 保存 16 个参数 tensor、32 个 moment tensor、每参数 step、线性 schedule、数据 fingerprint、permutation/cursor/epoch、data-generator RNG 与 dropout 的 Torch CPU RNG，6 步的 split run 与不中断路径逐位相同。它没有使用或保存 Python/NumPy/CUDA RNG，也不支持 AMP、accumulation、worker、distributed/sharded 或目标 LoRA/QLoRA；这说明 checkpoint 的“完整”总是相对具体训练契约，而不是文件里字段越多越完整。

仓库另有统一的跨进程 CPU AMP 反例：phase-1 写入 model/AdamW/StepLR/GradScaler、Torch+Python RNG、shuffle permutation/cursor/epoch 后退出，不同 PID 恢复并与 uninterrupted trajectory bit-exact。故意在 overflow 后仍推进 scheduler，或分别漏 scheduler/scaler/RNG/data state，都会出现不同且可定位的轨迹。它比“文件能读”更强，但仍只覆盖 tiny authored model、custom stream、zero-grad boundary 和 CPU；没有 worker/prefetch、accumulation 中间态、distributed/CUDA 或目标 Trainer，`weights_only=True` 也不等于来源认证。

### 31.4.1 为什么 DataLoader sampler cursor 可能不能直接写进 checkpoint？

多 worker loader 会预先从 sampler 取 index 并放进 worker queue，所以 sampler-emitted cursor 可能领先于 main loop 真正收到的 batch；main-loop consumed 又可能领先于 backward/optimizer committed。若在 emitted=7、consumed=3 时直接从 7 重启，queue 中尚未消费的 3–6 位置会静默丢失。正确回答要先命名三种 cursor，选择一致性边界，再用 sample/source ID 做 uninterrupted-vs-kill/reload 对账；不能用“保存 epoch/global step”代替。

顺序恢复也不保证随机增强恢复。fresh workers 的 worker-local RNG 会从新的 stream 起点运行，worker assignment 也可能改变。可以保存可公开恢复的 worker/pipeline state，或把随机变换改成由 dataset/transform revision、epoch/visit、sample ID 等派生的 stateless key；后者只适用于变换确实由该 key 完全决定。仓库当前 CPU fixture 在 `num_workers=2,prefetch_factor=2,batch_size=1` 时观察 emitted/consumed=`7/3`：从 3 恢复 ID exact，worker RNG tail 不同，sample-ID-keyed tail exact。它没有 optimizer、persistent worker、IterableDataset、DistributedSampler 或 queue checkpoint，也没有证明具体 ahead=4 是跨版本 API；高质量回答必须主动限定这些边界。

### 31.4.2 崩溃时 consumed=3、optimizer-committed=2，应从哪里恢复？

取决于 checkpoint 是否保存并恢复了第三条对应的 accumulation position、未提交 gradients、缩放分母和所有相关 RNG。若像常见 model/optimizer `state_dict` 一样不含 parameter `.grad`，就必须回到 committed=2，重放已消费但未提交的 sample；从 consumed=3 起步会把“交付过”误当成“训练已提交”。仓库六进程 2-worker/Float64/SGD/StepLR control 在第三条 stochastic backward 后崩溃：从 2 恢复 commit-boundary RNG并重放，与 uninterrupted 的 ledger 和终态 bit-exact。只恢复正确 crash RNG却漏 gradients/sample `1` 的反例，未来 RNG 与 baseline 相同，optimizer/scheduler step 同为 5、LR 同为 `0.0125`，参数最大差仍为 `0.005767858566116724`。

若显式 sidecar 保存 pending `[1]`、position/divisor、逐参数 gradients 与 crash-observed RNG，则可以从 consumed=3 继续；仓库第五个 PID 先要求最后发布的 canonical manifest 为 complete，核对 base/sidecar name/schema/size/hash 与 digest binding，再对实际反序列化 bytes 重查 identity，把保存的 `1` 与新 sample `7` 完成同一窗口，终态也 bit-exact。第六个 PID 恢复完整 gradients/ledger 却保留 commit-boundary RNG，step/LR 仍正确，参数却漂移 `0.017878893573032573`，终态 RNG 也不同。fault matrix 还在 `torch.load` 前拒绝 base-only、两 payload 无 manifest、manifest 缺 sidecar 与 sidecar tamper；但 base-only 仍是可从 committed=2 replay 的有效 checkpoint。

生产回答还要说明 delivery semantics：commit-boundary checkpoint 通常选择 at-least-once replay并要求幂等；保存半窗口扩大状态面。manifest-last 是 completeness marker，不是原子 manifest transaction：仓库没有证明 base/sidecar/manifest、loader、optimizer 与 sample commit 原子，也没有 directory `fsync`、断电/文件系统 fault、来源认证/不可变快照、GradScaler、worker/Python/NumPy/CUDA RNG、任意随机模型、distributed/CUDA 或目标 Trainer。当前 control 只覆盖 main-process Torch RNG 与 StepLR。关键是先定义 state inventory、commit receipt 与“完整可见”的判据，再用 sample ledger 和终态组件对账，而不是机械回答“总是从 2”或“总是从 3”。

### 31.4.3 JAX checkpoint 能打开，为什么仍可能无法精确续训？

因为“可解析”只说明部分 bytes 符合某个 loader，不说明训练状态完整、身份相同或消费位置连续。至少要核对参数 **PyTree treedef** 与逐叶 name/order/shape/dtype/sharding、完整 Optax transformation state（计数器与 moments）、schedule/global step、dropout/采样/数据增强的 typed PRNG key data、数据集 identity、shuffle permutation/cursor/epoch，以及代码和依赖契约。只保存 seed 也不够：必须恢复实际 key/消费位置；只恢复 iterator cursor 也不能补回已预取或随机变换状态。

高质量回答还要给因果验收：独立进程在切分点写入并退出，恢复后的 sample IDs、loss、gradient 和完整终态与 uninterrupted 路径逐项比较。仓库的 6-step CPU fixture 在 step 3 跨两个 spawn process 恢复后 bit-exact；只重置 dropout PRNG 的负例使参数最大差为 `0.037261832505464554`，**wrong-cursor** 负例使差为 `0.03700308472616598`。这只证明 authored JAX/Optax 单设备路径；它没有 Orbax/TensorStore、distributed arrays、CUDA/TPU、目标模型、directory durability、来源认证、收敛或性能证据。

### 31.4.4 共享 dropout mask 对齐，能否证明 PyTorch/JAX 原生 RNG 等价？

不能。共享 mask 是把随机性变成双方共同输入，适合定位 forward/backward/optimizer 差异；它绕开而不是验证两套 native PRNG 的算法、key/state 表示、split/消费顺序与 device/process 派生。仓库用 NumPy **PCG64** 物化三张 embedding mask，三步对账 raw/clipped gradients、AdamW moments/count、schedule、参数与 forward，并用 wrong-mask 反例得到 `0.06900620367377996` 参数漂移。能下的结论只是 shared-mask authored trajectory parity。

若题目要求原生随机训练可重放，应分别定义每个框架的 key/generator state、每个随机 site 的消费顺序、step/device/process folding、checkpoint 恢复与拓扑变化，再在同框架内做 uninterrupted/resume 对照；跨框架未必要 bit-identical。当前 control 没有验证 native RNG state advance、JIT、CUDA/TPU、sharding 或长训练收敛。

### 32. 如何做一次可信消融？

固定数据、token 预算、模型、seed、调参预算和评测；只改变目标组件，多次运行并报告方差。若计算预算不同，明确回答的是“同成本谁更好”还是“最高质量谁更好”。

### 33. 模型版本相同，为什么仍不能保证请求可重放？

模型名称不足以重放。还需要精确 revision、tokenizer/chat template、system/developer/user 消息、工具 schema、检索索引与 ACL 版本、generation 参数、随机性、runtime/kernel 和外部工具状态。配置 fingerprint 只能证明所序列化字段的 canonical bytes 相同，不能证明遗漏字段、外部状态、语义等价或位级确定性。

### 34. 对话摘要能否当作长期记忆的真实状态？

不能。摘要是有损、由模型生成的派生表示，可能遗漏否定、时间、权限和未决事项。事实状态应有 typed schema、来源、时间和用户修正；摘要用于节省上下文，并需要从原始事件重建或审计。

### 35. Prompt delimiter 为什么不是安全边界？

XML tag、Markdown fence 或“以下是不可信内容”能帮助模型区分结构，但外部文本仍在同一推理上下文中。权限、秘密、工具参数、网络访问和副作用必须由模型外执行层校验。结构化输出 schema-valid 也不等于语义正确或授权通过。

### 36. 为什么云模型调用不能“429/5xx 一律重试”？

HTTP class 不足以决定重试：`400/401/403/404` 通常需要修正请求或权限；`501/505` 也不会因属于 5xx 自动变成瞬时故障。先按固定 provider/endpoint/version 建 allowlist，再同时检查请求 replay-safe、远端 outcome 是否确定、attempt/deadline/费用预算和 `Retry-After`。有效 `Retry-After` 超过预算时应停止或上抛，不能提前轰击；timeout 后若远端可能已完成，自动重放可能重复副作用或计费。Exponential backoff、jitter 和 idempotency key 都不能替代对 provider 语义的验证。

实现时至少区分 pool/connect 前失败与 write/read/protocol 后失败：后者仅凭客户端异常不能证明服务端没执行。使用 monotonic deadline 而不是墙上时钟计算预算；exact origin allowlist、HTTPS 和 no-redirect 防止凭据被带到意外 endpoint。非流式客户端在完整 body 已缓冲后检查长度，只是 acceptance cap，不能回答峰值接收内存。

## 代码题建议

能够现场写并测试：

- stable softmax 与 causal attention；
- top-k/top-p sampling；
- BM25 或 RRF；
- Recall@k、MRR、token F1；
- LoRA Linear；
- 有并发上限、monotonic deadline、bounded retry、Retry-After 与 replay guard 的异步调用；
- 幂等工具执行与参数 schema；
- KV Cache 容量估算。

# Single-GPU Fine-tuning

目标：在单张消费级 GPU 上完成可比较、可回归的领域 SFT/LoRA/QLoRA 与偏好优化，而不是只得到一个 adapter 文件。

第一次学习不要从下面几十个机制实验逐项跑。先按这条主线：审计 `train.example.jsonl`，打印 assistant labels，
运行 `smoke_trl_sft.py`，最后独立重载 adapter。输入是两条教学记录和本地 tiny model；输出应让你看清
`messages → token IDs → labels → loss → adapter`。DDP、AMP、PPO、DPO 和精确恢复分别回答后续问题，
完整学习顺序见[项目教学页](../../docs/practice/projects/single-gpu-finetuning.md)。

## 第一次运行

~~~powershell
python -m about_llm.finetuning_cli audit `
  --jsonl projects/single-gpu-finetuning/audit.example.jsonl `
  --require-splits train,validation,test `
  --output outputs/sft-split-audit.json
python projects/single-gpu-finetuning/smoke_trl_sft.py
~~~

第一条命令检查 split、对话结构和重复数据；第二条使用本地随机 tiny GPT-2，真实执行 assistant-only labels、
一个 optimizer step 和 adapter 保存。它们帮助你先看懂数据与训练的连接方式，不代表目标 Qwen 已完成微调。

## 机制实验索引

下面每个脚本只隔离一个问题。遇到对应问题时再运行，不要把所有绿色输出合并成一次“生产级训练”结论。

src/about_llm/finetuning/lora.py 从零实现 LoRA Linear：

- 基座权重冻结；
- B 零初始化，初始函数与基座一致；
- 只保存 adapter 和必要元数据；
- 合并为普通 Linear 后数值等价；
- 测试证明 optimizer 不会更新基座。

~~~powershell
pytest tests/test_lora.py
~~~

同目录还提供一个与目标 LLM 训练解耦的持续学习控制实验：

~~~powershell
python projects/single-gpu-finetuning/continual_replay_toy.py
~~~

它实际执行 task A→task B 的 PyTorch SGD，并输出完整 accuracy matrix、final ACC、BWT、FWT 与非负 peak-to-final forgetting。任务带显式 task-id feature，模型容量足以联合求解；固定 seed 下 no replay 会严重遗忘 A，而在 B 阶段混入全部 A 样本的 1:1 full replay 可同时拟合两项任务。该结果只验证指标实现、顺序更新与全量 replay 对照：输入是单 seed、full batch 的合成 CPU 样例，不代表有限 replay buffer、真实 LLM/语料、安全 retention 或生产收益；全量保存旧样本的隐私、删除、存储和训练成本也尚未建模。

`--benchmark` 进一步执行 seed 0–19 的 no replay、64-example uniform reservoir 和 full replay 配对实验，并保存每个 seed 的完整矩阵、有限 buffer 索引、样本呈现量及 seed-level percentile bootstrap：

~~~powershell
python projects/single-gpu-finetuning/continual_replay_toy.py --benchmark
~~~

在当前固定样例中，64-example reservoir 相对 no replay 的旧任务 accuracy 平均增益为 0.4758，95% paired seed-level interval 为 [0.3389, 0.6104]；但新任务差值为 -0.0072，区间 [-0.0102, -0.0045]。三条路径只匹配 Task B 的 100 个 optimizer steps，每步总样本为 256/320/512，不能写成 compute-matched。任务数据没有跨 seed 重采样，因此区间只覆盖初始化与 buffer 选择，不覆盖任务/数据、目标 LLM、硬件、隐私或部署不确定性。

## 实验协议

至少比较四个系统：

1. base + zero/few-shot；
2. base + RAG（若任务依赖事实）；
3. PEFT LoRA/QLoRA；
4. 全参数微调或高预算参考（显存允许时）。

保持同一 chat template、生成参数和测试集。报告任务指标、格式合法率、通用能力回归、训练/峰值显存、耗时和 adapter 大小。

## 单卡数据契约

每条样本严格保留 `id/messages/source/license/task/language/risk/group_id/split`，可选 `metadata/tools`。tool-aware v2 还要求 function definition、call ID/name、并行 response 和 pending-call lifecycle 完整匹配。训练只对 assistant 区域计算 loss；padding、system、user、tool response 是否 mask 必须通过 token 级检查。按来源或用户划分测试，禁止近重复跨集合。

先审计包含 train/validation/test 的 combined artifact：

~~~powershell
python -m about_llm.finetuning_cli audit --jsonl projects/single-gpu-finetuning/audit.example.jsonl --require-splits train,validation,test --output outputs/sft-split-audit.json
~~~

该命令拒绝宽松 JSON、未知字段和不合法对话，报告重复 id、exact messages、跨 split group/content、切片计数和 canonical fingerprints。它只覆盖 exact/group 规则，不检查 lexical/semantic near duplicate、许可、PII/secret、token 长度或 assistant mask。

将 source/license 标签变成显式决策，并运行有限敏感候选扫描：

~~~powershell
python -m about_llm.finetuning_cli governance-audit --jsonl projects/single-gpu-finetuning/audit.example.jsonl --policy projects/single-gpu-finetuning/governance-policy.example.json --evaluated-at 2026-08-06T12:00:00Z --output outputs/sft-governance-audit.json
~~~

policy 对 `source + license` 默认拒绝，显式绑定 training/evaluation purpose、evidence、review time、expiry 和允许的 risk labels。扫描只覆盖固定的 email、若干 key/token/JWT、private-key header 与 Luhn-valid card-like pattern；报告不含命中原文，人工 exception 绑定精确 record/span identity，陈旧 exception 也失败。这只是治理工作流与候选 reference，不是法律意见、consent 证明、完整 PII/secret 检测或 detector 精度证据。

再显式选择 lossy normalization 与阈值，运行字符 n-gram candidate gate：

~~~powershell
python -m about_llm.finetuning_cli near-audit --jsonl projects/single-gpu-finetuning/audit.example.jsonl --profile nfc_whitespace --ngram-size 5 --threshold 0.9 --output outputs/sft-near-duplicate-audit.json
~~~

它分别检查 full/user/assistant view，并给出 Jaccard 分子、分母与比较总数。finding 只是 lexical candidate，不是 semantic duplicate；无 finding 也不证明没有改写、翻译或答案片段污染。Readiness gate 是全对 \(O(N^2)\) reference，profile/threshold 尚未替真实领域校准。`audit.example.jsonl` 的有序 train 子集与 `train.example.jsonl` 完全相同；两者都不是有效训练语料。

规模化 candidate reference：

~~~powershell
python projects/single-gpu-finetuning/minhash_lsh_toy.py
~~~

它用稳定 SHA-256 shingle mapping、seeded affine MinHash 和 exact band tuple 生成候选，再逐候选复算精确 Jaccard。
默认 64 hashes/16 bands×4 rows 的固定输入将 10 个 pair 缩成 3 个候选；阈值 0.8 下 1 个 true positive、
2 个 false positive，snapshot recall=1、precision=1/3。Exhaustive recall audit 为了得到 ground truth 又做了 10 次
全对比较，因此不是可规模化验证；单独测试还表明 1-hash 配置会漏掉 Jaccard=2/3 的 pair。LSH 不保证召回、
也不发现语义/翻译重复，理想 `1-(1-s^r)^b` 只是 banding 调参曲线；当前 core 仍会在 readiness 不满足时停止。
无密钥 hash/fingerprint 也不认证数据来源。

## 推荐递进

- 机制：本仓库 LoRALinear + 微型 GPT；
- 实用：Transformers + PEFT，对 0.5B–3B 模型做短序列 LoRA；
- 显存优化：4-bit 基座、gradient checkpointing、paged optimizer；
- 完整实验：数据卡、seed、checkpoint、早停、回归和合并/加载测试。

## 变长 masked token 的 gradient accumulation 控制

先用一个不依赖目标 checkpoint 的精确反例检查 loss reduction：

~~~powershell
python projects/single-gpu-finetuning/gradient_accumulation_toy.py
python -m pytest tests/test_gradient_accumulation.py -q
~~~

两个 micro-batch 分别有 1 和 3 个监督 token，另有 3 个 ignored/padding 位置。若目标是全窗口 token mean，micro-batch mean 的正确权重是 `1/4` 与 `3/4`；把两个 mean 等权平均则错误地给成 `1/2` 与 `1/2`。用 `Fraction` 手算可得，正确的 class-aggregate logit gradient 为 `(+23/40,-23/40)`，错误路径为 `(+7/20,-7/20)`，差为 `(-9/40,+9/40)`。

同一脚本真实执行 PyTorch CPU Float64 `cross_entropy.backward()`：full batch 与逐 micro-batch `reduction="sum"`、最后除以全局有效 token count 的梯度 `torch.equal`；等权 micro-batch mean 与 full batch 不同，ignored rows 的梯度严格为零。这个结果只证明本仓库准备的 logits/targets/mask 在单进程 reduction 下的行为。脚本没有 optimizer step、dropout/BatchNorm、AMP、DDP/FSDP/ZeRO collective、`no_sync`、CUDA、目标 tokenizer/model、性能或质量评测；生产训练还必须核对 reducer 的 sum/mean 与 world-size scaling。

### 双进程 CPU/Gloo DDP token-mean 控制

这个实验与上面的单进程 toy 提供两层独立证据；它不把项目改写成多 GPU 训练项目，只在 CPU 上实测默认 DDP reducer：

~~~powershell
python projects/single-gpu-finetuning/ddp_token_mean_control.py
~~~

固定 `world_size=2`、全局有效 token 数 `N=4`、rank counts `[1,3]`。精确手算表明，默认 DDP 对 rank gradient 取 mean 时，每 rank 应把 local loss sum 乘 `D/N=1/2`；真实 Gloo/DDP 得到 `(+23/40,-23/40)` = `(0.575,-0.575)`，与单进程 full batch 的最大误差约 `1.11e-16`。漏乘 world size、只用 `1/N=1/4` 时结果为 `(+23/80,-23/80)` = `(0.2875,-0.2875)`，恰为 full 的一半；rank-local mean 则为 `(+7/20,-7/20)` = `(0.35,-0.35)`。两个真实 OS 进程都通过 count `all_reduce` 观察到 4，并看到相同同步梯度。

这只证明当前 PyTorch 2.13.0+cpu、Gloo、temporary FileStore、`spawn`、默认 `DistributedDataParallel` 和仓库准备的 shared-bias 固定输入。它没有执行 optimizer/parameter update/clipping、gradient accumulation + `no_sync`、AMP/scaler、FSDP/ZeRO/TP/PP/EP、CUDA/GPU、多节点/远端、目标 tokenizer/model/dataset、性能、传输安全或质量评测；跨硬件/world-size bitwise 等价也未证明。

### 双进程 DDP accumulation、`no_sync`、clip 与 SGD 控制

在单次同步 reduction 通过后，再运行完整的极小 update window：

~~~powershell
python projects/single-gpu-finetuning/ddp_accumulation_no_sync_control.py
~~~

两个 rank 各处理两个 micro-batch，监督 token counts 为 `[[1,2],[3,1]]`，全局 `N=7`，所以 local loss sum 统一乘 `D/N=2/7`。精确手算得到，full/one-final-sync/sync-every-microbatch 的 pre-clip gradient 都为 `(+19/35,-19/35)`；若不裁剪，plain SGD `lr=7/20` 的参数 delta 是 `(-19/100,+19/100)`。

真实 CPU Float64/Gloo 运行先用未改写的 built-in DDP，将首批 forward+backward 一起放入 `no_sync`，末批同步；再用 PyTorch 官方 `default_hooks.allreduce_hook` reference 路径计数。正确 scope 观察 1 次两元素 bucket hook；故意把 forward 留在 context 外、只包 backward，观察 2 次 hook。三条路径的 pre-clip gradient 均约为 `(0.542857,-0.542857)`，同步后以 `max_grad_norm=0.5` clipping，再做 plain SGD `lr=0.35`，最终 bias 约 `(-0.123744,+0.123744)`；pre/post-clip gradient 和 update 与单进程 full batch 的最大误差均为 0。

backward-only 负对照在这个线性固定样例上仍保持数值正确，只是没有省通信。built-in reducer 本身没有被直接插桩计数；计数来自另一个语义等价的官方 reference hook 路径。这个实验只有一个两元素参数和单个 bucket，没有覆盖 dropout/BatchNorm/RNG、AMP/scaler、AdamW/optimizer state、checkpoint resume、FSDP/ZeRO/TP/PP/EP、CUDA/GPU、多节点、目标 tokenizer/model/Trainer、通信字节、性能或质量。

### CPU AMP/GradScaler overflow、clip 顺序与 state resume 控制

下面的控制与 DDP 实验独立，真实执行 CPU FP16 autocast、CPU GradScaler、SGD/AdamW，而不是模拟 scale 数值：

~~~powershell
python projects/single-gpu-finetuning/amp_grad_scaler_control.py
python -m pytest tests/test_amp_grad_scaler.py -q
~~~

两个 micro-batch 的 scaled accumulated gradient 为 24。正确顺序 `scaler.unscale_(optimizer) → clip_grad_norm_(max_norm=0.5)` 先得到 3，再 clip 为约 0.5，SGD 参数与 full batch reference exact；负对照先对 24 clipping，再 unscale，optimizer-visible gradient 只有约 0.0625，参数更新随之改变。报告用字段要求明确的 JSON 表示 intentional non-finite：`finite=false,value=null`，不会输出非标准 `Infinity/NaN`。

AdamW 路径先做一次 finite step，得到 step=1、`exp_avg=0.1`、`exp_avg_sq=0.001`。随后三个包含一个 finite 和一个 `inf` micro-batch 的窗口使 scale `8→4→2→1`；每次都跳过整个 update，参数与 AdamW moments/step exact 不变。进程内 split point 同时深拷贝 model、optimizer 与 scaler。恢复 scale=1 后，unscaled gradient 10000 保持 finite、执行 step=2，并与不中断路径的参数/moments/scaler exact；漏恢复 scaler、回到 initial scale=8 时 FP16 scaled gradient overflow，scale `8→4`，参数和 step=1 不动。

范围只到当前 PyTorch 2.13.0+cpu、单 FP32 scalar 参数、CPU FP16 autocast 和进程内 state replay。没有检查字段严格的文件 checkpoint、真实进程重启、scheduler/RNG/DataLoader、DDP/FSDP/ZeRO、CUDA/GPU kernel、目标 tokenizer/model/Trainer、性能、收敛或质量证据。MiniGPT 文件恢复实验与本实验互补但彼此独立；这里没有把两者事后拼成一条更强的结论。

### 双进程 DDP + AMP overflow 共识实验

下一个实验在同一 update path 中真实组合 built-in DDP reducer、`no_sync`、CPU FP16 autocast/GradScaler、AdamW 与 StepLR：

~~~powershell
python projects/single-gpu-finetuning/ddp_amp_overflow_consensus_control.py
~~~

三条路径都先执行一次 finite warm-up，建立 parameter≈0.99、AdamW step=1、scheduler epoch=1/LR=0.005、scale=8、growth tracker=1。第一条让 rank 0 在首个 `no_sync` micro-batch 产生 non-finite；末批 built-in DDP reduction 后两个 rank 都 non-finite，均跳过 AdamW/StepLR，scale `8→4`、tracker `1→0`，训练状态保持一致。

第二、三条先让 DDP 在两边得到 finite scaled gradient=8，再由脚本在 rank 0 的 `unscale_` **之前**把 gradient 改成 Inf。这是仓库故意注入的 post-reduction fault，不是 DDP 正常行为。无额外共识时，rank 0 skip、保持 step=1/parameter≈0.99/LR=0.005/scale=4；rank 1 update 到 step=2/parameter≈0.985/LR=0.0025/scale=8，moments 与 scheduler 也分叉。加入 optimizer-pre `all_reduce(MAX)` finite gate 后，local flags `[1,0]` 变成两个 global flag 1；两边都不调用 scaler/optimizer/scheduler step，并显式把 scale 统一到 4，parameter/optimizer/scheduler/scaler state 保持一致。non-finite value 仍只以 `finite=false,value=null` 写入 JSON。

这里的 `scaler.update(new_scale=4)` 是仓库定义的共同 scale policy，不等于同步了 GradScaler 的 native found-inf transition：两边 growth tracker 都保留为 1，而 native overflow 会重置为 0。生产代码应使用目标框架公开的 distributed overflow/scaler 协议，或定义并验证完整 scaler state；不能把这个单参数示例当 drop-in wrapper。第一条也说明当前默认 reducer 会传播 reduction 前的 Inf，不能反向声称 vanilla DDP 在所有情况下都缺一次 flag collective。这个实验没有覆盖 clipping、自然 overflow、多参数/bucket、custom hook、conditional graph、checkpoint/elastic restart、FSDP/ZeRO、CUDA/NCCL、多节点、目标 Trainer/model、性能、收敛或质量；built-in collective count 也未直接插桩。

### 跨进程恢复 AMP、scheduler、RNG 与数据游标

第三个实验在同一条训练轨迹中统一验证这些状态，而不是借用前两个实验的结论：

~~~powershell
python projects/single-gpu-finetuning/checkpoint_resume_control.py
~~~

固定 6 参数线性模型使用 CPU FP16 autocast、真实 GradScaler/AdamW/`StepLR(step_size=2,gamma=0.5)`、Torch 全局 RNG 生成的显式 inverted-dropout mask、Python RNG loss factor，以及独立 generator 驱动的 8-example stateful shuffle。attempt 0 成功，attempt 1–3 intentional non-finite，使 scale `8→4→2→1`，但 AdamW step 保持 1、scheduler `last_epoch/step_count` 保持 `1/2`。phase-1 随后把 model/optimizer/scheduler/scaler、Torch CPU/Python RNG、permutation/cursor/epoch/generator、attempt/update progress 与 dataset hash 写入 21,747-byte checkpoint 并退出；不同 PID 的进程重开后完成 attempt 4–7。resume tail 的 batch、随机因子、mask、loss、gradient norm、scale、optimizer/scheduler/LR trace 和最终所有组件 fingerprint 与独立 uninterrupted worker exact。

五条负对照分别在同一个 split point 错误地：overflow 仍推进 scheduler、漏 scheduler、漏 scaler、漏 RNG、漏 data stream。它们依次造成 scheduler 与 optimizer update 数脱钩、LR 衰减错位、scale=8 的边界 overflow、同 batch 不同随机 trace、同 RNG 不同 batch。完整 checkpoint schema 不允许缺字段；“漏状态”是 loader 验证字段存在后故意不恢复该组件的 counterfactual，不是生产 loader 的宽松行为。

该固定样例只证明 PyTorch 2.13.0+cpu、本机不同 Python PID、仓库准备的 tiny model、custom shuffle 与 zero-grad boundary。`torch.save` 是 pickle-backed 容器；`weights_only=True`、16 MiB 上限和 closed fields 不能认证或保护不可信文件。same-directory temp/file `fsync`/replace 没有经过 crash/power-loss 注入，也没有目录 `fsync`。没有真实 DataLoader worker/prefetch、NumPy/CUDA RNG、gradient accumulation 中间态、DDP/FSDP/ZeRO、目标 tokenizer/model/Trainer、CUDA、质量、性能或远程 checkpoint 证据。

### 跨进程 DataLoader prefetch、cursor 与 worker RNG 控制

这条数据控制不训练模型，而是隔离 custom shuffle 尚未覆盖的真实 worker/prefetch 语义：

~~~powershell
python projects/single-gpu-finetuning/dataloader_prefetch_resume_control.py
~~~

四个不同顶层 PID 分别执行 uninterrupted、phase-1、正确 resume 和错误 resume；每段用 `DataLoader(num_workers=2,prefetch_factor=2,batch_size=1,multiprocessing_context="spawn",in_order=True)` 启动真实 CPU workers。固定 permutation 为 `[8,3,1,7,0,9,4,2,6,5]`。phase-1 主循环只收到前三条时，tracking sampler 已 emitted 到 cursor 7，所以 `[7,0,9,4]` 虽已送进 worker 队列，却尚未交付给训练循环。

490-byte 左右、字段要求明确的 JSON checkpoint 同时保存 consumed/committed cursor=3 与 observed emitted cursor=7；具体大小会随 PID 位数变化。不同 PID 从 3 重建得到完整 sequence `[8,3,1,7,0,9,4,2,6,5]`，从 7 重建的负对照只得到组合 `[8,3,1,2,6,5]`，静默漏四条。这里的 ahead=4 是当前 PyTorch 2.13.0+cpu 固定样例的观察值；脚本没有读私有 queue 字段，不能把它当任意版本/配置的公开 prefetch 深度保证。

顺序 exact 仍不等于数据 tensor exact。同一 loader seed 的独立 phase-1 可重放 prefix worker RNG；restart 后 fresh workers 从各自 RNG 序列开头继续，resume tail 相对 uninterrupted 的最大差约 0.654431。脚本同时计算 `(namespace,sample_id)` 派生的局部 generator，tail 最大差为 0。该 sample-key 只适用于当前单 epoch 固定样例；生产多 epoch/重复采样应加入 dataset/transform revision、epoch/visit 等，否则每次同 ID 都会得到同一增强。这个实验未保存 queue payload/worker state，也没有 persistent workers、pin memory、IterableDataset、collator、model、optimizer、DistributedSampler、CUDA、吞吐或质量证据；“主循环收到 sample”更不等于 optimizer 已原子提交。不能与前一 model checkpoint 事后拼成完整 exact-resume 声明。

### 跨进程 consumed—optimizer-committed 崩溃窗口实验

下一项实验把上一节未知的 optimizer cursor 变成真实训练事件：

~~~powershell
python projects/single-gpu-finetuning/optimizer_commit_resume_control.py
~~~

六个不同顶层 PID 分别执行 uninterrupted、phase-1、从 optimizer-committed cursor 正确恢复、从 consumed cursor 只恢复 crash RNG 却漏 gradients、从 consumed cursor 加载完整 sidecar 正确恢复，以及 gradients 完整但误用 commit-boundary RNG 的隔离负例；每段仍启动两个 spawn DataLoader workers。CPU Float64 `Linear(2,1)` 的输入先乘 main-process inverted-Bernoulli mask（固定 seed `20260815`），再执行 MSE、`SGD(momentum=0.9)`、`StepLR(step_size=2,gamma=0.5)` 和 accumulation steps=2。phase-1 在第三个 microbatch `[8,3,1]` 已交付且 stochastic forward/backward 后模拟崩溃：sampler emitted=7、main loop consumed=3，但只有 `[8,3]` 已进入 optimizer step/scheduler step，所以 committed cursor=2，模型上还有两个未提交 gradient tensors。

当前 8,985-byte `torch.save` base checkpoint 保存 commit-boundary model/SGD momentum/StepLR、commit-boundary Torch CPU RNG、两种应用 cursor、sampler cursor、sample ledger、数据与 loader contract，故意不把 `.grad` 混入普通 model/optimizer state。第一种正确协议从 committed=2 同时恢复 RNG 并重放 sample `1`，其 mask tail、完整 ledger、model/optimizer/scheduler/RNG fingerprint 与 uninterrupted bit-exact、参数最大差为 0。第一个隔离负例从 sidecar 恢复正确 crash RNG，却故意不恢复 pending gradients并从 consumed=3 起步；未来 mask trace 与终态 RNG 仍和 baseline 相同，ledger 漏 `1`。即使末尾 singleton 重缩放后仍是 5 次 optimizer/StepLR step、终态 LR 同为 `0.0125`，参数最大差仍为 `0.005767858566116724`，因此差异可归因于漏掉半窗口而不是 RNG shift 或 step 数。

第二种正确协议另发布当前 7,905-byte gradient sidecar：它绑定 base checkpoint SHA-256、数据/permutation、三种 cursor、pending window `[1]`、accumulation position=1、steps/loss divisor=2、crash-observed Torch RNG，并按参数名保存两个 finite Float64 gradient tensors。随后最后发布当前 827-byte strict canonical JSON bundle manifest；closed schema 同时绑定数据 identity、两个固定文件名/schema/size/SHA-256、sidecar→base digest 与 `base→sidecar→manifest` 顺序。第五个 PID 必须先看到 `publication_state=complete`，在任何 `torch.load` 前校验 manifest 与两个文件 identity，再对实际送入 `BytesIO` 反序列化的 bytes 重查相同 size/hash，之后才从 consumed=3 恢复；首个 optimizer window 是 `[1,7]`，mask tail、完整 ledger 与 model/optimizer/scheduler/RNG fingerprint 同样和 uninterrupted bit-exact、参数最大差为 0。

第六个 PID 恢复同一完整 gradients 与 sample ledger，却故意保留 base 的 commit-boundary RNG、不加载 sidecar 的 crash RNG。它仍执行 5 次 optimizer/StepLR step并得到 LR `0.0125`，但新样本从错误 mask offset 开始；终态 RNG 不同、参数最大差为 `0.017878893573032573`。这个负例把“state inventory 不完整”与“漏 sample/gradients”分开：只看 ledger、global step、LR 或 gradient sidecar 完整都不足以声称 exact resume。

父进程另构造四种 publication fault snapshots：仅 base、base+sidecar 但无 manifest、manifest 存在但 sidecar 缺失，以及 manifest 发布后 sidecar 同长度篡改；sidecar 协议分别因缺完成标记、缺 artifact 或 digest drift 在反序列化前 fail closed，完整 bundle 则通过。仅 base 仍可用于第一种 commit-boundary replay，不能把“sidecar bundle 未完成”误写成“base checkpoint 无效”。当前 8 个测试还覆盖 duplicate/non-canonical/unknown manifest、manifest no-overwrite、base digest mismatch、loader contract drift、non-finite model 与 missing momentum。

这个实验执行了真实 worker/prefetch、stochastic forward、backward、gradient accumulation、SGD momentum、StepLR、Torch CPU RNG commit/crash snapshots，以及两种跨 PID exact-resume 协议，但 manifest-last 只是完整性门禁，仍不是“消费 sample 与 optimizer/checkpoint 原子提交”的实现。base、sidecar、manifest 分别 temp-file + file-`fsync` + `os.replace`；最后发布 manifest 可识别当前进程故障快照，却没有目录 `fsync`、断电/文件系统故障注入、原子目录切换或远程对象存储语义。无密钥 hashes 不认证来源，协同替换整套 internally consistent bundle 仍可能通过；也没有证明并发目录替换/不可变快照。它未保存 queue/worker RNG、Python/NumPy/CUDA RNG，也没有原生 Dropout/任意随机模型、随机数据增强、多 epoch、GradScaler/CUDA AMP、DistributedSampler/DDP/FSDP/ZeRO、目标 LLM/Trainer、质量或性能证据。

## 在固定 Qwen 版本上检查 SFT token、mask 与 final label { #target-qwen-sft-label-control }

`qwen2.5-0.5b-sft-label.control.json` 将固定 `Qwen/Qwen2.5-0.5B-Instruct` revision、三条 tool-aware 训练样例、held-out-free readiness 和审核后的本地 Jinja template 绑定到同一份字段封闭的契约。原生 Qwen 模板不含 `{% generation %}`；在多轮、并行 tool calls、带 preamble 的 tool call 三条固定记录上真实请求 mask 时，47 / 301 / 200 个 input token 的 assistant mask 均全零，不能直接用于 assistant-only supervision。

审核模板 `qwen2.5-generation-aware-sft.jinja` 保留 checkpoint 原生 system/tool schema/tool response 序列化，只给 assistant payload 加 generation span。对三条固定记录，它与原生模板的 input IDs 逐 token 相同，并把所有 assistant turn 的正文、Qwen tool-call markup 与 `<|im_end|>\n` 精确圈成 8 / 51 / 31 个 assistant token。该结果只适用于这组三条记录及其 Qwen schema，不代表任意 provider tool schema、multimodal、任意新消息，也不证明 tool 执行或结果真实。

异构 tool arguments 若先进入 `Dataset.from_list`，Arrow 会把对象统一为 struct 并向其他调用注入 `null` 字段，从而改变 Qwen prompt。因此入口先在 Python 中得到整数 `input_ids/assistant_masks`，再构造只含这两列的 Dataset。TRL 0.29.1 对已预分词数据会拒绝 `assistant_only_loss=True`；这里显式设置 `assistant_only_loss=False`，但真实 configured collator 仍独立消费预计算 masks，并在训练前做 final-label audit。固定 batch 为 `[3, 301]`，共 548 attention token、355 padding token、90 个监督 label 和 813 个 `-100`；每个监督 label 等于对应 input ID，其他有效 token 与 padding 全为 `-100`。固定 Qwen CPU FP32 eager no-grad loss 为 `1.251716136932373`。验证清单的指纹为 `sha256:b1c1a6b3…936e6`，录制报告的指纹为 `sha256:8b61fa58…10421a`。

~~~powershell
python projects/single-gpu-finetuning/run_qwen_target_sft_label_control.py --verify projects/single-gpu-finetuning/qwen2.5-0.5b-sft-label.recorded-report.json
~~~

这个实验没有执行 backward、optimizer、adapter export、LoRA、QLoRA、CUDA、vLLM 或 serving，也没有测量 loss 下降、训练收敛、数据泛化或生产可用性。训练记录、readiness 和 template 都由本仓库准备；size/SHA-256 和字段检查能发现已定义范围内的漂移，但不证明数据合法性、语义质量、发布者/审核者身份或来源真实性。无密钥 hash 可被有写权限的攻击者协同重算，verify→loader reopen 的 TOCTOU 也没有消除。

## PEFT 离线保存、重载与合并验证

`smoke_peft.py` 用随机初始化 tiny GPT-2 实际训练 PEFT LoRA，不下载模型。它先保存 exact base safetensors，再保存 adapter safetensors，从独立重载的 base 加载 adapter，执行 `safe_merge`，保存 merged full weights 并再次从磁盘重载：

~~~powershell
python projects/single-gpu-finetuning/smoke_peft.py
python projects/single-gpu-finetuning/smoke_peft.py --steps 8 --artifact-root artifacts/peft-export-control
~~~

这个 8-step 固定样例的 base/adapter/merged weight 文件分别为 110,632/4,608/110,632 bytes；训练前后所有 frozen base 参数 exact，构建时与 verify 后的 adapter reload 最大 logit error 都为 0，merge error 约 $8.94\times10^{-8}$，verified merged reload error 为 0。发布目录还保存 32-token WordLevel tokenizer、special tokens 和 chat template；保存/重载后 `tok5 tok7 <eos> tok9 <eos>` 的 token IDs 都是 `[5,7,2,9,2]`。

`about-llm-export-manifest.json` 是 strict canonical manifest。默认目录共有 13 个被覆盖文件、payload 236,589 bytes、manifest 2,297 bytes；descriptor 按 POSIX relative path 排序并绑定每个文件的 size/SHA-256，再绑定整个 descriptor set。Verifier 在 published-artifact reload 前运行，要求三个 safetensors 均可解析、base/merged 的完整 config payload 与 tensor key/dtype/shape signature 一致，并确认每个 target module 同时存在 LoRA A/B tensor；它拒绝额外或缺失文件、symlink、路径穿越、duplicate/non-canonical manifest、资源上限、size/hash 漂移，以及协同重算 hash 后的 weight/config/adapter/tokenizer 漂移。已有输出目录和已有 manifest 均拒绝覆盖。

这是标准 Transformers/PEFT artifact 加本仓库结构验证程序的 CPU 实验，不是通用 checkpoint。Adapter config
使用 immutable base identity string，manifest 另绑定 exact base 文件；但路径或 identity string 不是内容认证，
PEFT 自身仍不会自动强制仓库 manifest，调用方必须先执行 verifier。可解析、同 key/dtype/shape 和 LoRA A/B tensor
覆盖都只是结构证据，不证明权重数值正确或确由声明的训练产生。目录没有 optimizer/scheduler/RNG/training-resume
state，也未执行量化基座 merge、目标 checkpoint 或 CUDA。随机 tiny loss 下降、hash 一致和数值等价不说明
license、runtime 兼容、任务质量、跨版本可移植性、来源认证或断电原子发布；unkeyed SHA-256 可被攻击者协同重算，
单文件 exclusive-create+`fsync` 也不构成目录级原子发布。当前 verifier 还没有锁住随后由 Transformers 打开的文件，
不能防止 verify 与 load 之间的并发替换（TOCTOU）；生产消费要配合不可变目录、访问控制、
lease/content-addressed handle 或等价机制。

## 固定 Qwen 真实权重 LoRA 单步控制

`run_qwen_target_lora_control.py` 把上述 tiny artifact plumbing 推进到固定的 `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae5576…9a775`。它复用 Transformers Basics 已审核的 7-file、999,586,347-byte checkpoint manifest 与真实 forward report，在每次模型加载前重新核对全部选定 bytes；随后以 `trust_remote_code=False`、CPU FP32 eager、8 threads 加载 494,032,768 个基座参数。

~~~powershell
python projects/single-gpu-finetuning/run_qwen_target_lora_control.py --local-files-only --artifact-directory projects/single-gpu-finetuning/target-adapters/qwen2.5-0.5b-instruct-step1 --output-report artifacts/qwen-target-lora-report.json
python projects/single-gpu-finetuning/run_qwen_target_lora_control.py --verify projects/single-gpu-finetuning/qwen2.5-0.5b-lora.recorded-report.json
~~~

实验用目标 chat template 分别渲染 user-only generation prefix 与 user+assistant 完整对话；完整 44 tokens 必须逐 token 以前 41-token prefix 开头，loss 只保留 3 个 assistant-side tokens。LoRA 只注入 24 层的 `q_proj/v_proj`，`r=4, alpha=8, dropout=0`，得到 270,336 个 trainable parameters，占带 adapter 总参数的约 0.05469%。B 零初始化使注入后、step 前的 last-token logits 与基座 exact 相同。

2026-08-13 的录制运行真实执行一次 backward 与 AdamW step。96 个 trainable tensors 都得到 finite gradient，frozen base 没有 gradient；基座全参数 byte fingerprint 前后均为 `sha256:716454a9…e7092`。48 个 A 与 48 个 B tensors 完整导出，全部 B tensors 非零，共 98,304 个非零 B elements。标准 PEFT payload 为 README 5,404 bytes、adapter config 1,161 bytes、safetensors 1,093,728 bytes；strict artifact manifest 为 1,488 bytes，指纹 `sha256:ffab4958…c96c46`。重新核对 checkpoint bytes、重新加载基座并用 PEFT 加载 adapter 后，last-token logits 与训练态保存前 bit-exact，最大误差为 0。recorded report 指纹为 `sha256:8a3897b1…026230`。

这个结果**不证明 loss 改善**。固定单样本单步的 initial loss 约为 0.003864，step 后反而升到约 0.584557；报告保留该结果，不事后换样本或只展示好看的曲线。它证明 target checkpoint 上的 assistant-only mask、PEFT backward、optimizer、frozen-base、adapter save/reload 链路确实执行；不证明收敛、任务/通用质量、代表性数据或合理超参数。

Artifact verifier 会拒绝额外/缺失文件、symlink、duplicate/non-canonical manifest、size/hash、base revision、PEFT config、layer/module、A/B coverage、shape/dtype 和非有限/全零 tensor 漂移；测试还覆盖攻击者同步重算无密钥 manifest 后的 config、tensor、report 与 scope 漂移。但无密钥 hash 不认证模型发布者或训练者，framework verify 后按路径 reopen 的 TOCTOU 仍存在。该目录没有 optimizer/scheduler/RNG/resume state，也没有 merged full weights；没有执行量化基座、QLoRA、CUDA、AMP、vLLM、峰值内存/性能或生产发布。1.09 MB adapter 大小也不能当作训练峰值内存或部署 resident memory。

## TRL 单卡入口

先运行完全离线的 TRL 闭环。它用随机 tiny GPT-2、本地 WordLevel tokenizer、带 generation 标记的模板和仓库准备的 SFT 固定样例，实际验证 `messages → assistant_masks → collator labels → optimizer step`；同时断言非 assistant labels 全为 `-100`、assistant labels 保留且 tiny batch loss 下降：

~~~powershell
python projects/single-gpu-finetuning/smoke_trl_sft.py
~~~

这只是 CPU 控制流/label 契约证据，不代表任何目标模型质量、CUDA 兼容、真实数据合法性或生产收敛。

## 用 CPU 小模型观察 reward model 的捷径

`reward_model_toy.py` 用 NumPy 从零训练线性 Bradley–Terry scorer。本仓库准备的两个特征分别表示 quality signal 与 length proxy；`confounded` 训练集让二者总是同向，所以模型能取得 1.0 的训练 strict pair accuracy，却在长度方向反转的 counterfactual held-out pair 上得到 0.0。补入长度正反两种 pair 后，拟合出的 length 权重约为 0，训练与 held-out accuracy 都为 1.0：

~~~powershell
python projects/single-gpu-finetuning/reward_model_toy.py
~~~

输出同时保存初始/最终 Bradley–Terry objective、margin、tie count、偏好概率与权重。strict accuracy 只把正 margin 计为正确，zero-margin 单独计 tie。该实验使用作者构造的数值特征和 preference，不读取文本，不执行 tokenizer/Transformer，也不证明真实人类标签、目标 RM、OOD 鲁棒性、reward hacking 或 policy optimization；训练准确率满分不是上线证据。

### 文本与 tiny Transformer RM 闭环

下一项实验实际执行完整 prompt+response 的 chat-template tokenization、随机 tiny GPT-2 scalar reward head、Bradley–Terry backward 和 AdamW。reward head 从全零开始，因此初始两个 pair 都是 tie、loss≈`log(2)`；4 步后 reward head 与 Transformer token embedding 均改变，仓库准备的 train pair strict accuracy=1：

~~~powershell
python projects/single-gpu-finetuning/smoke_transformer_reward_model.py
~~~

脚本只读取 `preference.train.example.jsonl` 与 `preference-training-readiness.example.json`，复用 binary train ordered identity、lexical/governance readiness 和 prompt-prefix/截断 audit；tokenizer vocabulary 也只从 train pair 构建。测试在没有 combined 文件的临时目录中运行，并在模型初始化前拒绝缺失/篡改 readiness 与顺序漂移 train。一个故意把已见 `good/bad` 线索反转的反例得到 strict accuracy=0，说明真实 Transformer optimizer 同样可能学习词面捷径。这个反转标签本身没有自然语言质量含义；无密钥 readiness 也不认证签发者。随机 tiny 权重、本地 tokenizer 和两条 pair 不证明人类 preference、目标 RM、广泛 OOD/counterfactual 鲁棒性、CUDA、reward hacking 或 policy optimization。

### 目标模型 LoRA/QLoRA RM 入口

`train_reward_model.py` 与 DPO 入口使用相同的 train-only 权限边界。`--data-preflight-only` 不导入训练依赖、下载 tokenizer 或加载模型，只验证严格 readiness 与 train ordered identity。`--tokenization-preflight-only` 才加载固定 revision 的目标 tokenizer，真实渲染完整 prompt+chosen/rejected，并在模型加载前拒绝 prefix mismatch、空 completion、两侧相同 token 或任何超过 `max_length` 的 pair。这样避免 TRL `RewardTrainer` 将过长 pair 静默过滤后仍继续训练：

~~~powershell
python projects/single-gpu-finetuning/train_reward_model.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --readiness-json projects/single-gpu-finetuning/preference-training-readiness.example.json --output-dir outputs/rm-run --data-preflight-only
python projects/single-gpu-finetuning/train_reward_model.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --readiness-json projects/single-gpu-finetuning/preference-training-readiness.example.json --output-dir outputs/rm-run --tokenization-preflight-only
python projects/single-gpu-finetuning/train_reward_model.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --readiness-json projects/single-gpu-finetuning/preference-training-readiness.example.json --output-dir outputs/rm-run --qlora --target-modules q_proj,k_proj,v_proj,o_proj --modules-to-save score
~~~

正式入口配置 `AutoModelForSequenceClassification(num_labels=1)`、`SEQ_CLS` LoRA、可选 reward centering，并在 trainer 准备数据后再次断言 pair 数不变。自动化用本地随机 tiny GPT-2 实际贯通正式 `RewardTrainer`、一个 optimizer step、global-step/metrics 工件和 adapter 保存，并验证保存的 `lora_B` 已非全零；这与前面的手写 optimizer 实验是两条不同证据。`modules-to-save=score` 和默认 attention module 名不是跨 checkpoint 的普适真理，必须检查目标模型 module tree。当前没有下载目标 checkpoint，也没有执行目标 module mapping、CUDA/QLoRA、真实人类 preference、显存或生产 convergence。

## 用固定轨迹手算 PPO/GAE objective

`ppo_objective_toy.py` 是 PPO rollout 之前的数学控制层：它从零计算 masked GAE 与 clipped sampled-action surrogate，并分别输出 TD residual、advantage/return、bootstrap/continuation mask、unclipped/clipped objective、clip fraction、ratio 与 sampled KL proxy：

~~~powershell
python projects/single-gpu-finetuning/ppo_objective_toy.py
~~~

固定两步轨迹用手算验证 padding 不进递推或均值，terminated 不 bootstrap；对 truncated transition，脚本显式比较“有可靠 next value 因而 bootstrap”和“不 bootstrap”两种约定，但两种都阻断 advantage 跨 episode 传播。PPO 部分同时验证正 advantage 的 ratio 上界和负 advantage 的下界。三分类反例保持已采样动作 probability ratio=1，却把未采样概率质量移到极小尾部，使完整 \(D_{KL}(old\|new)>10\)；因此 clip fraction=0 和 sampled proxy=0 都不是全分布 KL 保证。

这只是仓库准备的 reward/value/log-prob/distribution 上的 CPU NumPy 参考计算。它没有执行 tokenizer、语言模型、value/reward model、rollout engine、optimizer、reference KL controller、value clipping/entropy loss、真实 EOS/truncation 来源或 CUDA，也不证明稳定 PPO 训练、目标策略质量或安全对齐。

### PyTorch 两状态 PPO 闭环

`smoke_torch_ppo.py` 再向前一步，实际执行 categorical on-policy sampling、policy/value forward、GAE、advantage normalization、clipped policy loss、value MSE、entropy bonus、多 epoch minibatch autograd 与 Adam：

~~~powershell
python projects/single-gpu-finetuning/smoke_torch_ppo.py
~~~

环境只有两个可观察状态，每个 episode 固定两步：正确动作各得 1 分，第二步 terminated。因为精确期望可直接由两个正确动作概率相加，测试不拿有限 rollout 均值代替 ground truth。默认 6 轮、每轮 128 episode、4 epoch、64-action minibatch，共 96 次 optimizer step；固定 seed 下精确 expected return 从均匀策略的 1.0 提升到 1.8 以上，policy/value 参数均改变。每轮 old log-prob 被 detached 保存，多 epoch 后仍逐元素不变，并与 rollout policy snapshot 精确一致。

这证明 tiny tabular MDP 上的 PyTorch rollout/GAE/optimizer 控制路径，不是语言模型 RLHF：没有 token generation、learned RM、reference KL controller、value clipping、time-limit truncation、checkpoint/resume、CUDA 或分布式 actor/learner。单环境单 seed 的改善不能外推到目标 LLM 的稳定性、样本效率、质量或安全。

### Tiny Transformer token PPO

`smoke_transformer_ppo.py` 使用随机 tiny `GPT2Model`、policy/value heads 与冻结 reference，真实执行两步 integer-token autoregressive sampling、sampled reference log-ratio reward、GAE 和多 epoch PPO optimizer：

~~~powershell
python projects/single-gpu-finetuning/smoke_transformer_ppo.py
~~~

词表大小为 6，BOS 固定，生成目标 token 每步得 1 分，第二步 terminated。脚本枚举全部 first-token branch 并精确积分第二步条件概率，得到等价于汇总 \(6^2\) 条 trajectory 的 expected task reward；因此测试可以要求它从均匀 policy 的 \(1/3\) 提升到 1.8 以上，而不是依赖有噪声的 rollout mean。默认 6 轮共 36 个 optimizer steps，验证 GPT‑2 backbone、policy/value heads 都改变，reference 逐参数不变，每轮 old log-prob 与冻结 behavior snapshot 一致。

Reward penalty 中保存的是 sampled action 的 `log π_old - log π_ref`，单条可为负；其在 behavior policy 下的条件期望才是 categorical forward KL。输出另对每个 sampled state 的全部 action 显式求和报告 exact KL。该实验没有 tokenizer/自然语言、learned RM、目标 checkpoint、variable-length EOS、time-limit truncation、checkpoint/resume、CUDA 或 distributed rollout；重复目标 token 只是 authored verifier，不代表真实偏好、质量或安全对齐。

### 本地 tokenizer/chat-template 文本 PPO

`smoke_text_ppo.py` 使用本地 WordLevel tokenizer 与 chat template 渲染 `Say good.`，随机 tiny GPT-2 policy 最多生成两个 token，并真实覆盖第一步 EOS termination、第二步未 EOS 的 `max_new_tokens` truncation，以及 termination 后的 padding：

~~~powershell
python projects/single-gpu-finetuning/smoke_text_ppo.py
~~~

作者构造的 dense task 是首 token 为 `good` 得 1 分、若继续则第二 token 为 EOS 再得 1 分。脚本用分离的 policy/value tiny backbones、冻结 reference 与 behavior snapshot，执行 sampled reference log-ratio shaping、GAE、clipped policy/value/entropy loss 和 96 次 optimizer step；同时为每个被长度截断的 post-action state 计算 value，并逐轮报告 EOS、truncation、padding、exact categorical KL、ratio 和 snapshot 不变量。

这里报告的是**到两 token generation cap 为止**的有限时域 return，因此默认不对 cap 后 continuation 做 value bootstrap。生成 API 的 `length`/truncated finish reason 不能单独推出 GAE 的 \(b_t=1\)：只有训练 objective 定义了 cap 后的 return，且 next value 与它匹配时才能 bootstrap。`bootstrap_truncated=True` 只作为反事实诊断开放，报告会明确标记 optimizer 与有限时域精确 objective 不一致。

13 词表的均匀初始 policy 可精确得到 expected reward \(25/169\) 与 `good, EOS` 概率 \(1/169\)；测试要求训练后分别超过 1.9 与 0.95。该精确枚举比有限 rollout mean 更适合验证这个短任务，但脚本没有 learned RM、真实人类 preference、目标 checkpoint、长 response、checkpoint/resume、CUDA 或 distributed rollout，不证明自然语言质量、目标 LLM PPO 或安全对齐。

### 冻结 learned RM 的 PPO proxy-exploitation 对照

`smoke_learned_rm_ppo.py` 把 RM 与 PPO 真正串成一条 CPU 闭环，并刻意保留独立 authored target verifier：

~~~powershell
python projects/single-gpu-finetuning/smoke_learned_rm_ppo.py
~~~

脚本只用一个 sparse pair 训练随机 tiny Transformer RM：chosen=`good, EOS`、rejected=`bad, EOS`。零 reward head 从 loss \(\log2\) 起步，30 step 后训练准确率为 1、margin 为 5.57；随后逐参数冻结 RM。因为 Bradley–Terry pairwise loss 不识别全局 score offset，脚本明确减去训练 pair midpoint；这个 centering 不改变排序，也不能修复未覆盖 response 的错误外推。

生成端 suppress `[UNK]`、`[PAD]` 与 role markers，只允许 EOS 和 7 个普通词 token；allowed-action mask 同时进入 rollout、old/new log-prob、entropy、KL 与精确枚举。两 token cap 的完整 support 为 \(1+7\times8=57\) 条。穷举发现训练 chosen 只排第 38，最高 RM score 是未见过的 `good., good`，55 条 response 没有训练覆盖。PPO 把精确 centered-RM expectation 从 2.739 提到 4.652，但严格 `good, EOS` 成功率从 \(1/64\) 降至 \(4.99\times10^{-4}\)；dense partial-credit reward 则从 \(15/64\) 升至 0.566。RM/reference 均保持不变，old log-prob/snapshot、EOS/truncation/padding 也有自动化验证。

因此该实验可写成“在仓库准备的 tiny support 上，learned proxy 提升而预先声明的严格 objective 恶化的 reward-hacking 反例”，不能写成所有指标都下降，更不能写成真实人类偏好或目标模型已经发生 reward hacking。它没有 held-out human labels、目标 checkpoint、长 response、真实 reward normalization、checkpoint/resume、CUDA 或 distributed rollout；训练准确率、RM reward、partial credit 和 strict success 必须分别报告。

## Preference 数据与 DPO 离线验证

字段要求明确的 preference 固定样例保留 prompt、A/B 原始候选、展示顺序、winner/tie/invalid、强度、rubric、annotator/adjudication、generator revision 与治理/split 字段。审计会把交换 A/B 后内容相同的 pair 仍视为重复，并阻断 exact prompt/pair/group 跨 split 泄漏：

~~~powershell
python -m about_llm.preference_cli audit --jsonl projects/single-gpu-finetuning/preference.example.jsonl --require-splits train,validation,test --output outputs/preference-audit.json
~~~

生产准备不把 combined 文件直接交给 trainer。审计身份读取 `preference.train.example.jsonl` 与含三种 split 的 combined artifact，要求前者逐记录、顺序敏感地等于后者的 **binary train subset**；train 中的 tie/invalid 会留在审计 artifact，但不会被强制改成 winner。它还要求显式 lexical profile/阈值、source policy 与固定 decision time。生成的 readiness v2 不含 held-out 原文：

~~~powershell
python -m about_llm.preference_cli prepare-training --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --audit-jsonl projects/single-gpu-finetuning/preference.example.jsonl --profile nfc_whitespace --ngram-size 5 --threshold 0.9 --governance-policy projects/single-gpu-finetuning/governance-policy.example.json --governance-evaluated-at 2026-08-06T12:00:00Z --output-dir outputs/preference-prepare
~~~

Preference lexical gate 比较 prompt↔prompt，以及跨记录 candidate A/B 的四种组合，避免候选从 A 换到 B 后漏检；它不比较 prompt↔candidate。报告给出 Unicode code-point character n-gram set 的精确 Jaccard 分子/分母、record pair 与 comparison count。实现仍是 O(N²) 全对 reference，normalization/阈值未按真实域校准，不能发现所有语义改写、翻译或污染。Governance gate 对 prompt 和两侧原始 candidate 执行同一 source/license/purpose/expiry/risk registry 与有限敏感候选扫描，报告不保存命中原文；registry 不是法律意见，未命中不证明无 PII/secret。

原始 judgment 固定样例独立于最终 pair 标签。下面的 gate 绑定 validation/test case，要求每个 pair 恰好 4 个不同 annotator、A-first/B-first 各至少 2 个，并阻断未知/train pair、重复 annotator-pair、rubric mismatch 与未盲化/非独立声明。通过后报告 raw pairwise agreement、Fleiss’ κ、明确二元分母的 A-selection position effect，以及按 case 聚类的 percentile bootstrap：

~~~powershell
python -m about_llm.preference_cli evaluate-judgments --cases-jsonl projects/single-gpu-finetuning/preference.example.jsonl --judgments-jsonl projects/single-gpu-finetuning/preference-judgments.example.jsonl --case-splits validation,test --judgments-per-pair 4 --minimum-per-order 2 --bootstrap-samples 10000 --bootstrap-seed 17 --output outputs/preference-judgment-report.json
~~~

这 8 条 judgment 是本仓库为检查统计口径准备的固定样例，不是人类标注。字段中的 blind/independent 是声明而非外部证明，顺序覆盖也不证明真正随机分配；因此示例 effect 不是因果 position bias，agreement/κ 也不证明 rubric 正确。

`smoke_trl_dpo.py` 使用真实 TRL 0.29 DPOTrainer、本地 tokenizer、随机 tiny GPT-2 和冻结 reference，不下载模型。它验证 train/combined readiness、`label → chosen/rejected` 映射、目标 tokenization prefix gate、collator 的 chosen-first/rejected-second 顺序、prompt/completion mask、相同 policy/reference 的初始 loss≈`log(2)`、optimizer 后 tiny-pair loss 下降以及 reference 参数不变：

~~~powershell
python projects/single-gpu-finetuning/smoke_trl_dpo.py
~~~

固定样例中的 `good/bad` 是本仓库准备的对照信号，不是人类偏好、对齐质量或安全标签。该闭环不证明目标模型 DPO、CUDA、真实域 length/position bias、annotator agreement 或生产收敛；tie/invalid 只被保留用于审计，不会被静默转成 DPO winner。

## 固定 Qwen 真实权重 TRL DPO 单步控制

`qwen2.5-0.5b-dpo.control.json` 把同一套 binary train/readiness artifact 与固定 Qwen2.5-0.5B-Instruct revision 绑定。它先重哈希 7-file/999,586,347-byte checkpoint、1,325-byte train JSONL 与 2,895-byte readiness JSON，再以目标 tokenizer 逐 token 核对两条 prompt/chosen/rejected。TRL collator 必须得到 `[4,28]`，四条 completion mask 各只覆盖 5 个 completion token，不能静默截断。

~~~powershell
python projects/single-gpu-finetuning/run_qwen_target_dpo_control.py --verify projects/single-gpu-finetuning/qwen2.5-0.5b-dpo.recorded-report.json
~~~

2026-08-13 录制的实验在 CPU FP32 eager、TRL 0.29.1/PEFT 0.20.0 下，为 24 层 `q_proj/v_proj` 注入 `r=4, alpha=8` LoRA，共 270,336 个 trainable parameters。一次真实 backward/AdamW step 的 96 个梯度张量全部 finite，48 个 B tensors 从全零变为全非零，共 98,304 个非零 elements；同 batch loss 从 `0.693147` 降至 `0.333352`，两条 chosen-relative margin 为 `8.566292/10.016453`。基座参数、排除 LoRA 后的完整 `state_dict`、规范化后的 model config 与 generation config 前后指纹分别 exact；report 指纹为 `sha256:3cafbade…b549b7bc`。

Transformers 4.57 会在 `train()` 前把 model/generation 的 BOS/PAD/EOS 对齐到 tokenizer；实验在 baseline 前显式执行并核对这一步，避免把配置变化误当训练效果。两次 reference forward 内 PEFT adapter layer 状态都实测为 disabled，但前后 reference log-prob replay 的 max-abs drift 为 `0.547077`，并非 bitwise equal；由于冻结参数、non-adapter state 与 config 指纹均未变，报告把它作为数值 replay drift 单列，**不能**改写成 reference 权重改变或确定性复现。该结果仍只使用本仓库准备的 `good/bad` pair；不证明人类偏好有效、数据代表性、收敛、泛化、安全、QLoRA/CUDA/vLLM 或生产就绪，也没有导出 adapter/optimizer/RNG/resume artifact。

真实 LoRA/QLoRA DPO 入口只读取 binary train JSONL 与 preference readiness。加 `--data-preflight-only` 时不会导入训练依赖、下载 tokenizer 或加载模型；正式运行会先下载固定 revision 的 tokenizer，再复现 TRL 0.29 conversational tokenization：prompt 使用 `add_generation_prompt=True`，prompt+chosen/rejected 各自完整渲染。仓库把 TRL 只记录 warning 的 prompt-prefix mismatch 升级为失败，并拒绝空 completion、chosen/rejected token 完全相同以及任何会触发 `max_length` 截断的 pair；通过仍不证明 template 的语义正确或目标模型质量。`--qlora` 使用 NF4/double quant 与单模型 PEFT adapter-disabled reference 路径，当前无 CUDA 环境未实跑：

~~~powershell
python projects/single-gpu-finetuning/train_trl_dpo.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --readiness-json outputs/preference-prepare/preference-training-readiness.json --output-dir outputs/dpo-run --data-preflight-only
python projects/single-gpu-finetuning/train_trl_dpo.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/preference.train.example.jsonl --readiness-json outputs/preference-prepare/preference-training-readiness.json --output-dir outputs/dpo-run --qlora
~~~

Preference readiness 绑定 exact/group split audit、binary train identity、声明的 lexical candidate policy 与 source/sensitive governance 决策。正式数据仍需由有权限的人复核候选，并执行 consent、法律许可、语义近重复、position/length bias 和人工一致性审查；不能把 readiness pass 写成“数据已获法律许可”“无敏感信息”或“没有语义污染”。

SFT 数据准备与训练同样是两个权限域。SFT `prepare-training` 接受严格的 train-only JSONL、combined JSONL、显式 near-duplicate policy、governance policy 和固定 decision time，在可读 validation/test 的审计进程验证有序 train 子集绑定并生成 readiness v3；v3 的 exact identity、lexical view 与有限 sensitive-candidate surface 都纳入 tool calls 和 tool schemas。两个 trainer 只接受 train JSONL 与 readiness，不读取 combined 原文。readiness 严格拒绝重复/未知字段、错误版本、失败 gate、指纹篡改和陈旧绑定；通过不等于语义无重复、法律许可或无敏感信息，无密钥 SHA-256 也不认证签发者。

tokenizer 下载后、模型权重加载前，入口对每条 Python record 调用目标 `apply_chat_template(..., tools=..., return_assistant_tokens_mask=True)`，拒绝缺失/全零/错长/非二值 mask 与静默右截断，并写出 `sft-template-mask-audit.json`。随后只把预计算整数特征交给 Arrow；Trainer 建立后，入口实际调用 configured collator，逐位置核对监督 label、非 assistant 与 padding 的 `-100`，写出 `sft-final-label-audit.json` 后才开始训练。报告仍不能独立证明模板作者选择了语义正确的 span，正式数据应继续抽样可视化 token/mask/label。

若 checkpoint 模板不支持 generation mask，先审核并版本化一个本地 Jinja template，再通过 `--chat-template-path <template.jinja>` 同时交给 preflight 与 trainer；不要在代码里临时拼接另一个格式。自定义模板仍必须匹配该 checkpoint 的 special tokens 和部署格式。

~~~powershell
python -m about_llm.finetuning_cli prepare-training --train-jsonl projects/single-gpu-finetuning/train.example.jsonl --audit-jsonl projects/single-gpu-finetuning/audit.example.jsonl --profile nfc_whitespace --ngram-size 5 --threshold 0.9 --governance-policy projects/single-gpu-finetuning/governance-policy.example.json --governance-evaluated-at 2026-08-06T12:00:00Z --output-dir outputs/sft-prepare
python projects/single-gpu-finetuning/train_trl_sft.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/train.example.jsonl --readiness-json outputs/sft-prepare/sft-training-readiness.json --output-dir outputs/sft-run
~~~

先给第二条命令加 `--data-preflight-only`，可在不导入训练依赖、不下载模型时验证 train/readiness 边界。生产部署应让审计身份可读 combined，让训练身份只读 train 与经受控通道发布的 readiness；若攻击者能整体替换 readiness，它也能重算其中所有无密钥 hash，需用 ACL、签名或独立审计日志补足真实性。

示例数据只用于检查 schema、exact/binding、未校准 lexical candidate 和 governance 控制流，不能训练出有用模型。
真实实验仍必须人工校准/复核 near duplicate 与敏感检测器，由有权限的负责人核验许可、consent、隐私和目标分布。

## 容量与风险

QLoRA 不是全部 4-bit；adapter、梯度、optimizer、激活和部分算子仍是高精度。序列长度与 batch 会显著增加激活。先用极小 batch dry-run，再逐步增大，并记录峰值。

`train_qlora.py` 提供无需 GPU/下载的 `--estimate-only`，拆分量化基座、adapter/optimizer、激活和运行时预留。估算用于筛掉明显不可行配置，不能替代目标 GPU 上的峰值实测：模型结构、attention kernel、词表 logits、bitsandbytes 版本和内存碎片都会改变结果。

~~~powershell
python projects/single-gpu-finetuning/train_qlora.py --model-id <model> --revision <commit> --num-parameters 7000000000 --num-layers 32 --hidden-size 4096 --max-length 1024 --estimate-only
~~~

真实训练去掉 `--estimate-only` 并增加 `--train-jsonl`、`--readiness-json` 与 `--output-dir`；readiness 先由上面的 `prepare-training` 生成。入口固定 NF4、double quant、BF16/FP16 compute、gradient checkpointing、预计算 assistant-only labels 和显式 target modules。当前环境没有 CUDA，因此只验证了估算、参数路径、Arrow 前预分词与 CPU collator 测试；真实 QLoRA 成功与峰值显存仍必须在目标消费级 GPU 上记录。

OOM 降级顺序是：micro-batch 降到 1（用梯度累积保持有效 batch）、启用 checkpoint/高效 attention、基于长度分布缩短序列、减少 target/rank，最后才换小模型。每次变化都要进入实验配置，不能一边降级一边沿用旧基线名称。

微调不能替代最新事实检索，也不能单独保证“无幻觉”。领域提升必须与通用能力、安全拒答和未见模板一起评测。

## MiniGPT 精确训练恢复控制

在把目标 LoRA/QLoRA 扩展为可恢复训练前，先运行一个状态面可完整枚举的 CPU 实验：

~~~powershell
python projects/single-gpu-finetuning/minigpt_resume_toy.py
python projects/single-gpu-finetuning/minigpt_resume_toy.py --artifact-path artifacts/minigpt/training.allmtrn
~~~

默认固定样例使用 7 条×5 token 数据、batch 2、dropout 0.2 和 6 次 AdamW update；一条路径不中断运行，另一条在第 3 次 update 后保存、重载再继续。固定输出的 batch 顺序为 `(6,5),(2,1),(4,0),(1,0),(6,5),(3,4)`，epoch 为 `0,0,0,1,1,1`，LR 为 `0.003,0.0026,0.0022,0.0018,0.0014,0.001`。53,917-byte artifact 包含 11,341-byte manifest、42,520-byte payload 和 51 个 tensor；恢复时和最终态的模型参数、每参数 AdamW step/一阶/二阶矩、当前 LR、permutation/cursor/epoch、data-generator RNG 与 Torch CPU dropout RNG 均逐位相等，loader 也不改变调用者的 Torch RNG。

格式绑定 Byte-BPE payload/config/tied weights、训练 identity 与数据 shape+content fingerprint，但不嵌入 7×5 数据 payload；调用者必须提供完全相同的数据。它只允许 zero-grad optimizer boundary，且只覆盖当前 CPU FP32 MiniGPT、单 AdamW param group、线性 per-update scheduler；因为 AdamW step 是 FP32 tensor，总 update 限制为 $2^{24}$，避免不可精确表示的 step。没有 Python/NumPy/CUDA RNG、AMP scaler、gradient accumulation、DataLoader worker/prefetch、distributed/sharded state、目标 checkpoint 或 CUDA 证据。六个 loss 为 `5.560535, 5.561857, 5.515058, 5.525568, 5.465405, 5.535903`，并不单调下降；bit-exact resume 只证明当前训练状态恢复契约，不证明收敛或质量。SHA-256 未加密钥，不认证来源；exclusive-create 与 file `fsync` 也不证明 crash/断电原子发布。

## 后续里程碑

1. 把已完成的固定 Qwen 多轮/tool SFT final-label 与 DPO 机制控制扩展到代表性真实数据、更多 provider schema 和独立 held-out 评测；
2. 在目标 CUDA 环境记录 QLoRA 实测峰值和 OOM 降级曲线；
3. 在已完成的目标 Qwen CPU LoRA 单步 save/reload 之上补 optimizer/RNG exact-resume、merge、QLoRA/CUDA 和 tokenizer/runtime 一体化发布；tiny CPU 实验已证明 consumed 领先 committed 时必须 replay，并为 gradient sidecar 增加 manifest-last completeness gate，但尚未实现 worker/adapter/optimizer/sample 的原子一致提交；
4. 与 RAG/Prompt 基线的统一评测报告。

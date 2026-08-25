# Single-GPU Fine-tuning

这个项目帮助你在一张消费级显卡上走通 `数据 → labels → LoRA/QLoRA → Adapter → held-out 评测`。
重点不是尽快得到一个 Adapter 文件，而是知道它学了什么、是否真的优于基线，以及能否在新进程中可靠重载。

第一次学习请从[项目教学页](../../docs/practice/projects/single-gpu-finetuning.md)开始。那里用一条售后对话微调任务串起完整过程；本页只保留运行入口、输入输出和排错索引。

## 第一次运行

先安装 CPU 与常规 LoRA 路径所需的依赖：

```powershell
python -m pip install -e ".[dev,torch,transformers,finetune]"
python scripts/doctor.py
```

然后运行两个不下载公开模型的检查：

```powershell
python -m about_llm.finetuning_cli audit `
  --jsonl projects/single-gpu-finetuning/audit.example.jsonl `
  --require-splits train,validation,test `
  --output outputs/sft-split-audit.json

python projects/single-gpu-finetuning/smoke_trl_sft.py
```

第一条命令检查数据划分、对话结构和重复记录。第二条使用本地随机 tiny GPT-2，真实执行
assistant-only labels、一次参数更新和 Adapter 保存。你应当能沿输出追踪：

```text
messages → token IDs → labels → loss → adapter
```

这两条记录只是教学样例。它们能检查训练链路是否连通，不能证明 Qwen 已经完成微调或模型质量有所提升。

如果你使用 RTX 3070 Laptop 和 Qwen3-0.6B，请先按教学页第 1 步生成
`artifacts/sft-prepare/sft-training-readiness.json`。然后运行目标 tokenizer 预检：

```powershell
python projects/single-gpu-finetuning/train_trl_sft.py `
  --model-id Qwen/Qwen3-0.6B `
  --revision c1899de289a04d12100db370d81485cdf75e47ca `
  --train-jsonl projects/single-gpu-finetuning/train.example.jsonl `
  --readiness-json artifacts/sft-prepare/sft-training-readiness.json `
  --chat-template-path projects/single-gpu-finetuning/qwen3-0.6b-c1899de-generation-aware-sft.jinja `
  --output-dir artifacts/qwen3-sft-tokenization-preflight `
  --max-length 512 `
  --tokenization-preflight-only
```

完整的一步训练命令、三个输出报告和显存数字的含义见[项目教学页](../../docs/practice/projects/single-gpu-finetuning.md#run)。

## 完整主线

需要做一个真实项目时，按下面的顺序推进：

1. 写下 base、Prompt 和 RAG 基线，并固定 held-out cases。
2. 审计 split、重复数据、来源、许可和敏感信息。
3. 用目标 tokenizer 与 chat template 检查最终 labels。
4. 先跑 tiny smoke，再在目标 GPU 上只做一次 update。
5. 在新进程中重新加载 base 和 Adapter。
6. 用同一组 held-out cases 比较基线与 Adapter，最后决定发布或回退。

每一步的解释、命令和完成信号见[端到端主线](../../docs/practice/projects/single-gpu-finetuning.md#run)。

## 根据当前问题选择脚本

不要按文件名把所有脚本依次跑一遍。先找你正在回答的问题，再运行对应的最小实验。

| 你想确认什么 | 入口 |
|---|---|
| Split、结构和 exact duplicate 是否正确 | `python -m about_llm.finetuning_cli audit ...` |
| 训练数据是否满足近重复与治理要求 | `python -m about_llm.finetuning_cli prepare-training ...` |
| Assistant 区域是否真的进入 labels | `smoke_trl_sft.py`、`run_qwen_target_sft_label_control.py` |
| Base 是否冻结，Adapter 能否保存、重载和合并 | `smoke_peft.py`、`run_qwen_target_lora_control.py` |
| 目标模型的 LoRA/QLoRA 能否启动 | `train_trl_sft.py`、`train_qlora.py` |
| Preference 数据和 DPO 路径是否连通 | `smoke_trl_dpo.py`、`run_qwen_target_dpo_control.py` |
| 变长序列的梯度累积是否按 token 加权 | `gradient_accumulation_toy.py` |
| DDP、`no_sync` 或 AMP 为何产生不同梯度 | `ddp_*_control.py`、`amp_grad_scaler_control.py` |
| 恢复训练后为何漏样本或参数漂移 | `checkpoint_resume_control.py`、`dataloader_prefetch_resume_control.py`、`optimizer_commit_resume_control.py` |
| PPO、reward model 与 reward hacking 如何形成最小闭环 | `smoke_*ppo.py`、`reward_model_toy.py`、`train_reward_model.py` |

机制实验的推荐顺序和现象解释集中在[深挖机制实验](../../docs/practice/projects/single-gpu-finetuning.md#controls)。
精确数值、固定样例和适用范围见[项目实验台账](../../docs/evidence/project-controls.md)、
[Qwen 证据台账](../../docs/evidence/qwen-controls.md)和[对齐证据台账](../../docs/evidence/alignment-controls.md)。

## 主要输入与输出

| 文件或目录 | 用途 |
|---|---|
| `train.example.jsonl` | 两条 train-only 教学记录，不能作为有效训练语料 |
| `audit.example.jsonl` | 包含 train、validation、test 的数据审计样例 |
| `preference*.example.jsonl` | Preference 数据、判断和 readiness 样例 |
| `governance-policy.example.json` | 来源、许可和风险标签的示例决策规则 |
| `qwen2.5-generation-aware-sft.jinja` | 标出 assistant generation span 的教学模板 |
| `qwen3-0.6b-c1899de-generation-aware-sft.jinja` | 为固定 Qwen3 revision 标出 assistant 监督区间 |
| `*.control.json` | 固定实验所使用的输入和身份 |
| `*.recorded-report.json` | 已录制结果，可在没有目标权重时离线核对 |
| `sft-training-run.json` | 记录单次训练终态、步数、依赖版本、参数量和进程内 CUDA 显存 |
| `artifacts/`、`outputs/` | 本地运行生成的报告、Adapter 和训练工件 |

正式发布包至少要绑定底座版本、tokenizer、对话模板、生成配置、数据身份、训练配置和 Adapter 文件。
仅保存 `adapter_model.safetensors` 不足以复现训练或推理行为。

## 用固定 Qwen 版本核对训练链路 { #target-qwen-sft-label-control }

仓库保存了 Qwen2.5-0.5B-Instruct 的三组录制报告。它们用于离线核对 SFT labels、LoRA 和 DPO 的执行路径，
不表示新项目必须选用 Qwen2.5。

```powershell
python projects/single-gpu-finetuning/run_qwen_target_sft_label_control.py `
  --verify projects/single-gpu-finetuning/qwen2.5-0.5b-sft-label.recorded-report.json

python projects/single-gpu-finetuning/run_qwen_target_lora_control.py `
  --verify projects/single-gpu-finetuning/qwen2.5-0.5b-lora.recorded-report.json

python projects/single-gpu-finetuning/run_qwen_target_dpo_control.py `
  --verify projects/single-gpu-finetuning/qwen2.5-0.5b-dpo.recorded-report.json
```

这三组报告分别回答“labels 是否正确”“LoRA 能否更新并重载”和“DPO 的单步路径是否连通”。它们使用不同输入，
不能拼成一次完整的 SFT → DPO → 部署实验。

切换到 Qwen3、Llama 或其他 checkpoint 时，还要重新确认模块名称、对话模板、特殊 token、推理 runtime 支持和许可。

## 常见故障

| 现象 | 先检查 |
|---|---|
| Assistant mask 全零 | Chat template 是否提供 generation span，最终 collator batch 中 labels 是否仍为 `-100` |
| Loss 在下降，但输出没有改善 | 是否只看了 train loss；改用固定 held-out cases 比较 base 与 Adapter |
| Base 参数发生变化 | Optimizer 参数组是否只包含 Adapter；保存前后比较 base fingerprint |
| Adapter 能保存但不能独立加载 | Base revision、target modules、tokenizer/template 或 PEFT config 是否漂移 |
| 显存估算能放下，实际仍 OOM | 记录 peak allocated/reserved；依次降低长度、micro-batch、rank，不要同时修改所有变量 |
| 梯度累积结果随 batch 切法变化 | 检查有效 token 总数和 reduction，运行 `gradient_accumulation_toy.py` |
| `no_sync` 没有减少通信 | Forward 是否也位于 `no_sync` 上下文内 |
| AMP 后梯度异常偏小 | 是否先 `unscale_` 再 clip；overflow 时 optimizer 与 scheduler 是否一起跳过 |
| Resume 后漏数据 | 区分 emitted、consumed 和 optimizer-committed cursor |
| 某个 rank overflow 后状态分叉 | Overflow/skip 决策是否在所有 rank 之间达成一致 |

更完整的故障—实验映射见[项目教学页](../../docs/practice/projects/single-gpu-finetuning.md#controls)。

## 运行检查

数据、LoRA 和导出路径的快速回归：

```powershell
python -m pytest `
  tests/test_finetuning_cli.py `
  tests/test_sft_readiness.py `
  tests/test_lora.py `
  tests/test_peft_export.py -q
```

内容与链接检查：

```powershell
python scripts/check_docs.py
python scripts/check_content_accuracy.py
```

GPU、QLoRA、目标模型和长时间训练不进入默认 CPU 检查。你需要在自己的硬件上保存成功配置、失败边界、
峰值显存、运行时间和 held-out 评测结果，才能把它们写成该环境下的实验结论。

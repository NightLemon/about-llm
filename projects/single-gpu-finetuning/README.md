# Single-GPU Fine-tuning

目标：在单张消费级 GPU 上完成可比较、可回归的领域 SFT/LoRA/QLoRA，而不是只得到一个 adapter 文件。

## 已完成的机制基线

src/about_llm/finetuning/lora.py 从零实现 LoRA Linear：

- 基座权重冻结；
- B 零初始化，初始函数与基座一致；
- 只保存 adapter 和必要元数据；
- 合并为普通 Linear 后数值等价；
- 测试证明 optimizer 不会更新基座。

~~~powershell
pytest tests/test_lora.py
~~~

## 实验协议

至少比较四个系统：

1. base + zero/few-shot；
2. base + RAG（若任务依赖事实）；
3. PEFT LoRA/QLoRA；
4. 全参数微调或高预算参考（显存允许时）。

保持同一 chat template、生成参数和测试集。报告任务指标、格式合法率、通用能力回归、训练/峰值显存、耗时和 adapter 大小。

## 单卡数据契约

每条样本保留 id、来源、许可、messages、任务类型与切分。训练只对 assistant 区域计算 loss；padding、system、user、tool 是否 mask 必须通过 token 级检查。按来源或用户划分测试，禁止近重复跨集合。

## 推荐递进

- 机制：本仓库 LoRALinear + 微型 GPT；
- 实用：Transformers + PEFT，对 0.5B–3B 模型做短序列 LoRA；
- 显存优化：4-bit 基座、gradient checkpointing、paged optimizer；
- 完整实验：数据卡、seed、checkpoint、早停、回归和合并/加载测试。

## PEFT 离线验证

smoke_peft.py 用随机初始化 tiny GPT-2 实际运行 PEFT LoRA，检查 trainable fraction、loss 下降、adapter state 和 merge 数值误差，不下载模型：

~~~powershell
python projects/single-gpu-finetuning/smoke_peft.py
~~~

## TRL 单卡入口

train_trl_sft.py 接受 messages JSONL、固定 model revision 和显式 LoRA target modules。它启用 assistant_only_loss。TRL 当前要求模板能产生 generation/assistant mask；对部分已知模型族可自动修补模板，其他模板需显式包含 `{% generation %}` / `{% endgeneration %}`。无论是否自动修补，都应在正式训练前检查实际 `assistant_masks`/labels，不能只凭模板名称假设正确。

~~~powershell
python projects/single-gpu-finetuning/train_trl_sft.py --model-id <model> --revision <commit> --train-jsonl projects/single-gpu-finetuning/train.example.jsonl --output-dir outputs/sft-run
~~~

示例数据只验证 schema，不能训练出有用模型。真实实验必须划分验证/测试集并检查近重复。

## 容量与风险

QLoRA 不是全部 4-bit；adapter、梯度、optimizer、激活和部分算子仍是高精度。序列长度与 batch 会显著增加激活。先用极小 batch dry-run，再逐步增大，并记录峰值。

`train_qlora.py` 提供无需 GPU/下载的 `--estimate-only`，拆分量化基座、adapter/optimizer、激活和运行时预留。估算用于筛掉明显不可行配置，不能替代目标 GPU 上的峰值实测：模型结构、attention kernel、词表 logits、bitsandbytes 版本和内存碎片都会改变结果。

~~~powershell
python projects/single-gpu-finetuning/train_qlora.py --model-id <model> --revision <commit> --num-parameters 7000000000 --num-layers 32 --hidden-size 4096 --max-length 1024 --estimate-only
~~~

真实训练去掉 `--estimate-only` 并增加 `--train-jsonl` 与 `--output-dir`。入口固定 NF4、double quant、BF16/FP16 compute、gradient checkpointing、assistant-only loss 和显式 target modules。本仓库当前环境没有 CUDA，因此只验证了估算、参数路径和 CPU 测试；真实 QLoRA 成功与峰值显存仍必须在目标消费级 GPU 上记录。

OOM 降级顺序是：micro-batch 降到 1（用梯度累积保持有效 batch）、启用 checkpoint/高效 attention、基于长度分布缩短序列、减少 target/rank，最后才换小模型。每次变化都要进入实验配置，不能一边降级一边沿用旧基线名称。

微调不能替代最新事实检索，也不能单独保证“无幻觉”。领域提升必须与通用能力、安全拒答和未见模板一起评测。

## 后续里程碑

1. 对话模板与 assistant-only label 可视化；
2. 在目标 CUDA 环境记录 QLoRA 实测峰值和 OOM 降级曲线；
3. checkpoint 恢复、adapter 合并和导出；
4. 与 RAG/Prompt 基线的统一评测报告。

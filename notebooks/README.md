# Notebooks

Notebook 用于观察现象，不承载唯一实现。核心逻辑位于 src/about_llm/ 并由 tests/ 覆盖。

四本 Notebook 使用同一学习节奏：**先预测 → 运行透明基线 → 与另一实现对账 → 故意破坏一个条件 → 解释能证明与不能证明什么**。不要跳过预测和反例直接执行全部单元格；那会把实验重新变成演示。

执行约定：

- 从仓库根目录启动 Jupyter。
- 安装 Notebook profile：`python -m pip install -c constraints/ci.txt -e ".[dev,torch,jax]"`。
- 先运行 `python scripts/doctor.py --profile notebooks`。
- 每本必须能 Restart & Run All。
- 默认不访问网络、不下载模型、不使用付费 API。
- 输出只展示小型结果，不提交大型二进制或模型权重。

## 逐本要求与成功特征

| Notebook | 直接依赖 | 当前 Windows CPU 参考时间 | 成功特征 |
|---|---|---:|---|
| `01_attention_three_ways.ipynb` | NumPy、PyTorch、JAX、about_llm | 约 17 秒 | 三方前向对账；移除 mask 出现未来概率；移除缩放改变分布 |
| `02_minigpt_forward.ipynb` | PyTorch、JAX、about_llm | 约 18 秒 | byte round-trip、logits shape 与 causal 正反对照通过 |
| `03_rag_retrieval_and_evaluation.ipynb` | NumPy、about_llm | 约 5 秒 | 安全融合无跨租户结果；故意混合会泄漏；错误 case 让指标下降 |
| `04_sft_sample_lifecycle.ipynb` | PyTorch、about_llm | 约 17 秒 | assistant-only label 边界可见；LoRA 包装后初始函数不变；冻结基座逐元素不变；adapter 独立重载 bit-exact |

时间是 2026-08-28 在 Windows 11、Python 3.12.10、无 CUDA 的当前仓库环境实测，不是性能承诺；首次 JAX/PyTorch 导入、CPU 和杀毒软件都会改变耗时。逐本执行：

~~~powershell
python scripts/execute_notebooks.py --pattern "notebooks/01_attention_three_ways.ipynb"
python scripts/execute_notebooks.py --pattern "notebooks/02_minigpt_forward.ipynb"
python scripts/execute_notebooks.py --pattern "notebooks/03_rag_retrieval_and_evaluation.ipynb"
python scripts/execute_notebooks.py --pattern "notebooks/04_sft_sample_lifecycle.ipynb"
~~~

全量执行使用 `python scripts/execute_notebooks.py`，每本默认超时 180 秒。Windows 上 pyzmq 可能打印 Proactor selector thread 或本机未加密 TCP kernel 警告；本地受信机器上的这类警告不等于单元执行失败，但共享主机或远程 kernel 必须使用受保护的连接配置。

## 覆盖边界与下一步

四本 Notebook 覆盖“观察张量/模型前向/检索现象/SFT 样本生命周期”的入门主线，不代表仓库全部主题都有 Notebook。训练恢复、PEFT 导出、云 API、Agent runtime、推理调度和评测门禁需要文件、进程、数据库或故障注入，优先使用 `projects/` 中的可运行入口与 `tests/`，不要为了 Notebook 形式复制唯一实现。

| 想学习的主题 | 首选入口 |
|---|---|
| Attention、MiniGPT、RAG 检索、SFT 样本 | 本目录四本 Notebook |
| Tokenizer、MoE、generation contract | `projects/transformers-basics/` |
| LoRA/QLoRA、偏好优化、训练恢复 | 先看 `notebooks/04`，再进 `projects/single-gpu-finetuning/` |
| Agent 权限、审批、恢复、outbox | `projects/safe-agent/` |
| batching、量化、KV cache、服务指标 | `projects/inference-serving/` |
| 评测统计、证据验证、发布门禁 | `projects/evaluation-gate/` |

完整的实验顺序、交付物和证据边界见 `docs/practice/labs.md`；项目成熟度见 `docs/practice/project-index.md`。

# 环境与硬件矩阵

**相关导航**：[学习路径](learning-paths.md) · [仓库地图](repo-map.md) · [实验与项目](../practice/labs.md) · [工程项目索引](../practice/project-index.md)
{ .doc-nav }

## 推荐基线

- Python 3.10–3.12；优先使用独立虚拟环境。
- 文档/单测：CPU、8 GB RAM 即可。
- 从零微型 GPT：CPU 可跑机制，8 GB+ GPU 可加速。
- 量化推理/LoRA：建议 NVIDIA GPU；实际显存依模型、序列、batch 和实现变化。
- vLLM 主要面向 Linux 与受支持加速器；Windows 用户建议 WSL2 或远程 Linux。

## 显存不是只看权重文件

推理显存 = 权重 + KV Cache + 激活/工作区 + runtime。训练还需梯度、优化器状态和保存的激活。以下只作实验选型量级，不是容量承诺：

| 可用显存 | 建议路线 |
|---|---|
| 无 GPU | 微型模型、BM25/RAG、API、CPU 量化 runtime |
| 6–8 GB | 0.5B–3B 量化推理；极小模型 LoRA |
| 10–16 GB | 3B–8B 4-bit 推理；短序列 QLoRA |
| 20–24 GB | 7B–14B 量化推理；7B 级 QLoRA 的受控实验 |

上下文变长、batch 增加、视觉输入或 optimizer 配置都会显著改变需求。运行前用目标框架的 dry-run 测峰值，不用表格替代实测。

## 安装策略

核心包按需安装，避免把 JAX、CUDA、vLLM 和多个 Agent 框架塞进一个不可维护环境：

| Profile | 用途 | Extras | Doctor profile |
|---|---|---|---|
| docs | 只读/构建教材 | `docs` | `docs` |
| cpu-starter | Tokenizer 与 NumPy 机制 | 无 | `cpu-starter` |
| notebooks | 四本离线 Notebook | `dev,torch,jax` | `notebooks` |
| full-ci | 本地 CPU 全门禁 | 见下方完整命令 | `full-ci` |

安装时都使用 `constraints/ci.txt`。例如：

```powershell
python -m pip install -c constraints/ci.txt -e ".[docs]"
python scripts/doctor.py --profile docs
```

需要在本机运行全部 CPU 门禁时：

```powershell
python -m pip install -c constraints/ci.txt -e `
  ".[docs,dev,torch,jax,transformers,finetune,rag,langchain,llamaindex,api,evaluation,agents]"
python scripts/doctor.py --profile full-ci
```

`doctor` 的 `fail` 会返回非零退出码并给出修复命令；`warn` 表示任务仍可运行，但环境没有完全隔离。
`constraints/ci.txt` 是 CI 直接依赖的复核快照，不是传递依赖或 CUDA wheels 的跨平台 lock。

MkDocs 保持 1.x，以匹配当前 Material 9.x theme、plugins 与 overrides。JAX GPU 版按官方平台说明安装；CPU 版
可使用 `jax` extras。vLLM 的平台约束变化较快，不固定进通用依赖组，具体项目单独记录已验证组合。

## 云 API

Provider adapter 统一消息、超时、重试、usage 和错误，但不会假装不同供应商所有功能一致。密钥只放环境变量或秘密管理服务。运行付费实验前：

1. 检查模型 id、区域和数据保留政策。
2. 设置最大请求数、输入/输出 token 和费用预算。
3. 用 2–3 条样本 smoke test。
4. 保存请求 id、模型版本和 usage，不记录不必要敏感正文。

## 可复现信息

实验报告必须保存 Python、包版本、OS、设备、dtype、模型 revision、tokenizer、随机种子、输入/输出长度和配置。CUDA 还记录 driver、runtime 与 GPU 名称。

~~~powershell
python scripts/doctor.py --profile cpu-starter
python -m pip freeze > outputs/environment-lock.txt
~~~

## 常见环境错误

- torch.cuda.is_available 为 false：wheel 可能是 CPU 版，或驱动不兼容。
- CUDA OOM 后仍 OOM：旧进程/张量未释放，或显存碎片；先确认进程和峰值位置。
- JAX 看不到 GPU：JAX wheel/plugin 必须匹配平台与 CUDA。
- vLLM Windows 安装失败：使用官方支持的 Linux/WSL2 环境。
- FAISS Windows wheel 不稳定：教学项目提供 NumPy/SQLite 基线，或使用 WSL2。

## 环境就绪后

- CPU 环境：从[注意力、MiniGPT、SFT 与 RAG 实验](../practice/labs.md)开始。
- NVIDIA GPU：进入[单卡微调](../training/peft-qlora-engineering.md)或[推理服务](../systems/vllm-serving.md)，先按目标 workload 做显存 dry-run。
- 云 API：先阅读[云模型 API 契约](../models/cloud-api-contracts.md)，再设置请求、token 与费用预算。
- 出现版本或平台差异：回到[内容准确性台账](../reference/accuracy.md)，区分文档核对与目标环境实测。

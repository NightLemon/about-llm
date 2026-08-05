# 环境与硬件矩阵

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

~~~powershell
python -m pip install -e ".[docs,dev,torch]"
python -m pip install -e ".[transformers,finetune]"
python -m pip install -e ".[rag,api]"
~~~

JAX 单独按官方平台说明安装；CPU 版可使用 jax 依赖组。vLLM 更新快且平台约束强，不固定进通用依赖组，在对应项目记录已验证组合。

## 云 API

Provider adapter 统一消息、超时、重试、usage 和错误，但不会假装不同供应商所有功能一致。密钥只放环境变量或秘密管理服务。运行付费实验前：

1. 检查模型 id、区域和数据保留政策。
2. 设置最大请求数、输入/输出 token 和费用预算。
3. 用 2–3 条样本 smoke test。
4. 保存请求 id、模型版本和 usage，不记录不必要敏感正文。

## 可复现信息

实验报告必须保存 Python、包版本、OS、设备、dtype、模型 revision、tokenizer、随机种子、输入/输出长度和配置。CUDA 还记录 driver、runtime 与 GPU 名称。

~~~powershell
python scripts/doctor.py
python -m pip freeze > outputs/environment-lock.txt
~~~

## 常见环境错误

- torch.cuda.is_available 为 false：wheel 可能是 CPU 版，或驱动不兼容。
- CUDA OOM 后仍 OOM：旧进程/张量未释放，或显存碎片；先确认进程和峰值位置。
- JAX 看不到 GPU：JAX wheel/plugin 必须匹配平台与 CUDA。
- vLLM Windows 安装失败：使用官方支持的 Linux/WSL2 环境。
- FAISS Windows wheel 不稳定：教学项目提供 NumPy/SQLite 基线，或使用 WSL2。

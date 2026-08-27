"""Qwen3-0.6B × nano-vLLM 实验的轻量命令行入口。

``collect`` 在真实 CUDA 环境采集请求 trace，``verify`` 离线检查报告不变量，``explain``
输出学习导览。完整实现位于 ``about_llm.inference.nano_vllm_study``，这里负责暴露可运行命令。
"""

from __future__ import annotations

# 复用 package 中经过测试的实现，避免项目入口与库 API 复制两套逻辑。
from about_llm.inference.nano_vllm_study import main

if __name__ == "__main__":
    raise SystemExit(main())

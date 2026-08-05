# 单卡推理服务与压测

目标：用同一组 workload 比较 Transformers 与 vLLM，并正确区分 TTFT、TPOT、端到端延迟和系统吞吐。

## vLLM 服务

vLLM 的平台和版本兼容变化较快。先按官方说明在 Linux/WSL2 安装，再选择适合显存和许可证的模型。示例：

~~~bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
~~~

不要未经测量就把上下文和 memory utilization 拉满。先单请求 smoke test，再逐步增加输入长度和并发。

## OpenAI-compatible 压测

安装 API 依赖后运行：

~~~powershell
python -m pip install -e ".[api]"
python projects/inference-serving/benchmark_openai.py --model Qwen/Qwen2.5-0.5B-Instruct --requests 50 --concurrency 4
~~~

客户端使用流式 SSE：

- 首个非空 content delta 记录 first-token time；
- completion_tokens 必须来自服务端 usage；
- SSE chunk 不是 token：缺少 usage 时脚本明确失败，不输出伪精确 TPOT；
- throughput 使用整轮 benchmark wall time，不把单请求速度相加；
- 单 token 输出的 TPOT 为未定义，不伪报为 0。

## 公平比较协议

固定模型 revision、tokenizer、量化、prompt 集、输入/输出上限、温度、硬件与并发。先做质量等价检查，再比较性能。至少扫描：

- 输入长度：短、中、目标上限；
- 输出长度：短回答与长生成；
- 并发：1、2、4、8，直到尾延迟或 OOM 不可接受；
- prefix cache 开关；
- 量化与 KV dtype；
- Transformers generate 与 vLLM continuous batching。

报告 p50/p95 TTFT、TPOT、E2E、请求/秒、输出 token/秒、峰值显存、错误率和任务质量。平均延迟不足以做容量规划。

## 容量与故障

压测同时记录服务队列、GPU 利用率、KV cache usage 和 preemption。失败请求不能从统计中静默删除；应单列超时、429、OOM 与取消。客户端断开时验证服务停止生成。

## 已验证范围

度量聚合和 SSE 解析由 CPU 单元测试覆盖。实际 vLLM 命令需要受支持的 Linux/GPU 环境，当前 Windows CPU 验证不会伪装成 GPU 实测。

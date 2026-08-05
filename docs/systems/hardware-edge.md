# 硬件、性能模型与端侧部署

## 硬件资源

- FLOPs/s：算术吞吐，需看目标精度而非单一峰值。
- HBM/内存容量：决定权重、KV 和 batch 能否容纳。
- 内存带宽：权重/激活每秒可移动多少，decode 常受其限制。
- 互联带宽与延迟：NVLink、PCIe、InfiniBand 等影响并行通信。
- CPU、主存、存储与网络：tokenize、数据加载、offload 和请求传输也会成为瓶颈。

GPU、TPU、NPU/ASIC 在编程栈、支持精度、内核和成本上不同。规格表峰值只在特定数据类型和稠密结构成立。

## Roofline 直觉

算术强度 = 运算次数 / 搬运字节。若强度低，性能受内存带宽上限；强度高，才接近计算峰值。LLM prefill 的大矩阵乘容易 compute-bound；batch 较小的 decode 每步读取大量权重，常 memory-bound。提高 batch 能复用权重并增加算术强度，但也提高排队和 KV 内存。

理论延迟下界可粗估为：

\[
\max\left(\frac{\text{FLOPs}}{\text{有效计算吞吐}},
\frac{\text{搬运字节}}{\text{有效带宽}}\right)
\text{通信与调度}
\]

“有效”远低于峰值时，应分析 kernel、shape、融合、同步和数据布局。

## Kernel 与编译

算子融合减少启动和 HBM 往返；tiling 让数据进入更快的 shared memory/cache；Tensor Core 要求适当 dtype 和维度对齐；CUDA Graph 可减少重复 launch 开销。编译器通过图捕获、常量折叠、布局与 kernel 选择优化，但动态 shape、控制流和自定义算子可能导致 fallback。

性能报告必须含模型、dtype/量化、输入/输出长度、batch/并发、硬件、框架和版本。

## 多卡与拓扑

TP 的逐层 collective 延迟敏感，优先在高带宽域；PP 的 stage 边界传激活，适合跨较慢链路但有 bubble；DP 服务副本彼此独立，容易横向扩容。拓扑感知 placement 避免 collective 绕过慢链路。

## CPU 与端侧

CPU 推理依赖 SIMD、缓存、线程/NUMA、量化和内存映射。消费设备常使用 GGUF 等分片/量化格式与专用 runtime。端侧优势是离线、数据本地和低网络延迟；约束是 RAM、带宽、功耗、温度、电池和应用包大小。

移动 NPU/GPU 只支持特定算子/shape；未支持算子回退 CPU 会产生复制和延迟。选择小模型、短上下文、GQA、KV 量化和按需加载；实测冷启动、持续生成降频和后台资源竞争。

## Offload

权重或 KV 可在 GPU、CPU RAM、SSD 间分层，扩大可运行模型，但 PCIe/存储带宽会显著拖慢每 token。适合吞吐要求低或专家稀疏访问的场景。异步预取只有在传输能被计算隐藏时有效。

## 能源

功耗 × 时间得到能量，但完整环境影响还含硬件制造、冷却、水和地区电力结构。报告测量边界、PUE、硬件利用率和测量方法，避免用训练 FLOPs 直接声称精确碳排。

## 自测

1. 为什么 decode 常比 prefill 更受内存带宽限制？
2. 量化减少权重字节后，什么条件下能显著加速 CPU 推理？
3. 多卡模型 fit 进显存后，为什么仍可能比单卡小模型慢很多？

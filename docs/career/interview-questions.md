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

### 4. Prefill 与 decode 的瓶颈为何不同？

Prefill 对多个 token 做大矩阵乘，较易 compute-bound；decode 每步 batch/序列维小，却反复读权重和 KV，常 memory-bound。用 TTFT、TPOT、GPU 利用率和 memory bandwidth 验证，不能只看总延迟。

### 5. temperature、top-k、top-p 分别做什么？

temperature 缩放整个 logit 分布；top-k 固定候选数；top-p 保留累计概率集合。说明处理顺序、temperature=0 的 greedy 特例、随机种子与 GPU 非确定性。

## 训练与微调

### 6. 预训练 loss 下降为何不保证产品质量？

next-token loss 是代理目标，可能改善流畅度却不改善事实、工具、拒答或业务成功。需要域外验证、任务评测、安全与成本共同判断。

### 7. LoRA 的参数量和前向公式是什么？

冻结 W，学习 B 与 A，更新为 W 加 alpha/r 乘 BA。参数量约为 r 乘输入输出维度之和。B 零初始化保证初始函数不变。追问 target modules、rank、合并和多 adapter 服务。

### 8. QLoRA 为什么不等于“4-bit 训练”？

通常只有冻结基座权重低位存储；反量化计算、LoRA、梯度、optimizer 和激活仍用更高精度。显存还受序列长度和 checkpointing 影响。

### 9. 如何构造 assistant-only loss？

先用真实 chat template 序列化消息，再通过 token 边界生成 label mask；user/system/tool/padding 设 ignore index。用 token 级可视化和特殊 token 单测，不能按字符串长度猜边界。

### 10. 何时用 RAG，何时微调？

易变、私有、需引用事实优先 RAG；行为、格式、风格和稳定领域模式可微调。先做 Prompt 基线，按错误 taxonomy 决策；二者可组合。

## RAG

### 11. chunk 越大还是越小？

小块匹配精确但上下文不足；大块语义完整但噪声与 token 成本高。按文档结构、答案跨度和检索模型实验，比较 Recall@k、上下文覆盖、冗余和最终忠实度。

### 12. 为什么 dense retrieval 不能完全替代 BM25？

Embedding 擅长语义相似，BM25 擅长型号、错误码、姓名等精确稀有词。混合召回后用 RRF 或 reranker。向量分数跨 query/模型不可直接用统一阈值。

### 13. RAG 返回正确文档但答案仍错，怎样排查？

依次检查 chunk 是否含完整证据、重排是否截断、上下文是否重复/冲突、Prompt 是否要求引用、模型是否正确使用证据、引用是否真正蕴含主张。分层指标比只看最终答案有效。

### 14. 多租户 RAG 怎样防泄漏？

身份传到检索层，ACL 在召回查询前执行；索引、cache key、trace 和评测也隔离。用交叉租户 canary 做负向测试。不能先全局召回再让模型忽略。

### 15. Recall@k、MRR、nDCG 各测什么？

Recall@k 看相关文档覆盖；MRR 看第一个相关结果位置；nDCG 支持分级相关性并惩罚位置。指标选择取决于生成需要一个证据还是多个证据。

## Agent

### 16. Agent 和 workflow 的边界？

分支确定、风险高的流程用代码状态机；模型只处理开放语义决策。开放性越高，越需要预算、审批、幂等、恢复和审计。

### 17. 如何避免重复转账或重复发消息？

模型生成稳定 call_id 不够；执行层用幂等键、参数指纹、数据库唯一约束和业务事务。崩溃恢复先查 ledger/外部状态，不能盲目重放。

### 18. 提示注入为什么不能靠 system prompt 解决？

外部内容与高优先级指令共享模型上下文，模型可能错误服从。真正边界在最小权限、秘密隔离、参数/ACL 校验、域名限制和人工审批。

### 19. Agent 怎样判断停止？

完成条件、最大步数、时间/token/费用预算、重复动作与无进展检测共同决定。评测成功率、步骤数、副作用、安全和成本。

## 评测与实验

### 20. LLM-as-judge 有哪些偏差？

位置、长度、风格、自我偏好、知识和提示敏感。交换顺序、匿名、结构化 rubric、多 judge，并用专家集校准一致性。

### 21. 为什么用 paired bootstrap？

基线和候选回答同一批 case，差值天然配对。对 case 级差值重采样得到均值差置信区间，比独立均值更贴近问题。

### 22. 测试集反复调 Prompt 有什么问题？

测试集变成事实上的开发集，结果乐观。保留隐藏集、滚动新鲜集和时间切片，记录试验次数与版本。

### 23. 线上总体提升但中文用户下降怎么办？

分层结果必须显式报告；关键群体设 guardrail，不能用总体均值抵消。检查数据、路由、tokenization、Prompt 和 judge 语言偏差，再决定阻断或局部发布。

## 系统与故障

### 24. LLM 服务 p95 TTFT 突然升高如何排查？

先分解网关、队列、prefill、外部检索与网络；看输入长度/并发分布、batch 调度、长 prompt、GPU 利用率、KV 容量和 preemption。p50 正常而 p95 高通常提示排队或长请求干扰。

### 25. 4-bit 模型为何不一定更快？

取决于硬件内核、反量化、group size、未量化层和 batch。若计算不是权重带宽瓶颈或 kernel 不成熟，文件更小不代表端到端更快。

### 26. 如何做一次可信消融？

固定数据、token 预算、模型、seed、调参预算和评测；只改变目标组件，多次运行并报告方差。若计算预算不同，明确回答的是“同成本谁更好”还是“最高质量谁更好”。

## 代码题建议

能够现场写并测试：

- stable softmax 与 causal attention；
- top-k/top-p sampling；
- BM25 或 RRF；
- Recall@k、MRR、token F1；
- LoRA Linear；
- 有并发上限、超时和重试的异步调用；
- 幂等工具执行与参数 schema；
- KV Cache 容量估算。

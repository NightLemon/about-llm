"""Check durable fact boundaries and executable numeric claims in the textbook."""

from __future__ import annotations

import hashlib
import json
import math
import runpy
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from scripts.content_quality import OFFICIAL_URLS, check_encoding, check_ledger, text_files
except ModuleNotFoundError:
    from content_quality import OFFICIAL_URLS, check_encoding, check_ledger, text_files

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

MODEL_BOUNDARIES = {
    "landscape.md": (
        "不要选择一个品牌名",
        "榜单只能帮助生成候选",
        "开放权重不等于开源软件",
        "feasible(c)",
        "unknown` 不应自动当作 `true",
        "cost per successful task",
        "工具调用只是候选动作",
        "OpenAI-compatible",
        "immutable_revision=null",
        "不要维护没有日期和来源的",
    ),
    "gpt.md": (
        "未披露",
        "时间敏感",
        "2026-08-14",
        "GPT-5.6 Sol",
        "response/output item/content part",
        "response.created",
        "本地 replay 契约",
        "openai_responses_replay.py",
        "不证明真实 OpenAI API",
        "OpenAI SDK",
        "f2947212c1f67adf6f35bc976264db28c30abe1a32310daa284df42ca5a54686",
        "c4829c19895dcb4013141da3d11b5dc9befee8189210a0901f0cb14c19942579",
    ),
    "llama.md": (
        "以所选 checkpoint",
        "config",
        "authored_standard_gqa",
        "不是任何 Llama checkpoint",
        "Llama 3.2 官方 model card",
        "vendor-reported",
    ),
    "qwen.md": (
        "不能用一个架构",
        "检查 checkpoint",
        "authored_moe_gqa",
        "不对应任何 Qwen",
        "三方 special-token IDs",
        "Qwen2.5-0.5B-Instruct",
        "402,653,184",
    ),
    "deepseek.md": (
        "具体 checkpoint",
        "不能",
        "estimate_refused: true",
        "不是 DeepSeek-V2/V3/R1 配置快照",
        "三类 JSON 不能各看各的",
        "DeepSeek-V3",
        "MLA markers",
    ),
    "claude.md": ("保持未知", "不要"),
    "gemini.md": ("API 状态于", "Interactions API", "generateContent"),
    "cloud-api-contracts.md": (
        "Canonical core",
        "Typed extension",
        "response → output item → content part",
        "三个独立问题",
        "arbitrary network byte chunk",
        "logical-call:attempt:1",
        "cost per successful task",
        "不要对所有 4xx/5xx 自动重试",
        "501/505",
        "outcome 是否确定",
        "有效值若超过 policy/deadline 就停止",
        "第二次生成仍可能产生另一份 usage 与费用",
        "不证明真实 provider 的当前错误、配额、幂等、计费或 endpoint 语义",
        "exact origin allowlist",
        "write/read/protocol 或执行器 attempt timeout",
        "不是下载过程的 ingress/memory 上限",
        "httpx.MockTransport",
        "不执行真实 DNS/TLS",
        "OpenAI `[DONE]`、Anthropic `message_stop` 与 Gemini finishReason+EOF",
        "该测试仍不执行真实 DNS/TLS/TCP/HTTP2",
        "仅在非 2xx headers 阶段允许重试",
        "关闭 client response 不证明服务端已收到取消、停止生成或停止计费",
        "Crash 后 active reservation 继续占额度",
        "SQLite commit 不可能与远程 HTTP/provider billing 原子",
        "强制 `max_attempts=1`",
        "生产 retry 必须给每个 attempt 独立 reservation/tombstone",
    ),
}

FOUNDATION_BOUNDARIES = {
    "ml-dl.md": (
        "训练损失只是这个闭环中的一个量",
        "代理目标",
        "全局 token mean",
        "不能只把每个 rank",
        "不是“整段回答为真”",
    ),
    "nlp.md": (
        "Tokenizer 是模型契约的一部分",
        "不能直接横比",
        "训练时仍可并行",
        "结构合法不保证语义正确",
    ),
}

GENERATION_BOUNDARIES = (
    "必须保留第一个让累计概率达到或超过阈值的 token",
    "只计生成 token",
    "包含已发出的 EOS",
    "不计 prompt",
    "保存所有从 active prefix 产生的 EOS",
    "完整转移判断",
    "constraint dead end",
    "EOS 只有在当前状态接受完整输出时",
    "不是 JSON Schema、CFG",
    "一个 chunk 可含零个、一个或多个 token",
    "Stop string 可能跨 token 边界",
    "usage 缺失时标记未知",
    "真实调用 Transformers `generate()`",
    "token 由测试 processor 强制",
    "报告只能由受控路径推断",
)

CORE_BOUNDARIES = {
    "tokenization.md": (
        "Python 的 `len(text)`",
        "BPE 编码必须使用已学习的 merge rank",
        "不跨文档边界",
        "不是 GPT-2",
        "NFKC 会折叠 compatibility character",
        "不能从 token 数下降直接声称端到端延迟按平方下降",
        "无密钥 hash 只能标识",
    ),
    "transformer.md": (
        "`input_ids`、attention visibility 和 labels/loss mask 是三个不同契约",
        "每个有效 query 至少能看到一个 key",
        "仅插入 EOS 不会自动切断 attention",
        "不能在加载 checkpoint 时互换",
        "同时平移相同常数",
        "优化 kernel 不应真的复制",
        "Parity 仍不覆盖 NumPy 完整模型",
        "配置声称启用不证明没有 fallback",
    ),
    "scaling.md": (
        "`6ND` 适合做同口径的一阶预算",
        "只在已验证区间附近外推",
        "unique tokens 与 consumed tokens",
        "不能用 active parameters 解释 checkpoint 内存",
    ),
    "architectures-interpretability.md": (
        "可解码性不是因果使用证据",
        "高 attention weight 不等于",
        "分母非零且 metric 方向一致",
        "post-residual output",
        "事后选择",
        "未来位置负对照",
        "不能写成“发现了模型的自然 circuit”",
        "固定 Qwen2.5-0.5B-Instruct 的单事实、单模板 source-position",
    ),
}

TRAINING_BOUNDARIES = {
    "finetuning.md": (
        "先按失败类型选择干预",
        "机制证据只回答训练路径是否按声明执行",
        "同 batch loss 下降不是 held-out 行为证据",
        "训练 checkpoint 和服务 artifact 是两种不同契约",
        "量化或 merge 会产生新的部署 artifact",
    ),
    "data.md": (
        "用于 exact dedup 的规范化文本不必等于最终训练文本",
        "插入 EOS 本身并不会自动阻止 attention 跨文档",
        "unique raw/normalized tokens",
        "污染检测报告应给覆盖范围与漏检边界",
        "P(\\text{candidate}\\mid s)=1-(1-s^r)^b",
        "不是单个 pair 的召回保证",
        "1-hash 反例",
    ),
    "alignment.md": (
        "PPO clip 限制的是 sampled action probability ratio",
        "bootstrap mask",
        "continuation mask",
        "sampled KL proxy",
        "不能写成“已实现 PPO RLHF”",
        "PyTorch rollout 与 optimizer control",
        "没有 tokenizer/语言模型 token rollout",
        "Tiny Transformer token PPO control",
        "sampled action log-ratio",
        "不执行 tokenizer 或自然语言",
        "response token log-prob 的**和**",
        "没有真实人类 preference dataset",
        "confounded",
        "counterfactual",
        "strict pair accuracy",
        "不是 text/Transformer reward model",
        "训练准确率",
        "GPT2ForSequenceClassification",
        "reward head 与 token embedding 都发生更新",
        "不需要 held-out plaintext",
        "超过 `max_length` 的 pair 会被 trainer **过滤**",
        "保存后的 `lora_B` 不再是全零",
        "没有下载任一目标 checkpoint",
        "completion_mask",
        "有序 binary-train/combined binding",
        "prompt-prefix mismatch",
        "四种跨记录 candidate surface",
        "registry 不是法律意见",
        "不证明任一目标模型已经完成偏好对齐",
        "权限和副作用必须由系统强制执行",
        "blind_model_identity=true",
        "Fleiss",
        "以 **case** 而不是单条 judgment",
        "不是人类标注",
        "不自动是因果效应",
    ),
    "continual-learning.md": (
        "相同架构但独立随机初始化",
        "冻结底座不代表旧任务质量绝对不变",
        "只支持这十个行为",
        "包含最终阶段的非负 peak-to-final drop",
        "显式 task-id feature",
        "uniform reservoir",
        "optimizer-step matched",
        "不是 example/compute matched",
        "区间**不覆盖**新任务、数据采样",
        "没有在目标 LLM、真实时间序列、多任务/数据采样",
    ),
    "synthetic-data.md": (
        "q_{accept}(x)",
        "同一 revision 既生成又验证是需报告的相关性风险",
        "Tokenizer 不同不能逐 token 直接 KL",
        "使用 synthetic 必然 collapse",
        "没有调用真实 teacher",
    ),
    "sft-data-pipeline.md": (
        "字符统计单位是 Unicode code point",
        "candidate”不是人工确认的 duplicate",
        "Readiness 的 fail-closed gate 仍使用全对精确比较",
        "候选召回不是保证",
        "当前 core 尚未替代 readiness gate",
        "阈值必须按语言、长度和任务",
        "不判断许可是否合法",
        "不是外部律师意见",
        "无候选不证明无敏感信息",
        "不保存命中原文",
        "hash 只证明显式 canonical 字段",
        "不是 tokenizer/mask 验证报告",
        "不独立证明 mask 语义或最终 collator labels 正确",
    ),
}

QUALITY_BOUNDARIES = {
    "evaluation-methodology.md": (
        "ECE 强依赖 bin 数/边界",
        "相同 confidence 的样本一起接受",
        "模型自述",
        "校准良好也不证明事实",
        "raw judgment artifact",
        "固定 rater 数前提下的 Fleiss",
        "以 case 为重采样单位",
        "不是可发表的人类结论",
        "只比较 `case_id` 集合仍不够",
        "`compare` 在 bootstrap 前",
        "不是来源认证",
        "不能证明输出来自所称模型",
        "最终 gate JSON 也必须是 artifact",
        "comparison v2",
        "v2 不静默加载旧 v1",
        "`verify-comparison`",
        "`verification_scope: artifact_only`",
        "不会重新打开 cases/results/run manifests",
        "不能检测没有可信 head 时的历史截断",
        "合法前缀必然仍可验证",
        "HMAC 不证明 key custody 或不可否认性",
        "verify 后还存在 TOCTOU",
        "`verify-evidence` 提供更强但仍有限的本地复算",
        "`artifact_authentication_verified=false`",
        "`model_execution_replayed=false`",
        "`render-comparison-html`",
        "`artifact_only_render`",
        "所有动态文本 HTML escape",
    ),
    "safety.md": (
        "Prompt hierarchy 是行为约束",
        "任何单例都是 incident",
        "它不证明真实供应商日志或网络路径安全",
        "未成功提取不证明样本对参数没有影响",
        "cache replay 前重新授权",
        "仍不是集中 IAM",
        "planner 的结构化 `tool/finish/escalate` 输出仍是不可信 proposal",
        "无密钥 hash 不是认证",
        "opaque_reasoning_block_count == 0",
    ),
    "reasoning-artifact-security.md": (
        "签名有效",
        "截至 2026 年 8 月",
        "原攻击方法已不能复现",
        "没有 ground-truth plaintext reasoning",
        "unsafe_acceptance_count",
        "secret_pii_scan_performed: false",
        "不是 raw provider response sanitizer",
        "不解析或生成任何真实供应商",
        "内存 nonce/replay ledger",
    ),
    "governance-impact.md": (
        "只有已经实现并有测试证据的控制",
        "二手博客可用于导航",
        "不能把本仓库通过自动化测试等同于法规合规",
        "审核者必须有能力、时间和 override 权限",
    ),
    "governance-templates.md": (
        "计划中的控制不降低 residual risk",
        "Fingerprint 只证明 manifest 中显式 canonical bytes 的身份",
        "模板存在不等于治理完成",
    ),
    "agent-evaluation.md": (
        "分母为零报告 N/A",
        "`handler_attempted=true` 不证明远端动作发生",
        "不能证明 supplied observation 真实",
        "安全指标是 guardrail",
        "policy judgment 缺失/indeterminate",
        "`completed` 仅指 completion verifier passed",
        "checkpoint 自带 hash 的通过率不是安全指标",
        "provider success 后 ack 前 crash",
        "SQLite + 模拟 provider 的通过只证明本地状态机",
    ),
}

SYSTEM_BOUNDARIES = {
    "inference-optimization.md": (
        "4-bit 时这个约定使用 `[-7, 7]`",
        "理想 dense-bitstream 账本",
        "不能把它的 `nbytes` 当真实 int4 packing",
        "全 1 的 unsigned code",
        "self-contained model artifact",
        "Weight RMSE 小不保证 logits",
        "per-token/per-KV-head",
        "\\frac{4D}{D+4}",
        "先 dequantize、再 float32 attention",
        "forward positions 是 \\(L_p+L_o-1\\)",
        "不是机械的 \\(L_p+L_o\\)",
        "拒绝概率满足",
        "一步接受率正好是",
        "第一个拒绝位置发出 residual token",
        "Monte Carlo 只作直觉展示",
        "Greedy speculative decoding 是另一份契约",
        "未填满的共享 tail",
        "整个 append 应在填充旧 tail",
        "physical_token_values",
        "没有存储或复制真实 K/V tensor",
    ),
    "serving.md": (
        "数据面\uFF08data plane\uFF09",
        "`eligible` 必须在看结果前由合同定义",
        "Error budget 是变更速度与可靠性的治理工具",
        "成功请求的 latency percentile 是条件统计",
        "429 可以是正确的过载保护",
        "不能把 429 从 availability 分母删除",
        "concurrency semaphore 也是队列",
        "Client queue 不等于服务端 queue",
        "快速 429",
        "scheduled timestamp 写成 `offered_at`",
        "不证明发生器实际按时执行",
        "Request id 用于关联一次 attempt",
        "进程内 `Semaphore(8)` 不等于 4 worker 的服务总并发是 8",
        "Autoscaling 不是只看 GPU utilization",
        "Capacity 只在确定的 terminal/resource-release 事件后归还",
        "不能把 client `perf_counter()` 与 server monotonic 数值直接相减",
        "回滚成功是用户请求恢复",
        "这些 controls 是因果链中的局部证据",
    ),
    "vllm-serving.md": (
        "成功延迟是条件统计",
        "没有成功样本时 percentile 应为 unavailable/null",
        "Timeout 是 right-censored experience",
        "该 fixture 是合成 client trace",
        "p95 不能由两个 p95 相加得到",
        "client-side coordinated omission",
        "--arrival-process constant",
        "不等于负载生成器无误差",
        "scheduled timestamp 不证明事件循环按时执行",
        "一次性物化有限 `--requests`",
        "shared partial tail append 必须 COW",
        "只证明 metadata 状态机",
        "W=\\sum_i(P_i+O_i-1)",
        "prefill 的最后一个 prompt position 已给出首个输出 token 的分布",
        "离散 step 不是秒",
        "不证明 vLLM scheduler equivalence",
        "generation EOS 可以是 tokenizer EOS 的有意 superset",
        "normalized snapshots",
        "Transformers generation runtime control",
        "它不运行 vLLM",
    ),
    "hardware-edge.md": (
        "结果是理想下界",
        "当前实跑 device 是 CPU",
        "本地执行减少网络传输",
        "不是实测性能声明",
    ),
}

FRONTIER_BOUNDARIES = {
    "multimodal.md": (
        "只验证 metric convention",
        "CER 可因大量 insertion 大于 1",
        "不证明任何具体多模态模型能力或成本",
        "不能随机跨 train/test",
    ),
    "reasoning-long-context-moe.md": (
        "接受长度、训练长度和有效利用不同",
        "弱 verifier 会选择高分漏洞",
        "长上下文侧没有目标 checkpoint 的全长度矩阵",
        "active parameters 只包含当前 token 使用部分",
        "C=\\left\\lceil\\phi\\frac{Nk}{E}\\right\\rceil",
        "dropped assignment",
        "all-assignments-dropped token",
        "广义诊断",
        "不是所有论文/框架的 training loss",
        "不是目标模型或生产 EP 复现",
    ),
    "embodied-small-models.md": (
        "不能把“模型输出了动作”当“动作已成功”",
        "仿真成功只能证明",
        "只有满足特定接受/残差算法",
        "没有机器人 simulator",
        "CPU 循环也不是 verification kernel",
    ),
}

APPLICATION_BOUNDARIES = {
    "agent-interoperability.md": (
        "Tool discovery 不等于 tool authorization",
        "Agent Card 当作可验证的声明",
        "远端返回 `completed` 只代表协议状态",
        "五个 MCP control 都固定协议",
        "official-SDK memory | 官方 client/server/types/schema validation + memory stream",
        "没有创建 subprocess、OS pipe、socket、HTTP、SSE",
        "unknown_tool_handler_delta=1",
        "可运行的官方 MCP SDK stdio control",
        "mcp.client.stdio.stdio_client",
        "没有独立注入 missing LF",
        "不能从 SDK 源码存在相应分支就写成已测试",
        "可运行的官方 MCP SDK Streamable HTTP control",
        "mcp.client.streamable_http.streamable_http_client",
        "固定 HTTP profile 是 7 次 POST、1 次 GET 与 1 次 DELETE",
        "不是 MCP auth",
        "可运行的 MCP Streamable HTTP control",
        "不是 MCP Authorization/OAuth flow",
        "取消 in-flight request 要另发 `notifications/cancelled`",
        "两个 official-SDK transport controls 分别把 SDK 与真实 pipe",
        "官方 `a2a-sdk==1.1.2`",
        "真实 IPv4 loopback TCP/HTTP",
        "不等于完整 A2A conformance",
        "核对日期",
    ),
    "agent-architecture.md": (
        "typed loop 把 `finish` 视为 proposal",
        "supplied fixture 数字",
        "不能发现任意长周期",
        "checkpoint 的 SHA-256 只检测 canonical 内容漂移",
        "文件与 SQLite ledger 没有原子事务",
        "lease 防并发领取",
    ),
    "agent-runtime.md": (
        "拒绝 `NaN`/`Infinity`",
        "通用 JSON 序列化不会替你定义这些语义",
        "既不是加密保险箱",
        "实验 ledger 都不兼容",
        "DefaultDenyPolicy",
        "cache hit 也不能绕过",
        "只是进程内 typed contract",
        "JSON-valid 只证明可移植的值域和稳定快照",
        "不能让客户端自报已用次数",
        "本地业务状态",
        "lease 只表示本地并发所有权",
        "receipt 是 provider supplied artifact",
        "只证明 reference 状态机的 at-least-once delivery",
    ),
    "rag-retrieval.md": (
        "Precision@k 分母定义为实际返回且被检查",
        "zero-result accuracy 只是一条检索层信号",
        "未标注文档不自动等于不相关",
        "`rerank-recorded`",
        "跨租户或 ACL-blocked 文本不会传给 scorer",
        "分数由作者构造",
        "不能把 packing 决策写成“生成已使用证据”",
        "不能从 BM25、recorded score 或 greedy packing 声称目标语料上的最优排序",
    ),
    "rag-generation.md": (
        "检索零结果正确不证明最终拒答正确",
        "不能替代这些端到端分母",
        "每条 claim verdict 由工件显式提供",
        "不执行语义蕴含",
        "不证明 answer completeness",
        "各组件分别 tokenize 后的长度能精确相加",
        "不能强制 `base_cost <= used_cost`",
        "--budget-bytes` 只计算 UTF-8 serialized bytes",
        "每个候选都在 budget/quota 判断前重新检查",
        "`pack-tokenized`",
        "model_context_window_verified=false",
        "不会把 query/context 中碰巧出现的",
        "Tokenizer revision/hash 也不是来源认证",
        "`audit-traces`",
        "该命令不重新 tokenize",
        "不能阻止攻击者协同重写所有文件",
    ),
    "rag-production.md": (
        "`SQLiteChunkStore`",
        "`BEGIN IMMEDIATE`",
        "chunker revision/`max_chars`",
        "`about-llm-rag store-upsert/store-delete/store-retrieve`",
        "`cross_store_atomicity_proved: false`",
        "`store-backup`",
        "有序 source/chunk row fingerprint",
        "不等于达到目标 RPO/RTO",
        "trigger 注入失败测试证明单库 rollback",
        "不证明远端向量库、object store 与 source DB 的分布式原子性",
        "之后才允许构造 scorer",
        "同一 canonical BM25",
        "不证明框架默认提供 ACL",
        "不能认证 supplied results 的来源",
        "tenant/principals 必须由注入的 `AuthResolver` 提供",
        "wait_for(asyncio.to_thread(...))` 超时不能终止同步 thread",
        "不能冒充全局 admission",
        "不能仅因“有 FastAPI endpoint”就提升为完整 L3",
    ),
    "prompting.md": (
        "template 缺失时失败",
        "delimiters 只是语义提示",
        "结构合法不代表语义正确",
        "它不证明组件列表完整、语义等价、安全、来源可信或远程模型能 bitwise 重放",
    ),
    "code-conversation-llmops.md": (
        "不是单次线上成功率",
        "1\\le k\\le n",
        "摘要会遗漏否定、时间、谁说的和不确定性",
        "只记录 model name 无法回放/归因",
        "Fingerprint 只识别已列配置",
        "correction/retraction 不能跨 tenant",
        "只是单进程内存 reference",
    ),
    "product-design.md": (
        "模型文本不是状态转移证据",
        "不展示虚构 confidence",
        "Progressive disclosure 用于管理认知负担",
        "没有真实 UI 实现",
    ),
}

CAREER_BOUNDARIES = {
    "roadmap.md": (
        "本页不是周计划",
        "按失败责任选岗位",
        "岗位名称不是统一标准",
        "知道、实现、验证、负责",
        "不按工作年限",
        "JD 中的技术栈是约束",
        "一个主项目 + 一个相邻项目 + 一项底层实现",
        "仍不自动证明真实生产流量",
        "没有原始 artifact",
        "CPU fixture 不证明 GPU 吞吐",
        "完成这些检查不保证录用",
    ),
    "system-design.md": (
        "公式估算",
        "降级必须保留安全不变量",
        "不要把离线 L2 的重放结果描述成生产 availability",
        "每次 cache replay 先重新授权",
        "`finish` 必须经过独立 completion verifier",
        "checkpoint hash 不提供认证",
        "semaphore 后计时会漏 client queue",
        "exactly-once external effect",
        "先预留 COW replacement",
        "physical materialized positions",
    ),
    "interview-questions.md": (
        "pass@k 测的是什么",
        "Prompt delimiter 为什么不是安全边界",
        "配置 fingerprint 只能证明所序列化字段的 canonical bytes 相同",
        "proposal fingerprint 只标识模型提出的 tool + arguments",
        "handler 返回的 dict 仍要 JSON snapshot",
        "approval pause 后怎样安全恢复",
        "MCP stdio 接通后",
        "tools/list 成功",
        "discovery 和 schema-valid 不建立",
        "负载生成器 `offered_at`",
        "Closed-loop 与 open-loop",
        "不代表请求真的准时 dispatch",
        "为什么云模型调用不能“429/5xx 一律重试”",
        "远端 outcome 是否确定",
        "Transactional outbox 能否保证 exactly-once",
        "lease 不是远端 exactly-once",
        "Speculative decoding 为什么能保持 target sampling distribution",
        "一步接受率因此是 `1-TV(p,q)`",
        "Paged KV 为什么仍会碎片化",
        "CPU metadata 模拟通过不等于真实 KV copy",
        "是否已经证明评测发布历史没有被回滚",
        "否则任何有效前缀都能通过",
        "timestamp 只是 caller 字符串",
        "Artifact-only 校验与完整证据图复算有什么区别",
        "作为五个不同命题",
        "deterministic extractive non-LLM answer",
        "FastAPI 返回 504 后",
        "不能把 504 当作服务端 work 已停止或费用为零的证据",
        "评测 HTML 报告怎样避免成为安全漏洞或“证据升级”",
        "XSS-safe 只证明渲染边界",
        "tokenizer、model config 与 generation config 的 EOS 不同",
        "静态一致仍不证明运行时正确",
        "强制 `[4,3]`",
        "不让模型权重选择 token",
    ),
    "resume-projects.md": (
        "每个数字都要有证据账本",
        "CPU replay 不证明 GPU 吞吐",
        "如果当前只有离线 L2 证据",
        "cache 重新授权",
        "应明确写“控制流回归”",
        "不能写成“防篡改、原子持久化”",
        "arrival process/长度分布/并发",
        "不能声称测得用户排队",
        "仅有 UTF-8 bytes 不能写成模型 token 窗口",
        "成功 tokenize 也不证明 tokenizer/权重匹配",
        "若简历写“实现 MCP”",
        "不能缩写成“通用 MCP 平台”",
    ),
}

PAPER_BOUNDARIES = {
    "index.md": (
        "榜单只解释“为什么现在读”",
        "预印本写成“预印本”",
        "热门不等于重要",
        "只有实际加入代码、环境、原始输出和失败测试后",
        "来源超过 90 天未复核时",
    ),
    "2026-08.md": (
        "快照日期**：2026-08-12",  # noqa: RUF001
        "六篇榜单入选",
        "两篇编辑精选",
        "Hugging Face 页面上的互动数字归属容易",
        "https://arxiv.org/abs/2607.24653",
        "https://arxiv.org/abs/2608.05466",
        "https://arxiv.org/abs/2608.01964",
        "https://arxiv.org/abs/2608.02023",
        "https://arxiv.org/abs/2606.30534",
        "https://arxiv.org/abs/2607.19191",
        "https://arxiv.org/abs/2608.10296",
        "https://arxiv.org/abs/2608.09867",
        "实际预训练只使用约十分之一",
        "16 FPS 是激进低比特配置的上界",
        "摘要与正文对一项 PPO 最终数字存在版本内不一致",
        "OSWorld subset 的摘要与正文数字不一致",
        "截至 2026 年 8 月",
        "作者报告，不等于本仓库复现",  # noqa: RUF001
    ),
}


def check_model_boundaries() -> list[str]:
    errors: list[str] = []
    model_dir = ROOT / "docs" / "models"
    for filename, markers in MODEL_BOUNDARIES.items():
        text = (model_dir / filename).read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            errors.append(f"docs/models/{filename}: missing boundary marker(s): {missing}")
    return errors


def check_foundation_boundaries() -> list[str]:
    errors: list[str] = []
    foundation_dir = ROOT / "docs" / "foundations"
    for filename, markers in FOUNDATION_BOUNDARIES.items():
        text = (foundation_dir / filename).read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            errors.append(
                f"docs/foundations/{filename}: missing accuracy boundary marker(s): {missing}"
            )
    return errors


def check_generation_boundaries() -> list[str]:
    generation = ROOT / "docs" / "core" / "generation.md"
    text = generation.read_text(encoding="utf-8")
    missing = [marker for marker in GENERATION_BOUNDARIES if marker not in text]
    if missing:
        return [f"docs/core/generation.md: missing accuracy boundary marker(s): {missing}"]
    return []


def check_core_boundaries() -> list[str]:
    errors: list[str] = []
    core_dir = ROOT / "docs" / "core"
    for filename, markers in CORE_BOUNDARIES.items():
        text = (core_dir / filename).read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            errors.append(
                f"docs/core/{filename}: missing accuracy boundary marker(s): {missing}"
            )
    return errors


def check_training_boundaries() -> list[str]:
    errors: list[str] = []
    training_dir = ROOT / "docs" / "training"
    for filename, markers in TRAINING_BOUNDARIES.items():
        text = (training_dir / filename).read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            errors.append(
                f"docs/training/{filename}: missing accuracy boundary marker(s): {missing}"
            )
    return errors


def _check_directory_boundaries(
    *, path: Path, boundaries: dict[str, tuple[str, ...]], label: str
) -> list[str]:
    errors: list[str] = []
    for filename, markers in boundaries.items():
        text = (path / filename).read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            errors.append(
                f"{label}/{filename}: missing accuracy boundary marker(s): {missing}"
            )
    return errors


def check_quality_boundaries() -> list[str]:
    return _check_directory_boundaries(
        path=ROOT / "docs" / "quality",
        boundaries=QUALITY_BOUNDARIES,
        label="docs/quality",
    )


def check_system_boundaries() -> list[str]:
    return _check_directory_boundaries(
        path=ROOT / "docs" / "systems",
        boundaries=SYSTEM_BOUNDARIES,
        label="docs/systems",
    )


def check_frontier_boundaries() -> list[str]:
    return _check_directory_boundaries(
        path=ROOT / "docs" / "frontier",
        boundaries=FRONTIER_BOUNDARIES,
        label="docs/frontier",
    )


def check_application_boundaries() -> list[str]:
    return _check_directory_boundaries(
        path=ROOT / "docs" / "applications",
        boundaries=APPLICATION_BOUNDARIES,
        label="docs/applications",
    )


def check_career_boundaries() -> list[str]:
    return _check_directory_boundaries(
        path=ROOT / "docs" / "career",
        boundaries=CAREER_BOUNDARIES,
        label="docs/career",
    )


def check_paper_boundaries() -> list[str]:
    return _check_directory_boundaries(
        path=ROOT / "docs" / "papers",
        boundaries=PAPER_BOUNDARIES,
        label="docs/papers",
    )


def check_stream_token_accounting() -> list[str]:
    benchmark = ROOT / "projects" / "inference-serving" / "benchmark_openai.py"
    text = benchmark.read_text(encoding="utf-8")
    errors: list[str] = []
    forbidden = ("observed_chunks", 'usage.get("completion_tokens") or')
    for fragment in forbidden:
        if fragment in text:
            errors.append(f"{benchmark.relative_to(ROOT)}: forbidden token fallback: {fragment}")
    required = (
        "SSE chunks are not tokens",
        'usage.get("completion_tokens") is None',
        "summarize_attempts",
        "classify_http_failure",
        '"attempts":',
        "offered_at = benchmark_started_at + offset_seconds",
        "offered_at=offered_at",
    )
    for fragment in required:
        if fragment not in text:
            errors.append(f"{benchmark.relative_to(ROOT)}: missing strict token check: {fragment}")
    offered_position = text.find("offered_at = benchmark_started_at + offset_seconds")
    semaphore_position = text.find("async with semaphore")
    if (
        offered_position == -1
        or semaphore_position == -1
        or offered_position > semaphore_position
    ):
        errors.append(
            f"{benchmark.relative_to(ROOT)}: offered_at must be captured before "
            "the client semaphore"
        )
    return errors


def check_tokenization_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.from_scratch import ByteBPETokenizer

    errors: list[str] = []
    tokenizer = ByteBPETokenizer.train(
        ["abab"], vocab_size=258, min_pair_frequency=1
    )
    if not (
        tokenizer.merges == ((ord("a"), ord("b")), (256, 256))
        and tokenizer.encode("abab") == [257]
        and tokenizer.token_bytes(257) == b"abab"
        and tokenizer.decode([257]) == "abab"
    ):
        errors.append("byte-BPE ranked merge example mismatch")
    boundary_model = ByteBPETokenizer.train(
        ["a", "b"], vocab_size=257, min_pair_frequency=1
    )
    if boundary_model.merges:
        errors.append("byte-BPE trainer merged across document boundaries")
    raw_bytes = ByteBPETokenizer()
    if raw_bytes.encode("e\u0301") == raw_bytes.encode("é"):
        errors.append("byte tokenizer unexpectedly normalized distinct Unicode sequences")
    return errors


def check_transformer_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import numpy as np

    from about_llm.from_scratch import (
        apply_rope,
        blockwise_online_attention,
        causal_mask,
        grouped_query_attention,
        rms_norm,
        scaled_dot_product_attention,
    )

    errors: list[str] = []
    normalized = rms_norm(np.array([[3.0, 4.0]]), epsilon=1e-12)
    if not np.isclose(np.mean(np.square(normalized)), 1.0):
        errors.append("RMSNorm definition example mismatch")

    rng = np.random.default_rng(41)
    query = rng.normal(size=(1, 4, 4, 6))
    key = rng.normal(size=(1, 2, 4, 6))
    value = rng.normal(size=(1, 2, 4, 5))
    rotated_query, rotated_key = apply_rope(
        query, key, np.arange(4, dtype=np.int64)
    )
    if not (
        np.allclose(
            np.linalg.norm(rotated_query, axis=-1),
            np.linalg.norm(query, axis=-1),
        )
        and np.allclose(
            np.linalg.norm(rotated_key, axis=-1),
            np.linalg.norm(key, axis=-1),
        )
    ):
        errors.append("RoPE norm-preservation example mismatch")

    gqa_output, _ = grouped_query_attention(
        query, key, value, mask=causal_mask(4)
    )
    repeated_output, _ = scaled_dot_product_attention(
        query,
        np.repeat(key, 2, axis=1),
        np.repeat(value, 2, axis=1),
        mask=causal_mask(4),
    )
    if not np.allclose(gqa_output, repeated_output):
        errors.append("GQA explicit K/V repetition equivalence mismatch")

    full_output, _ = scaled_dot_product_attention(
        query,
        np.repeat(key, 2, axis=1),
        np.repeat(value, 2, axis=1),
        mask=causal_mask(4),
    )
    steps = []
    repeated_key = np.repeat(key, 2, axis=1)
    repeated_value = np.repeat(value, 2, axis=1)
    for position in range(4):
        step, _ = scaled_dot_product_attention(
            query[:, :, position : position + 1],
            repeated_key[:, :, : position + 1],
            repeated_value[:, :, : position + 1],
            mask=causal_mask(1, position + 1),
        )
        steps.append(step)
    if not np.allclose(np.concatenate(steps, axis=-2), full_output):
        errors.append("incremental/full causal attention equivalence mismatch")

    online_query = rng.normal(size=(5, 4))
    online_key = rng.normal(size=(7, 4))
    online_value = rng.normal(size=(7, 3))
    online_mask = causal_mask(query_length=5, key_length=7)
    dense_online_reference, _ = scaled_dot_product_attention(
        online_query,
        online_key,
        online_value,
        mask=online_mask,
    )
    online_result = blockwise_online_attention(
        online_query,
        online_key,
        online_value,
        block_size=3,
        mask=online_mask,
    )
    if not (
        np.allclose(online_result.output, dense_online_reference, rtol=1e-12, atol=1e-12)
        and online_result.key_block_count == 3
        and online_result.logical_peak_score_elements == 15
        and online_result.full_score_elements == 35
        and np.all(np.isfinite(online_result.running_row_max))
        and np.all(np.isfinite(online_result.row_normalizer))
        and not hasattr(online_result, "probabilities")
    ):
        errors.append("blockwise online-softmax recurrence/dense equivalence mismatch")

    online_boundary_files = (
        ROOT / "docs" / "foundations" / "math.md",
        ROOT / "docs" / "systems" / "inference-optimization.md",
        ROOT / "projects" / "transformers-basics" / "README.md",
    )
    online_boundary_text = "\n".join(
        path.read_text(encoding="utf-8") for path in online_boundary_files
    )
    required_online_boundaries = (
        "实数算术下",
        "不返回完整 probability matrix",
        "不是进程峰值内存测量",
        "不证明 FlashAttention/CUDA/vLLM backend",
    )
    missing_online_boundaries = [
        marker
        for marker in required_online_boundaries
        if marker not in online_boundary_text
    ]
    if missing_online_boundaries:
        errors.append(
            "online-softmax docs missing evidence boundary marker(s): "
            f"{missing_online_boundaries}"
        )
    return errors


def check_gpt_cross_framework_parity_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.from_scratch.gpt_cross_framework import (
        GPT_CROSS_FRAMEWORK_PARITY_VERSION,
        run_gpt_cross_framework_parity_control,
    )

    report = run_gpt_cross_framework_parity_control()
    runtime = report.get("runtime", {})
    fixture = report.get("fixture", {})
    config = fixture.get("config", {}) if isinstance(fixture, dict) else {}
    observation = report.get("observation", {})
    comparison = report.get("comparison", {})
    assertions = report.get("assertions", {})
    scope = report.get("scope", {})
    errors: list[str] = []

    if not (
        report.get("schema_version") == GPT_CROSS_FRAMEWORK_PARITY_VERSION
        and runtime.get("jax_backend") == "cpu"
        and runtime.get("torch_device") == "cpu"
        and runtime.get("dtype") == "float32"
        and config
        == {
            "vocab_size": 11,
            "context_length": 5,
            "model_dim": 8,
            "num_heads": 2,
            "num_layers": 2,
            "mlp_ratio": 2,
            "dropout": 0.0,
            "linear_bias": False,
            "normalization": "LayerNorm with affine scale/bias",
            "normalization_epsilon": 1e-5,
            "gelu": "tanh approximation",
            "token_embedding_lm_head_tied": True,
        }
        and fixture.get("input_ids") == [[0, 1, 2, 3], [3, 2, 1, 0]]
        and fixture.get("targets") == [[1, 2, 3, 4], [2, -100, 0, 5]]
        and fixture.get("ignored_target_count") == 1
        and fixture.get("optimizer")
        == "plain SGD without momentum or weight decay"
        and fixture.get("learning_rate") == 0.025
        and fixture.get("comparison_tolerance") == 2e-6
        and isinstance(assertions, dict)
        and assertions
        and all(assertions.values())
    ):
        errors.append("PyTorch/JAX GPT parity identity/runtime mismatch")

    scalar_checks = (
        (
            "torch_loss_before_step",
            observation.get("torch_loss_before_step", -1),
            2.36346173286438,
        ),
        (
            "jax_loss_before_step",
            observation.get("jax_loss_before_step", -1),
            2.36346173286438,
        ),
        (
            "torch_loss_after_step",
            observation.get("torch_loss_after_step", -1),
            2.312957525253296,
        ),
        (
            "jax_loss_after_step",
            observation.get("jax_loss_after_step", -1),
            2.312957286834717,
        ),
    )
    for name, actual, expected in scalar_checks:
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=5e-7):
            errors.append(
                f"PyTorch/JAX GPT parity {name} mismatch: {actual}"
            )

    if not (
        comparison.get("initial_logits_max_abs_difference", math.inf)
        <= 2e-6
        and comparison.get("initial_loss_abs_difference", math.inf) <= 2e-6
        and comparison.get("gradient_global_max_abs_difference", math.inf)
        <= 2e-6
        and comparison.get(
            "post_update_parameter_global_max_abs_difference",
            math.inf,
        )
        <= 2e-6
        and comparison.get("post_update_logits_max_abs_difference", math.inf)
        <= 2e-6
        and comparison.get("post_update_loss_abs_difference", math.inf)
        <= 2e-6
        and comparison.get(
            "native_rmsnorm_counterfactual_logits_max_abs_difference",
            -1,
        )
        > 0.3
        and isinstance(
            comparison.get("gradient_max_abs_difference_by_parameter"),
            dict,
        )
        and len(comparison["gradient_max_abs_difference_by_parameter"]) == 20
        and isinstance(
            comparison.get(
                "post_update_parameter_max_abs_difference_by_parameter"
            ),
            dict,
        )
        and len(
            comparison[
                "post_update_parameter_max_abs_difference_by_parameter"
            ]
        )
        == 20
    ):
        errors.append("PyTorch/JAX GPT numerical parity comparison mismatch")

    expected_scope = {
        "same_initial_parameter_values_compared": True,
        "layernorm_bias_epsilon_gelu_mask_and_tying_aligned": True,
        "masked_cross_entropy_forward_compared": True,
        "every_unique_parameter_gradient_compared": True,
        "plain_sgd_one_step_compared": True,
        "post_update_forward_compared": True,
        "native_rmsnorm_architecture_counterfactual_executed": True,
        "jax_cpu_execution_forced": True,
        "framework_rng_equivalence_claimed": False,
        "adamw_optimizer_state_or_schedule_compared": False,
        "dropout_rng_or_stochastic_sampling_compared": False,
        "jit_compile_or_async_timing_compared": False,
        "cuda_tpu_multi_device_or_sharding_executed": False,
        "large_model_training_convergence_or_performance_proved": False,
    }
    if scope != expected_scope:
        errors.append("PyTorch/JAX GPT parity evidence scope mismatch")

    source = (
        SRC / "about_llm" / "from_scratch" / "gpt_cross_framework.py"
    ).read_text(encoding="utf-8")
    script = (
        ROOT / "projects" / "jax-minigpt" / "cross_framework_parity.py"
    ).read_text(encoding="utf-8")
    test = (
        ROOT / "tests" / "test_gpt_cross_framework_parity.py"
    ).read_text(encoding="utf-8")
    required_code_markers = {
        "source": (
            source,
            (
                '"about-llm.gpt-cross-framework-parity.v1"',
                "def layernorm_jax_forward(",
                "def torch_model_to_layernorm_jax_params(",
                "native_rmsnorm_counterfactual_logits_max_abs_difference",
                "with jax.default_device(cpu_devices[0])",
                "allow_nan=False",
            ),
        ),
        "script": (
            script,
            (
                "run_gpt_cross_framework_parity_control",
                "allow_nan=False",
            ),
        ),
        "test": (
            test,
            (
                "test_pytorch_jax_layernorm_forward_backward_and_sgd_parity",
                "native_rmsnorm_counterfactual_logits_max_abs_difference",
                "_canonical_bytes(report)",
            ),
        ),
    }
    for label, (document, markers) in required_code_markers.items():
        missing = [marker for marker in markers if marker not in document]
        if missing:
            errors.append(
                f"PyTorch/JAX GPT parity {label} missing marker(s): {missing}"
            )

    required_docs = {
        "readme": (
            ROOT / "README.md",
            (
                "PyTorch↔JAX MiniGPT parity control",
                "2.384185791015625e-07",
                "LayerNorm/RMSNorm",
            ),
        ),
        "project": (
            ROOT / "projects" / "jax-minigpt" / "README.md",
            (
                "cross_framework_parity.py",
                "0.37747739627957344",
                "不比较 AdamW state",
            ),
        ),
        "project_page": (
            ROOT / "docs" / "practice" / "projects" / "jax-minigpt.md",
            (
                "cross_framework_parity.py",
                "20 个 unique parameters",
                "RMSNorm 反事实",
            ),
        ),
        "training": (
            ROOT / "docs" / "training" / "jax-optax.md",
            (
                "PyTorch↔JAX 同权重 parity control",
                "LayerNorm 的 mean subtraction",
                "2e-6",
            ),
        ),
        "transformer": (
            ROOT / "docs" / "core" / "transformer.md",
            (
                "跨框架等价不是同名模块等价",
                "RMSNorm 反事实 logits 最大差",
            ),
        ),
        "accuracy": (
            ROOT / "docs" / "reference" / "accuracy.md",
            (
                "PyTorch/JAX MiniGPT parity control",
                "0.37747739627957344",
                "2.384185791015625e-07",
            ),
        ),
        "knowledge_map": (
            ROOT / "docs" / "guide" / "knowledge-map.md",
            (
                "PyTorch↔JAX parity",
                "原生 RMSNorm 反事实",
            ),
        ),
        "repo_map": (
            ROOT / "docs" / "guide" / "repo-map.md",
            (
                "PyTorch/JAX LayerNorm parity",
                "RMSNorm 反事实",
            ),
        ),
        "project_index": (
            ROOT / "docs" / "practice" / "project-index.md",
            (
                "JAX MiniGPT 又新增 PyTorch↔JAX",
                "20 个 unique parameter gradients",
            ),
        ),
        "changelog": (
            ROOT / "CHANGELOG.md",
            (
                "PyTorch↔JAX MiniGPT forward/backward/SGD parity control",
                "0.37747739627957344",
                "2.384185791015625e-07",
            ),
        ),
    }
    for name, (path, markers) in required_docs.items():
        document = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in document]
        if missing:
            errors.append(
                f"PyTorch/JAX GPT parity {name} docs missing marker(s): {missing}"
            )

    stale_claims = {
        ROOT / "projects" / "jax-minigpt" / "README.md": (
            "做 PyTorch/JAX 同权重小模型的 logits、loss 和单步更新对照",
        ),
    }
    for path, forbidden_markers in stale_claims.items():
        document = path.read_text(encoding="utf-8")
        present = [
            marker for marker in forbidden_markers if marker in document
        ]
        if present:
            errors.append(
                f"PyTorch/JAX GPT parity stale claim(s) in {path.name}: {present}"
            )
    return errors


def check_gpt_cross_framework_training_parity_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.from_scratch.gpt_cross_framework_training import (
        GPT_CROSS_FRAMEWORK_TRAINING_PARITY_VERSION,
        run_gpt_cross_framework_training_parity_control,
    )

    report = run_gpt_cross_framework_training_parity_control()
    runtime = report.get("runtime", {})
    fixture = report.get("fixture", {})
    dropout = fixture.get("dropout", {}) if isinstance(fixture, dict) else {}
    optimizer = fixture.get("optimizer", {}) if isinstance(fixture, dict) else {}
    steps = report.get("steps", [])
    comparison = report.get("comparison", {})
    assertions = report.get("assertions", {})
    scope = report.get("scope", {})
    errors: list[str] = []

    expected_masks = [
        "sha256:7277dcc5670adf12d8eacb46155845890aa0272e54c51be11df570e6aa40287e",
        "sha256:46a4c1fbd64af64766839d4992e097b78e1eabf2cf4469804a07ba816db025cf",
        "sha256:47b90e2328e71d62e2c3d718b7714109176c7083db9d1e5f040978f3b6dfc67c",
    ]
    if not (
        report.get("schema_version")
        == GPT_CROSS_FRAMEWORK_TRAINING_PARITY_VERSION
        and report.get("report_fingerprint")
        == (
            "sha256:68ffa8093a1f2b986fa7c0c8d9c45075"
            "dcb17c1cf5c0b92d0852e874e175c609"
        )
        and runtime.get("jax_backend") == "cpu"
        and runtime.get("torch_device") == "cpu"
        and runtime.get("dtype") == "float32"
        and fixture.get("steps") == 3
        and fixture.get("comparison_tolerance") == 5e-6
        and dropout
        == {
            "site": "embedding sum only",
            "kind": "externally materialized inverted dropout",
            "rate": 0.25,
            "generator": "NumPy PCG64",
            "seed": 20260814,
            "mask_shape": [2, 4, 8],
            "mask_sha256": expected_masks,
        }
        and optimizer
        == {
            "kind": "AdamW",
            "learning_rates": (0.02, 0.01, 0.005),
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "weight_decay": 0.03,
            "weight_decay_mask": "all parameters",
            "max_grad_norm": 0.08,
        }
        and isinstance(assertions, dict)
        and assertions
        and all(assertions.values())
    ):
        errors.append("PyTorch/JAX training parity identity/runtime mismatch")

    maximum = (
        comparison.get("maximum_difference_across_steps", {})
        if isinstance(comparison, dict)
        else {}
    )
    expected_maximum = {
        "loss_before_step_abs_difference": 2.384185791015625e-07,
        "preclip_gradient_norm_abs_difference": 2.384185791015625e-07,
        "raw_gradient_global_max_abs_difference": 3.129243850708008e-07,
        "clipped_gradient_global_max_abs_difference": 1.862645149230957e-08,
        "parameter_global_max_abs_difference": 2.5480985641479492e-06,
        "first_moment_global_max_abs_difference": 2.561137080192566e-09,
        "second_moment_global_max_abs_difference": 8.003553375601768e-11,
        "post_step_logits_max_abs_difference": 1.564621925354004e-07,
        "post_step_loss_abs_difference": 2.384185791015625e-07,
    }
    if not (
        isinstance(steps, list)
        and len(steps) == 3
        and [step.get("step") for step in steps] == [1, 2, 3]
        and [step.get("learning_rate") for step in steps]
        == [0.02, 0.01, 0.005]
        and [step.get("kept_elements") for step in steps] == [54, 50, 45]
        and [step.get("mask_sha256") for step in steps] == expected_masks
        and [step.get("torch_adam_step") for step in steps] == [1, 2, 3]
        and [step.get("jax_adam_count") for step in steps] == [1, 2, 3]
        and [step.get("jax_schedule_count") for step in steps] == [1, 2, 3]
        and all(
            step.get("torch_preclip_gradient_norm", 0) > 0.08
            and step.get("jax_preclip_gradient_norm", 0) > 0.08
            for step in steps
        )
        and maximum == expected_maximum
        and math.isclose(
            comparison.get(
                "wrong_mask_final_parameter_max_abs_difference",
                math.inf,
            ),
            0.06900620367377996,
            rel_tol=0,
            abs_tol=1e-15,
        )
    ):
        errors.append("PyTorch/JAX training parity observation mismatch")

    expected_scope = {
        "same_initial_parameter_values_compared": True,
        "shared_materialized_embedding_dropout_masks_compared": True,
        "framework_native_rng_equivalence_claimed": False,
        "dropout_prng_state_advance_compared": False,
        "raw_and_global_norm_clipped_gradients_compared": True,
        "adamw_first_second_moments_and_count_compared": True,
        "learning_rate_schedule_compared": True,
        "all_parameter_weight_decay_compared": True,
        "norm_or_bias_weight_decay_mask_compared": False,
        "three_post_update_forwards_compared": True,
        "wrong_materialized_mask_counterfactual_executed": True,
        "checkpoint_resume_or_artifact_serialization_compared": False,
        "jit_compile_or_async_timing_compared": False,
        "cuda_tpu_multi_device_or_sharding_executed": False,
        "large_model_training_convergence_or_performance_proved": False,
    }
    if scope != expected_scope:
        errors.append("PyTorch/JAX training parity evidence scope mismatch")

    source = (
        SRC
        / "about_llm"
        / "from_scratch"
        / "gpt_cross_framework_training.py"
    ).read_text(encoding="utf-8")
    script = (
        ROOT
        / "projects"
        / "jax-minigpt"
        / "cross_framework_training_parity.py"
    ).read_text(encoding="utf-8")
    test = (
        ROOT / "tests" / "test_gpt_cross_framework_training_parity.py"
    ).read_text(encoding="utf-8")
    required_code_markers = {
        "source": (
            source,
            (
                '"about-llm.gpt-cross-framework-training-parity.v1"',
                "np.random.PCG64(DROPOUT_SEED)",
                "torch.nn.utils.clip_grad_norm_",
                "optax.adamw(",
                '"wrong_materialized_mask_counterfactual_executed": True',
                'with jax.default_device(cpu_devices[0])',
            ),
        ),
        "script": (
            script,
            (
                "run_gpt_cross_framework_training_parity_control",
                "allow_nan=False",
            ),
        ),
        "test": (
            test,
            (
                "test_pytorch_jax_adamw_clipping_schedule_and_mask_parity",
                "68ffa8093a1f2b986fa7c0c8d9c45075",
                "wrong_mask_final_parameter_max_abs_difference",
            ),
        ),
    }
    for label, (document, markers) in required_code_markers.items():
        missing = [marker for marker in markers if marker not in document]
        if missing:
            errors.append(
                f"PyTorch/JAX training parity {label} missing marker(s): "
                f"{missing}"
            )

    required_docs = {
        "readme": (
            ROOT / "README.md",
            (
                "PyTorch↔JAX 三步 AdamW parity",
                "0.06900620367377996",
            ),
        ),
        "project": (
            ROOT / "projects" / "jax-minigpt" / "README.md",
            (
                "cross_framework_training_parity.py",
                "materialized dropout masks",
                "2.5480985641479492e-06",
            ),
        ),
        "project_page": (
            ROOT / "docs" / "practice" / "projects" / "jax-minigpt.md",
            (
                "三步 AdamW trajectory",
                "native RNG equivalence",
            ),
        ),
        "training": (
            ROOT / "docs" / "training" / "jax-optax.md",
            (
                "三步 AdamW trajectory parity",
                "first/second moments",
                "68ffa8093a1f2b98",
            ),
        ),
        "interview": (
            ROOT / "docs" / "career" / "interview-questions.md",
            (
                "共享 dropout mask 对齐\uff0c能否证明 PyTorch/JAX 原生 RNG 等价",
                "PCG64",
            ),
        ),
        "accuracy": (
            ROOT / "docs" / "reference" / "accuracy.md",
            (
                "PyTorch/JAX stochastic AdamW trajectory control",
                "0.06900620367377996",
            ),
        ),
        "knowledge_map": (
            ROOT / "docs" / "guide" / "knowledge-map.md",
            ("AdamW/clipping/schedule parity",),
        ),
        "repo_map": (
            ROOT / "docs" / "guide" / "repo-map.md",
            ("shared-mask AdamW trajectory",),
        ),
        "project_index": (
            ROOT / "docs" / "practice" / "project-index.md",
            ("JAX MiniGPT 的三步 AdamW parity control",),
        ),
        "changelog": (
            ROOT / "CHANGELOG.md",
            ("PyTorch↔JAX stochastic AdamW trajectory parity control",),
        ),
    }
    for name, (path, markers) in required_docs.items():
        document = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in document]
        if missing:
            errors.append(
                f"PyTorch/JAX training parity {name} docs missing marker(s): "
                f"{missing}"
            )
    stale_claims = {
        ROOT / "projects" / "jax-minigpt" / "README.md": (
            "将 parity 扩展到明确对齐的 AdamW state、schedule 与 dropout PRNG",
        ),
        ROOT / "docs" / "training" / "jax-optax.md": (
            "并把 AdamW state、schedule 与显式 PRNG 纳入新的对账",
        ),
    }
    for path, forbidden_markers in stale_claims.items():
        document = path.read_text(encoding="utf-8")
        present = [
            marker for marker in forbidden_markers if marker in document
        ]
        if present:
            errors.append(
                f"PyTorch/JAX training parity stale claim(s) in "
                f"{path.name}: {present}"
            )
    return errors


def check_jax_training_resume_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.from_scratch.jax_training_resume import (
        JAX_TRAINING_RESUME_VERSION,
        run_jax_training_resume_control,
    )

    report = run_jax_training_resume_control()
    runtime = report.get("runtime", {})
    fixture = report.get("fixture", {})
    artifact = report.get("artifact", {})
    process_observation = report.get("process_observation", {})
    uninterrupted = report.get("uninterrupted", {})
    resumed = report.get("resumed", {})
    counterfactuals = report.get("counterfactuals", {})
    assertions = report.get("assertions", {})
    scope = report.get("scope", {})
    errors: list[str] = []

    if not (
        report.get("schema_version") == JAX_TRAINING_RESUME_VERSION
        and report.get("report_fingerprint")
        == "sha256:652c22d525598adfbd473738c6b3ef4cbffaf13c0e2ae06a8de63b1d467e6fee"
        and runtime.get("jax_backend") == "cpu"
        and runtime.get("dtype") == "float32"
        and runtime.get("process_start_method") == "spawn"
        and fixture.get("model_config")
        == {
            "context_length": 4,
            "mlp_ratio": 2,
            "model_dim": 8,
            "num_heads": 2,
            "num_layers": 1,
            "vocab_size": 13,
        }
        and fixture.get("optimizer")
        == {
            "kind": "clip-global-norm+adamw",
            "learning_rate": 0.01,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
        }
        and fixture.get("dropout_rate") == 0.2
        and fixture.get("batch_size") == 2
        and fixture.get("dataset_examples") == 7
        and fixture.get("dataset_fingerprint")
        == "sha256:d91b77df27af887b21758e4b2f0cb69004db9da08b99942a4ba085526bf99da3"
        and fixture.get("total_steps") == 6
        and fixture.get("split_step") == 3
        and artifact
        == {
            "artifact_bytes": 13476,
            "artifact_sha256": (
                "sha256:e9252e5dddfa4aa507bfaa864cd205f9b"
                "a5a7c0aef7a03b4b98366f770568a35"
            ),
        }
        and process_observation
        == {
            "distinct_phase_worker_count": 2,
            "raw_process_ids_published": False,
        }
        and isinstance(assertions, dict)
        and assertions
        and all(assertions.values())
    ):
        errors.append("JAX checkpoint-resume identity/runtime mismatch")

    expected_trace = {
        "sample_ids": [[0, 4], [3, 2], [5, 1], [6, 3], [2, 1], [6, 4]],
        "losses": [
            2.584887981414795,
            2.517184019088745,
            2.497069835662842,
            2.4534246921539307,
            2.3306925296783447,
            2.374443292617798,
        ],
        "gradient_norms": [
            1.6905815601348877,
            1.9288957118988037,
            1.3948036432266235,
            1.6974924802780151,
            1.3795976638793945,
            1.1869877576828003,
        ],
    }
    expected_final_fingerprint = (
        "sha256:720817cca4c067cf1e532a5ce73e13d0dd1eba1c7b65c964445f67171d058f33"
    )
    if not (
        uninterrupted == resumed
        and uninterrupted.get("final_state_fingerprint")
        == expected_final_fingerprint
        and uninterrupted.get("trace") == expected_trace
    ):
        errors.append("JAX checkpoint-resume bit-exact trace mismatch")

    reset_rng = (
        counterfactuals.get("reset_dropout_prng", {})
        if isinstance(counterfactuals, dict)
        else {}
    )
    reset_cursor = (
        counterfactuals.get("reset_data_cursor", {})
        if isinstance(counterfactuals, dict)
        else {}
    )
    if not (
        reset_rng.get("final_state_fingerprint")
        == "sha256:7847080bbaccde12815f0b98b7d66893052304b6d0e9791a23dea73539d44b53"
        and reset_rng.get("parameter_max_abs_difference")
        == 0.037261832505464554
        and reset_cursor.get("final_state_fingerprint")
        == "sha256:7985745ff2d03704f39ee640f9c9207ad15760799620cfe7c4c2559be2dff635"
        and reset_cursor.get("parameter_max_abs_difference")
        == 0.03700308472616598
        and reset_cursor.get("trace", {}).get("sample_ids")
        == [[0, 4], [3, 2], [5, 1]]
    ):
        errors.append("JAX checkpoint-resume counterfactual mismatch")

    expected_scope = {
        "strict_canonical_manifest_and_outer_digest_executed": True,
        "parameter_and_optax_state_restored": True,
        "dropout_prng_and_data_shuffle_state_restored": True,
        "cross_process_split_resume_executed": True,
        "bit_exact_full_state_and_trace_compared": True,
        "wrong_prng_and_cursor_counterfactuals_executed": True,
        "exclusive_create_and_file_fsync_executed": True,
        "directory_fsync_or_power_loss_atomicity_proved": False,
        "orbax_flax_tensorstore_or_distributed_checkpoint_executed": False,
        "cuda_tpu_multi_device_or_sharding_executed": False,
        "python_numpy_worker_or_accelerator_rng_restored": False,
        "target_model_dataset_convergence_or_performance_proved": False,
        "artifact_origin_authentication_or_confidentiality_proved": False,
    }
    if scope != expected_scope:
        errors.append("JAX checkpoint-resume evidence scope mismatch")

    source = (
        SRC / "about_llm" / "from_scratch" / "jax_training_resume.py"
    ).read_text(encoding="utf-8")
    script = (
        ROOT / "projects" / "jax-minigpt" / "checkpoint_resume_control.py"
    ).read_text(encoding="utf-8")
    test = (ROOT / "tests" / "test_jax_training_resume.py").read_text(
        encoding="utf-8"
    )
    required_code_markers = {
        "source": (
            source,
            (
                '"about-llm.jax-training-resume.v1"',
                'MAGIC = b"ALLMJAX1"',
                "object_pairs_hook=_strict_object",
                "jax.random.key_data(state.dropout_key)",
                "jax.random.wrap_key_data",
                "with target.open(\"xb\")",
                "os.fsync(handle.fileno())",
                "mp.get_context(\"spawn\")",
                "allow_nan=False",
            ),
        ),
        "script": (
            script,
            (
                "run_jax_training_resume_control",
                "allow_nan=False",
            ),
        ),
        "test": (
            test,
            (
                "test_cross_process_jax_checkpoint_resume_is_bit_exact",
                "inner_payload",
                "FileExistsError",
                "652c22d525598adfbd473738c6b3ef4cbffaf13c0e2ae06a8de63b1d467e6fee",
            ),
        ),
    }
    for label, (document, markers) in required_code_markers.items():
        missing = [marker for marker in markers if marker not in document]
        if missing:
            errors.append(
                f"JAX checkpoint-resume {label} missing marker(s): {missing}"
            )

    required_docs = {
        "readme": (
            ROOT / "README.md",
            (
                "JAX/Optax cross-process bit-exact resume",
                "13,476-byte",
                "0.037261832505464554",
            ),
        ),
        "project": (
            ROOT / "projects" / "jax-minigpt" / "README.md",
            (
                "checkpoint_resume_control.py",
                "`ALLMJAX1`",
                "720817cca4c067cf",
            ),
        ),
        "project_page": (
            ROOT / "docs" / "practice" / "projects" / "jax-minigpt.md",
            (
                "checkpoint_resume_control.py",
                "wrong PRNG",
                "wrong cursor",
            ),
        ),
        "training": (
            ROOT / "docs" / "training" / "jax-optax.md",
            (
                "独立进程 bit-exact resume",
                "Optax state、typed PRNG key data",
                "13,476 bytes",
            ),
        ),
        "interview": (
            ROOT / "docs" / "career" / "interview-questions.md",
            (
                "JAX checkpoint 能打开\uff0c为什么仍可能无法精确续训",
                "PyTree treedef",
                "wrong-cursor",
            ),
        ),
        "accuracy": (
            ROOT / "docs" / "reference" / "accuracy.md",
            (
                "JAX/Optax checkpoint-resume control",
                "e9252e5dddfa4aa5",
                "0.03700308472616598",
            ),
        ),
        "knowledge_map": (
            ROOT / "docs" / "guide" / "knowledge-map.md",
            (
                "JAX cross-process resume",
                "dropout/data PRNG",
            ),
        ),
        "repo_map": (
            ROOT / "docs" / "guide" / "repo-map.md",
            (
                "JAX strict checkpoint/resume",
                "Optax/PRNG/data cursor",
            ),
        ),
        "project_index": (
            ROOT / "docs" / "practice" / "project-index.md",
            (
                "JAX MiniGPT 的 strict resume control",
                "13,476-byte artifact",
            ),
        ),
        "changelog": (
            ROOT / "CHANGELOG.md",
            (
                "JAX/Optax strict checkpoint + cross-process bit-exact resume control",
                "e9252e5dddfa4aa5",
                "0.037261832505464554",
            ),
        ),
    }
    for name, (path, markers) in required_docs.items():
        document = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in document]
        if missing:
            errors.append(
                f"JAX checkpoint-resume {name} docs missing marker(s): {missing}"
            )

    stale_project_marker = (
        "将数据 iterator/RNG state 与 optimizer state 一起保存和恢复"
    )
    project_readme = (
        ROOT / "projects" / "jax-minigpt" / "README.md"
    ).read_text(encoding="utf-8")
    if stale_project_marker in project_readme:
        errors.append("JAX checkpoint-resume stale project next-step claim")
    return errors


def check_model_config_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.model_config import (
        estimate_standard_kv_cache,
        inspect_decoder_config,
        load_model_config_json,
    )

    config_dir = ROOT / "projects" / "transformers-basics" / "configs"
    standard = inspect_decoder_config(
        load_model_config_json(config_dir / "standard-gqa.example.json")
    )
    standard_estimate = estimate_standard_kv_cache(
        standard, token_count=4096, batch_size=1, element_bytes=2
    )
    moe = inspect_decoder_config(
        load_model_config_json(config_dir / "moe-gqa.example.json")
    )
    moe_estimate = estimate_standard_kv_cache(
        moe, token_count=4096, batch_size=2, element_bytes=2
    )
    mla = inspect_decoder_config(
        load_model_config_json(config_dir / "mla-moe.example.json")
    )

    errors: list[str] = []
    if not (
        standard.model_type == "authored_standard_gqa"
        and standard.config_fingerprint
        == "sha256:16839fe12b7e1280d1a5fd60102387e1aed21d6dbf0a03c148c53da46b731e46"
        and standard.standard_kv_layout.attention_kind == "gqa"
        and standard.standard_kv_layout.head_dim == 128
        and standard_estimate.bytes_per_token_per_layer == 4096
        and standard_estimate.total_bytes == 536_870_912
    ):
        errors.append("authored standard-GQA config/KV fixture mismatch")
    if not (
        moe.model_type == "authored_moe_gqa"
        and bool(moe.moe_marker_fields)
        and moe.standard_kv_layout.attention_kind == "gqa"
        and moe_estimate.bytes_per_token_per_layer == 2048
        and moe_estimate.total_bytes == 402_653_184
        and moe.to_dict()["parameter_count_estimated"] is False
    ):
        errors.append("authored MoE-GQA config/KV fixture mismatch")
    if not (
        mla.model_type == "authored_mla_moe"
        and bool(mla.mla_marker_fields)
        and bool(mla.moe_marker_fields)
        and mla.standard_kv_layout.applicable is False
        and "standard dense K/V formula must not be applied"
        in mla.standard_kv_layout.reason
    ):
        errors.append("authored MLA config fail-closed fixture mismatch")
    return errors


def check_model_release_evidence() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.model_release_evidence import (
        MODEL_RELEASE_EVIDENCE_BOUNDARY,
        verify_model_release_evidence,
    )

    manifest = (
        ROOT
        / "projects"
        / "transformers-basics"
        / "release-evidence"
        / "manifest.json"
    )
    report = verify_model_release_evidence(manifest)
    records = {str(record.get("record_id")): record for record in report.records}
    llama = records.get("llama-3.2-text-model-card", {})
    qwen = records.get("qwen2.5-0.5b-instruct-config", {})
    deepseek = records.get("deepseek-v3-config", {})
    qwen_contract = qwen.get("contract", {})
    deepseek_contract = deepseek.get("contract", {})
    qwen_estimates = qwen.get("standard_kv_estimates", [])

    errors: list[str] = []
    if not (
        report.manifest_fingerprint
        == "sha256:74166133716bfebddb444587e9f9a012b4beada923f5209482308ff61194953b"
        and report.projection_fingerprint
        == "sha256:40b3fe7b2a9c054ea6aa17e9e747d1831b8ae41ee3d55130c916f818acbe4638"
        and report.upstream_verified is False
        and len(report.records) == 3
    ):
        errors.append("model release evidence manifest/report identity mismatch")
    if not (
        llama.get("source_fragment_count") == 6
        and llama.get("source_fragments_verified") is False
        and llama.get("vendor_reported", {}).get("evidence_type")
        == "vendor_model_card_claims_not_independent_measurements"
        and llama.get("vendor_reported", {}).get("reported_context_length") == "128k"
    ):
        errors.append("Llama vendor-model-card projection mismatch")
    if not (
        qwen.get("config_fingerprint")
        == "sha256:ee6f9831a4c4729cf094af9a76a53dfe1dde8e34a8251889f527d2179c7d918d"
        and qwen_contract.get("attention_kind") == "gqa"
        and qwen_contract.get("head_dim") == 64
        and qwen_contract.get("query_heads_per_kv_head") == 7
        and isinstance(qwen_estimates, list)
        and len(qwen_estimates) == 1
        and qwen_estimates[0].get("total_bytes") == 402_653_184
    ):
        errors.append("Qwen immutable config/KV release evidence mismatch")
    if not (
        deepseek.get("config_fingerprint")
        == "sha256:fed8c13b4637058cd68e600bd4bf7dc734bda4594dd583e3b49fa27c6e123cc6"
        and deepseek_contract.get("known_mla_markers_present") is True
        and deepseek_contract.get("known_moe_markers_present") is True
        and deepseek_contract.get("standard_kv_applicable") is False
        and deepseek.get("estimate_refused") is True
        and deepseek.get("standard_kv_estimates") == []
    ):
        errors.append("DeepSeek-V3 immutable config fail-closed evidence mismatch")
    if "Vendor model-card claims are not independent measurements" not in (
        MODEL_RELEASE_EVIDENCE_BOUNDARY
    ):
        errors.append("model release evidence boundary drift")
    return errors


def check_transformers_checkpoint_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.integrations.transformers_checkpoint_control import (
        TRANSFORMERS_CHECKPOINT_EVIDENCE_BOUNDARY,
        load_checkpoint_control_spec,
        verify_recorded_checkpoint_report,
    )

    directory = ROOT / "projects" / "transformers-basics" / "target-checkpoints"
    manifest = directory / "qwen2.5-0.5b-instruct.control.json"
    recorded_report = directory / "qwen2.5-0.5b-instruct.recorded-report.json"
    spec = load_checkpoint_control_spec(manifest)
    report = verify_recorded_checkpoint_report(
        recorded_report,
        expected_manifest_fingerprint=spec.manifest_fingerprint,
    )
    source = report.get("source", {})
    artifacts = report.get("artifacts", {})
    model = report.get("model", {})
    parameter_report = model.get("parameter_report", {})
    execution = report.get("execution", {})
    runtime = report.get("runtime", {})
    scope = report.get("scope", {})
    files = artifacts.get("files", [])
    weights = next(
        (
            item
            for item in files
            if isinstance(item, dict) and item.get("filename") == "model.safetensors"
        ),
        {},
    )

    errors: list[str] = []
    if not (
        spec.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
        and spec.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
        and spec.manifest_fingerprint
        == "sha256:ddf41f2cff963bc2a8fc186c28369abba8a920b850152fc815e2b17c7d037876"
        and report.get("report_fingerprint")
        == "sha256:56528a3e02ed6ef9d205dcf83ba456658d639d7681ab6a1ad9eb110211edba62"
        and source.get("model_id") == spec.model_id
        and source.get("revision") == spec.revision
        and source.get("loader_input") == "verified_local_snapshot_directory"
        and source.get("all_selected_file_bytes_verified_before_load") is True
    ):
        errors.append("Qwen target-checkpoint identity/report binding mismatch")
    if not (
        artifacts.get("selected_file_count") == 7
        and artifacts.get("selected_total_bytes") == 999_586_347
        and weights.get("size_bytes") == 988_097_824
        and weights.get("sha256")
        == "sha256:fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
        and weights.get("verified") is True
    ):
        errors.append("Qwen target-checkpoint immutable artifact evidence mismatch")
    if not (
        model.get("class") == "Qwen2ForCausalLM"
        and model.get("model_type") == "qwen2"
        and model.get("verified_raw_config_semantic_fingerprint")
        == "sha256:ee6f9831a4c4729cf094af9a76a53dfe1dde8e34a8251889f527d2179c7d918d"
        and model.get("parameter_dtypes") == ["torch.float32"]
        and model.get("eval_mode") is True
        and parameter_report.get("total_parameters") == 494_032_768
        and parameter_report.get("trainable_parameters") == 0
        and parameter_report.get("trainable_fraction") == 0.0
        and parameter_report.get("parameter_storage_bytes") == 1_976_131_072
        and execution.get("parameters_frozen_for_control") is True
        and runtime.get("device") == "cpu"
        and runtime.get("dtype") == "float32"
        and runtime.get("attention_implementation") == "eager"
        and runtime.get("cuda_executed") is False
    ):
        errors.append("Qwen target-checkpoint loaded-model/runtime evidence mismatch")
    max_abs_error = execution.get("cached_full_max_abs_error")
    tolerance = execution.get("cached_full_tolerance")
    if not (
        execution.get("prompt_token_count") == 31
        and execution.get("prefill_logits_shape") == [1, 31, 151_936]
        and execution.get("generated_token_ids") == [17, 151_645]
        and execution.get("decoded_continuation") == "2<|im_end|>"
        and execution.get("manual_prefill_argmax_matches_generate") is True
        and execution.get("manual_cached_argmax_matches_generate") is True
        and execution.get("cached_full_argmax_match") is True
        and execution.get("past_key_values_executed") is True
        and execution.get("generated_ended_with_eos") is True
        and execution.get("generation_eos_token_id") == 151_645
        and execution.get("generation_sampling_disabled") is True
        and isinstance(max_abs_error, float)
        and math.isclose(max_abs_error, 3.719329833984375e-05)
        and isinstance(tolerance, float)
        and tolerance == 1e-4
        and max_abs_error <= tolerance
    ):
        errors.append("Qwen target-checkpoint forward/cache/generate evidence mismatch")
    if not (
        scope.get("training_or_backward_executed") is False
        and scope.get("gpu_or_vllm_executed") is False
        and scope.get("model_quality_proven") is False
        and scope.get("performance_benchmark_performed") is False
        and scope.get("verification_to_loader_reopen_toctou_eliminated") is False
        and "does not authenticate the publisher"
        in TRANSFORMERS_CHECKPOINT_EVIDENCE_BOUNDARY
        and "verification-to-loader-reopen TOCTOU"
        in TRANSFORMERS_CHECKPOINT_EVIDENCE_BOUNDARY
    ):
        errors.append("Qwen target-checkpoint evidence boundary drift")
    return errors


def check_target_weight_quantization_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.integrations.transformers_weight_quantization_control import (
        TARGET_WEIGHT_QUANTIZATION_EVIDENCE_BOUNDARY,
        verify_recorded_target_weight_quantization_report,
    )

    report_path = (
        ROOT
        / "projects"
        / "transformers-basics"
        / "target-checkpoints"
        / "qwen2.5-0.5b-instruct.weight-int4.recorded-report.json"
    )
    report = verify_recorded_target_weight_quantization_report(report_path)
    selection = report.get("selection", {})
    artifact = report.get("artifact", {})
    execution = report.get("execution", {})
    scope = report.get("scope", {})
    last_logits_error = execution.get("last_logits_error", {})

    errors: list[str] = []
    if not (
        report.get("manifest_fingerprint")
        == "sha256:ddf41f2cff963bc2a8fc186c28369abba8a920b850152fc815e2b17c7d037876"
        and report.get("report_fingerprint")
        == "sha256:df9ee045be4bf2e2ab4441bacfe24ffd1f903e9a0715bda0f35219ac3928f5cb"
        and selection.get("module_name")
        == "model.layers.0.self_attn.o_proj"
        and selection.get("selected_parameters") == 802_816
        and selection.get("total_model_parameters") == 494_032_768
        and selection.get("weight_shape") == [896, 896]
    ):
        errors.append("Qwen selected-weight INT4 identity/selection mismatch")
    if not (
        artifact.get("serialized_bundle_sha256")
        == "sha256:006cc9a2abdd62b8513926fe822f892b58fc309ab4141fd7a3d6a1acac470bf7"
        and artifact.get("reference_fp32_weight_bytes") == 3_211_264
        and artifact.get("ideal_packed_weight_bytes") == 401_408
        and artifact.get("scale_metadata_bytes") == 25_088
        and artifact.get("serialized_bundle_bytes") == 427_328
        and isinstance(artifact.get("serialized_compression_ratio"), float)
        and math.isclose(
            artifact["serialized_compression_ratio"],
            7.514752134192002,
            rel_tol=1e-15,
        )
        and artifact.get("tamper_rejected_before_decode") is True
    ):
        errors.append("Qwen selected-weight INT4 artifact/arithmetic mismatch")
    if not (
        isinstance(last_logits_error, dict)
        and isinstance(last_logits_error.get("relative_l2_error"), float)
        and math.isclose(
            last_logits_error["relative_l2_error"],
            0.08513807180570929,
            rel_tol=1e-15,
        )
        and execution.get("baseline_last_argmax_token_id") == 17
        and execution.get("partial_quantized_last_argmax_token_id") == 17
        and execution.get("last_argmax_match") is True
        and execution.get("source_weight_restored_exactly") is True
    ):
        errors.append("Qwen selected-weight INT4 forward observation mismatch")
    expected_false = (
        "full_checkpoint_quantized",
        "quantized_runtime_loaded",
        "fused_low_bit_kernel_executed",
        "gptq_awq_or_calibration_executed",
        "generation_executed",
        "gpu_cuda_or_vllm_executed",
        "whole_model_storage_or_runtime_memory_proven",
        "model_quality_or_effective_context_proven",
        "performance_benchmark_performed",
        "publisher_authenticated_by_signature",
        "license_compatibility_proven",
        "production_safety_proven",
    )
    if not (
        all(scope.get(name) is False for name in expected_false)
        and "only model.layers.0.self_attn.o_proj.weight"
        in TARGET_WEIGHT_QUANTIZATION_EVIDENCE_BOUNDARY
        and "does not produce or load a full low-bit checkpoint"
        in TARGET_WEIGHT_QUANTIZATION_EVIDENCE_BOUNDARY
        and "dequantized FP32" in TARGET_WEIGHT_QUANTIZATION_EVIDENCE_BOUNDARY
    ):
        errors.append("Qwen selected-weight INT4 evidence boundary drift")
    return errors


def check_target_activation_patching_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.integrations.transformers_activation_patching_control import (
        QWEN2_5_0_5B_ACTIVATION_PATCHING_PROTOCOL,
        TARGET_ACTIVATION_PATCHING_EVIDENCE_BOUNDARY,
        verify_recorded_activation_patching_report,
    )
    from about_llm.integrations.transformers_checkpoint_control import (
        load_checkpoint_control_spec,
    )

    directory = ROOT / "projects" / "transformers-basics" / "target-checkpoints"
    manifest = directory / "qwen2.5-0.5b-instruct.control.json"
    recorded_report = directory / (
        "qwen2.5-0.5b-instruct.activation-patching.recorded-report.json"
    )
    spec = load_checkpoint_control_spec(manifest)
    report = verify_recorded_activation_patching_report(
        recorded_report,
        expected_checkpoint_manifest_fingerprint=spec.manifest_fingerprint,
    )
    source = report.get("source", {})
    model = report.get("model", {})
    result = report.get("result", {})
    baseline = result.get("baseline", {})
    execution = result.get("execution", {})
    controls = result.get("structural_controls", {})
    scope = report.get("scope", {})
    conditions = result.get("conditions", [])
    by_name = {
        item.get("name"): item for item in conditions if isinstance(item, dict)
    }

    errors: list[str] = []
    if not (
        QWEN2_5_0_5B_ACTIVATION_PATCHING_PROTOCOL.fingerprint
        == "sha256:e34b2bfe2999fe52acb18e8f1908d89db286db042be67ad4f2343d7b83ed6702"
        and report.get("report_fingerprint")
        == "sha256:3f8410f5c31666b1be4f83e343a5b849a0545b2f635f7d415da85a195eebb18c"
        and report.get("checkpoint_manifest_fingerprint")
        == spec.manifest_fingerprint
        and source.get("model_id") == spec.model_id
        and source.get("revision") == spec.revision
        and source.get("selected_total_bytes") == 999_586_347
        and source.get("all_selected_file_bytes_verified_before_load") is True
    ):
        errors.append("Qwen activation-patching source/protocol/report binding mismatch")
    if not (
        model.get("class") == "Qwen2ForCausalLM"
        and model.get("model_type") == "qwen2"
        and model.get("hidden_size") == 896
        and model.get("decoder_layer_count") == 24
        and model.get("parameter_dtypes") == ["torch.float32"]
        and model.get("parameter_report", {}).get("total_parameters")
        == 494_032_768
        and model.get("parameter_report", {}).get("trainable_parameters") == 0
    ):
        errors.append("Qwen activation-patching loaded model contract mismatch")
    if not (
        math.isclose(baseline.get("clean_metric", math.nan), 9.210310935974121)
        and math.isclose(
            baseline.get("corrupt_metric", math.nan), -7.7003021240234375
        )
        and math.isclose(
            baseline.get("clean_minus_corrupt_gap", math.nan),
            16.91061305999756,
        )
        and baseline.get("clean_top_token_id") == 59_604
        and baseline.get("clean_top_token_text") == "Paris"
        and baseline.get("corrupt_top_token_id") == 94_409
        and baseline.get("corrupt_top_token_text") == "Berlin"
    ):
        errors.append("Qwen activation-patching fixed behavior baseline mismatch")
    expected_recovery = {
        "source_position_layer_0": 1.0000241370674128,
        "source_position_layer_11": 0.9922442752431005,
        "source_position_layer_23": 0.0,
        "full_prefix_first_layer_positive_control": 1.0,
        "readout_position_final_layer_positive_control": 1.0,
        "future_position_first_layer_negative_control": 0.0,
    }
    if set(by_name) != set(expected_recovery) or any(
        not math.isclose(
            by_name[name].get("normalized_recovery", math.nan),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for name, expected in expected_recovery.items()
    ):
        errors.append("Qwen activation-patching condition/recovery mismatch")
    if not (
        controls.get("all_passed") is True
        and controls.get("control_tolerance") == 1e-5
        and all(controls.get("checks", {}).values())
        and execution.get("total_forward_count") == 10
        and execution.get("real_forward_hooks_executed") is True
        and execution.get("parameters_frozen_for_control") is True
        and execution.get("gradient_or_backward_executed") is False
        and execution.get("hook_count_after_control") == 0
    ):
        errors.append("Qwen activation-patching structural/execution control mismatch")
    if not (
        scope.get("target_checkpoint_weights_loaded") is True
        and scope.get("real_transformers_forward_hooks_executed") is True
        and scope.get("external_timestamped_preregistration") is False
        and scope.get("unique_natural_circuit_proven") is False
        and scope.get("unbiased_effect_estimate_proven") is False
        and scope.get("model_quality_or_factual_reliability_proven") is False
        and scope.get("cuda_gpu_or_vllm_executed") is False
        and "not an externally timestamped preregistration"
        in TARGET_ACTIVATION_PATCHING_EVIDENCE_BOUNDARY
        and "unique or natural circuit"
        in TARGET_ACTIVATION_PATCHING_EVIDENCE_BOUNDARY
    ):
        errors.append("Qwen activation-patching evidence boundary drift")
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "docs" / "core" / "architectures-interpretability.md",
            ROOT / "projects" / "transformers-basics" / "README.md",
            ROOT / "docs" / "reference" / "accuracy.md",
        )
    )
    required_markers = (
        "authored fixed protocol",
        "不是外部可信时间戳 preregistration",
        "不证明事实存储层、唯一自然 circuit",
        "final-layer source recovery=0",
        "sha256:3f8410f5…ebb18c",
    )
    missing = [marker for marker in required_markers if marker not in docs]
    if missing:
        errors.append(
            f"Qwen activation-patching docs missing boundary marker(s): {missing}"
        )
    return errors


def check_target_service_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.inference.target_service_control import (
        TARGET_SERVICE_EVIDENCE_BOUNDARY,
        load_and_verify_recorded_target_service_report,
        load_target_service_control_spec,
    )
    from about_llm.integrations.transformers_checkpoint_control import (
        load_checkpoint_control_spec,
    )

    project = ROOT / "projects" / "inference-serving"
    service_manifest = project / "qwen2.5-0.5b-service.control.json"
    checkpoint_manifest = (
        ROOT
        / "projects"
        / "transformers-basics"
        / "target-checkpoints"
        / "qwen2.5-0.5b-instruct.control.json"
    )
    recorded_report = project / "qwen2.5-0.5b-service.recorded-report.json"
    spec = load_target_service_control_spec(service_manifest)
    checkpoint_spec = load_checkpoint_control_spec(checkpoint_manifest)
    report = load_and_verify_recorded_target_service_report(
        service_manifest,
        checkpoint_manifest,
        recorded_report,
    )
    artifacts = report.get("artifacts", {})
    api = report.get("api", {})
    execution = report.get("execution", {})
    network = report.get("network", {})
    process = report.get("server_process", {})
    scope = report.get("scope", {})

    errors: list[str] = []
    if not (
        spec.manifest_fingerprint
        == "sha256:cfb9b5409c1ccec7267d85e5adca2ae8f8e9e80c0ff4301f0414f659728fb4ea"
        and spec.checkpoint_manifest_fingerprint
        == checkpoint_spec.manifest_fingerprint
        and report.get("report_fingerprint")
        == "sha256:63e566ca60126c09c0f97f23b591e879d6efe7991b646f72bcc96ec493617ddb"
        and report.get("checked_at") == "2026-08-13"
        and artifacts.get("selected_file_count") == 7
        and artifacts.get("selected_total_bytes") == 999_586_347
    ):
        errors.append("Qwen target-service manifest/artifact/report binding mismatch")
    if not (
        network.get("scheme") == "http"
        and network.get("address_scope") == "IPv4 loopback"
        and network.get("real_tcp_http") is True
        and network.get("tls") is False
        and process.get("subprocess_used") is True
        and process.get("stdout_stderr_empty") is True
        and api.get("models_endpoint_executed") is True
        and api.get("chat_nonstream_executed") is True
        and api.get("chat_sse_executed") is True
        and api.get("unauthorized_status") == 401
        and api.get("unknown_field_status") == 422
        and api.get("wrong_model_status") == 404
        and api.get("sse_done_observed") is True
        and api.get("stream_usage_observed") is True
        and api.get("raw_request_or_response_published") is False
    ):
        errors.append("Qwen target-service network/API control evidence mismatch")
    if not (
        execution.get("model_class") == "Qwen2ForCausalLM"
        and execution.get("model_type") == "qwen2"
        and execution.get("parameter_count") == 494_032_768
        and execution.get("parameters_frozen") is True
        and execution.get("framework_generate_call_count") == 2
        and execution.get("prompt_token_count") == 31
        and execution.get("completion_token_ids") == [17, 151_645]
        and execution.get("completion_text_fingerprint")
        == "sha256:f734df76252d8e1047f3dcca7ecbcef3e8d07c1e24c28dd62eb023b88ffac4a5"
        and execution.get("finish_reason") == "stop"
        and execution.get("nonstream_stream_content_match") is True
        and execution.get("nonstream_stream_usage_match") is True
        and execution.get("stream_content_delta_count") == 2
        and execution.get("generation_completed_before_sse_emission") is True
    ):
        errors.append("Qwen target-service framework/token/response evidence mismatch")
    if not (
        scope.get("target_checkpoint_weights_loaded") is True
        and scope.get("transformers_generate_executed") is True
        and scope.get("real_ipv4_loopback_tcp_http_executed") is True
        and scope.get("vllm_executed") is False
        and scope.get("cuda_executed") is False
        and scope.get("incremental_model_decode_streaming_proven") is False
        and scope.get("client_disconnect_cancellation_proven") is False
        and scope.get("performance_capacity_or_slo_proven") is False
        and scope.get("full_openai_api_compatibility_proven") is False
        and scope.get("model_quality_proven") is False
        and scope.get("production_safety_proven") is False
        and "does not use vLLM, CUDA, TLS" in TARGET_SERVICE_EVIDENCE_BOUNDARY
        and "verification-to-loader-reopen TOCTOU"
        in TARGET_SERVICE_EVIDENCE_BOUNDARY
    ):
        errors.append("Qwen target-service evidence boundary drift")

    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "README.md",
            project / "README.md",
            ROOT / "docs" / "systems" / "inference.md",
            ROOT / "docs" / "systems" / "serving.md",
            ROOT / "docs" / "systems" / "vllm-serving.md",
            ROOT / "docs" / "practice" / "labs.md",
            ROOT / "docs" / "career" / "interview-questions.md",
            ROOT / "docs" / "career" / "resume-projects.md",
            ROOT / "docs" / "reference" / "accuracy.md",
        )
    )
    required_markers = (
        "sha256:63e566ca…617ddb",
        "999,586,347",
        "[17,151645]",
        "两次 `GenerationMixin.generate()`",
        "SSE 在完整 generation 后",
        "不证明 incremental decode/cancel",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(f"target-service docs missing boundary marker(s): {missing}")
    return errors


def check_incremental_streaming_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.inference.incremental_streaming_control import (
        INCREMENTAL_STREAMING_EVIDENCE_BOUNDARY,
        SCRIPTED_BACKEND_FINGERPRINT,
        load_and_verify_incremental_streaming_report,
    )

    project = ROOT / "projects" / "inference-serving"
    recorded_report = project / "incremental-streaming.recorded-report.json"
    report = load_and_verify_incremental_streaming_report(recorded_report)
    complete = report.get("complete_stream", {})
    disconnect = report.get("disconnect_stream", {})
    audit = report.get("audit", {})
    process = report.get("server_process", {})
    scope = report.get("scope", {})

    errors: list[str] = []
    if not (
        report.get("report_fingerprint")
        == "sha256:258468229bb14af198f7a39a68999fb41375a9256ee7aa4b2c2c0e80f42b5d00"
        and report.get("checked_at") == "2026-08-13"
        and report.get("backend_fingerprint") == SCRIPTED_BACKEND_FINGERPRINT
        and report.get("evidence_boundary")
        == INCREMENTAL_STREAMING_EVIDENCE_BOUNDARY
    ):
        errors.append("incremental-streaming report identity/boundary mismatch")
    if not (
        complete.get("client_content_delta_count") == 3
        and complete.get("backend_completion_token_ids") == [101, 102, 103]
        and complete.get("client_finish_reason") == "stop"
        and complete.get("client_usage")
        == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        and complete.get("client_sse_done_observed") is True
        and complete.get("backend_completed") is True
    ):
        errors.append("incremental-streaming complete-case evidence mismatch")
    if not (
        disconnect.get("preclose_service_active_streams") == 1
        and disconnect.get("preclose_backend_completed") is False
        and disconnect.get("client_response_explicitly_closed") is True
        and disconnect.get("postclose_service_active_streams") == 0
        and disconnect.get("postclose_service_cancelled_streams") == 1
        and disconnect.get("postclose_backend_asyncio_cancelled_error_observed")
        is True
        and disconnect.get("postclose_backend_iterator_closed") is True
        and disconnect.get("postclose_backend_completed") is False
        and disconnect.get("postclose_backend_emitted_token_ids") == [201]
    ):
        errors.append("incremental-streaming disconnect evidence mismatch")
    if not (
        audit.get("accepted_requests") == 2
        and audit.get("completed_incremental_streams") == 1
        and audit.get("cancelled_incremental_streams") == 1
        and audit.get("failed_backend_requests") == 0
        and audit.get("single_process_admission_limit") == 1
        and process.get("subprocess_used") is True
        and process.get("stdout_stderr_empty") is True
    ):
        errors.append("incremental-streaming audit/process evidence mismatch")
    if not (
        scope.get("authored_async_backend_executed") is True
        and scope.get("real_ipv4_loopback_tcp_http_executed") is True
        and scope.get("content_observed_before_backend_completion") is True
        and scope.get("client_disconnect_cancelled_asgi_stream_task") is True
        and scope.get("cooperative_async_backend_cancellation_observed") is True
        and scope.get("later_authored_deltas_suppressed_after_disconnect") is True
        and scope.get("tokenizer_or_model_forward_executed") is False
        and scope.get("transformers_generation_thread_cancellation_proven") is False
        and scope.get("vllm_or_cuda_executed") is False
        and scope.get("kv_or_gpu_resource_release_proven") is False
        and scope.get("remote_provider_cancellation_or_billing_proven") is False
        and scope.get("performance_quality_or_slo_proven") is False
        and "does not execute a tokenizer, model forward" in INCREMENTAL_STREAMING_EVIDENCE_BOUNDARY
        and "does not authenticate the recorder"
        in INCREMENTAL_STREAMING_EVIDENCE_BOUNDARY
    ):
        errors.append("incremental-streaming evidence scope drift")

    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "README.md",
            project / "README.md",
            ROOT / "docs" / "systems" / "serving.md",
            ROOT / "docs" / "systems" / "vllm-serving.md",
            ROOT / "docs" / "practice" / "labs.md",
            ROOT / "docs" / "career" / "interview-questions.md",
            ROOT / "docs" / "career" / "resume-projects.md",
            ROOT / "docs" / "reference" / "accuracy.md",
        )
    )
    required_markers = (
        "sha256:25846822…2b5d00",
        "content-before-completion",
        "CancelledError",
        "Transformers blocking thread",
        "KV/GPU release",
        "provider billing",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            f"incremental-streaming docs missing boundary marker(s): {missing}"
        )
    return errors


def check_transformers_thread_cancellation_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.inference.transformers_thread_cancellation_control import (
        BACKEND_FINGERPRINT,
        TRANSFORMERS_THREAD_CANCELLATION_EVIDENCE_BOUNDARY,
        load_and_verify_transformers_thread_cancellation_report,
    )

    project = ROOT / "projects" / "inference-serving"
    report = load_and_verify_transformers_thread_cancellation_report(
        project / "transformers-thread-cancellation.recorded-report.json"
    )
    model = report.get("model", {})
    preclose = report.get("preclose", {})
    postclose = report.get("postclose", {})
    audit = report.get("audit", {})
    scope = report.get("scope", {})

    errors: list[str] = []
    if not (
        report.get("report_fingerprint")
        == "sha256:eadcab544cc78dabfc171446fd825992cc1c12edbbc478679c8bb10f7cf62bc7"
        and report.get("checked_at") == "2026-08-13"
        and report.get("backend_fingerprint") == BACKEND_FINGERPRINT
        and report.get("evidence_boundary")
        == TRANSFORMERS_THREAD_CANCELLATION_EVIDENCE_BOUNDARY
    ):
        errors.append("Transformers thread-cancellation report identity mismatch")
    if not (
        model.get("architecture") == "GPT2LMHeadModel"
        and model.get("parameter_count") == 1_272
        and model.get("device") == "cpu"
        and model.get("dtype") == "float32"
        and model.get("input_token_ids") == [1, 2, 3]
        and model.get("forced_token_id") == 7
        and model.get("tokenizer_or_chat_template_executed") is False
        and model.get("public_checkpoint_loaded") is False
    ):
        errors.append("Transformers thread-cancellation model evidence mismatch")
    if not (
        preclose.get("service_active_streams") == 1
        and preclose.get("generation_thread_alive") is True
        and preclose.get("generation_returned") is False
        and preclose.get("streamer_waiting_for_cancel") is True
        and preclose.get("stopping_criteria_observed_cancel") is False
        and preclose.get("generated_token_ids") == [7]
        and preclose.get("forward_call_count") == 1
    ):
        errors.append("Transformers thread-cancellation preclose evidence mismatch")
    if not (
        postclose.get("backend_asyncio_cancelled_error_observed") is True
        and postclose.get("cancellation_event_set") is True
        and postclose.get("streamer_wait_released_by_cancel") is True
        and postclose.get("stopping_criteria_observed_cancel") is True
        and postclose.get("stopping_criteria_call_count") == 1
        and postclose.get("generation_returned") is True
        and postclose.get("generation_thread_exited") is True
        and postclose.get("generation_thread_joined") is True
        and postclose.get("generation_thread_alive") is False
        and postclose.get("generation_error_type") is None
        and postclose.get("generated_token_ids") == [7]
        and postclose.get("generate_output_token_ids") == [7]
        and postclose.get("forward_call_count") == 1
        and postclose.get("logits_processor_call_count") == 1
        and audit.get("cancelled_incremental_streams") == 1
        and audit.get("failed_backend_requests") == 0
    ):
        errors.append("Transformers thread-cancellation postclose evidence mismatch")
    if not (
        scope.get("transformers_generation_mixin_generate_executed") is True
        and scope.get("real_model_forward_executed") is True
        and scope.get("blocking_python_generation_thread_executed") is True
        and scope.get("threading_event_observed_by_stopping_criteria") is True
        and scope.get("generation_thread_joined_before_postclose_audit") is True
        and scope.get("second_generated_token_suppressed") is True
        and scope.get("unmodified_transformers_cancellation_proven") is False
        and scope.get("tokenizer_or_chat_template_executed") is False
        and scope.get("public_checkpoint_or_target_logits_executed") is False
        and scope.get("vllm_or_cuda_executed") is False
        and scope.get("kv_cpu_or_gpu_memory_release_proven") is False
        and scope.get("arbitrary_thread_process_or_kernel_termination_proven")
        is False
        and scope.get("remote_provider_cancellation_or_billing_proven") is False
        and "deterministic streamer pause"
        in TRANSFORMERS_THREAD_CANCELLATION_EVIDENCE_BOUNDARY
        and "nor does it prove KV/CPU/GPU memory release"
        in TRANSFORMERS_THREAD_CANCELLATION_EVIDENCE_BOUNDARY
    ):
        errors.append("Transformers thread-cancellation evidence boundary drift")

    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "README.md",
            project / "README.md",
            ROOT / "docs" / "systems" / "serving.md",
            ROOT / "docs" / "systems" / "vllm-serving.md",
            ROOT / "docs" / "practice" / "labs.md",
            ROOT / "docs" / "career" / "interview-questions.md",
            ROOT / "docs" / "career" / "resume-projects.md",
            ROOT / "docs" / "reference" / "accuracy.md",
        )
    )
    required_markers = (
        "sha256:eadcab54…f62bc7",
        "1,272",
        "StoppingCriteria",
        "streamer pause",
        "未修改",
        "KV/CPU/GPU",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "Transformers thread-cancellation docs missing boundary marker(s): "
            f"{missing}"
        )
    return errors


def check_rag_transformers_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.integrations.transformers_checkpoint_control import (
        load_checkpoint_control_spec,
    )
    from about_llm.rag.transformers_control import (
        RAG_TRANSFORMERS_EVIDENCE_BOUNDARY,
        load_rag_transformers_control_spec,
        verify_recorded_rag_transformers_report,
    )

    rag_directory = ROOT / "projects" / "rag-foundations"
    checkpoint_manifest = (
        ROOT
        / "projects"
        / "transformers-basics"
        / "target-checkpoints"
        / "qwen2.5-0.5b-instruct.control.json"
    )
    checkpoint_spec = load_checkpoint_control_spec(checkpoint_manifest)
    spec = load_rag_transformers_control_spec(
        rag_directory / "qwen2.5-0.5b-rag.control.json"
    )
    report = verify_recorded_rag_transformers_report(
        rag_directory / "qwen2.5-0.5b-rag.recorded-report.json",
        spec=spec,
        checkpoint_spec=checkpoint_spec,
    )
    checkpoint = report.get("checkpoint", {})
    model = report.get("model", {})
    summary = report.get("summary", {})
    scope = report.get("scope", {})
    cases = report.get("cases", [])
    answerable = cases[0] if isinstance(cases, list) and len(cases) == 2 else {}
    no_answer = cases[1] if isinstance(cases, list) and len(cases) == 2 else {}
    answerable_generation = answerable.get("generation", {})
    answerable_verification = answerable.get("verification", {})
    no_answer_retrieval = no_answer.get("retrieval", {})
    no_answer_packing = no_answer.get("packing", {})
    no_answer_generation = no_answer.get("generation", {})
    no_answer_verification = no_answer.get("verification", {})

    errors: list[str] = []
    if not (
        spec.manifest_fingerprint
        == "sha256:4ee166171982118552fcc73e38902e653596b652fc645573583a5ef6ca609dfd"
        and report.get("report_fingerprint")
        == "sha256:829663e216828ad418ddf9a6c38ee487fe44b38d3939072d0ce443e8e8ee5b60"
        and report.get("checkpoint_manifest_fingerprint")
        == checkpoint_spec.manifest_fingerprint
        and checkpoint.get("selected_file_count") == 7
        and checkpoint.get("selected_total_bytes") == 999_586_347
    ):
        errors.append("real-weight RAG manifest/checkpoint/report identity mismatch")
    if not (
        model.get("class") == "Qwen2ForCausalLM"
        and model.get("model_type") == "qwen2"
        and model.get("total_parameters") == 494_032_768
        and model.get("trainable_parameters") == 0
        and model.get("parameter_storage_bytes") == 1_976_131_072
        and model.get("parameter_dtypes") == ["torch.float32"]
        and model.get("eval_mode") is True
    ):
        errors.append("real-weight RAG model/runtime evidence mismatch")
    if not (
        answerable.get("case_id") == "answerable-citation"
        and answerable.get("retrieval", {}).get("document_ids")
        == ["acl-order-v1", "citation-boundary-v1"]
        and answerable.get("packing", {}).get("used_cost_units") == 273
        and answerable.get("prompt", {}).get("prompt_token_count") == 209
        and answerable_generation.get("generated_ended_with_eos") is True
        and answerable_generation.get("stop_reason") == "eos"
        and answerable_verification.get("cited_source_ids") == []
        and answerable_verification.get("citation_syntax_passed") is False
        and answerable_verification.get("expected_behavior_gate_passed") is False
    ):
        errors.append("real-weight RAG answerable citation-failure evidence mismatch")
    if not (
        no_answer.get("case_id") == "empty-evidence-abstention"
        and no_answer_retrieval.get("document_ids") == []
        and no_answer_packing.get("document_ids") == []
        and no_answer_packing.get("used_cost_units") == 179
        and no_answer.get("prompt", {}).get("prompt_token_count") == 115
        and len(no_answer_generation.get("generated_token_ids", [])) == 64
        and no_answer_generation.get("generated_ended_with_eos") is False
        and no_answer_generation.get("stop_reason") == "max_new_tokens"
        and no_answer_verification.get("abstention_exact_match") is False
        and no_answer_verification.get("expected_behavior_gate_passed") is False
    ):
        errors.append("real-weight RAG empty-evidence hallucination mismatch")
    if not (
        summary.get("case_count") == 2
        and summary.get("expected_behavior_gate_passed_count") == 0
        and summary.get("all_expected_behavior_gates_passed") is False
        and scope.get("model_failures_recorded_without_output_repair") is True
        and scope.get("claim_evidence_entailment_verified") is False
        and scope.get("general_rag_quality_proven") is False
        and scope.get("verification_to_loader_reopen_toctou_eliminated") is False
        and "records model failures rather than repairing outputs"
        in RAG_TRANSFORMERS_EVIDENCE_BOUNDARY
        and "does not prove semantic entailment" in RAG_TRANSFORMERS_EVIDENCE_BOUNDARY
    ):
        errors.append("real-weight RAG scope/evidence boundary drift")
    required_docs = {
        ROOT / "docs" / "applications" / "rag-generation.md": (
            "行为门禁为 **0/2**",
            "verify→loader reopen",
        ),
        ROOT / "projects" / "rag-foundations" / "README.md": (
            "总体行为门禁是 **0/2**",
            "不再为追求漂亮结果",
        ),
    }
    for path, markers in required_docs.items():
        text = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            errors.append(
                f"real-weight RAG documentation boundary missing in {path.name}: {missing}"
            )
    return errors


def check_rag_guarded_transformers_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.integrations.transformers_checkpoint_control import (
        load_checkpoint_control_spec,
    )
    from about_llm.rag.guarded_transformers_control import (
        RAG_GUARDED_TRANSFORMERS_EVIDENCE_BOUNDARY,
        load_guarded_rag_transformers_control_spec,
        verify_recorded_guarded_rag_transformers_report,
    )

    rag_directory = ROOT / "projects" / "rag-foundations"
    checkpoint_spec = load_checkpoint_control_spec(
        ROOT
        / "projects"
        / "transformers-basics"
        / "target-checkpoints"
        / "qwen2.5-0.5b-instruct.control.json"
    )
    spec = load_guarded_rag_transformers_control_spec(
        rag_directory / "qwen2.5-0.5b-rag.guarded.control.json"
    )
    report = verify_recorded_guarded_rag_transformers_report(
        rag_directory / "qwen2.5-0.5b-rag.guarded.recorded-report.json",
        spec=spec,
        checkpoint_spec=checkpoint_spec,
    )
    checkpoint = report.get("checkpoint", {})
    model = report.get("model", {})
    summary = report.get("summary", {})
    scope = report.get("scope", {})
    cases = report.get("cases", [])
    answerable = cases[0] if isinstance(cases, list) and len(cases) == 2 else {}
    no_evidence = cases[1] if isinstance(cases, list) and len(cases) == 2 else {}
    answerable_generation = answerable.get("generation", {})
    answerable_decision = answerable.get("decision", {})
    answerable_public = answerable.get("public_decision", {})
    no_evidence_generation = no_evidence.get("generation", {})
    no_evidence_decision = no_evidence.get("decision", {})
    no_evidence_public = no_evidence.get("public_decision", {})

    errors: list[str] = []
    if not (
        spec.manifest_fingerprint
        == "sha256:9ead4c0655673117f62e154ac78f7fed8a3f0da6acec1ed874b80e17cf40778a"
        and report.get("report_fingerprint")
        == "sha256:00706d003921282625e7c8ad89291c64493d35c13faf4ad7e7553a1388f29ede"
        and report.get("checkpoint_manifest_fingerprint")
        == checkpoint_spec.manifest_fingerprint
        and checkpoint.get("selected_file_count") == 7
        and checkpoint.get("selected_total_bytes") == 999_586_347
    ):
        errors.append("guarded Qwen RAG manifest/checkpoint/report identity mismatch")
    if not (
        model.get("class") == "Qwen2ForCausalLM"
        and model.get("model_type") == "qwen2"
        and model.get("total_parameters") == 494_032_768
        and model.get("trainable_parameters") == 0
        and model.get("parameter_storage_bytes") == 1_976_131_072
        and model.get("parameter_dtypes") == ["torch.float32"]
        and model.get("eval_mode") is True
    ):
        errors.append("guarded Qwen RAG model evidence mismatch")
    if not (
        answerable.get("case_id") == "guarded-answerable-citation"
        and answerable.get("retrieval", {}).get("document_ids")
        == ["acl-order-v1", "citation-boundary-v1"]
        and answerable.get("packing", {}).get("used_cost_units") == 272
        and answerable.get("prompt", {}).get("prompt_token_count") == 208
        and answerable_generation.get("framework_generate_invocation_count") == 1
        and answerable_generation.get("generator_callback_invocation_count") == 1
        and answerable_generation.get("raw_output") == "无权文档不得进行排序"
        and answerable_generation.get("generated_ended_with_eos") is True
        and answerable_decision.get("stage") == "post_generation"
        and answerable_decision.get("action") == "reject"
        and answerable_decision.get("reason_code") == "missing_citation"
        and answerable_public.get("action") == "reject"
        and "raw_output" not in answerable_public
    ):
        errors.append("guarded Qwen RAG answerable invocation/rejection mismatch")
    if not (
        no_evidence.get("case_id") == "guarded-empty-evidence"
        and no_evidence.get("retrieval", {}).get("document_ids") == []
        and no_evidence.get("packing", {}).get("document_ids") == []
        and no_evidence.get("packing", {}).get("used_cost_units") == 180
        and no_evidence.get("prompt", {}).get("prompt_token_count") == 116
        and no_evidence.get("prompt", {}).get("prompt_transmitted_to_model") is False
        and no_evidence_generation.get("framework_generate_invocation_count") == 0
        and no_evidence_generation.get("generator_callback_invocation_count") == 0
        and no_evidence_generation.get("generated_token_ids") == []
        and no_evidence_decision.get("stage") == "pre_generation"
        and no_evidence_decision.get("action") == "abstain"
        and no_evidence_decision.get("raw_output") is None
        and no_evidence_public.get("action") == "abstain"
        and "raw_output" not in no_evidence_public
    ):
        errors.append("guarded Qwen RAG empty-evidence suppression mismatch")
    if not (
        summary
        == {
            "case_count": 2,
            "framework_generate_invocation_count": 1,
            "publish_count": 0,
            "pre_generation_abstention_count": 1,
            "post_generation_rejection_count": 1,
            "public_raw_output_field_count": 0,
        }
        and scope.get("publication_policy_wrapped_generation_callback") is True
        and scope.get("framework_generate_invocation_executed_for_evidence") is True
        and scope.get(
            "framework_generate_invocation_suppression_observed_for_empty_evidence"
        )
        is True
        and scope.get("audit_public_projection_separation_executed") is True
        and scope.get("manual_greedy_logits_cross_check_executed") is False
        and scope.get("claim_evidence_entailment_verified") is False
        and scope.get("provider_billing_or_cancellation_verified") is False
        and scope.get("production_integration_proven") is False
        and "does not replay model generation"
        in RAG_GUARDED_TRANSFORMERS_EVIDENCE_BOUNDARY
        and "does not count internal model forward calls"
        in RAG_GUARDED_TRANSFORMERS_EVIDENCE_BOUNDARY
    ):
        errors.append("guarded Qwen RAG scope/evidence boundary drift")
    required_docs = {
        ROOT / "README.md": ("真实 guarded Qwen runtime", "00706d00"),
        rag_directory / "README.md": (
            "run_qwen_guarded_rag_control.py",
            "framework_generate_invocation_count",
            "不是代表性质量集",
        ),
        ROOT / "docs" / "applications" / "rag-generation.md": (
            "真实 guarded runtime",
            "00706d00",
        ),
        ROOT / "docs" / "practice" / "labs.md": (
            "run_qwen_guarded_rag_control.py",
            "GenerationMixin.generate",
        ),
        ROOT / "docs" / "career" / "interview-questions.md": (
            "真实 guarded control",
            "内部 forward",
        ),
        ROOT / "docs" / "career" / "resume-projects.md": (
            "真实 guarded runtime",
            "0 次",
        ),
        ROOT / "docs" / "practice" / "production-checklist.md": (
            "pre-generation 0-call",
            "public projection",
        ),
        ROOT / "docs" / "reference" / "accuracy.md": (
            "9ead4c06",
            "00706d00",
        ),
    }
    for path, markers in required_docs.items():
        text = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            errors.append(
                f"guarded Qwen RAG documentation missing in {path.name}: {missing}"
            )
    return errors


def check_rag_publication_policy() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.integrations.transformers_checkpoint_control import (
        load_checkpoint_control_spec,
    )
    from about_llm.rag.generation_policy import (
        RAG_PUBLICATION_REPLAY_EVIDENCE_BOUNDARY,
        build_publication_policy_replay_report,
        verify_publication_policy_replay_report,
    )
    from about_llm.rag.transformers_control import (
        load_rag_transformers_control_spec,
        verify_recorded_rag_transformers_report,
    )

    rag_directory = ROOT / "projects" / "rag-foundations"
    checkpoint_spec = load_checkpoint_control_spec(
        ROOT
        / "projects"
        / "transformers-basics"
        / "target-checkpoints"
        / "qwen2.5-0.5b-instruct.control.json"
    )
    spec = load_rag_transformers_control_spec(
        rag_directory / "qwen2.5-0.5b-rag.control.json"
    )
    source_report = verify_recorded_rag_transformers_report(
        rag_directory / "qwen2.5-0.5b-rag.recorded-report.json",
        spec=spec,
        checkpoint_spec=checkpoint_spec,
    )
    expected = build_publication_policy_replay_report(
        spec=spec,
        source_report=source_report,
    )
    replay = verify_publication_policy_replay_report(
        rag_directory / "qwen2.5-0.5b-rag.publication-policy-replay.json",
        spec=spec,
        source_report=source_report,
    )
    summary = replay.get("summary", {})
    scope = replay.get("scope", {})
    policy = replay.get("policy", {})
    cases = replay.get("cases", [])
    answerable = cases[0] if isinstance(cases, list) and len(cases) == 2 else {}
    no_evidence = cases[1] if isinstance(cases, list) and len(cases) == 2 else {}
    answerable_decision = answerable.get("decision", {})
    no_evidence_decision = no_evidence.get("decision", {})

    errors: list[str] = []
    if not (
        replay == expected
        and replay.get("report_fingerprint")
        == "sha256:ed4d16ad762d7cb8dbd66f8c51ce1ac4972c0f26679d7c36e085954a30b13239"
        and replay.get("source_rag_report_fingerprint")
        == "sha256:829663e216828ad418ddf9a6c38ee487fe44b38d3939072d0ce443e8e8ee5b60"
        and replay.get("rag_control_manifest_fingerprint")
        == "sha256:4ee166171982118552fcc73e38902e653596b652fc645573583a5ef6ca609dfd"
        and policy.get("policy_fingerprint")
        == "sha256:4e59d11cefc5ed9e6cc55a4c36a572e0ed698a8583527bcfbb4eb78b99722449"
    ):
        errors.append("RAG publication-policy replay identity mismatch")
    if summary != {
        "case_count": 2,
        "publish_count": 0,
        "pre_generation_abstention_count": 1,
        "post_generation_rejection_count": 1,
        "unsafe_baseline_outputs_published_count": 0,
    }:
        errors.append("RAG publication-policy action summary mismatch")
    if not (
        answerable.get("case_id") == "answerable-citation"
        and answerable.get("policy_generator_call_count") == 1
        and answerable_decision.get("stage") == "post_generation"
        and answerable_decision.get("action") == "reject"
        and answerable_decision.get("reason_code") == "missing_citation"
        and answerable_decision.get("raw_output_sha256")
        == answerable.get("baseline_raw_output_sha256")
        and answerable_decision.get("semantic_entailment_verified") is False
    ):
        errors.append("RAG publication-policy answerable rejection mismatch")
    if not (
        no_evidence.get("case_id") == "empty-evidence-abstention"
        and no_evidence.get("policy_generator_call_count") == 0
        and no_evidence.get("baseline_raw_output_sha256")
        == "sha256:9f50666396921a1bf45d06c1e52ecd6cef7a158effa45e67771c1b8b1e67f85b"
        and no_evidence_decision.get("stage") == "pre_generation"
        and no_evidence_decision.get("action") == "abstain"
        and no_evidence_decision.get("reason_code") == "no_authorized_evidence"
        and no_evidence_decision.get("raw_output") is None
        and no_evidence_decision.get("semantic_entailment_verified") is False
    ):
        errors.append("RAG publication-policy no-evidence abstention mismatch")
    if not (
        scope.get("counterfactual_policy_replay_on_recorded_attempt") is True
        and scope.get("no_evidence_model_call_would_be_suppressed") is True
        and scope.get("invalid_citation_output_would_be_rejected") is True
        and scope.get("guarded_runtime_model_call_suppression_observed") is False
        and scope.get("claim_evidence_entailment_verified") is False
        and scope.get("artifact_origin_authenticated_by_signature") is False
        and scope.get("general_rag_quality_proven") is False
        and scope.get("production_integration_proven") is False
        and "counterfactual policy replay"
        in RAG_PUBLICATION_REPLAY_EVIDENCE_BOUNDARY
        and "does not prove claim-evidence entailment"
        in RAG_PUBLICATION_REPLAY_EVIDENCE_BOUNDARY
    ):
        errors.append("RAG publication-policy evidence boundary drift")
    required_docs = {
        ROOT / "README.md": ("反事实回放", "不冒充 guard"),
        rag_directory / "README.md": (
            "counterfactual policy replay",
            "call count 为 0",
            "decision.to_public_dict()",
        ),
        ROOT / "docs" / "applications" / "rag-generation.md": (
            "semantic_entailment_verified=false",
            "sha256:ed4d16ad",
            "raw_output_included=false",
        ),
        ROOT / "docs" / "practice" / "labs.md": (
            "policy_generator_call_count=1",
            "counterfactual replay",
        ),
        ROOT / "docs" / "career" / "interview-questions.md": (
            "pre_generation/abstain",
            "counterfactual policy replay",
            "audit/public projection",
        ),
        ROOT / "docs" / "career" / "resume-projects.md": (
            "counterfactual replay",
            "零调用 pre-generation abstain",
            "audit/public allowlist projection",
        ),
        ROOT / "docs" / "practice" / "production-checklist.md": (
            "Publication decision 分离 audit/public projection",
            "直接序列化内部 decision",
        ),
        ROOT / "docs" / "reference" / "accuracy.md": (
            "sha256:ed4d16ad",
            "没有观察到 guard",
            "allowlist `to_public_dict()`",
        ),
    }
    for path, markers in required_docs.items():
        document = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in document]
        if missing:
            errors.append(
                "RAG publication-policy documentation boundary missing in "
                f"{path.name}: {missing}"
            )
    return errors


def check_generation_protocol_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.generation_contract import (
        inspect_generation_protocol_document,
        load_generation_protocol_json,
    )

    protocol_dir = ROOT / "projects" / "transformers-basics" / "protocols"
    aligned = inspect_generation_protocol_document(
        load_generation_protocol_json(
            protocol_dir / "aligned-superset-eos.example.json"
        )
    )
    drift = inspect_generation_protocol_document(
        load_generation_protocol_json(
            protocol_dir / "drift-out-of-range.example.json"
        )
    )
    aligned_eos = aligned.special_tokens[1]
    drift_pad = drift.special_tokens[2]
    errors: list[str] = []
    if not (
        aligned.contract_id == "authored-aligned-superset-eos@v1"
        and aligned.contract_fingerprint
        == "sha256:fc3d4f4477d59687fd5b311badb212efb4fb5bf808d2ecc3ae0b28959aa6f807"
        and aligned_eos.field == "eos_token_id"
        and aligned_eos.tokenizer_ids == (2,)
        and aligned_eos.model_config_ids == (2,)
        and aligned_eos.generation_config_ids == (2, 3)
        and aligned_eos.tokenizer_vs_generation == "left_strict_subset"
        and aligned.observations
        == (
            "generation_config_contains_both_max_length_and_max_new_tokens; "
            "runtime precedence is not inferred",
        )
    ):
        errors.append("authored aligned generation-protocol fixture mismatch")
    if not (
        drift.contract_id == "authored-drift-out-of-range@v1"
        and drift.contract_fingerprint
        == "sha256:9a33ae14d2035794f17a0d0ead561baab647585b409c5e4f4f4e17e3f5422e52"
        and drift_pad.field == "pad_token_id"
        and drift_pad.generation_config_ids == (9,)
        and drift_pad.tokenizer_vs_generation == "disjoint"
        and drift_pad.model_vs_generation == "disjoint"
        and drift_pad.ids_outside_tokenizer_size == (9,)
        and drift_pad.ids_outside_model_vocab == (9,)
        and "pad_token_id:ids_outside_model_vocab=9" in drift.observations
    ):
        errors.append("authored drift generation-protocol fixture mismatch")
    checkpoint_script = (
        ROOT / "projects" / "transformers-basics" / "inspect_checkpoint.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"unavailable_or_load_error"',
        "NORMALIZED_GENERATION_CONFIG_SNAPSHOT_SOURCE",
        '"generation_protocol_contract"',
    )
    missing_scope = [marker for marker in required_scope if marker not in checkpoint_script]
    if missing_scope:
        errors.append(
            "checkpoint generation-protocol inspection missing scope marker(s): "
            f"{missing_scope}"
        )
    return errors


def check_transformers_generation_runtime_control() -> list[str]:
    script = (
        ROOT
        / "projects"
        / "transformers-basics"
        / "generation_runtime_control.py"
    )
    namespace = runpy.run_path(str(script))
    report = namespace["run_control"]()
    cases = report.get("cases", {})
    eos_case = cases.get("generation_config_eos_set", {})
    override_case = cases.get("call_level_eos_override", {})
    length_case = cases.get("call_level_length_cap", {})
    scope = report.get("scope", {})
    assertions = report.get("assertions", {})
    errors: list[str] = []
    if not (
        report.get("parameter_report")
        == {
            "total_parameters": 3824,
            "trainable_parameters": 3824,
            "trainable_fraction": 1.0,
            "parameter_storage_bytes": 15296,
        }
        and eos_case.get("generated_token_ids") == [4, 3]
        and override_case.get("generated_token_ids") == [3, 5]
        and override_case.get("call_overrides")
        == {"eos_token_id": 5, "max_new_tokens": 4}
        and length_case.get("generated_token_ids") == [4, 6]
        and length_case.get("call_overrides") == {"max_new_tokens": 2}
        and all(assertions.values())
    ):
        errors.append("Transformers generation runtime control trace mismatch")
    if not (
        scope.get("real_transformers_generation_mixin_executed") is True
        and scope.get("real_tiny_gpt2_forward_executed") is True
        and scope.get("authored_logits_processor_overrode_all_next_token_scores")
        is True
        and scope.get("random_untrained_model_used") is True
        and scope.get("real_tokenizer_or_chat_template_executed") is False
        and scope.get("public_checkpoint_or_remote_code_loaded") is False
        and scope.get("vllm_or_provider_runtime_executed") is False
        and scope.get("model_quality_latency_throughput_or_gpu_behavior_proved")
        is False
        and scope.get("provider_style_finish_reason_observed") is False
    ):
        errors.append("Transformers generation runtime control scope mismatch")
    return errors


def check_rag_framework_parity_control() -> list[str]:
    script = ROOT / "projects" / "rag-framework-adapters" / "parity_control.py"
    namespace = runpy.run_path(str(script))
    report = namespace["run_control"]()
    cases = report.get("cases", {})
    engineering = cases.get("engineering", {})
    anonymous = cases.get("anonymous", {})
    metrics = report.get("metrics", {})
    assertions = report.get("assertions", {})
    scope = report.get("scope", {})
    errors: list[str] = []
    if not (
        report.get("implementation")
        == "about-llm.rag-framework-parity-control.v1"
        and engineering.get("canonical_document_ids")
        == ["acl-before-ranking", "citation-binding"]
        and engineering.get("langchain_document_ids")
        == engineering.get("canonical_document_ids")
        and engineering.get("llamaindex_document_ids")
        == engineering.get("canonical_document_ids")
        and engineering.get("prompt_sha256")
        == "b9c8cb77ec15536c4ff38fcdcb397596d23b278aee04423aa99d705ee1e8e19c"
        and engineering.get("answer_artifact_fingerprint")
        == "sha256:d1045446a8b984fb4b81653e75fe78569d6c4d82c49cc8447d2cc2ab48180cca"
        and engineering.get("answer_coverage") == 1.0
        and anonymous.get("canonical_document_ids") == ["acl-before-ranking"]
        and anonymous.get("prompt_sha256")
        == "1e33ed1346c4a8f7da9cd26b7a9b1e5e46b5d85ca01fee4d26a1b5f0e396d8fd"
        and anonymous.get("answer_artifact_fingerprint")
        == "sha256:ed8e3f4562df2241b0cb969313505c8b33445fb53426bfa9cc0591b078441e8c"
        and metrics
        == {
            "engineering_recall_at_4": 1.0,
            "engineering_ndcg_at_4": 1.0,
        }
        and assertions
        and all(assertions.values())
    ):
        errors.append("LangChain/LlamaIndex RAG parity control trace mismatch")
    if not (
        scope.get("real_langchain_and_llamaindex_core_executed") is True
        and scope.get("canonical_bm25_authorization_and_ranking_used") is True
        and scope.get("deterministic_extractive_non_llm_answer_used") is True
        and scope.get("learned_embedding_vector_index_or_reranker_executed") is False
        and scope.get("provider_or_local_llm_generation_executed") is False
        and scope.get("framework_default_acl_or_security_proved") is False
        and scope.get("model_quality_latency_scalability_or_production_safety_proved")
        is False
    ):
        errors.append("LangChain/LlamaIndex RAG parity control scope mismatch")
    return errors


def check_rag_service_asgi_control() -> list[str]:
    script = ROOT / "projects" / "rag-foundations" / "rag_service_control.py"
    namespace = runpy.run_path(str(script))
    report = namespace["run_control"]()
    engineering = report.get("engineering", {})
    anonymous = report.get("anonymous", {})
    negative = report.get("negative_cases", {})
    scope = report.get("scope", {})
    errors: list[str] = []
    if not (
        report.get("implementation") == "about-llm.rag-service-asgi-control.v1"
        and report.get("health", {}).get("body")
        == {
            "service_revision": "about-llm.rag-extractive-asgi.v1",
            "status": "ready",
        }
        and engineering
        == {
            "status_code": 200,
            "request_id": "control-request-1",
            "source_ids": ["public-security", "engineering-citations"],
            "action": "answer",
            "artifact_fingerprint": (
                "sha256:cdc57ac0c4f54562b2d3e595046febd78cd635476d067c227ebafc98f73fbe89"
            ),
        }
        and anonymous
        == {
            "status_code": 200,
            "request_id": "control-request-2",
            "source_ids": ["public-security"],
            "action": "answer",
            "artifact_fingerprint": (
                "sha256:5bc0701cb8b5d54705541273a2200327965e1572af631a80396d8a5b1f37d91a"
            ),
        }
        and negative
        == {
            "body_tenant_injection_status": 422,
            "body_tenant_injection_code": "invalid_request",
            "missing_auth_status": 401,
            "missing_auth_code": "unauthorized",
        }
    ):
        errors.append("persistent extractive RAG ASGI control trace mismatch")
    if not (
        scope.get("real_fastapi_starlette_httpx_asgi_dispatch_executed") is True
        and scope.get("real_sqlite_persistence_reopened_per_query") is True
        and scope.get("authorization_context_resolved_outside_json_body") is True
        and scope.get("authorization_filtered_before_bm25_scoring") is True
        and scope.get("deterministic_extractive_non_llm_answer_executed") is True
        and scope.get("real_tcp_tls_reverse_proxy_or_remote_identity_executed") is False
        and scope.get("learned_retriever_reranker_or_llm_executed") is False
        and scope.get("multi_process_global_admission_or_production_slo_proved") is False
    ):
        errors.append("persistent extractive RAG ASGI control scope mismatch")
    service_source = (ROOT / "src" / "about_llm" / "rag" / "service.py").read_text(
        encoding="utf-8"
    )
    server_source = (
        ROOT / "projects" / "rag-foundations" / "serve_extractive.py"
    ).read_text(encoding="utf-8")
    required_service_markers = (
        "asyncio.shield(work)",
        "work.add_done_callback(self._release_after_background_work)",
        'headers["WWW-Authenticate"] = "Bearer"',
        'response.headers["Cache-Control"] = "no-store"',
        "docs_url=None",
    )
    missing_service = [
        marker for marker in required_service_markers if marker not in service_source
    ]
    if missing_service:
        errors.append(f"RAG ASGI service missing boundary marker(s): {missing_service}")
    required_server_markers = (
        "allow-non-loopback-demo-auth",
        "proxy_headers=False",
        "workers=1",
        "os.environ.get(args.token_env)",
    )
    missing_server = [marker for marker in required_server_markers if marker not in server_source]
    if missing_server:
        errors.append(f"RAG demo server missing boundary marker(s): {missing_server}")
    return errors


def check_recorded_model_planner_control() -> list[str]:
    script = ROOT / "projects" / "safe-agent" / "model_planner_control.py"
    namespace = runpy.run_path(str(script))
    report = namespace["run_control"]()
    loop = report.get("loop", {})
    records = report.get("planner_records", [])
    negative = report.get("negative_controls", {})
    unauthorized_loop = negative.get("unauthorized_loop", {})
    unauthorized_events = unauthorized_loop.get("events", [])
    scope = report.get("scope", {})
    expected_requests = [
        "sha256:108e39c169a1c9fcef55aefb48980ed80a225cbd32e34f6b9294366581253896",
        "sha256:8f13990f21a0b193ccfc6fb0fc108a431954b7155bba830b600f2f5af1a6c139",
    ]
    expected_responses = [
        "sha256:af4cf1b51c4b8599803053152ef09bcb7259f8b441e5a31771bbc327eeff9238",
        "sha256:234d407a4c5e3bfe31a878d22658e444fccccc84d9c55b385c107f2e7f898370",
    ]
    expected_decisions = [
        "sha256:9ea9f6d0e7ff6a2709c89ee3de138d37a9e2c20f47e0969b526a152d3ce67c0a",
        "sha256:a13668d18d9f9afd321295a343428d5963453736d61b90912edf6b9ed04ea4bb",
    ]
    errors: list[str] = []
    if not (
        report.get("implementation")
        == "about-llm.recorded-model-planner-control.v1"
        and report.get("mode") == "offline_recorded_provider_responses"
        and report.get("expected_request_fingerprints") == expected_requests
        and report.get("expected_response_fingerprints") == expected_responses
        and report.get("expected_decision_ids") == expected_decisions
        and report.get("tool_contract")
        == {
            "draft": "https://json-schema.org/draft/2020-12/schema",
            "schema_revision": "fixture-tool-arguments@v1",
            "validator_revision": (
                "about-llm.closed-tool-json-schema.v1+jsonschema-4.26.0"
                "+formats-annotation"
            ),
            "schema_fingerprint": (
                "sha256:5542cbcc48890d768f5934ceb008fef72a6f75b92387ec8c03f7d014bd273579"
            ),
            "formats_enforced": False,
        }
        and [record.get("request_fingerprint") for record in records]
        == expected_requests
        and [record.get("response_fingerprint") for record in records]
        == expected_responses
        and [record.get("decision_id") for record in records] == expected_decisions
        and [record.get("action_kind") for record in records] == ["tool", "finish"]
        and loop.get("termination") == "completed"
        and loop.get("final_answer") == "verified answer"
        and loop.get("model_tokens_used") == 62
        and loop.get("cost_units_used") == 0.03
        and loop.get("handler_attempts") == 1
        and report.get("effects") == ["evidence"]
        and [event.get("status") for event in loop.get("events", [])]
        == ["completed", "passed"]
    ):
        errors.append("recorded strict-JSON model planner control trace mismatch")
    if not (
        negative.get("recorded_request_drift_rejected") is True
        and negative.get("markdown_fenced_json_rejected") is True
        and negative.get("runtime_schema_rejected_before_resolver_policy_handler")
        is True
        and negative.get("missing_capability_denied_before_handler") is True
        and unauthorized_loop.get("handler_attempts") == 0
        and len(unauthorized_events) == 1
        and unauthorized_events[0].get("status") == "policy_denied"
    ):
        errors.append("recorded strict-JSON model planner negative control mismatch")
    if scope != {
        "network_or_live_model_called": False,
        "provider_usage_or_cost_independently_verified": False,
        "usage_and_cost_are_authored_fixture_metadata": True,
        "production_iam_or_policy_executed": False,
        "open_task_semantic_verifier_executed": False,
        "fingerprints_prove_authenticity_or_safety": False,
        "tool_observation_is_untrusted_prompt_data": True,
        "standard_jsonschema_runtime_validation_executed": True,
        "planner_and_runtime_schema_derived_from_same_contract": True,
    }:
        errors.append("recorded strict-JSON model planner scope mismatch")

    planner_source = (
        ROOT / "src" / "about_llm" / "agents" / "model_planner.py"
    ).read_text(encoding="utf-8")
    required_source_markers = (
        "parse_constant=reject_constant",
        "object_pairs_hook=pairs",
        "response.output_tokens > request.max_output_tokens",
        '"proposal_fingerprint": event.proposal_fingerprint',
        '"execution_fingerprint": event.execution_fingerprint',
        "response.model_revision != self.model_revision",
        "planner request fingerprint does not match recorded exchange",
    )
    missing_source = [
        marker for marker in required_source_markers if marker not in planner_source
    ]
    if missing_source:
        errors.append(
            f"strict-JSON model planner missing boundary marker(s): {missing_source}"
        )
    schema_source = (
        ROOT / "src" / "about_llm" / "agents" / "schema.py"
    ).read_text(encoding="utf-8")
    required_schema_markers = (
        'DRAFT_2020_12_URI = "https://json-schema.org/draft/2020-12/schema"',
        'schema.get("additionalProperties") is False',
        'schema.get("unevaluatedProperties") is False',
        '("$ref", "$dynamicRef")',
        'if "$id" in node',
        "len(instance_bytes) > self.max_instance_bytes",
        "validator_type.check_schema(schema)",
        "unknown_formats = sorted",
        "instance_pointer=_json_pointer",
        "return PlannerToolContract(",
        "validate=self.validate",
    )
    missing_schema = [
        marker for marker in required_schema_markers if marker not in schema_source
    ]
    if missing_schema:
        errors.append(
            f"executable tool schema missing boundary marker(s): {missing_schema}"
        )
    project_docs = (ROOT / "projects" / "safe-agent" / "README.md").read_text(
        encoding="utf-8"
    )
    required_doc_markers = (
        "Strict JSON model planner boundary",
        "recorded provider response",
        "authored metadata",
        "不证明真实 API schema、目标模型能遵循协议",
        "无密钥 SHA-256",
        "不做字符串转数字等 coercion、不插入 `default`",
        "`$ref/$dynamicRef` 只允许 local fragment",
        "同一份 immutable schema",
    )
    missing_docs = [
        marker for marker in required_doc_markers if marker not in project_docs
    ]
    if missing_docs:
        errors.append(
            f"strict-JSON model planner docs missing boundary marker(s): {missing_docs}"
        )
    return errors


def check_mcp_stdio_control() -> list[str]:
    from about_llm.agents.mcp_stdio import run_stdio_control
    from about_llm.llmops import canonical_json_bytes

    report = run_stdio_control(cwd=ROOT)
    errors: list[str] = []
    calls = report.get("calls", {})
    transcript = report.get("transcript", {})
    if not (
        report.get("implementation") == "about-llm.mcp-stdio-control.v1"
        and report.get("protocol_version") == "2025-11-25"
        and report.get("transport")
        == {
            "client_launched_server_subprocess": True,
            "utf8_jsonrpc_lf_framing_executed": True,
            "server_stdout_protocol_only": True,
            "server_stderr": "",
        }
        and report.get("handshake", {}).get("negotiated_protocol_version")
        == "2025-11-25"
        and report.get("handshake", {}).get("tools_capability_negotiated") is True
        and calls.get("valid", {}).get("structuredContent") == {"sum": 12}
        and calls.get("valid", {}).get("isError") is False
        and calls.get("invalid_arguments", {}).get("rejected_value_disclosed") is False
        and calls.get("invalid_arguments", {}).get("result", {}).get("isError") is True
        and calls.get("unknown_tool_protocol_error", {}).get("code") == -32602
        and transcript
        == {
            "message_count": 11,
            "request_ids": [1, 2, 3, 4, 5],
            "client_methods": [
                "initialize",
                "notifications/initialized",
                "tools/list",
                "tools/call",
                "tools/call",
                "tools/call",
            ],
            "projection_fingerprint": (
                "sha256:5be5bed393fd66ef3269fe10452164423c29adfaa5cb59e2a5086b8eb7f64256"
            ),
            "projection_fields": [
                "direction",
                "jsonrpc",
                "id",
                "method",
                "response_kind",
                "tool_is_error",
                "error_code",
            ],
            "raw_messages_published": False,
        }
    ):
        errors.append("MCP stdio lifecycle/tool control trace mismatch")
    if report.get("scope") != {
        "real_local_subprocess_and_os_pipes_executed": True,
        "external_network_or_remote_server_called": False,
        "official_mcp_sdk_used": False,
        "full_mcp_schema_or_conformance_suite_executed": False,
        "streamable_http_or_authentication_executed": False,
        "a2a_client_or_server_executed": False,
        "cross_vendor_interoperability_proven": False,
        "business_authorization_or_human_approval_executed": False,
        "tool_is_bounded_local_read_only_fixture": True,
        "transcript_projection_fingerprint_proves_authenticity": False,
    }:
        errors.append("MCP stdio control scope mismatch")
    if b"TOP-SECRET-MCP-VALUE" in canonical_json_bytes(report):
        errors.append("MCP stdio public report disclosed rejected fixture value")

    source = (SRC / "about_llm" / "agents" / "mcp_stdio.py").read_text(
        encoding="utf-8"
    )
    required_source_markers = (
        'MCP_PROTOCOL_VERSION = "2025-11-25"',
        "line.endswith(b\"\\n\")",
        'payload.decode("utf-8", errors="strict")',
        "object_pairs_hook=_unique_object",
        "parse_constant=_reject_nonfinite",
        'method == "notifications/initialized"',
        'method == "tools/list"',
        'method == "tools/call"',
        "process: subprocess.Popen[bytes] = subprocess.Popen(",
        '"projection_fields": [',
        '"raw_messages_published": False',
    )
    missing_source = [
        marker for marker in required_source_markers if marker not in source
    ]
    if missing_source:
        errors.append(f"MCP stdio source missing boundary marker(s): {missing_source}")
    return errors


def check_mcp_sdk_memory_control() -> list[str]:
    from about_llm.agents.mcp_sdk_memory import (
        MCP_SDK_MEMORY_CONTROL_VERSION,
        MCP_SDK_MEMORY_EVIDENCE_BOUNDARY,
        MCP_SDK_PROTOCOL_VERSION,
        MCP_SDK_REVIEWED_VERSION,
        run_mcp_sdk_memory_control,
    )

    report = run_mcp_sdk_memory_control()
    errors: list[str] = []
    if not (
        report.get("control_version") == MCP_SDK_MEMORY_CONTROL_VERSION
        and report.get("checked_at") == "2026-08-14"
        and report.get("runtime")
        == {
            "sdk_distribution": "mcp",
            "sdk_version": MCP_SDK_REVIEWED_VERSION,
            "latest_protocol": MCP_SDK_PROTOCOL_VERSION,
            "supported_protocols": [
                "2024-11-05",
                "2025-03-26",
                "2025-06-18",
                "2025-11-25",
            ],
        }
        and report.get("transport")
        == {
            "kind": "official_sdk_anyio_memory_object_stream",
            "official_sdk_memory_stream": True,
            "os_stdio": False,
            "tcp_http": False,
            "subprocess": False,
        }
        and report.get("initialization")
        == {
            "protocol_version": MCP_SDK_PROTOCOL_VERSION,
            "server_name": "about-llm-mcp-sdk-memory",
            "server_version": "1.0.0",
            "tools_capability": True,
        }
        and report.get("discovery", {}).get("tool_count") == 1
        and report.get("discovery", {}).get("tool_name") == "fixture.add"
        and report.get("discovery", {}).get("closed_input_schema") is True
        and report.get("discovery", {}).get("closed_output_schema") is True
        and report.get("calls")
        == {
            "successful_sum": 5,
            "success_is_error": False,
            "invalid_schema_is_error": True,
            "invalid_schema_handler_delta": 0,
            "unknown_tool_is_error": True,
            "unknown_tool_handler_delta": 1,
            "recognized_handler_calls": 1,
            "total_handler_calls": 2,
            "raw_error_content_published": False,
        }
        and report.get("scope")
        == {
            "official_sdk_client_executed": True,
            "official_sdk_server_executed": True,
            "mcp_2025_11_25_negotiated": True,
            "official_generated_types_executed": True,
            "sdk_json_schema_validation_executed": True,
            "application_unknown_tool_gate_executed": True,
            "stdio_transport_executed": False,
            "streamable_http_transport_executed": False,
            "remote_or_cross_vendor_interop_proven": False,
            "official_conformance_suite_executed": False,
            "authentication_or_authorization_proven": False,
            "production_readiness_proven": False,
        }
        and report.get("evidence_boundary") == MCP_SDK_MEMORY_EVIDENCE_BOUNDARY
    ):
        errors.append("official MCP SDK memory control semantic boundary mismatch")

    source = (SRC / "about_llm" / "agents" / "mcp_sdk_memory.py").read_text(
        encoding="utf-8"
    )
    required_source_markers = (
        'MCP_SDK_REVIEWED_VERSION: Final = "1.29.0"',
        'MCP_SDK_PROTOCOL_VERSION: Final = "2025-11-25"',
        "create_client_server_memory_streams",
        "ClientSession(",
        "Server(",
        "@server.call_tool(validate_input=True)",
        'if after_invalid != after_success:',
        'if after_unknown != {"total": 2, "recognized": 1}:',
        '"raw_error_content_published": False',
        '"authentication_or_authorization_proven": False',
    )
    missing_source = [
        marker for marker in required_source_markers if marker not in source
    ]
    if missing_source:
        errors.append(
            f"official MCP SDK memory source missing boundary marker(s): {missing_source}"
        )

    project = (ROOT / "projects" / "safe-agent" / "README.md").read_text(
        encoding="utf-8"
    )
    required_doc_markers = (
        "MCP 2025-11-25 official-SDK memory control",
        "handler delta 为 0",
        "handler delta 为 1",
        "不替代应用的工具名、资源和授权 gate",
        "没有启动 subprocess",
        "下一节会把相同 SDK fixture 接到真实 stdio",
        "memory control 仍只证明 in-process 路径",
    )
    missing_docs = [marker for marker in required_doc_markers if marker not in project]
    if missing_docs:
        errors.append(
            f"official MCP SDK memory docs missing boundary marker(s): {missing_docs}"
        )
    return errors


def check_mcp_sdk_stdio_control() -> list[str]:
    from about_llm.agents.mcp_sdk_stdio import (
        MCP_SDK_STDIO_CONTROL_VERSION,
        MCP_SDK_STDIO_EVIDENCE_BOUNDARY,
        run_mcp_sdk_stdio_control,
    )
    from about_llm.llmops import canonical_json_bytes

    report = run_mcp_sdk_stdio_control()
    errors: list[str] = []
    if not (
        report.get("control_version") == MCP_SDK_STDIO_CONTROL_VERSION
        and report.get("checked_at") == "2026-08-14"
        and report.get("runtime", {}).get("sdk_distribution") == "mcp"
        and report.get("runtime", {}).get("sdk_version") == "1.29.0"
        and report.get("runtime", {}).get("latest_protocol") == "2025-11-25"
        and report.get("transport")
        == {
            "kind": "official_sdk_stdio_subprocess",
            "client_transport": "mcp.client.stdio.stdio_client",
            "server_transport": "mcp.server.stdio.stdio_server",
            "client_launched_server_subprocess": True,
            "os_stdin_stdout_pipes": True,
            "encoding_profile": "client=utf-8-strict;server-stdin=utf-8-replace",
            "server_process_distinct": True,
            "graceful_eof_shutdown_observed": True,
            "server_stderr_empty": True,
            "raw_transcript_published": False,
        }
        and report.get("initialization")
        == {
            "protocol_version": "2025-11-25",
            "server_name": "about-llm-mcp-sdk-stdio",
            "server_version": "1.0.0",
            "tools_capability": True,
        }
        and report.get("discovery", {}).get("tool_count") == 1
        and report.get("discovery", {}).get("tool_name") == "fixture.add"
        and report.get("discovery", {}).get("closed_input_schema") is True
        and report.get("discovery", {}).get("closed_output_schema") is True
        and report.get("calls")
        == {
            "successful_sum": 5,
            "success_is_error": False,
            "invalid_schema_is_error": True,
            "invalid_schema_handler_delta": 0,
            "unknown_tool_is_error": True,
            "unknown_tool_handler_delta": 1,
            "raw_error_content_published": False,
        }
        and report.get("server_receipt", {}).get("handler_events")
        == ["fixture.add", "fixture.missing"]
        and report.get("server_receipt", {}).get("recognized_handler_calls") == 1
        and report.get("server_receipt", {}).get("total_handler_calls") == 2
        and report.get("server_receipt", {}).get("server_run_completed") is True
        and report.get("server_receipt", {}).get(
            "raw_arguments_or_results_published"
        )
        is False
        and report.get("scope")
        == {
            "official_sdk_client_executed": True,
            "official_sdk_server_executed": True,
            "official_sdk_stdio_client_executed": True,
            "official_sdk_stdio_server_executed": True,
            "real_subprocess_and_os_pipes_executed": True,
            "mcp_2025_11_25_negotiated": True,
            "official_generated_types_executed": True,
            "sdk_json_schema_validation_executed": True,
            "application_unknown_tool_gate_executed": True,
            "malformed_raw_framing_controls_executed": False,
            "streamable_http_transport_executed": False,
            "remote_or_cross_vendor_interop_proven": False,
            "official_conformance_suite_executed": False,
            "authentication_or_authorization_proven": False,
            "production_readiness_proven": False,
        }
        and report.get("evidence_boundary") == MCP_SDK_STDIO_EVIDENCE_BOUNDARY
    ):
        errors.append("official MCP SDK real-stdio semantic boundary mismatch")

    serialized = canonical_json_bytes(report)
    if b"server_pid" in serialized or b"receipt_path" in serialized:
        errors.append("official MCP SDK stdio public report disclosed local identity")

    source = (SRC / "about_llm" / "agents" / "mcp_sdk_stdio.py").read_text(
        encoding="utf-8"
    )
    required_source_markers = (
        "StdioServerParameters(",
        "stdio_client(",
        "stdio_server()",
        "TemporaryDirectory(",
        'with path.open("xb") as handle:',
        "os.fsync(handle.fileno())",
        '"handler_events": [ADD_INPUT_CONTRACT.name, "fixture.missing"]',
        '"malformed_raw_framing_controls_executed": False',
        '"authentication_or_authorization_proven": False',
        '"raw_transcript_published": False',
    )
    missing_source = [
        marker for marker in required_source_markers if marker not in source
    ]
    if missing_source:
        errors.append(
            f"official MCP SDK stdio source missing boundary marker(s): {missing_source}"
        )

    project = (ROOT / "projects" / "safe-agent" / "README.md").read_text(
        encoding="utf-8"
    )
    required_doc_markers = (
        "MCP 2025-11-25 official-SDK stdio control",
        "真实 OS stdin/stdout pipe",
        "handler 序列精确为 `fixture.add, fixture.missing`",
        "公开报告不含 PID、receipt path、raw transcript、raw 参数/result content",
        "没有独立注入 missing LF",
        "不能把 SDK 源码中存在的分支写成已测",
        "不认证进程、来源或真实执行",
    )
    missing_docs = [marker for marker in required_doc_markers if marker not in project]
    if missing_docs:
        errors.append(
            f"official MCP SDK stdio docs missing boundary marker(s): {missing_docs}"
        )
    return errors


def check_mcp_sdk_streamable_http_control() -> list[str]:
    from about_llm.agents.mcp_sdk_streamable_http import (
        MCP_SDK_HTTP_CONTROL_VERSION,
        MCP_SDK_HTTP_EVIDENCE_BOUNDARY,
        run_mcp_sdk_http_control,
    )
    from about_llm.llmops import canonical_json_bytes

    report = run_mcp_sdk_http_control()
    errors: list[str] = []
    if not (
        report.get("control_version") == MCP_SDK_HTTP_CONTROL_VERSION
        and report.get("checked_at") == "2026-08-14"
        and report.get("runtime", {}).get("sdk_distribution") == "mcp"
        and report.get("runtime", {}).get("sdk_version") == "1.29.0"
        and report.get("runtime", {}).get("latest_protocol") == "2025-11-25"
        and report.get("transport")
        == {
            "kind": "official_sdk_streamable_http_subprocess",
            "client_transport": (
                "mcp.client.streamable_http.streamable_http_client"
            ),
            "server_transport": (
                "mcp.server.streamable_http_manager.StreamableHTTPSessionManager"
            ),
            "control_launched_server_subprocess": True,
            "real_ipv4_loopback_tcp_http": True,
            "stateful_session": True,
            "post_response_mode": "sse",
            "server_process_distinct": True,
            "client_session_id_observed": True,
            "mcp_session_termination_delete_observed": True,
            "server_shutdown_via_separate_control_endpoint": True,
            "private_control_unauthorized_status": 401,
            "server_process_graceful_shutdown_observed": True,
            "server_stderr_empty": True,
            "raw_http_or_protocol_payload_published": False,
        }
        and report.get("initialization")
        == {
            "protocol_version": "2025-11-25",
            "server_name": "about-llm-mcp-sdk-streamable-http",
            "server_version": "1.0.0",
            "tools_capability": True,
        }
        and report.get("discovery", {}).get("tool_count") == 1
        and report.get("discovery", {}).get("tool_name") == "fixture.add"
        and report.get("discovery", {}).get("closed_input_schema") is True
        and report.get("discovery", {}).get("closed_output_schema") is True
        and report.get("calls")
        == {
            "successful_sum": 5,
            "success_is_error": False,
            "invalid_schema_is_error": True,
            "invalid_schema_handler_delta": 0,
            "unknown_tool_is_error": True,
            "unknown_tool_handler_delta": 1,
            "raw_error_content_published": False,
        }
        and report.get("http_observations")
        == {
            "mcp_response_count": 9,
            "post_count": 7,
            "get_count": 1,
            "delete_count": 1,
            "status_200_count": 8,
            "status_202_count": 1,
            "sse_response_count": 7,
            "json_response_count": 2,
            "unexpected_method_status_or_media_type_count": 0,
            "raw_headers_bodies_or_session_id_published": False,
        }
        and report.get("server_receipt", {}).get("handler_events")
        == ["fixture.add", "fixture.missing"]
        and report.get("server_receipt", {}).get("recognized_handler_calls") == 1
        and report.get("server_receipt", {}).get("total_handler_calls") == 2
        and report.get("server_receipt", {}).get(
            "session_manager_run_completed"
        )
        is True
        and report.get("server_receipt", {}).get("shutdown_control_received")
        is True
        and report.get("server_receipt", {}).get(
            "raw_arguments_or_results_published"
        )
        is False
        and report.get("scope")
        == {
            "official_sdk_client_executed": True,
            "official_sdk_server_executed": True,
            "official_sdk_streamable_http_client_executed": True,
            "official_sdk_streamable_http_session_manager_executed": True,
            "real_loopback_tcp_http_executed": True,
            "stateful_session_and_delete_executed": True,
            "post_sse_responses_executed": True,
            "get_sse_stream_opened": True,
            "mcp_2025_11_25_negotiated": True,
            "official_generated_types_executed": True,
            "sdk_json_schema_validation_executed": True,
            "application_unknown_tool_gate_executed": True,
            "private_control_token_gate_executed": True,
            "malformed_http_controls_executed": False,
            "session_resumption_executed": False,
            "tls_or_oauth_executed": False,
            "remote_or_cross_vendor_interop_proven": False,
            "official_conformance_suite_executed": False,
            "authentication_or_authorization_proven": False,
            "production_readiness_proven": False,
        }
        and report.get("evidence_boundary") == MCP_SDK_HTTP_EVIDENCE_BOUNDARY
    ):
        errors.append("official MCP SDK Streamable HTTP semantic boundary mismatch")

    serialized = canonical_json_bytes(report)
    forbidden = (
        b'"server_pid"',
        b'"receipt_path"',
        b"MCP-Session-Id",
        b"ABOUT_LLM_MCP_SDK_HTTP_CONTROL_TOKEN",
    )
    if any(marker in serialized for marker in forbidden):
        errors.append("official MCP SDK HTTP public report disclosed local identity")

    source = (
        SRC / "about_llm" / "agents" / "mcp_sdk_streamable_http.py"
    ).read_text(encoding="utf-8")
    required_source_markers = (
        "streamable_http_client(",
        "StreamableHTTPSessionManager(",
        "StreamableHTTPASGIApp(manager)",
        'host != "127.0.0.1"',
        '"post_count": 7',
        '"get_count": 1',
        '"delete_count": 1',
        '"session_resumption_executed": False',
        '"authentication_or_authorization_proven": False',
        '"raw_http_or_protocol_payload_published": False',
    )
    missing_source = [
        marker for marker in required_source_markers if marker not in source
    ]
    if missing_source:
        errors.append(
            "official MCP SDK HTTP source missing boundary marker(s): "
            f"{missing_source}"
        )

    project = (ROOT / "projects" / "safe-agent" / "README.md").read_text(
        encoding="utf-8"
    )
    required_doc_markers = (
        "MCP 2025-11-25 official-SDK Streamable HTTP control",
        "7 次 POST、1 次 GET 与 1 次 DELETE",
        "独立 server subprocess",
        "私有 control endpoint",
        "缺失 token 的真实负例为 401",
        "不是 MCP auth",
        "不发布 PID、session id、token、header、raw HTTP/protocol payload",
        "没有执行 MCP endpoint 的 malformed body、Host/Origin failure、resumption、TLS 或 OAuth",
    )
    missing_docs = [marker for marker in required_doc_markers if marker not in project]
    if missing_docs:
        errors.append(
            f"official MCP SDK HTTP docs missing boundary marker(s): {missing_docs}"
        )
    return errors


def check_mcp_streamable_http_control() -> list[str]:
    from about_llm.agents.mcp_streamable_http import run_streamable_http_control
    from about_llm.llmops import canonical_json_bytes

    report = run_streamable_http_control(cwd=ROOT)
    errors: list[str] = []
    if not (
        report.get("implementation")
        == "about-llm.mcp-streamable-http-control.v1"
        and report.get("protocol_version") == "2025-11-25"
        and report.get("binding") == "Streamable HTTP"
        and report.get("network")
        == {
            "scheme": "http",
            "address_scope": "IPv4 loopback",
            "real_tcp_http": True,
            "tls": False,
        }
        and report.get("transport")
        == {
            "single_endpoint_path": "/mcp",
            "post_json_response_executed": True,
            "post_sse_response_executed": True,
            "get_sse_executed": True,
            "delete_session_executed": True,
            "notification_empty_202_verified": True,
            "sse_priming_event_with_id_verified": True,
            "event_ids_unique_within_session": True,
        }
        and report.get("security_controls")
        == {
            "origin_allowlist_executed": True,
            "invalid_origin_status": 403,
            "bearer_header_gate_executed": True,
            "missing_or_wrong_bearer_status": 401,
            "oauth_flow_executed": False,
        }
        and report.get("session")
        == {
            "server_assigned_on_initialize": True,
            "visible_ascii_and_minimum_length_verified": True,
            "included_on_subsequent_requests": True,
            "missing_session_status": 400,
            "missing_or_unsupported_protocol_version_status": 400,
            "terminated_session_status": 404,
        }
        and report.get("cancellation")
        == {
            "concurrent_request_and_notification_executed": True,
            "notification_status": 202,
            "jsonrpc_response_after_cancellation_count": 0,
            "stream_closed_after_cancellation": True,
        }
        and report.get("tool_result")
        == {
            "structured_output_local_verifier_passed": True,
            "raw_arguments_or_result_published": False,
        }
        and report.get("projection_fingerprint")
        == "sha256:5a5cc3be24268d3dec80edb3613e51ffed3dc0d0d6535f7039c74386ce7c8915"
        and report.get("raw_http_messages_published") is False
        and report.get("secret_or_session_identifiers_published") is False
        and report.get("server_process", {}).get("stdout_stderr_empty") is True
    ):
        errors.append("MCP Streamable HTTP lifecycle/transport control mismatch")
    if not all(value is False for value in report.get("evidence_limits", {}).values()):
        errors.append("MCP Streamable HTTP evidence limits mismatch")
    serialized = canonical_json_bytes(report)
    forbidden = (
        b"REJECTED-MCP-HTTP-TOKEN",
        b"about-llm-mcp-http-client",
        b'"sum":12',
        b'"a":7',
    )
    if any(marker in serialized for marker in forbidden):
        errors.append("MCP Streamable HTTP public report disclosed private content")

    source = (
        SRC / "about_llm" / "agents" / "mcp_streamable_http.py"
    ).read_text(encoding="utf-8")
    required_source_markers = (
        'MCP_STREAMABLE_HTTP_CONTROL_VERSION: Final = (',
        'MCP_ENDPOINT_PATH: Final = "/mcp"',
        'host != "127.0.0.1"',
        'message.get("method") == "notifications/cancelled"',
        '"MCP-Protocol-Version": MCP_PROTOCOL_VERSION',
        '"MCP-Session-Id": session_id',
        'media_type="text/event-stream"',
        'raise MCPHTTPControlError(403)',
        '"WWW-Authenticate": \'Bearer realm="about-llm-mcp-control"\'',
        '"raw_http_messages_published": False',
        '"oauth_or_protected_resource_metadata_proven": False',
    )
    missing_source = [
        marker for marker in required_source_markers if marker not in source
    ]
    if missing_source:
        errors.append(
            f"MCP Streamable HTTP source missing boundary marker(s): {missing_source}"
        )
    return errors


def check_a2a_loopback_control() -> list[str]:
    from about_llm.agents.a2a_loopback import run_loopback_control

    report = run_loopback_control()
    errors: list[str] = []
    if not (
        report.get("implementation") == "about-llm.a2a-loopback-control.v1"
        and report.get("protocol_version") == "1.0"
        and report.get("binding") == "JSONRPC"
        and report.get("official_sdk")
        == {
            "distribution": "a2a-sdk",
            "runtime_version": "1.1.2",
            "reviewed_version": "1.1.2",
            "client_used": True,
            "server_used": True,
            "generated_proto_models_validated": True,
        }
        and report.get("network")
        == {
            "scheme": "http",
            "address_scope": "IPv4 loopback",
            "real_tcp_http": True,
            "tls": False,
        }
        and report.get("agent_card", {}).get("well_known_resolved") is True
        and report.get("agent_card", {}).get("interface_binding") == "JSONRPC"
        and report.get("agent_card", {}).get("interface_protocol_version") == "1.0"
        and [item.get("method") for item in report.get("operations", [])]
        == ["SendMessage", "GetTask"]
        and report.get("task", {}).get("remote_state") == "TASK_STATE_COMPLETED"
        and report.get("task", {}).get("local_verifier_passed") is True
        and report.get("task", {}).get("remote_completed_treated_as_sufficient") is False
        and report.get("negative_controls", {}).get("legacy_kind_error_code") == -32602
        and report.get("negative_controls", {}).get("unsupported_version_error_code")
        == -32009
        and report.get("official_schema", {}).get("validated") is False
        and report.get("server_process", {}).get("stdout_stderr_empty") is True
        and report.get("projection_fingerprint")
        == "sha256:f1ad7ae1c0e18c91caa710d6448f6a503eaf8c8cbf0c0e689166d3f1af4b099e"
        and report.get("raw_messages_published") is False
    ):
        errors.append("A2A 1.0 official-SDK loopback control trace mismatch")
    if not all(value is False for value in report.get("evidence_limits", {}).values()):
        errors.append("A2A loopback evidence limits mismatch")

    source = (SRC / "about_llm" / "agents" / "a2a_loopback.py").read_text(
        encoding="utf-8"
    )
    required_source_markers = (
        'A2A_PROTOCOL_VERSION: Final = "1.0"',
        'A2A_SCHEMA_URL: Final = "https://a2a-protocol.org/v1.0.0/spec/a2a.json"',
        '"sha256:6b6560c726289734799b7d5883be84e4cc0452600736db0f811341bac43b8d62"',
        "A2ACardResolver(http_client, base_url)",
        '"method": "SendMessage"',
        "client.get_task(",
        'legacy_parts[0]["kind"] = "data"',
        'headers={"A2A-Version": "9.9"}',
        '"remote_completed_treated_as_sufficient": False',
        '"complete_a2a_conformance_proven": False',
        'report["raw_messages_published"] = False',
    )
    missing_source = [marker for marker in required_source_markers if marker not in source]
    if missing_source:
        errors.append(f"A2A loopback source missing boundary marker(s): {missing_source}")
    return errors


def check_moe_routing_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import numpy as np

    from about_llm.from_scratch import (
        route_topk_capacity,
        routed_linear_expert_forward,
    )
    from about_llm.from_scratch.moe_training import (
        TRAINABLE_MOE_CONTROL_VERSION,
        run_trainable_moe_control,
    )

    logits = np.array(
        [
            [4.0, 1.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 3.0, 0.0],
            [0.0, 4.0, 3.0],
        ]
    )
    routing = route_topk_capacity(logits, top_k=2, capacity_factor=0.75)
    hidden = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, -1.0], [3.0, 1.0]])
    expert_weights = np.stack(
        [np.eye(2), 2 * np.eye(2), np.array([[1.0, 1.0], [-1.0, 1.0]])]
    )
    output = routed_linear_expert_forward(hidden, expert_weights, routing)
    errors: list[str] = []
    if not (
        routing.expert_capacity == 2
        and routing.expert_counts_before_capacity == (3, 4, 1)
        and routing.expert_counts_after_capacity == (2, 2, 1)
        and routing.assignments_before_capacity == 8
        and routing.kept_assignments == 5
        and routing.dropped_assignments == 3
        and routing.tokens_with_all_assignments_dropped == 0
        and routing.selected_expert_indices.tolist()
        == [[0, 1], [0, 1], [0, 1], [1, 2]]
        and routing.kept_mask.tolist()
        == [[True, False], [True, False], [False, True], [True, True]]
        and np.allclose(routing.combine_weights.sum(axis=1), 1)
        and np.allclose(output[0], [1, 2])
        and np.allclose(output[2], [2, -2])
    ):
        errors.append("MoE top-k/capacity/drop/combine fixture mismatch")
    uniform = route_topk_capacity(np.zeros((2, 2)), top_k=2)
    if not (
        math.isclose(uniform.load_balance_diagnostic, 1)
        and math.isclose(uniform.router_z_loss, math.log(2) ** 2)
        and math.isclose(uniform.mean_router_entropy, math.log(2))
    ):
        errors.append("MoE router balance/z-loss/entropy fixture mismatch")
    demo = (
        ROOT / "projects" / "transformers-basics" / "moe_routing.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"actual_topk_routing_and_capacity_drop_executed": True',
        '"actual_linear_expert_dispatch_and_combine_executed": True',
        '"trained_expert_mlp_or_router_used": False',
        '"expert_parallel_all_to_all_or_gpu_kernel_executed": False',
        '"deepseek_qwen_or_other_checkpoint_reproduced": False',
        '"quality_throughput_or_memory_proved": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(f"MoE routing toy missing scope marker(s): {missing_scope}")

    training_report = run_trainable_moe_control()
    training_fixture = training_report.get("fixture", {})
    training_oracle = training_report.get("sparse_dense_oracle", {})
    training_capacity = training_report.get("capacity_training_control", {})
    capacity_all_drop = training_capacity.get("all_drop_fixture", {})
    training_grouped = training_report.get("routing_group_padding_control", {})
    training_overflow = training_report.get("overflow_policy_control", {})
    training_step = training_report.get("optimizer_step", {})
    detached = training_report.get("detached_gate_negative_control", {})
    balance = training_report.get("balance_gradient_control", {})
    assertions = training_report.get("assertions", {})
    scope = training_report.get("scope", {})
    expected_selected = [[0, 2], [1, 0], [2, 1], [2, 0], [0, 1]]
    if not (
        training_report.get("schema_version") == TRAINABLE_MOE_CONTROL_VERSION
        and training_fixture.get("assignment_counts") == [4, 3, 3]
        and training_fixture.get("selected_expert_indices") == expected_selected
        and training_oracle.get("output_max_abs_difference") == 0.0
        and math.isclose(
            training_oracle.get("all_parameter_gradient_max_abs_difference", -1),
            6.938893903907228e-18,
            rel_tol=1e-12,
            abs_tol=1e-30,
        )
        and math.isclose(
            training_step.get("task_loss_before", -1),
            0.08864729306070791,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and math.isclose(
            training_step.get("task_loss_after", -1),
            0.08755795603512319,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and training_step.get("router_parameter_max_abs_delta", 0) > 0
        and all(
            value > 0
            for value in training_step.get("expert_parameter_max_abs_deltas", [])
        )
        and detached.get("router_task_gradient_is_missing") is True
        and all(value > 0 for value in detached.get("expert_task_gradient_norms", []))
        and balance.get("selected_expert_indices_before") == [[0], [0], [0], [0]]
        and balance.get("selected_expert_indices_after") == [[0], [0], [0], [0]]
        and math.isclose(
            balance.get("load_balance_loss_before", -1),
            2.5677239505126708,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and math.isclose(
            balance.get("load_balance_loss_after", -1),
            2.552750704884368,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and isinstance(assertions, dict)
        and assertions
        and all(assertions.values())
    ):
        errors.append("trainable MoE forward/backward control mismatch")

    expected_preserved_mass = (
        0.8053384164084222,
        0.6681877721681662,
        1.0,
        0.638763175148842,
        0.5498339973124778,
    )
    preserved_mass = training_capacity.get("preserved_mass_weight_sums", [])
    if not (
        training_capacity.get("capacity_factor") == 0.5
        and training_capacity.get("expert_capacity") == 2
        and training_capacity.get("expert_counts_before_capacity") == [4, 3, 3]
        and training_capacity.get("expert_counts_after_capacity") == [2, 2, 2]
        and training_capacity.get("kept_mask")
        == [
            [True, False],
            [True, False],
            [True, True],
            [True, False],
            [True, False],
        ]
        and training_capacity.get("dropped_assignments") == 4
        and training_capacity.get("tokens_with_all_assignments_dropped") == 0
        and training_capacity.get("sparse_dense_output_max_abs_difference") == 0.0
        and training_capacity.get(
            "sparse_dense_all_parameter_gradient_max_abs_difference"
        )
        == 0.0
        and training_capacity.get("renormalized_weight_sums") == [1.0] * 5
        and isinstance(preserved_mass, list)
        and len(preserved_mass) == len(expected_preserved_mass)
        and all(
            math.isclose(actual, expected, rel_tol=0, abs_tol=1e-15)
            for actual, expected in zip(
                preserved_mass,
                expected_preserved_mass,
                strict=True,
            )
        )
        and math.isclose(
            training_capacity.get(
                "renormalize_vs_preserve_output_max_abs_difference",
                -1,
            ),
            0.1255417263895207,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and capacity_all_drop.get("expert_capacity") == 1
        and capacity_all_drop.get("selected_expert_indices")
        == [[0, 1], [0, 1], [0, 1]]
        and capacity_all_drop.get("kept_mask")
        == [[True, True], [False, False], [False, False]]
        and capacity_all_drop.get("expert_counts_before_capacity") == [3, 3, 0]
        and capacity_all_drop.get("expert_counts_after_capacity") == [1, 1, 0]
        and capacity_all_drop.get("tokens_with_all_assignments_dropped") == 2
        and capacity_all_drop.get("routed_output", [])[1:]
        == [[0.0, 0.0], [0.0, 0.0]]
    ):
        errors.append("trainable MoE capacity/drop control mismatch")

    grouped_balance_losses = training_grouped.get("load_balance_loss_by_group", [])
    grouped_z_losses = training_grouped.get("router_z_loss_by_group", [])
    if not (
        training_grouped.get("token_mask") == [True, True, True, True, False]
        and training_grouped.get("routing_group_ids") == [10, 10, 20, 20, 999]
        and training_grouped.get("routing_group_labels") == [10, 20]
        and training_grouped.get("active_tokens_per_group") == [2, 2]
        and training_grouped.get("capacity_factor") == 0.5
        and training_grouped.get("expert_capacity") is None
        and training_grouped.get("expert_capacities_by_group") == [1, 1]
        and training_grouped.get("expert_counts_before_capacity") == [3, 2, 3]
        and training_grouped.get("expert_counts_after_capacity") == [2, 2, 2]
        and training_grouped.get("expert_counts_before_capacity_by_group")
        == [[2, 1, 1], [1, 1, 2]]
        and training_grouped.get("expert_counts_after_capacity_by_group")
        == [[1, 1, 1], [1, 1, 1]]
        and training_grouped.get("kept_mask")
        == [
            [True, True],
            [True, False],
            [False, True],
            [True, True],
            [False, False],
        ]
        and training_grouped.get("dropped_assignments") == 2
        and training_grouped.get("tokens_with_all_assignments_dropped") == 0
        and training_grouped.get("selection_fractions_by_group")
        == [[0.5, 0.25, 0.25], [0.25, 0.25, 0.5]]
        and isinstance(grouped_balance_losses, list)
        and len(grouped_balance_losses) == 2
        and all(
            math.isclose(actual, expected, rel_tol=0, abs_tol=1e-15)
            for actual, expected in zip(
                grouped_balance_losses,
                [1.108214748077347, 1.1401893426112772],
                strict=True,
            )
        )
        and isinstance(grouped_z_losses, list)
        and len(grouped_z_losses) == 2
        and all(
            math.isclose(actual, expected, rel_tol=0, abs_tol=1e-15)
            for actual, expected in zip(
                grouped_z_losses,
                [2.368472188216285, 2.345999773471999],
                strict=True,
            )
        )
        and math.isclose(
            training_grouped.get("active_token_weighted_load_balance_loss", -1),
            1.1242020453443122,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and math.isclose(
            training_grouped.get("active_token_weighted_router_z_loss", -1),
            2.357235980844142,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and training_grouped.get("sparse_dense_output_max_abs_difference") == 0.0
        and training_grouped.get(
            "sparse_dense_all_parameter_gradient_max_abs_difference"
        )
        == 0.0
        and training_grouped.get("single_group_expert_capacity") == 2
        and math.isclose(
            training_grouped.get(
                "grouped_vs_single_group_output_max_abs_difference",
                -1,
            ),
            0.3293871976258794,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and training_grouped.get("padding_routed_output") == [[0.0, 0.0]]
        and training_grouped.get("padding_hidden_gradient_max_abs") == 0.0
        and training_grouped.get(
            "padding_value_and_group_id_mutation_active_output_max_abs_difference"
        )
        == 0.0
        and training_grouped.get(
            "padding_value_and_group_id_mutation_balance_abs_difference"
        )
        == 0.0
        and training_grouped.get(
            "padding_value_and_group_id_mutation_z_abs_difference"
        )
        == 0.0
    ):
        errors.append("trainable MoE routing-group/padding control mismatch")

    overflow_drop = training_overflow.get("drop", {})
    overflow_reroute = training_overflow.get("reroute", {})
    overflow_dropless = training_overflow.get("dropless", {})
    selected_probabilities = training_overflow.get("selected_probabilities", [])
    rerouted_probabilities = overflow_reroute.get(
        "dispatched_probabilities",
        [],
    )
    rerouted_preserved_mass = overflow_reroute.get(
        "preserved_mass_weight_sums",
        [],
    )
    if not (
        training_overflow.get("hidden_states") == [[1.0, 0.0, 0.0]] * 4
        and training_overflow.get("capacity_factor") == 1.0
        and training_overflow.get("top_k") == 1
        and training_overflow.get("expert_capacity") == 2
        and training_overflow.get("ranked_expert_indices") == [[0, 2, 1]] * 4
        and training_overflow.get("selected_expert_indices") == [[0]] * 4
        and isinstance(selected_probabilities, list)
        and len(selected_probabilities) == 4
        and all(
            len(row) == 1
            and math.isclose(
                row[0],
                0.5896483941044577,
                rel_tol=0,
                abs_tol=1e-15,
            )
            for row in selected_probabilities
        )
        and overflow_drop.get("dispatched_expert_indices") == [[0]] * 4
        and overflow_drop.get("expert_counts_before_capacity") == [4, 0, 0]
        and overflow_drop.get("expert_counts_after_capacity") == [2, 0, 0]
        and overflow_drop.get("pre_policy_capacity_excess_by_group")
        == [[2, 0, 0]]
        and overflow_drop.get("post_policy_capacity_excess_by_group")
        == [[0, 0, 0]]
        and overflow_drop.get("rerouted_assignments") == 0
        and overflow_drop.get("dropped_assignments") == 2
        and overflow_drop.get("combine_weight_sums") == [1.0, 1.0, 0.0, 0.0]
        and overflow_reroute.get("dispatched_expert_indices")
        == [[0], [0], [2], [2]]
        and isinstance(rerouted_probabilities, list)
        and len(rerouted_probabilities) == 4
        and all(
            len(row) == 1
            and math.isclose(
                row[0],
                expected,
                rel_tol=0,
                abs_tol=1e-15,
            )
            for row, expected in zip(
                rerouted_probabilities,
                [
                    0.5896483941044577,
                    0.5896483941044577,
                    0.2649461021163392,
                    0.2649461021163392,
                ],
                strict=True,
            )
        )
        and overflow_reroute.get("expert_counts_after_capacity") == [2, 0, 2]
        and overflow_reroute.get("post_policy_capacity_excess_by_group")
        == [[0, 0, 0]]
        and overflow_reroute.get("rerouted_assignments") == 2
        and overflow_reroute.get("dropped_assignments") == 0
        and overflow_reroute.get("renormalized_weight_sums") == [1.0] * 4
        and isinstance(rerouted_preserved_mass, list)
        and len(rerouted_preserved_mass) == 4
        and all(
            math.isclose(actual, expected, rel_tol=0, abs_tol=1e-15)
            for actual, expected in zip(
                rerouted_preserved_mass,
                [1.0, 1.0, 0.44932896411722156, 0.44932896411722156],
                strict=True,
            )
        )
        and math.isclose(
            overflow_reroute.get(
                "renormalize_vs_preserve_output_max_abs_difference",
                -1,
            ),
            0.06399997177521191,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and overflow_reroute.get("sparse_dense_output_max_abs_difference") == 0.0
        and overflow_reroute.get(
            "sparse_dense_materialized_zero_gradient_max_abs_difference"
        )
        == 0.0
        and overflow_reroute.get("sparse_parameters_with_missing_zero_gradient")
        == ["experts.1.0.weight", "experts.1.2.weight"]
        and overflow_reroute.get("dense_corresponding_gradients_are_zero") is True
        and overflow_dropless.get("dispatched_expert_indices") == [[0]] * 4
        and overflow_dropless.get("expert_counts_after_capacity") == [4, 0, 0]
        and overflow_dropless.get("post_policy_capacity_excess_by_group")
        == [[2, 0, 0]]
        and overflow_dropless.get("rerouted_assignments") == 0
        and overflow_dropless.get("dropped_assignments") == 0
        and overflow_dropless.get("sparse_dense_output_max_abs_difference") == 0.0
        and overflow_dropless.get(
            "sparse_dense_materialized_zero_gradient_max_abs_difference"
        )
        == 0.0
        and overflow_dropless.get("sparse_parameters_with_missing_zero_gradient")
        == [
            "experts.1.0.weight",
            "experts.1.2.weight",
            "experts.2.0.weight",
            "experts.2.2.weight",
        ]
        and overflow_dropless.get("dense_corresponding_gradients_are_zero") is True
        and math.isclose(
            training_overflow.get(
                "drop_vs_reroute_output_max_abs_difference",
                -1,
            ),
            0.11622178688336826,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and math.isclose(
            training_overflow.get(
                "reroute_vs_dropless_output_max_abs_difference",
                -1,
            ),
            0.10698215447093767,
            rel_tol=0,
            abs_tol=1e-15,
        )
    ):
        errors.append("trainable MoE overflow-policy control mismatch")

    expected_true_scope = (
        "trainable_router_and_expert_mlp_forward_backward_executed",
        "sparse_dispatch_dense_oracle_forward_backward_compared",
        "hard_topk_indices_treated_as_nondifferentiable",
        "selected_probability_task_gradient_to_router_executed",
        "detached_gate_missing_router_task_gradient_negative_control_executed",
        "authored_balance_and_z_loss_gradients_executed",
        "score_priority_capacity_drop_in_training_graph_executed",
        "post_drop_renormalize_and_preserve_mass_policies_executed",
        "all_assignments_dropped_zero_routed_output_executed",
        "padding_aware_capacity_aux_and_gradient_executed",
        "routing_group_scoped_capacity_and_aux_executed",
        "deterministic_full_ranking_reroute_policy_executed",
        "dropless_nominal_capacity_excess_policy_executed",
    )
    expected_false_scope = (
        "distributed_capacity_group_collective_executed",
        "shared_or_fine_grained_experts_executed",
        "expert_parallel_all_to_all_grouped_gemm_or_gpu_executed",
        "deepseek_qwen_or_other_checkpoint_reproduced",
        "convergence_quality_throughput_memory_or_scaling_proved",
    )
    if not (
        isinstance(scope, dict)
        and set(scope) == set(expected_true_scope) | set(expected_false_scope)
        and all(scope.get(field) is True for field in expected_true_scope)
        and all(scope.get(field) is False for field in expected_false_scope)
    ):
        errors.append("trainable MoE evidence scope mismatch")

    training_source = (
        SRC / "about_llm" / "from_scratch" / "moe_training.py"
    ).read_text(encoding="utf-8")
    training_script = (
        ROOT / "projects" / "transformers-basics" / "moe_training_control.py"
    ).read_text(encoding="utf-8")
    training_tests = (ROOT / "tests" / "test_moe_training.py").read_text(
        encoding="utf-8"
    )
    source_markers = (
        'TRAINABLE_MOE_CONTROL_VERSION = "about-llm.trainable-moe-control.v3"',
        'OverflowPolicy: TypeAlias = Literal["drop", "reroute", "dropless"]',
        "torch.argsort(",
        "stable=True",
        "combine_weights.detach()",
        "math.ceil(",
        "assignments.sort(key=lambda item: (-item[0], item[1], item[2]))",
        "(dispatched_expert_indices == expert_id) & kept_mask",
        "expert_capacities_by_group",
        "group_mask = active_token_mask & (routing_group_ids == group_label)",
        "def _reroute_dropped_assignments(",
        "for candidate in ranked_expert_indices[token_index].cpu().tolist():",
        "if candidate_id in occupied_experts:",
        'elif overflow_policy == "dropless":',
        "post_policy_capacity_excess_by_group",
        "output.index_add(0, token_indices, weighted)",
        "selection_fractions = selection_fractions.detach()",
        '"score_priority_capacity_drop_in_training_graph_executed": True',
        '"padding_aware_capacity_aux_and_gradient_executed": True',
        '"routing_group_scoped_capacity_and_aux_executed": True',
        '"deterministic_full_ranking_reroute_policy_executed": True',
        '"dropless_nominal_capacity_excess_policy_executed": True',
        '"distributed_capacity_group_collective_executed": False',
        '"deepseek_qwen_or_other_checkpoint_reproduced": False',
        "allow_nan=False",
    )
    script_markers = (
        "run_trainable_moe_control",
        "allow_nan=False",
    )
    test_markers = (
        "test_stable_topk_tie_break_prefers_lower_expert_id",
        "test_constructor_rejects_invalid_dimensions",
        "test_forward_contract_fails_closed",
        "test_sparse_dispatch_matches_dense_oracle_forward_and_backward",
        "test_capacity_score_priority_counts_and_post_drop_renormalization",
        "test_preserving_dropped_mass_is_distinct_from_post_drop_renormalization",
        "test_all_dropped_tokens_have_zero_routed_expert_output",
        "test_padding_and_routing_groups_scope_capacity_competition",
        "test_grouped_padding_sparse_dense_forward_and_backward_match",
        "test_padding_values_and_group_ids_do_not_affect_active_path_or_gradient",
        "test_overflow_policies_pin_dispatch_drop_and_capacity_excess",
        "test_reroute_scans_full_ranking_without_duplicate_experts_per_token",
        "test_reroute_capacity_is_scoped_to_each_routing_group",
        "test_reroute_can_preserve_selected_mass_or_renormalize_after_dispatch",
        "test_overflow_policy_sparse_dense_forward_and_backward_match_with_zero_fill",
        "test_detaching_selected_gate_blocks_only_main_task_router_gradient",
        "test_control_report_pins_gradient_semantics_and_scope",
        "test_project_control_emits_strict_finite_json",
    )
    missing_source = [
        marker for marker in source_markers if marker not in training_source
    ]
    missing_script = [
        marker for marker in script_markers if marker not in training_script
    ]
    missing_tests = [marker for marker in test_markers if marker not in training_tests]
    if missing_source:
        errors.append(f"trainable MoE source missing marker(s): {missing_source}")
    if missing_script:
        errors.append(f"trainable MoE script missing marker(s): {missing_script}")
    if missing_tests:
        errors.append(f"trainable MoE tests missing marker(s): {missing_tests}")

    required_docs = {
        "frontier": (
            ROOT / "docs" / "frontier" / "reasoning-long-context-moe.md",
            (
                "可运行 router/MLP 梯度 fixture",
                "所有 router/expert 参数的 backward 最大差约为",
                "router 的 task gradient 为 `None`",
                "2.567724 降到 2.552751",
                "同一训练图还执行 score-priority capacity/drop",
                "两个 active groups 各有 2 tokens",
                "padding hidden gradient 也为 0",
                "deterministic full-ranking reroute",
                "dropless nominal-capacity-excess contract",
                "不是某个框架或模型的默认行为",
            ),
        ),
        "architecture": (
            ROOT / "docs" / "core" / "architectures-interpretability.md",
            (
                "selected-only sparse dispatch 与 dense masked oracle",
                "hard expert index 不会自行把 task gradient 送回 router",
                "在同一训练图中执行 trainable top-2 router",
                "全丢 token 的 routed expert 输出为零",
                "token mask 与两个 CPU-local group",
                "整数 group label 不是 distributed collective",
                "full-ranking、token 内去重的 deterministic `reroute`",
                "v3 policy 也不是 DeepSeek/Qwen/PyTorch 默认语义",
            ),
        ),
        "deepseek": (
            ROOT / "docs" / "models" / "deepseek.md",
            (
                "trainable router/三组 MLP experts",
                "在同一训练图真实执行",
                "post-drop 重归一化/保留丢失 mass",
                "padding mask/两个 CPU-local groups",
                "int64 group label 不是分布式通信证据",
                "deterministic full-ranking reroute",
                "dropless nominal-capacity-excess contract",
                "不能把结果标成 DeepSeek-V2/V3/R1 架构复现",
            ),
        ),
        "qwen": (
            ROOT / "docs" / "models" / "qwen.md",
            (
                "`moe_training_control.py`",
                "对齐 sparse—dense forward/backward",
                "score-priority capacity/drop",
                "padding 与两个 CPU-local routing groups",
                "int64 group label 不等于 distributed collective",
                "不读取任何 Qwen config/weight",
                "deterministic full-ranking reroute",
                "authored reroute/dropless 也不是 Qwen 默认策略",
            ),
        ),
        "project": (
            ROOT / "projects" / "transformers-basics" / "README.md",
            (
                "Trainable MoE router/MLP gradient control",
                "所有参数梯度最大差约 `6.94e-18`",
                "0.0886473` 降至 `0.0875580",
                "router 的主任务 gradient 却缺失",
                "2.567724` 降至 `2.552751",
                "counts 从 `[4,3,3]` 变为 `[2,2,2]`",
                "两组 pre/post counts 分别为 `[2,1,1]→[1,1,1]`",
                "输出最大差约 `0.329387`",
                "deterministic full-ranking `reroute`",
                "nominal-capacity excess `[[2,0,0]]`",
                "不是 PyTorch、DeepSeek、Qwen 或任意训练框架的默认 overflow 语义",
            ),
        ),
        "project_page": (
            ROOT / "docs" / "practice" / "projects" / "transformers-basics.md",
            (
                "moe_training_control.py",
                "所有参数 backward 最大差约 `6.94e-18`",
                "0.0886473→0.0875580",
                "score-priority capacity/drop 放入同一训练图",
                "全丢 token 的 routed expert 输出为零",
                "per-group capacities `[1,1]`",
                "CPU-local int64 label",
                "deterministic full-ranking `reroute`",
                "materialized-zero 全参数梯度差均为 0",
            ),
        ),
        "labs": (
            ROOT / "docs" / "practice" / "labs.md",
            (
                "5-token/top-2 assignments",
                "全参数梯度最大差约 `6.94e-18`",
                "router task gradient 却缺失",
                "2.567724` 降至 `2.552751",
                "4/10 assignment drop",
                "group 10/20 各有 2 active tokens",
                "输出最大差约 `0.329387`",
                "deterministic full-ranking reroute",
                "dropless 则保持 `[0,0,0,0]`",
                "不是 DeepSeek、Qwen 或框架默认策略",
            ),
        ),
        "interview": (
            ROOT / "docs" / "career" / "interview-questions.md",
            (
                "Hard top-k 不可导",
                "被选 gate 的 softmax probability",
                "router 的 task gradient却消失",
                "4/10 assignments 被丢",
                "全丢 token",
                "两个 2-token groups",
                "CPU int64 group IDs 不证明真实 distributed collective",
                "不证明最终负载均衡",
                "deterministic full-ranking reroute",
                "dropless 保持 `[4,0,0]`",
                "authored reroute/dropless 也不证明任意框架或目标 MoE 实现",
            ),
        ),
        "accuracy": (
            ROOT / "docs" / "reference" / "accuracy.md",
            (
                "Trainable MoE control",
                "6.938893903907228e-18",
                "0.08864729306070791→0.08755795603512319",
                "2.5677239505126708→2.552750704884368",
                "4/10 assignment drop",
                "0.1255417263895207",
                "0.3293871976258794",
                "CPU-local int64 groups 不执行 distributed collective",
                "Trainable MoE v3 overflow control",
                "0.06399997177521191",
                "68 个 MoE 专项测试",
                "不是 PyTorch、DeepSeek、Qwen 或任意框架的默认策略",
            ),
        ),
        "knowledge_map": (
            ROOT / "docs" / "guide" / "knowledge-map.md",
            (
                "同一训练图真实执行 top-2 router/三组 MLP experts",
                "detached-gate、collapsed top-1 balance",
                "padding exclusion 与 CPU-local routing groups",
                "v3 再执行 authored full-ranking reroute、dropless excess",
                "第一条 two-process CPU/Gloo control",
            ),
        ),
        "repo_map": (
            ROOT / "docs" / "guide" / "repo-map.md",
            (
                "PyTorch trainable top-k router/MLP",
                "MoE 单进程 controls 覆盖 assignment/drop/combine",
                "sparse—dense forward/backward",
                "gate/balance gradient",
                "padding/group competition",
                "full-ranking reroute",
                "dropless excess",
                "该 capacity control 仍无 expert ownership/token `all_to_all`",
                "后续同机 Gloo controls 分别执行 owner-only",
                "不声称 authored MoE policy 是框架默认",
            ),
        ),
        "project_index": (
            ROOT / "docs" / "practice" / "project-index.md",
            (
                "MoE routing/training",
                "同一训练图真正执行 top-2 router/三组 MLP experts",
                "padding exclusion 与 CPU-local routing groups",
                "int64 group IDs 也不是 distributed collective",
                "deterministic full-ranking reroute",
                "dropless nominal-capacity-excess policy",
                "没有目标 MoE checkpoint",
            ),
        ),
        "changelog": (
            ROOT / "CHANGELOG.md",
            (
                "trainable top-k MoE router/MLP gradient control",
                "6.938893903907228e-18",
                "router task gradient 却缺失",
                "4/10 assignment drop",
                "0.1255417263895207",
                "trainable MoE control 升级为 v2 padding/routing-group contract",
                "0.3293871976258794",
                "trainable MoE control 升级为 v3 explicit overflow-policy contract",
                "materialized-zero 全参数 gradient 差均为 0",
                "MoE 专项增至 68 tests",
            ),
        ),
    }
    for name, (path, markers) in required_docs.items():
        text = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            errors.append(f"trainable MoE {name} docs missing marker(s): {missing}")
    return errors


def check_distributed_moe_capacity_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.from_scratch.moe_distributed_capacity import (
        DISTRIBUTED_MOE_CAPACITY_CONTROL_VERSION,
        run_distributed_moe_capacity_control,
    )

    report = run_distributed_moe_capacity_control()
    runtime = report.get("runtime", {})
    fixture = report.get("fixture", {})
    process_observation = report.get("process_observation", {})
    rank_reports = report.get("rank_reports", [])
    comparison = report.get("comparison", {})
    assertions = report.get("assertions", {})
    scope = report.get("scope", {})
    rank_zero = (
        rank_reports[0]
        if isinstance(rank_reports, list)
        and len(rank_reports) == 2
        and isinstance(rank_reports[0], dict)
        else {}
    )
    rank_one = (
        rank_reports[1]
        if isinstance(rank_reports, list)
        and len(rank_reports) == 2
        and isinstance(rank_reports[1], dict)
        else {}
    )
    global_route = rank_zero.get("collective_global_route", {})
    rank_zero_local = rank_zero.get("local_independent_route", {})
    rank_one_local = rank_one.get("local_independent_route", {})
    rank_zero_slice = rank_zero.get("collective_local_slice", {})
    rank_one_slice = rank_one.get("collective_local_slice", {})
    errors: list[str] = []
    if not (
        report.get("schema_version")
        == DISTRIBUTED_MOE_CAPACITY_CONTROL_VERSION
        and report.get("report_fingerprint")
        == "sha256:9e342b0ba87b0e11ebf43eb41eaca0be165a3ee365cc9285bbe5a2f2923887be"
        and runtime.get("backend") == "gloo"
        and runtime.get("device") == "cpu"
        and runtime.get("dtype") == "torch.float64"
        and runtime.get("world_size") == 2
        and runtime.get("process_start_method") == "spawn"
        and runtime.get("rendezvous") == "temporary-file-store"
        and fixture.get("local_hidden_states_by_rank")
        == [[[2.0], [1.0]], [[3.0], [0.5]]]
        and fixture.get("router_weight") == [[1.0], [0.0]]
        and fixture.get("expert_count") == 2
        and fixture.get("top_k") == 1
        and fixture.get("capacity_factor") == 0.5
        and fixture.get("overflow_policy") == "drop"
        and process_observation
        == {
            "distinct_worker_process_count": 2,
            "raw_process_ids_published": False,
        }
        and isinstance(assertions, dict)
        and assertions
        and all(assertions.values())
    ):
        errors.append("distributed MoE capacity report identity/runtime mismatch")

    if not (
        rank_zero.get("rank") == 0
        and rank_one.get("rank") == 1
        and rank_zero.get("gathered_hidden_states")
        == [[2.0], [1.0], [3.0], [0.5]]
        and rank_one.get("gathered_hidden_states")
        == [[2.0], [1.0], [3.0], [0.5]]
        and rank_zero.get("global_active_token_count_after_all_reduce") == 4
        and rank_one.get("global_active_token_count_after_all_reduce") == 4
        and rank_zero.get("global_selected_counts_after_all_reduce") == [4, 0]
        and rank_one.get("global_selected_counts_after_all_reduce") == [4, 0]
        and rank_zero.get("collective_call_counts")
        == {"all_gather": 1, "all_reduce": 2, "barrier": 1}
        and rank_one.get("collective_call_counts")
        == {"all_gather": 1, "all_reduce": 2, "barrier": 1}
        and rank_zero.get("collective_global_route_fingerprint")
        == "sha256:71a66eeb27ebb1f218f7ff4e11eccadee27f0d05f5c977eeaa4633a8dd3b7249"
        and rank_one.get("collective_global_route_fingerprint")
        == rank_zero.get("collective_global_route_fingerprint")
        and rank_one.get("collective_global_route") == global_route
    ):
        errors.append("distributed MoE collective trace mismatch")

    expected_probabilities = [
        0.8807970779778823,
        0.7310585786300049,
        0.9525741268224334,
        0.6224593312018546,
    ]
    actual_probabilities = global_route.get("selected_probabilities", [])
    if not (
        global_route.get("expert_capacity") == 1
        and global_route.get("selected_expert_indices") == [[0]] * 4
        and isinstance(actual_probabilities, list)
        and len(actual_probabilities) == 4
        and all(
            len(row) == 1
            and math.isclose(
                row[0],
                expected,
                rel_tol=0,
                abs_tol=1e-15,
            )
            for row, expected in zip(
                actual_probabilities,
                expected_probabilities,
                strict=True,
            )
        )
        and global_route.get("kept_mask")
        == [[False], [False], [True], [False]]
        and global_route.get("expert_counts_before_capacity") == [4, 0]
        and global_route.get("expert_counts_after_capacity") == [1, 0]
        and global_route.get("pre_policy_capacity_excess_by_group") == [[3, 0]]
        and global_route.get("post_policy_capacity_excess_by_group") == [[0, 0]]
        and global_route.get("dropped_assignments") == 3
        and global_route.get("routed_output")
        == [[0.0], [0.0], [0.9950547536867305], [0.0]]
        and rank_zero_local.get("kept_mask") == [[True], [False]]
        and rank_one_local.get("kept_mask") == [[True], [False]]
        and rank_zero_slice.get("kept_mask") == [[False], [False]]
        and rank_one_slice.get("kept_mask") == [[True], [False]]
        and math.isclose(
            rank_zero_slice.get(
                "vs_local_independent_output_max_abs_difference",
                -1,
            ),
            0.9640275800758169,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and rank_one_slice.get(
            "vs_local_independent_output_max_abs_difference"
        )
        == 0.0
        and comparison
        == {
            "independent_rank_local_kept_assignments": 2,
            "collective_global_kept_assignments": 1,
            "collective_minus_independent_kept_assignments": -1,
            "rank_zero_collective_vs_local_output_max_abs_difference": (
                0.9640275800758169
            ),
            "rank_one_collective_vs_local_output_max_abs_difference": 0.0,
        }
    ):
        errors.append("distributed MoE local-vs-collective capacity mismatch")

    expected_scope = {
        "real_two_process_same_host_gloo_process_group_executed": True,
        "temporary_file_store_rendezvous_executed": True,
        "hidden_state_all_gather_for_replicated_global_routing_executed": True,
        "global_active_token_count_all_reduce_executed": True,
        "global_selected_assignment_count_all_reduce_executed": True,
        "collective_capacity_group_competition_executed": True,
        "replicated_router_and_experts_used": True,
        "distributed_autograd_or_ddp_backward_executed": False,
        "expert_parallel_all_to_all_or_reduce_scatter_executed": False,
        "cuda_nccl_multi_node_or_remote_host_executed": False,
        "deepseek_qwen_or_other_checkpoint_reproduced": False,
        "throughput_memory_scaling_convergence_or_quality_proved": False,
    }
    if scope != expected_scope:
        errors.append("distributed MoE evidence scope mismatch")

    source = (
        SRC / "about_llm" / "from_scratch" / "moe_distributed_capacity.py"
    ).read_text(encoding="utf-8")
    script = (
        ROOT
        / "projects"
        / "transformers-basics"
        / "moe_distributed_capacity_control.py"
    ).read_text(encoding="utf-8")
    test = (ROOT / "tests" / "test_moe_distributed_capacity.py").read_text(
        encoding="utf-8"
    )
    source_markers = (
        '"about-llm.distributed-moe-capacity-control.v1"',
        'backend="gloo"',
        "dist.all_gather(gathered_hidden, local_hidden)",
        "dist.all_reduce(active_token_count, op=dist.ReduceOp.SUM)",
        "dist.all_reduce(selected_counts, op=dist.ReduceOp.SUM)",
        '"raw_process_ids_published": False',
        '"collective_capacity_group_competition_executed": True',
        '"expert_parallel_all_to_all_or_reduce_scatter_executed": False',
        '"distributed_autograd_or_ddp_backward_executed": False',
        "allow_nan=False",
    )
    script_markers = (
        "run_distributed_moe_capacity_control",
        "allow_nan=False",
    )
    test_markers = (
        "test_two_process_gloo_moe_capacity_group_control",
        "global_selected_counts_after_all_reduce",
        "collective_minus_independent_kept_assignments",
        "raw_process_ids_published",
    )
    missing_source = [marker for marker in source_markers if marker not in source]
    missing_script = [marker for marker in script_markers if marker not in script]
    missing_test = [marker for marker in test_markers if marker not in test]
    if missing_source:
        errors.append(
            f"distributed MoE source missing marker(s): {missing_source}"
        )
    if missing_script:
        errors.append(
            f"distributed MoE script missing marker(s): {missing_script}"
        )
    if missing_test:
        errors.append(f"distributed MoE test missing marker(s): {missing_test}")

    required_docs = {
        "project": (
            ROOT / "projects" / "transformers-basics" / "README.md",
            (
                "Two-process Gloo capacity-group control",
                "active tokens=4 和 pre-capacity selected counts `[4,0]`",
                "合计 kept assignments=2",
                "global kept mask 为 `[F,F,T,F]`",
                "Router 与 experts 在两 rank 完全复制",
                "没有 token-to-expert `all_to_all`/`reduce_scatter`",
            ),
        ),
        "project_page": (
            ROOT / "docs" / "practice" / "projects" / "transformers-basics.md",
            (
                "moe_distributed_capacity_control.py",
                "selected counts=`[4,0]`",
                "rank-0 output counterfactual 差为 `0.9640275800758169`",
                "仍无 expert `all_to_all`",
            ),
        ),
        "frontier": (
            ROOT / "docs" / "frontier" / "reasoning-long-context-moe.md",
            (
                "Collective capacity group 与 expert parallel 不是同一件事",
                "真实 `all_gather`",
                "mask `[F,F,T,F]`",
                "不证明 expert parallel",
            ),
        ),
        "architecture": (
            ROOT / "docs" / "core" / "architectures-interpretability.md",
            (
                "two-process Gloo control",
                "local-only 的两个 kept assignments",
                "没有 expert ownership 与 `all_to_all`",
            ),
        ),
        "deepseek": (
            ROOT / "docs" / "models" / "deepseek.md",
            (
                "two-process CPU/Gloo control",
                "复制 router/experts",
                "不执行 expert `all_to_all` 或 backward",
            ),
        ),
        "qwen": (
            ROOT / "docs" / "models" / "qwen.md",
            (
                "two-process CPU/Gloo capacity-group control",
                "global capacity=1 只 kept=1",
                "只能证明 authored collective-capacity 反事实",
            ),
        ),
        "labs": (
            ROOT / "docs" / "practice" / "labs.md",
            (
                "moe_distributed_capacity_control.py",
                "active count=4、selected counts `[4,0]`",
                "hidden-state `all_gather` 形成 replicated routing input",
                "不得把 same-host Gloo/FileStore 外推",
            ),
        ),
        "interview": (
            ROOT / "docs" / "career" / "interview-questions.md",
            (
                "有 collective 是否就等于 expert parallel",
                "Local-only capacity=1 会跨 ranks 共保留 2 个",
                "capacity-group collective、expert dispatch collective、gradient collective",
            ),
        ),
        "accuracy": (
            ROOT / "docs" / "reference" / "accuracy.md",
            (
                "Distributed MoE capacity control",
                "0.9525741268224334",
                "sha256:71a66eeb27ebb1f218f7ff4e11eccadee27f0d05f5c977eeaa4633a8dd3b7249",
                "sha256:9e342b0ba87b0e11ebf43eb41eaca0be165a3ee365cc9285bbe5a2f2923887be",
                "不是生产 EP 的 token dispatch",
            ),
        ),
        "knowledge_map": (
            ROOT / "docs" / "guide" / "knowledge-map.md",
            (
                "two-process CPU/Gloo control",
                "local-only kept=2 与 global kept=1/mask `[F,F,T,F]`",
                "variable-split token `all_to_all` forward/return",
            ),
        ),
        "repo_map": (
            ROOT / "docs" / "guide" / "repo-map.md",
            (
                "two-process CPU/Gloo control",
                "replicated-global kept=1",
                "仍无 expert ownership/token `all_to_all`",
            ),
        ),
        "project_index": (
            ROOT / "docs" / "practice" / "project-index.md",
            (
                "第三条 distributed capacity fixture",
                "rank-0 output 反事实差 `0.9640275800758169`",
                "不能借给前两条升级为 distributed training/目标模型声明",
            ),
        ),
        "distributed": (
            ROOT / "docs" / "systems" / "distributed-training.md",
            (
                "moe_distributed_capacity_control.py",
                "replicated 4-token global routing input",
                "不具备 scalable EP 的 expert ownership",
                "capacity/routing group、expert dispatch group、gradient synchronization group",
            ),
        ),
        "changelog": (
            ROOT / "CHANGELOG.md",
            (
                "two-process CPU/Gloo distributed MoE capacity-group control",
                "global-route fingerprint 固定为 `sha256:71a66eeb",
                "strict report fingerprint 连续运行稳定为 `sha256:9e342b0b",
                "无 expert ownership、token `all_to_all`/`reduce_scatter`",
            ),
        ),
    }
    for name, (path, markers) in required_docs.items():
        document = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in document]
        if missing:
            errors.append(
                f"distributed MoE {name} docs missing marker(s): {missing}"
            )
    return errors


def check_moe_all_to_all_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.from_scratch.moe_all_to_all import (
        MOE_ALL_TO_ALL_CONTROL_VERSION,
        run_moe_all_to_all_control,
    )

    report = run_moe_all_to_all_control()
    runtime = report.get("runtime", {})
    fixture = report.get("fixture", {})
    process_observation = report.get("process_observation", {})
    workers = report.get("worker_reports", [])
    comparison = report.get("comparison", {})
    assertions = report.get("assertions", {})
    scope = report.get("scope", {})
    rank_zero = (
        workers[0]
        if isinstance(workers, list)
        and len(workers) == 2
        and isinstance(workers[0], dict)
        else {}
    )
    rank_one = (
        workers[1]
        if isinstance(workers, list)
        and len(workers) == 2
        and isinstance(workers[1], dict)
        else {}
    )
    rank_zero_trace = rank_zero.get("trace", {})
    rank_one_trace = rank_one.get("trace", {})
    errors: list[str] = []

    if not (
        report.get("schema_version") == MOE_ALL_TO_ALL_CONTROL_VERSION
        and report.get("report_fingerprint")
        == "sha256:51c77e2499d84d5cf5500a5f5c1143b2979f3d70f1755114c34900b55299a61c"
        and runtime.get("backend") == "gloo"
        and runtime.get("device") == "cpu"
        and runtime.get("dtype") == "torch.float64"
        and runtime.get("world_size") == 2
        and runtime.get("process_start_method") == "spawn"
        and runtime.get("rendezvous") == "temporary-file-store"
        and fixture.get("local_hidden_states_by_rank")
        == [[[-1.0], [2.0], [-2.0]], [[1.0]]]
        and fixture.get("router_weight") == [[1.0], [-1.0]]
        and fixture.get("expert_ownership") == {"expert_0": 0, "expert_1": 1}
        and fixture.get("top_k") == 1
        and fixture.get("combine_weight_policy")
        == "preserve selected softmax probability"
        and fixture.get("capacity_or_drop_policy")
        == "none; every assignment is dispatched"
        and process_observation
        == {
            "distinct_worker_process_count": 2,
            "raw_process_ids_published": False,
        }
        and isinstance(assertions, dict)
        and assertions
        and all(assertions.values())
    ):
        errors.append("MoE all-to-all report identity/runtime mismatch")

    expected_probabilities = [
        0.8807970779778823,
        0.9820137900379085,
        0.9820137900379085,
        0.8807970779778823,
    ]
    actual_probabilities = (
        rank_zero.get("selected_probabilities", [])
        + rank_one.get("selected_probabilities", [])
    )
    if not (
        rank_zero.get("rank") == 0
        and rank_one.get("rank") == 1
        and rank_zero.get("owned_expert_id") == 0
        and rank_one.get("owned_expert_id") == 1
        and rank_zero.get("selected_expert_indices") == [1, 0, 1]
        and rank_one.get("selected_expert_indices") == [0]
        and isinstance(actual_probabilities, list)
        and len(actual_probabilities) == 4
        and all(
            math.isclose(actual, expected, rel_tol=0, abs_tol=1e-15)
            for actual, expected in zip(
                actual_probabilities,
                expected_probabilities,
                strict=True,
            )
        )
        and rank_zero.get("all_to_all_single_call_count") == 5
        and rank_one.get("all_to_all_single_call_count") == 5
        and rank_zero.get("logical_tensor_payload_bytes_sent") == 256
        and rank_one.get("logical_tensor_payload_bytes_sent") == 160
    ):
        errors.append("MoE all-to-all worker route/payload mismatch")

    if not (
        rank_zero_trace.get("send_counts_by_owner") == [1, 2]
        and rank_one_trace.get("send_counts_by_owner") == [1, 0]
        and rank_zero_trace.get("received_counts_by_source") == [1, 1]
        and rank_one_trace.get("received_counts_by_source") == [2, 0]
        and rank_zero_trace.get("owner_received_metadata")
        == [[0, 1, 1, 0], [1, 0, 3, 0]]
        and rank_one_trace.get("owner_received_metadata")
        == [[0, 0, 0, 1], [0, 2, 2, 1]]
        and rank_zero_trace.get("owner_raw_expert_outputs") == [[4.5], [2.5]]
        and rank_one_trace.get("owner_raw_expert_outputs") == [[4.0], [7.0]]
        and rank_zero_trace.get("return_arrival_metadata")
        == [[0, 1, 1, 0], [0, 0, 0, 1], [0, 2, 2, 1]]
        and rank_one_trace.get("return_arrival_metadata") == [[1, 0, 3, 0]]
    ):
        errors.append("MoE all-to-all dispatch/return metadata mismatch")

    expected_outputs = [
        3.5231883119115293,
        4.419062055170588,
        6.874096530265359,
        2.201992694944706,
    ]
    distributed_outputs = comparison.get(
        "distributed_outputs_by_global_token_id",
        [],
    )
    oracle_outputs = comparison.get(
        "single_process_oracle_outputs_by_global_token_id",
        [],
    )
    if not (
        comparison.get("source_to_owner_token_counts")
        == [[1, 2], [1, 0]]
        and comparison.get("owner_from_source_token_counts")
        == [[1, 1], [2, 0]]
        and isinstance(distributed_outputs, list)
        and isinstance(oracle_outputs, list)
        and len(distributed_outputs) == len(expected_outputs)
        and len(oracle_outputs) == len(expected_outputs)
        and all(
            math.isclose(actual, expected, rel_tol=0, abs_tol=1e-15)
            for actual, expected in zip(
                distributed_outputs,
                expected_outputs,
                strict=True,
            )
        )
        and all(
            math.isclose(actual, expected, rel_tol=0, abs_tol=1e-15)
            for actual, expected in zip(
                oracle_outputs,
                expected_outputs,
                strict=True,
            )
        )
        and comparison.get("distributed_vs_oracle_max_abs_difference") == 0.0
        and math.isclose(
            comparison.get(
                "rank_zero_metadata_free_vs_correct_max_abs_difference",
                -1,
            ),
            0.8958737432590591,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and comparison.get(
            "rank_one_metadata_free_vs_correct_max_abs_difference"
        )
        == 0.0
        and comparison.get("logical_tensor_payload_bytes_sent_by_rank")
        == [256, 160]
        and comparison.get("logical_tensor_payload_bytes_sent_total") == 416
    ):
        errors.append("MoE all-to-all oracle/order/byte accounting mismatch")

    expected_scope = {
        "real_two_process_same_host_gloo_process_group_executed": True,
        "variable_split_all_to_all_single_count_exchange_executed": True,
        "token_to_owner_float_and_metadata_dispatch_executed": True,
        "owner_only_expert_parameter_placement_executed": True,
        "owner_to_source_output_and_metadata_return_executed": True,
        "source_metadata_scatter_and_gate_combine_executed": True,
        "single_process_forward_oracle_compared": True,
        "replicated_router_executed": True,
        "capacity_drop_reroute_or_dropless_executed": False,
        "distributed_autograd_backward_or_optimizer_executed": False,
        "ddp_fsdp_zero_tensor_or_pipeline_parallel_executed": False,
        "cuda_nccl_multi_node_or_remote_host_executed": False,
        "wire_bytes_protocol_overhead_or_packet_capture_measured": False,
        "deepseek_qwen_or_other_checkpoint_reproduced": False,
        "throughput_latency_memory_scaling_convergence_or_quality_proved": False,
    }
    if scope != expected_scope:
        errors.append("MoE all-to-all evidence scope mismatch")

    source = (
        SRC / "about_llm" / "from_scratch" / "moe_all_to_all.py"
    ).read_text(encoding="utf-8")
    script = (
        ROOT
        / "projects"
        / "transformers-basics"
        / "moe_all_to_all_control.py"
    ).read_text(encoding="utf-8")
    test = (ROOT / "tests" / "test_moe_all_to_all.py").read_text(
        encoding="utf-8"
    )
    source_markers = (
        '"about-llm.moe-all-to-all-control.v1"',
        'backend="gloo"',
        "dist.all_to_all_single(",
        "output_split_sizes=",
        "input_split_sizes=",
        "OwnedExpert(rank)",
        '"all_to_all_single_call_count": 5',
        '"capacity_drop_reroute_or_dropless_executed": False',
        '"wire_bytes_protocol_overhead_or_packet_capture_measured": False',
        "allow_nan=False",
    )
    script_markers = (
        "run_moe_all_to_all_control",
        "allow_nan=False",
    )
    test_markers = (
        "test_two_process_gloo_variable_split_moe_all_to_all_control",
        "51c77e2499d84d5cf5500a5f5c1143b2979f3d70f1755114c34900b55299a61c",
        "rank_zero_metadata_free_vs_correct_max_abs_difference",
        "logical_tensor_payload_bytes_sent_total",
    )
    missing_source = [marker for marker in source_markers if marker not in source]
    missing_script = [marker for marker in script_markers if marker not in script]
    missing_test = [marker for marker in test_markers if marker not in test]
    if missing_source:
        errors.append(f"MoE all-to-all source missing marker(s): {missing_source}")
    if missing_script:
        errors.append(f"MoE all-to-all script missing marker(s): {missing_script}")
    if missing_test:
        errors.append(f"MoE all-to-all test missing marker(s): {missing_test}")

    required_docs = {
        "readme": (
            ROOT / "README.md",
            (
                "token-to-owner `all_to_all_single`",
                "416 logical tensor-payload bytes",
                "不等于 wire bytes",
            ),
        ),
        "project": (
            ROOT / "projects" / "transformers-basics" / "README.md",
            (
                "Two-process Gloo token-to-owner all-to-all control",
                "source→owner counts matrix 为 `[[1,2],[1,0]]`",
                "global token 顺序 `[1,0,2]`",
                "0.8958737432590591",
            ),
        ),
        "project_page": (
            ROOT / "docs" / "practice" / "projects" / "transformers-basics.md",
            (
                "moe_all_to_all_control.py",
                "每 rank 五次 `all_to_all_single`",
                "owner-only expert parameters",
                "416 logical tensor-payload bytes",
            ),
        ),
        "frontier": (
            ROOT / "docs" / "frontier" / "reasoning-long-context-moe.md",
            (
                "真实 token-to-owner all-to-all control",
                "return arrival 的 global token 顺序为 `[1,0,2]`",
                "保留 selected softmax probability",
            ),
        ),
        "architecture": (
            ROOT / "docs" / "core" / "architectures-interpretability.md",
            (
                "token-to-owner `all_to_all_single`",
                "owner-only experts",
                "metadata scatter",
            ),
        ),
        "deepseek": (
            ROOT / "docs" / "models" / "deepseek.md",
            (
                "owner-only expert-0/expert-1",
                "不是 DeepSeek checkpoint",
                "不含 capacity/drop 或 backward",
            ),
        ),
        "qwen": (
            ROOT / "docs" / "models" / "qwen.md",
            (
                "variable-split token dispatch/return",
                "不是 Qwen checkpoint",
                "不能证明 Qwen MoE runtime",
            ),
        ),
        "labs": (
            ROOT / "docs" / "practice" / "labs.md",
            (
                "moe_all_to_all_control.py",
                "五次 `all_to_all_single`",
                "sha256:51c77e2499d84d5c",
            ),
        ),
        "interview": (
            ROOT / "docs" / "career" / "interview-questions.md",
            (
                "all-to-all 返回后仍要 metadata scatter",
                "source local index",
                "logical tensor payload 不等于 wire bytes",
            ),
        ),
        "accuracy": (
            ROOT / "docs" / "reference" / "accuracy.md",
            (
                "MoE token-to-owner all-to-all control",
                "3.5231883119115293",
                "0.8958737432590591",
                "sha256:51c77e2499d84d5cf5500a5f5c1143b2979f3d70f1755114c34900b55299a61c",
            ),
        ),
        "knowledge_map": (
            ROOT / "docs" / "guide" / "knowledge-map.md",
            (
                "token dispatch/return + metadata scatter",
                "source→owner `[[1,2],[1,0]]`",
                "仍不含 CUDA/NCCL",
            ),
        ),
        "repo_map": (
            ROOT / "docs" / "guide" / "repo-map.md",
            (
                "variable-split `all_to_all_single`",
                "owner-only expert placement",
                "416-byte logical payload",
            ),
        ),
        "project_index": (
            ROOT / "docs" / "practice" / "project-index.md",
            (
                "第四条 all-to-all fixture",
                "metadata-free 错序差 `0.8958737432590591`",
                "不证明 CUDA/NCCL 或生产性能",
            ),
        ),
        "distributed": (
            ROOT / "docs" / "systems" / "distributed-training.md",
            (
                "五次 `all_to_all_single`",
                "owner→source return",
                "416 logical tensor-payload bytes",
            ),
        ),
        "changelog": (
            ROOT / "CHANGELOG.md",
            (
                "two-process CPU/Gloo MoE token-to-owner all-to-all control",
                "`[[1,2],[1,0]]`",
                "strict report fingerprint 稳定为 `sha256:51c77e24",
                "不外推到 CUDA/NCCL、目标模型或生产性能",
            ),
        ),
    }
    for name, (path, markers) in required_docs.items():
        document = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in document]
        if missing:
            errors.append(f"MoE all-to-all {name} docs missing marker(s): {missing}")

    stale_global_claims = {
        ROOT / "docs" / "guide" / "repo-map.md": (
            "MoE 的真实 collective 只到 replicated global routing input",
        ),
        ROOT / "docs" / "guide" / "knowledge-map.md": (
            "expert-parallel token `all_to_all`/grouped GEMM、distributed backward",
        ),
        ROOT / "docs" / "practice" / "project-index.md": (
            "Transformers Basics 的 MoE 证据现分两条",
        ),
        ROOT / "docs" / "practice" / "labs.md": (
            "即使所有 control 都通过",
        ),
        ROOT / "docs" / "frontier" / "reasoning-long-context-moe.md": (
            "也没有训练 MoE router/MLP",
        ),
    }
    for path, forbidden_markers in stale_global_claims.items():
        document = path.read_text(encoding="utf-8")
        present = [
            marker for marker in forbidden_markers if marker in document
        ]
        if present:
            errors.append(
                f"MoE all-to-all stale global claim(s) in {path.name}: {present}"
            )
    return errors


def check_moe_all_to_all_training_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.from_scratch.moe_all_to_all_training import (
        MOE_ALL_TO_ALL_TRAINING_CONTROL_VERSION,
        run_moe_all_to_all_training_control,
    )

    report = run_moe_all_to_all_training_control()
    runtime = report.get("runtime", {})
    fixture = report.get("fixture", {})
    workers = report.get("worker_reports", [])
    comparison = report.get("comparison", {})
    oracle = report.get("single_process_oracle", {})
    assertions = report.get("assertions", {})
    scope = report.get("scope", {})
    rank_zero = (
        workers[0]
        if isinstance(workers, list)
        and len(workers) == 2
        and isinstance(workers[0], dict)
        else {}
    )
    rank_one = (
        workers[1]
        if isinstance(workers, list)
        and len(workers) == 2
        and isinstance(workers[1], dict)
        else {}
    )
    errors: list[str] = []

    if not (
        report.get("schema_version")
        == MOE_ALL_TO_ALL_TRAINING_CONTROL_VERSION
        and report.get("report_fingerprint")
        == "sha256:f577b29dd9e1ccc6def8c1fa156a7aba40a352d883646911a603c06f5adca67c"
        and runtime.get("backend") == "gloo"
        and runtime.get("device") == "cpu"
        and runtime.get("dtype") == "torch.float64"
        and runtime.get("world_size") == 2
        and runtime.get("process_start_method") == "spawn"
        and runtime.get("rendezvous") == "temporary-file-store"
        and fixture.get("local_hidden_states_by_rank")
        == [[[-1.0], [2.0], [-2.0]], [[1.0]]]
        and fixture.get("local_targets_by_rank")
        == [[[0.25], [-0.5], [1.0]], [[-1.5]]]
        and fixture.get("router_weight") == [[1.0], [-1.0]]
        and fixture.get("expert_ownership") == {"expert_0": 0, "expert_1": 1}
        and fixture.get("loss")
        == "global mean squared error over four scalar targets"
        and fixture.get("learning_rate") == 0.01
        and fixture.get("optimizer") == "SGD without momentum or weight decay"
        and isinstance(assertions, dict)
        and assertions
        and all(assertions.values())
    ):
        errors.append("MoE all-to-all training identity/runtime mismatch")

    expected_call_counts = {
        "autograd_payload_forward_all_to_all_single": 4,
        "autograd_payload_backward_all_to_all_single": 2,
        "nondifferentiable_count_or_metadata_all_to_all_single": 6,
        "router_gradient_all_reduce": 1,
    }
    if not (
        rank_zero.get("rank") == 0
        and rank_one.get("rank") == 1
        and rank_zero.get("owned_expert_id") == 0
        and rank_one.get("owned_expert_id") == 1
        and rank_zero.get("selected_expert_indices") == [1, 0, 1]
        and rank_one.get("selected_expert_indices") == [0]
        and rank_zero.get("source_to_owner_counts") == [1, 2]
        and rank_one.get("source_to_owner_counts") == [1, 0]
        and rank_zero.get("owner_from_source_counts") == [1, 1]
        and rank_one.get("owner_from_source_counts") == [2, 0]
        and rank_zero.get("authored_collective_call_counts")
        == expected_call_counts
        and rank_one.get("authored_collective_call_counts")
        == expected_call_counts
        and rank_zero.get("router_gradient_before_all_reduce")
        == [[1.8045724077794292], [-1.8045724077794323]]
        and rank_one.get("router_gradient_before_all_reduce")
        == [[0.48585685772479353], [-0.48585685772479265]]
        and rank_zero.get("router_gradient_after_all_reduce")
        == [[2.2904292655042227], [-2.290429265504225]]
        and rank_one.get("router_gradient_after_all_reduce")
        == rank_zero.get("router_gradient_after_all_reduce")
        and rank_zero.get("owned_expert_weight_gradient")
        == [[6.460938946431114]]
        and rank_one.get("owned_expert_weight_gradient")
        == [[-7.209951147135929]]
        and rank_zero.get("owned_expert_bias_gradient") == [4.045645560316248]
        and rank_one.get("owned_expert_bias_gradient") == [4.325729248768723]
    ):
        errors.append("MoE all-to-all training worker gradient/collective mismatch")

    if not (
        comparison.get("distributed_outputs_before_step_by_global_token_id")
        == [
            [3.5231883119115293],
            [4.419062055170588],
            [6.874096530265359],
            [2.201992694944706],
        ]
        and comparison.get("distributed_outputs_after_step_by_global_token_id")
        == [
            [3.4025704512978336],
            [4.245112885397256],
            [6.678486844293563],
            [2.097729901341357],
        ]
        and comparison.get("distributed_hidden_gradients_by_global_token_id")
        == [
            [-5.699177157478319],
            [5.2215645378941495],
            [-9.378932784079748],
            [4.232418063852349],
        ]
        and math.isclose(
            comparison.get("distributed_global_mean_loss_before_step", -1),
            20.78017329703821,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and math.isclose(
            comparison.get("distributed_global_mean_loss_after_step", -1),
            19.41091750734501,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and comparison.get("output_before_step_max_abs_difference") == 0.0
        and comparison.get("output_after_step_max_abs_difference") == 0.0
        and comparison.get("hidden_gradient_max_abs_difference") == 0.0
        and comparison.get("router_gradient_max_abs_difference_by_rank")
        == [0.0, 0.0]
        and comparison.get("owned_expert_gradient_max_abs_difference_by_rank")
        == [0.0, 0.0]
        and comparison.get("post_step_parameter_max_abs_difference_by_rank")
        == [0.0, 0.0]
        and oracle.get("router_weight_after_step")
        == [[0.9770957073449578], [-0.9770957073449578]]
        and oracle.get("expert_weights_after_step")
        == [[[1.9353906105356888]], [[-2.927900488528641]]]
        and oracle.get("expert_biases_after_step")
        == [[0.45954354439683753], [0.9567427075123127]]
    ):
        errors.append("MoE all-to-all training oracle/update mismatch")

    expected_scope = {
        "real_two_process_same_host_gloo_process_group_executed": True,
        "owner_only_expert_parameter_placement_executed": True,
        "variable_split_token_and_metadata_dispatch_return_executed": True,
        "authored_autograd_all_to_all_forward_backward_executed": True,
        "reverse_split_hidden_and_gate_gradient_communication_executed": True,
        "replicated_router_gradient_sum_all_reduce_executed": True,
        "owner_local_expert_parameter_gradients_executed": True,
        "one_sgd_optimizer_step_executed": True,
        "post_step_distributed_forward_evaluation_executed": True,
        "single_process_global_mean_mse_oracle_compared": True,
        "capacity_drop_reroute_or_dropless_executed": False,
        "pytorch_distributed_nn_functional_wrapper_executed": False,
        "torch_distributed_autograd_rpc_context_executed": False,
        "ddp_fsdp_zero_tensor_or_pipeline_parallel_executed": False,
        "optimizer_momentum_weight_decay_or_state_resume_executed": False,
        "cuda_nccl_multi_node_or_remote_host_executed": False,
        "wire_bytes_protocol_overhead_or_collective_profiler_measured": False,
        "deepseek_qwen_or_other_checkpoint_reproduced": False,
        "throughput_latency_memory_scaling_convergence_or_quality_proved": False,
    }
    if scope != expected_scope:
        errors.append("MoE all-to-all training evidence scope mismatch")

    source = (
        SRC / "about_llm" / "from_scratch" / "moe_all_to_all_training.py"
    ).read_text(encoding="utf-8")
    script = (
        ROOT
        / "projects"
        / "transformers-basics"
        / "moe_all_to_all_training_control.py"
    ).read_text(encoding="utf-8")
    test = (ROOT / "tests" / "test_moe_all_to_all_training.py").read_text(
        encoding="utf-8"
    )
    source_markers = (
        '"about-llm.moe-all-to-all-training-control.v1"',
        "class _VariableSplitAllToAll(torch.autograd.Function)",
        "output_split_sizes=list(input_splits)",
        "input_split_sizes=list(output_splits)",
        "dist.all_reduce(router_gradient, op=dist.ReduceOp.SUM)",
        '"capacity_drop_reroute_or_dropless_executed": False',
        '"pytorch_distributed_nn_functional_wrapper_executed": False',
        "allow_nan=False",
    )
    script_markers = (
        "run_moe_all_to_all_training_control",
        "allow_nan=False",
    )
    test_markers = (
        "test_two_process_gloo_moe_all_to_all_forward_backward_and_sgd",
        "f577b29dd9e1ccc6def8c1fa156a7aba40a352d883646911a603c06f5adca67c",
        "router_gradient_before_all_reduce",
        "distributed_hidden_gradients_by_global_token_id",
    )
    for label, text, markers in (
        ("source", source, source_markers),
        ("script", script, script_markers),
        ("test", test, test_markers),
    ):
        missing = [marker for marker in markers if marker not in text]
        if missing:
            errors.append(
                f"MoE all-to-all training {label} missing marker(s): {missing}"
            )

    required_docs = {
        "readme": (
            ROOT / "README.md",
            (
                "autograd-enabled reverse-split all-to-all",
                "20.78017329703821→19.41091750734501",
                "不外推 CUDA/NCCL",
            ),
        ),
        "project": (
            ROOT / "projects" / "transformers-basics" / "README.md",
            (
                "Two-process Gloo all-to-all forward/backward + SGD control",
                "global-mean MSE",
                "router gradient SUM all-reduce",
                "owner expert gradients",
            ),
        ),
        "project_page": (
            ROOT / "docs" / "practice" / "projects" / "transformers-basics.md",
            (
                "moe_all_to_all_training_control.py",
                "reverse-split backward",
                "20.78017329703821",
                "不等于 DDP 或生产 EP",
            ),
        ),
        "frontier": (
            ROOT / "docs" / "frontier" / "reasoning-long-context-moe.md",
            (
                "authored autograd Function",
                "backward 交换 reverse splits",
                "19.41091750734501",
            ),
        ),
        "architecture": (
            ROOT / "docs" / "core" / "architectures-interpretability.md",
            (
                "owner expert gradient 留在 owner",
                "replicated router gradient",
                "global-token mean",
            ),
        ),
        "deepseek": (
            ROOT / "docs" / "models" / "deepseek.md",
            (
                "不是 DeepSeekMoE training",
                "authored reverse-split autograd",
                "不含 capacity 或目标 checkpoint",
            ),
        ),
        "qwen": (
            ROOT / "docs" / "models" / "qwen.md",
            (
                "all-to-all backward + router-gradient all-reduce",
                "不是 Qwen MoE training",
                "不证明 Qwen optimizer",
            ),
        ),
        "labs": (
            ROOT / "docs" / "practice" / "labs.md",
            (
                "moe_all_to_all_training_control.py",
                "`4/2/6/1` collective ledger",
                "sha256:f577b29dd9e1ccc6",
            ),
        ),
        "interview": (
            ROOT / "docs" / "career" / "interview-questions.md",
            (
                "owner-only MoE backward 中哪些梯度需要 collective",
                "router 要跨 source ranks 求和",
                "expert 参数梯度不应再全局 all-reduce",
            ),
        ),
        "accuracy": (
            ROOT / "docs" / "reference" / "accuracy.md",
            (
                "MoE all-to-all training control",
                "2.2904292655042227",
                "20.78017329703821→19.41091750734501",
                "sha256:f577b29dd9e1ccc6def8c1fa156a7aba40a352d883646911a603c06f5adca67c",
            ),
        ),
        "knowledge_map": (
            ROOT / "docs" / "guide" / "knowledge-map.md",
            (
                "第三条 two-process Gloo training control",
                "reverse all-to-all hidden/gate gradients",
                "第四条 capacity-aware two-process Gloo training control",
            ),
        ),
        "repo_map": (
            ROOT / "docs" / "guide" / "repo-map.md",
            (
                "authored autograd reverse-split",
                "owner expert + synchronized replicated router",
                "它自身仍不含 capacity、DDP 或 CUDA",
            ),
        ),
        "project_index": (
            ROOT / "docs" / "practice" / "project-index.md",
            (
                "第五条 all-to-all training fixture",
                "global mean loss `20.78017329703821→19.41091750734501`",
                "不借用为 CUDA/NCCL 或目标模型训练证据",
            ),
        ),
        "distributed": (
            ROOT / "docs" / "systems" / "distributed-training.md",
            (
                "reverse-split `all_to_all_single` backward",
                "owner expert gradient 不做 data-parallel all-reduce",
                "router gradient 做 SUM all-reduce",
            ),
        ),
        "changelog": (
            ROOT / "CHANGELOG.md",
            (
                "two-process CPU/Gloo MoE all-to-all forward/backward + SGD control",
                "20.78017329703821→19.41091750734501",
                "strict report fingerprint 稳定为 `sha256:f577b29d",
                "不外推 CUDA/NCCL、目标模型、收敛或生产性能",
            ),
        ),
    }
    for name, (path, markers) in required_docs.items():
        document = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in document]
        if missing:
            errors.append(
                f"MoE all-to-all training {name} docs missing marker(s): {missing}"
            )
    return errors


def check_moe_all_to_all_capacity_training_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.from_scratch.moe_all_to_all_training import (
        MOE_ALL_TO_ALL_CAPACITY_TRAINING_CONTROL_VERSION,
        run_moe_all_to_all_capacity_training_control,
    )

    report = run_moe_all_to_all_capacity_training_control()
    runtime = report.get("runtime", {})
    fixture = report.get("fixture", {})
    workers = report.get("worker_reports", [])
    comparison = report.get("comparison", {})
    oracle = report.get("single_process_oracle", {})
    assertions = report.get("assertions", {})
    scope = report.get("scope", {})
    rank_zero = (
        workers[0]
        if isinstance(workers, list)
        and len(workers) == 2
        and isinstance(workers[0], dict)
        else {}
    )
    rank_one = (
        workers[1]
        if isinstance(workers, list)
        and len(workers) == 2
        and isinstance(workers[1], dict)
        else {}
    )
    errors: list[str] = []

    if not (
        report.get("schema_version")
        == MOE_ALL_TO_ALL_CAPACITY_TRAINING_CONTROL_VERSION
        and report.get("report_fingerprint")
        == "sha256:33f11f199b9668c3600ce870cd8369c965cf9daad4bb716fe57fdb751373042e"
        and runtime.get("backend") == "gloo"
        and runtime.get("device") == "cpu"
        and runtime.get("dtype") == "torch.float64"
        and runtime.get("world_size") == 2
        and runtime.get("process_start_method") == "spawn"
        and runtime.get("rendezvous") == "temporary-file-store"
        and fixture.get("local_hidden_states_by_rank")
        == [[[-1.0], [2.0], [-2.0]], [[1.0]]]
        and fixture.get("local_targets_by_rank")
        == [[[0.25], [-0.5], [1.0]], [[-1.5]]]
        and fixture.get("router_weight") == [[1.0], [-1.0]]
        and fixture.get("expert_ownership") == {"expert_0": 0, "expert_1": 1}
        and fixture.get("top_k") == 1
        and fixture.get("capacity_factor") == 0.5
        and fixture.get("expert_capacity") == 1
        and fixture.get("capacity_group")
        == "all four active tokens across both ranks"
        and fixture.get("overflow_policy")
        == "drop by selected probability, then global token id"
        and fixture.get("combine_weight_policy")
        == "preserve selected softmax probability"
        and fixture.get("loss")
        == "global mean squared error over kept and dropped token outputs"
        and fixture.get("learning_rate") == 0.01
        and fixture.get("optimizer") == "SGD without momentum or weight decay"
        and isinstance(assertions, dict)
        and assertions
        and all(assertions.values())
    ):
        errors.append("MoE capacity all-to-all training identity/runtime mismatch")

    expected_call_counts = {
        "autograd_payload_forward_all_to_all_single": 4,
        "autograd_payload_backward_all_to_all_single": 2,
        "nondifferentiable_count_or_metadata_all_to_all_single": 6,
        "router_gradient_all_reduce": 1,
        "capacity_route_all_gather": 4,
    }
    if not (
        rank_zero.get("rank") == 0
        and rank_one.get("rank") == 1
        and rank_zero.get("global_keep_mask")
        == [False, True, True, False]
        and rank_one.get("global_keep_mask")
        == [False, True, True, False]
        and rank_zero.get("local_keep_mask") == [False, True, True]
        and rank_one.get("local_keep_mask") == [False]
        and rank_zero.get("selected_counts_by_expert") == [2, 2]
        and rank_zero.get("kept_counts_by_expert") == [1, 1]
        and rank_zero.get("dropped_assignments") == 2
        and rank_one.get("dropped_assignments") == 2
        and rank_zero.get("source_to_owner_counts") == [1, 1]
        and rank_one.get("source_to_owner_counts") == [0, 0]
        and rank_zero.get("owner_from_source_counts") == [1, 0]
        and rank_one.get("owner_from_source_counts") == [1, 0]
        and rank_one.get("return_arrival_metadata") == []
        and rank_zero.get("authored_collective_call_counts")
        == expected_call_counts
        and rank_one.get("authored_collective_call_counts")
        == expected_call_counts
        and rank_zero.get("router_gradient_before_all_reduce")
        == [[1.1172448546425442], [-1.1172448546425469]]
        and rank_one.get("router_gradient_before_all_reduce")
        == [[0.0], [0.0]]
        and rank_zero.get("router_gradient_after_all_reduce")
        == [[1.1172448546425442], [-1.1172448546425469]]
        and rank_one.get("router_gradient_after_all_reduce")
        == rank_zero.get("router_gradient_after_all_reduce")
        and rank_zero.get("owned_expert_weight_gradient")
        == [[4.830586772229733]]
        and rank_one.get("owned_expert_weight_gradient")
        == [[-5.768443796734413]]
        and rank_zero.get("owned_expert_bias_gradient")
        == [2.4152933861148664]
        and rank_one.get("owned_expert_bias_gradient")
        == [2.8842218983672065]
    ):
        errors.append(
            "MoE capacity all-to-all training worker/collective mismatch"
        )

    if not (
        comparison.get("distributed_outputs_before_step_by_global_token_id")
        == [[0.0], [4.419062055170588], [6.874096530265359], [0.0]]
        and comparison.get("distributed_outputs_after_step_by_global_token_id")
        == [[0.0], [4.29693726711294], [6.726949482533174], [0.0]]
        and comparison.get("distributed_hidden_gradients_by_global_token_id")
        == [[0.0], [5.2215645378941495], [-9.378932784079748], [0.0]]
        and math.isclose(
            comparison.get("distributed_global_mean_loss_before_step", -1),
            15.253670387373656,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and math.isclose(
            comparison.get("distributed_global_mean_loss_after_step", -1),
            14.530264380025987,
            rel_tol=0,
            abs_tol=1e-15,
        )
        and comparison.get("output_before_step_max_abs_difference") == 0.0
        and comparison.get("output_after_step_max_abs_difference") == 0.0
        and comparison.get("hidden_gradient_max_abs_difference") == 0.0
        and comparison.get("router_gradient_max_abs_difference_by_rank")
        == [0.0, 0.0]
        and comparison.get("owned_expert_gradient_max_abs_difference_by_rank")
        == [0.0, 0.0]
        and comparison.get("post_step_parameter_max_abs_difference_by_rank")
        == [0.0, 0.0]
        and oracle.get("router_weight_after_step")
        == [[0.9888275514535746], [-0.9888275514535745]]
        and oracle.get("expert_weights_after_step")
        == [[[1.9516941322777026]], [[-2.9423155620326558]]]
        and oracle.get("expert_biases_after_step")
        == [[0.47584706613885136], [0.9711577810163279]]
    ):
        errors.append("MoE capacity all-to-all training oracle/update mismatch")

    expected_scope = {
        "real_two_process_same_host_gloo_process_group_executed": True,
        "global_score_priority_drop_capacity_collective_executed": True,
        "owner_only_expert_parameter_placement_executed": True,
        "kept_only_variable_split_dispatch_return_executed": True,
        "zero_assignment_source_rank_forward_backward_executed": True,
        "authored_autograd_reverse_all_to_all_backward_executed": True,
        "dropped_token_zero_output_and_task_gradient_executed": True,
        "replicated_router_gradient_sum_all_reduce_executed": True,
        "owner_local_expert_parameter_gradients_executed": True,
        "one_sgd_optimizer_step_executed": True,
        "post_step_distributed_capacity_forward_executed": True,
        "single_process_capacity_training_oracle_compared": True,
        "reroute_dropless_shared_or_fine_grained_experts_executed": False,
        "ddp_fsdp_zero_tensor_or_pipeline_parallel_executed": False,
        "optimizer_momentum_weight_decay_or_state_resume_executed": False,
        "cuda_nccl_multi_node_or_remote_host_executed": False,
        "wire_bytes_protocol_overhead_or_collective_profiler_measured": False,
        "deepseek_qwen_or_other_checkpoint_reproduced": False,
        "throughput_latency_memory_scaling_convergence_or_quality_proved": False,
    }
    if scope != expected_scope:
        errors.append("MoE capacity all-to-all training evidence scope mismatch")

    source = (
        SRC / "about_llm" / "from_scratch" / "moe_all_to_all_training.py"
    ).read_text(encoding="utf-8")
    script = (
        ROOT
        / "projects"
        / "transformers-basics"
        / "moe_all_to_all_capacity_training_control.py"
    ).read_text(encoding="utf-8")
    test = (
        ROOT / "tests" / "test_moe_all_to_all_capacity_training.py"
    ).read_text(encoding="utf-8")
    source_markers = (
        '"about-llm.moe-all-to-all-capacity-training-control.v1"',
        "def _collective_capacity_mask(",
        "def _pack_kept_dispatch(",
        "def _restore_kept_source_order(",
        "returned_float.sum() * 0.0",
        '"zero_assignment_source_rank_forward_backward_executed": True',
        '"cuda_nccl_multi_node_or_remote_host_executed": False',
        "allow_nan=False",
    )
    script_markers = (
        "run_moe_all_to_all_capacity_training_control",
        "allow_nan=False",
    )
    test_markers = (
        "test_capacity_drop_kept_only_all_to_all_backward_and_sgd",
        "33f11f199b9668c3600ce870cd8369c965cf9daad4bb716fe57fdb751373042e",
        "distributed_hidden_gradients_by_global_token_id",
        "router_gradient_before_all_reduce",
    )
    for label, text, markers in (
        ("source", source, source_markers),
        ("script", script, script_markers),
        ("test", test, test_markers),
    ):
        missing = [marker for marker in markers if marker not in text]
        if missing:
            errors.append(
                "MoE capacity all-to-all training "
                f"{label} missing marker(s): {missing}"
            )

    required_docs = {
        "readme": (
            ROOT / "README.md",
            (
                "capacity-aware all-to-all training control",
                "15.253670387373656→14.530264380025987",
                "zero-assignment source rank",
            ),
        ),
        "project": (
            ROOT / "projects" / "transformers-basics" / "README.md",
            (
                "capacity + all-to-all backward + SGD control",
                "global keep mask `[F,T,T,F]`",
                "sha256:33f11f199b9668c",
            ),
        ),
        "project_page": (
            ROOT / "docs" / "practice" / "projects" / "transformers-basics.md",
            (
                "moe_all_to_all_capacity_training_control.py",
                "15.253670387373656→14.530264380025987",
                "zero-assignment source rank",
            ),
        ),
        "frontier": (
            ROOT / "docs" / "frontier" / "reasoning-long-context-moe.md",
            (
                "Capacity、owner dispatch 与 backward 同图",
                "global keep mask `[F,T,T,F]`",
                "33f11f199b9668c",
            ),
        ),
        "architecture": (
            ROOT / "docs" / "core" / "architectures-interpretability.md",
            (
                "capacity-aware training fixture",
                "dropped token 的 routed output 与 task hidden gradient 都为 0",
            ),
        ),
        "deepseek": (
            ROOT / "docs" / "models" / "deepseek.md",
            (
                "capacity-aware all-to-all training control",
                "仍不是 DeepSeekMoE training",
            ),
        ),
        "qwen": (
            ROOT / "docs" / "models" / "qwen.md",
            (
                "capacity-aware all-to-all training control",
                "仍不是 Qwen MoE training",
            ),
        ),
        "labs": (
            ROOT / "docs" / "practice" / "labs.md",
            (
                "moe_all_to_all_capacity_training_control.py",
                "`[F,T,T,F]`",
                "sha256:33f11f199b9668c",
            ),
        ),
        "interview": (
            ROOT / "docs" / "career" / "interview-questions.md",
            (
                "capacity drop 与 expert-parallel backward 放在同一图",
                "zero-size collective graph edge",
            ),
        ),
        "accuracy": (
            ROOT / "docs" / "reference" / "accuracy.md",
            (
                "MoE capacity all-to-all training control",
                "15.253670387373656→14.530264380025987",
                "sha256:33f11f199b9668c3600ce870cd8369c965cf9daad4bb716fe57fdb751373042e",
            ),
        ),
        "knowledge_map": (
            ROOT / "docs" / "guide" / "knowledge-map.md",
            (
                "第四条 capacity-aware two-process Gloo training control",
                "zero-assignment source",
            ),
        ),
        "repo_map": (
            ROOT / "docs" / "guide" / "repo-map.md",
            (
                "capacity-aware training fixture",
                "kept-only all-to-all backward",
            ),
        ),
        "project_index": (
            ROOT / "docs" / "practice" / "project-index.md",
            (
                "第六条 capacity-aware all-to-all training fixture",
                "15.253670387373656→14.530264380025987",
            ),
        ),
        "distributed": (
            ROOT / "docs" / "systems" / "distributed-training.md",
            (
                "第四条 `moe_all_to_all_capacity_training_control.py`",
                "全零 source→owner splits `[0,0]`",
                "Dropped tokens 的 task hidden gradient 为 0",
            ),
        ),
        "changelog": (
            ROOT / "CHANGELOG.md",
            (
                "MoE global-capacity + all-to-all backward + SGD control",
                "15.253670387373656→14.530264380025987",
                "sha256:33f11f19",
            ),
        ),
    }
    for name, (path, markers) in required_docs.items():
        document = path.read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in document]
        if missing:
            errors.append(
                "MoE capacity all-to-all training "
                f"{name} docs missing marker(s): {missing}"
            )

    stale_aggregate_claims = {
        ROOT / "docs" / "guide" / "knowledge-map.md": (
            "训练 control仍无 capacity+backward 同图",
        ),
        ROOT / "docs" / "frontier" / "reasoning-long-context-moe.md": (
            "把 capacity+dispatch+distributed backward 合成同一训练图",
        ),
        ROOT / "docs" / "guide" / "repo-map.md": (
            "它仍是无 capacity/DDP/CUDA 的同机 Gloo control",
        ),
    }
    for path, forbidden_markers in stale_aggregate_claims.items():
        document = path.read_text(encoding="utf-8")
        present = [
            marker for marker in forbidden_markers if marker in document
        ]
        if present:
            errors.append(
                "MoE capacity all-to-all training stale aggregate claim(s) "
                f"in {path.name}: {present}"
            )
    return errors


def check_kv_example() -> list[str]:
    sys.path.insert(0, str(SRC))
    from about_llm.inference import estimate_kv_cache_bytes

    actual = estimate_kv_cache_bytes(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        tokens=8192,
        bytes_per_element=2,
    )
    expected = 1024**3
    if actual != expected:
        return [f"KV example mismatch: expected {expected} bytes, got {actual}"]
    return []


def check_kv_allocator_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.inference import KVCapacityError, PagedKVAllocator

    errors: list[str] = []
    allocator = PagedKVAllocator(total_blocks=3, block_size_tokens=4)
    allocator.create_sequence("accuracy-a")
    initial = allocator.append("accuracy-a", 6)
    allocator.fork_sequence("accuracy-a", "accuracy-b")
    after_fork = allocator.report()
    cow = allocator.append("accuracy-a", 1)
    allocator.append("accuracy-b", 2)
    before_failure = allocator.report()
    state_before_failure = allocator.sequence_state("accuracy-a")
    try:
        allocator.append("accuracy-a", 2)
    except KVCapacityError:
        pass
    else:
        errors.append("Paged KV capacity fixture unexpectedly succeeded")
    if not (
        initial.physical_block_ids == (0, 1)
        and after_fork.logical_block_references == 4
        and after_fork.sharing_saved_blocks == 2
        and after_fork.logical_tokens == 12
        and after_fork.physical_token_values == 6
        and cow.copied_partial_block == (1, 2)
        and before_failure.logical_tokens == 15
        and before_failure.physical_token_values == 11
        and before_failure.allocated_token_slots == 12
        and before_failure.internal_fragmentation_slots == 1
        and allocator.sequence_state("accuracy-a") == state_before_failure
        and allocator.report() == before_failure
    ):
        errors.append("Paged KV block sharing/COW/atomic-failure fixture mismatch")

    demo = (
        ROOT / "projects" / "inference-serving" / "kv_block_allocator_toy.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"metadata_only_cpu_simulation": True',
        '"real_kv_tensor_values_stored_or_copied": False',
        '"paged_attention_gpu_kernel_executed": False',
        '"eviction_preemption_or_swap_implemented": False',
        '"latency_throughput_or_vram_proved": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(f"Paged KV allocator toy missing scope marker(s): {missing_scope}")
    return errors


def check_scaling_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.scaling import (
        compute_optimal_under_power_law,
        estimate_dense_training_flops,
    )

    errors: list[str] = []
    flops = estimate_dense_training_flops(1e9, 20e9)
    if flops != 1.2e20:
        errors.append(f"6ND example mismatch: expected 1.2e20 FLOPs, got {flops}")

    symmetric = compute_optimal_under_power_law(
        100,
        parameter_coefficient=1,
        data_coefficient=1,
        parameter_exponent=1,
        data_exponent=1,
        flops_per_parameter_token=1,
    )
    if not (
        math.isclose(symmetric.num_parameters, 10)
        and math.isclose(symmetric.training_tokens, 10)
        and math.isclose(symmetric.modeled_loss, 0.2)
    ):
        errors.append(f"symmetric compute-optimal example mismatch: {symmetric}")
    return errors


def check_preference_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.finetuning.governance import load_sft_governance_policy
    from about_llm.finetuning.near_duplicate import NearDuplicateProfile
    from about_llm.finetuning.preference import (
        bradley_terry_loss,
        dpo_logit,
        dpo_loss,
        sequence_log_probability,
    )
    from about_llm.finetuning.preference_data import (
        audit_preference_records,
        load_preference_records,
        validate_dpo_training_records,
        validate_dpo_training_subset,
    )
    from about_llm.finetuning.preference_evaluation import (
        audit_preference_judgments,
        load_preference_judgments,
        summarize_preference_judgments,
    )
    from about_llm.finetuning.preference_governance import (
        audit_preference_governance,
    )
    from about_llm.finetuning.preference_near_duplicate import (
        audit_preference_near_duplicates,
    )
    from about_llm.finetuning.preference_readiness import (
        PreferenceTrainingReadinessReport,
        load_preference_training_readiness,
        validate_preference_training_readiness,
    )

    errors: list[str] = []
    if not math.isclose(bradley_terry_loss(1.0, 1.0), math.log(2)):
        errors.append("Bradley-Terry equal-reward example must have log(2) loss")
    if not math.isclose(
        dpo_loss(
            chosen_policy_logp=-2,
            rejected_policy_logp=-4,
            chosen_reference_logp=-3,
            rejected_reference_logp=-5,
            beta=0.1,
        ),
        math.log(2),
    ):
        errors.append("reference-equivalent DPO example must have log(2) loss")
    logit = dpo_logit(
        chosen_policy_logp=-2,
        rejected_policy_logp=-5,
        chosen_reference_logp=-2.5,
        rejected_reference_logp=-3.5,
        beta=0.2,
    )
    if not math.isclose(logit, 0.4):
        errors.append(f"DPO reference-relative logit mismatch: expected 0.4, got {logit}")
    if not (
        math.isclose(sequence_log_probability([-0.1, -0.2, -0.3]), -0.6)
        and math.isclose(
            sequence_log_probability([-0.1, -0.2, -0.3], reduction="mean"), -0.2
        )
    ):
        errors.append("sequence log-probability reduction example mismatch")
    records = load_preference_records(
        ROOT / "projects" / "single-gpu-finetuning" / "preference.example.jsonl"
    )
    report = audit_preference_records(records)
    training = load_preference_records(
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "preference.train.example.jsonl"
    )
    train_report = validate_dpo_training_records(training)
    binding = validate_dpo_training_subset(training, records)
    near = audit_preference_near_duplicates(
        records,
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=5,
        threshold=0.9,
    )
    governance = audit_preference_governance(
        records,
        policy=load_sft_governance_policy(
            ROOT
            / "projects"
            / "single-gpu-finetuning"
            / "governance-policy.example.json"
        ),
        evaluated_at=datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
    )
    readiness = PreferenceTrainingReadinessReport.from_reports(
        binding, near, governance
    )
    readiness_fixture = load_preference_training_readiness(
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "preference-training-readiness.example.json"
    )
    rebound = validate_preference_training_readiness(training, readiness)
    judgments = load_preference_judgments(
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "preference-judgments.example.jsonl"
    )
    judgment_audit = audit_preference_judgments(
        records,
        judgments,
        judgments_per_pair=4,
        minimum_judgments_per_order=2,
    )
    judgment_summary = summarize_preference_judgments(
        judgments,
        judgment_audit,
        bootstrap_samples=2_000,
        bootstrap_seed=17,
    )
    second_row = records[1].to_dpo_row()
    scope = report.to_dict()["scope"]
    readiness_scope = readiness.to_dict()["scope"]
    if not (
        report.gate_passed
        and report.record_count == 4
        and report.label_counts == {"a": 1, "b": 2, "tie": 1}
        and report.preferred_display_position_counts
        == {"first": 2, "second": 1, "tie": 1}
        and train_report.gate_passed
        and rebound.manifest_fingerprint == train_report.manifest_fingerprint
        and readiness_fixture == readiness
        and readiness.binary_train_record_count == 2
        and readiness_scope["trainer_needs_held_out_access"] is False
        and readiness_scope["legal_permission_or_consent_verified"] is False
        and readiness_scope["limited_sensitive_candidate_scan_executed"] is True
        and second_row["chosen"][0]["content"] == "good beta answer"
        and second_row["rejected"][0]["content"] == "bad beta answer"
        and scope["position_bias_estimated"] is False
        and scope["rubric_quality_verified"] is False
        and scope["tokenization_or_truncation_verified"] is False
    ):
        errors.append(f"preference data/audit example mismatch: {report}")
    judgment_scope = judgment_summary.to_dict()["scope"]
    if not (
        judgment_audit.gate_passed
        and judgment_audit.judgment_count == 8
        and judgment_audit.selected_case_ids
        == ("pref-validation-tie", "pref-test-gamma")
        and judgment_summary.label_counts == {"a": 1, "b": 5, "tie": 2}
        and judgment_summary.pairwise_agreement_numerator == 7
        and judgment_summary.pairwise_agreement_denominator == 12
        and math.isclose(judgment_summary.pairwise_agreement, 7 / 12)
        and math.isclose(judgment_summary.fleiss_expected_agreement, 30 / 64)
        and judgment_summary.fleiss_kappa is not None
        and math.isclose(judgment_summary.fleiss_kappa, 11 / 51)
        and math.isclose(judgment_summary.mean_pair_position_effect or -1, 0.5)
        and judgment_summary.position_effect_pair_count == 2
        and judgment_scope["authored_fixture_is_human_evidence"] is False
        and judgment_scope["causal_position_bias_identified"] is False
        and judgment_scope["random_assignment_verified"] is False
    ):
        errors.append(
            "preference judgment agreement/position example mismatch: "
            f"{judgment_summary}"
        )
    return errors


def check_roofline_example() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.inference import roofline_lower_bound

    bound = roofline_lower_bound(
        flop_count=100,
        bytes_moved=100,
        effective_flops_per_second=100,
        effective_bytes_per_second=10,
    )
    errors: list[str] = []
    if not (
        bound.bottleneck == "memory"
        and math.isclose(bound.compute_seconds, 1)
        and math.isclose(bound.memory_seconds, 10)
        and math.isclose(bound.lower_bound_seconds, 10)
        and math.isclose(bound.arithmetic_intensity, 1)
        and math.isclose(bound.ridge_point, 10)
    ):
        errors.append(f"roofline example mismatch: {bound}")
    return errors


def check_multimodal_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.evaluation import box_iou, character_error_rate, temporal_iou

    errors: list[str] = []
    if not math.isclose(character_error_rate("语言模型", "语言大模"), 0.5):
        errors.append("multimodal CER example mismatch")
    if not math.isclose(box_iou((0, 0, 2, 2), (1, 1, 3, 3)), 1 / 7):
        errors.append("continuous box IoU example mismatch")
    if not math.isclose(temporal_iou((0, 10), (5, 15)), 1 / 3):
        errors.append("continuous temporal IoU example mismatch")
    return errors


def check_llmops_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

    first = {"model": "m@rev", "generation": {"temperature": 0, "max_tokens": 64}}
    reordered = {"generation": {"max_tokens": 64, "temperature": 0}, "model": "m@rev"}
    errors: list[str] = []
    expected_bytes = (
        b'{"generation":{"max_tokens":64,"temperature":0},"model":"m@rev"}'
    )
    if canonical_json_bytes(first) != expected_bytes:
        errors.append("canonical JSON example mismatch")
    if artifact_fingerprint(first) != artifact_fingerprint(reordered):
        errors.append("artifact fingerprint must ignore mapping insertion order")
    changed = {"model": "m@rev-2", "generation": {"temperature": 0, "max_tokens": 64}}
    if artifact_fingerprint(first) == artifact_fingerprint(changed):
        errors.append("artifact fingerprint example failed to identify changed revision")
    return errors


def check_agent_safety_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.agents import (
        AgentRuntime,
        CapabilityPolicy,
        ExecutionContext,
        ExecutionStatus,
        ResourceRef,
        SideEffect,
        SQLiteLedger,
        Tool,
        ToolCall,
        ToolRegistry,
        evaluate_agent_traces,
        load_agent_loop_checkpoint,
        load_trace_cases,
    )
    from about_llm.agents.cli import (
        load_loop_fixtures,
        pause_loop_fixture,
        resume_loop_fixture,
        run_loop_fixtures,
    )

    errors: list[str] = []
    secret = "accuracy-gate-secret"
    mutable = {"nested": {"secret": secret, "items": [1, 2]}}
    call = ToolCall("accuracy-call", "demo", mutable)
    mutable["nested"]["secret"] = "changed"
    if call.arguments["nested"]["secret"] != secret:
        errors.append("Agent ToolCall deep snapshot drifted after caller mutation")
    if not call.fingerprint().startswith("sha256:") or secret in call.fingerprint():
        errors.append("Agent ToolCall fingerprint must be prefixed hash, not plaintext")

    effects: list[str] = []
    tool = Tool(
        "read",
        "accuracy-tool@v1",
        "Read a deterministic fixture resource.",
        SideEffect.READ_ONLY,
        lambda arguments: None,
        lambda arguments: effects.append("attempt") or {"ok": True},
        required_capability="fixture:read",
        resolve_resource=lambda arguments: ResourceRef(
            "tenant-a", "fixture", "resource-1", "fixture@v1"
        ),
    )
    proposal = ToolCall("policy-call", "read", {})
    authorized = ExecutionContext(
        "task", "user", "tenant-a", frozenset({"fixture:read"})
    )
    revoked = ExecutionContext("task", "user", "tenant-a", frozenset())
    runtime = AgentRuntime(
        ToolRegistry([tool]), policy=CapabilityPolicy("accuracy-policy@v1")
    )
    first = runtime.execute(proposal, context=authorized)
    after_revocation = runtime.execute(proposal, context=revoked)
    default_denied = AgentRuntime(ToolRegistry([tool])).execute(
        ToolCall("default-deny", "read", {}), context=authorized
    )
    if not (
        first.status is ExecutionStatus.COMPLETED
        and after_revocation.status is ExecutionStatus.POLICY_DENIED
        and after_revocation.policy_decision.reason_code == "missing_capability"
        and default_denied.status is ExecutionStatus.POLICY_DENIED
        and first.execution_fingerprint != proposal.fingerprint()
        and effects == ["attempt"]
    ):
        errors.append("Agent default-deny/cache re-authorization example mismatch")

    fixture = ROOT / "projects" / "safe-agent" / "trajectory.example.jsonl"
    report = evaluate_agent_traces(load_trace_cases(fixture))
    if not (
        report.gate_passed
        and report.task_success.numerator == 3
        and report.task_success.denominator == 3
        and report.blocked_unsafe_proposals.numerator == 1
        and report.executed_policy_violations.numerator == 0
        and report.policy_over_refusals.numerator == 0
        and report.policy_unjudged_steps.numerator == 0
        and report.unapproved_side_effect_attempts.numerator == 0
        and report.duplicate_applied_effects.numerator == 0
        and report.unresolved_pending_cases.numerator == 0
        and report.step_budget_violation_cases.numerator == 0
        and report.handler_budget_violation_cases.numerator == 0
    ):
        errors.append(f"Agent trajectory fixture mismatch: {report}")
    loop_fixture = ROOT / "projects" / "safe-agent" / "loop.example.jsonl"
    loop_report = run_loop_fixtures(load_loop_fixtures(loop_fixture))
    if not (
        loop_report["passed"] is True
        and loop_report["provider_usage_measured"] is False
        and [case["termination"] for case in loop_report["cases"]]
        == [
            "completed",
            "repeated_action",
            "action_cycle",
            "repeated_error",
            "needs_approval",
        ]
        and loop_report["cases"][0]["final_answer"] == "demo:answer"
        and loop_report["cases"][-1]["handler_attempts"] == 0
    ):
        errors.append(f"Agent typed-loop fixture mismatch: {loop_report}")
    approval_case = load_loop_fixtures(loop_fixture)[-1]
    with tempfile.TemporaryDirectory() as temporary_directory:
        ledger = SQLiteLedger(Path(temporary_directory) / "resume.db")
        paused = pause_loop_fixture(approval_case, ledger)
        if paused.checkpoint is None:
            errors.append("Agent approval pause omitted checkpoint")
        else:
            checkpoint = load_agent_loop_checkpoint(paused.checkpoint.to_dict())
            resumed = resume_loop_fixture(approval_case, checkpoint, ledger)
            if not (
                resumed.completed
                and resumed.model_tokens_used == 4
                and math.isclose(resumed.cost_units_used, 0.2)
                and resumed.handler_attempts == 1
                and [event.status for event in resumed.state.events]
                == ["completed", "passed"]
            ):
                errors.append(f"Agent checkpoint/resume fixture mismatch: {resumed}")
    return errors


def check_agent_outbox_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.agents import (
        EffectRequest,
        OutboxState,
        SQLiteTransactionalOutbox,
    )

    errors: list[str] = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        database = Path(temporary_directory) / "outbox.db"
        first_worker = SQLiteTransactionalOutbox(database)
        effect = EffectRequest(
            effect_id="accuracy-effect",
            execution_fingerprint="sha256:" + "a" * 64,
            destination="simulated-provider",
            payload={"message": "accuracy fixture"},
        )
        created = first_worker.commit_task_effect(
            "accuracy-task",
            {"status": "approved"},
            effect,
            now=10,
        )
        first = first_worker.claim_due(
            "accuracy-worker-a", now=10, lease_seconds=5
        )
        restarted_worker = SQLiteTransactionalOutbox(database)
        second = restarted_worker.claim_due(
            "accuracy-worker-b", now=16, lease_seconds=5
        )
        if first is None or second is None:
            errors.append("Agent outbox lease-expiry fixture omitted a delivery")
        else:
            restarted_worker.mark_delivered(
                second.effect_id,
                "accuracy-worker-b",
                {"provider_effect_id": "simulated-1"},
                now=17,
            )
            record = restarted_worker.get(second.effect_id)
            event_types = [
                event.event_type
                for event in restarted_worker.events(second.effect_id)
            ]
            if not (
                created
                and dict(restarted_worker.task_state("accuracy-task"))
                == {"status": "approved"}
                and record is not None
                and record.state is OutboxState.DELIVERED
                and record.attempt_count == 2
                and first.attempt == 1
                and second.attempt == 2
                and first.provider_idempotency_key
                == second.provider_idempotency_key
                == "accuracy-effect"
                and event_types
                == [
                    "enqueued",
                    "claimed",
                    "lease_expired",
                    "claimed",
                    "delivered",
                ]
            ):
                errors.append(f"Agent transactional-outbox fixture mismatch: {record}")

    demo = (ROOT / "projects" / "safe-agent" / "outbox_demo.py").read_text(
        encoding="utf-8"
    )
    required_scope = (
        '"local SQLite"',
        '"in-memory simulated idempotent provider"',
        '"at-least-once"',
        '"proves_exactly_once_external_effect": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(f"Agent outbox demo missing scope marker(s): {missing_scope}")
    return errors


def check_code_metric_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.evaluation import pass_at_k

    errors: list[str] = []
    if not math.isclose(pass_at_k(num_samples=10, num_correct=2, k=2), 17 / 45):
        errors.append("pass@k combinatorial example mismatch")
    if pass_at_k(num_samples=10, num_correct=0, k=10) != 0:
        errors.append("pass@k zero-correct boundary mismatch")
    if pass_at_k(num_samples=10, num_correct=1, k=10) != 1:
        errors.append("pass@k full-budget boundary mismatch")
    try:
        pass_at_k(num_samples=2, num_correct=1, k=3)
    except ValueError:
        pass
    else:
        errors.append("pass@k must reject k > num_samples")
    return errors


def check_retrieval_metric_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.evaluation import (
        all_evidence_recall_at_k,
        precision_at_k,
        recall_at_k,
    )
    from about_llm.rag import (
        Document,
        RAGTraceCaseBinding,
        RecordedRerankScorer,
        SearchResult,
        audit_rag_generation_traces,
        build_citation_context,
        load_rag_generation_traces,
        load_recorded_answers,
        make_rag_chat_prompt_cost,
        pack_citation_context,
        rerank_authorized_candidates,
        utf8_byte_length,
    )
    from about_llm.rag.cli import (
        MarkdownBM25Pipeline,
        evaluate_answers,
        evaluate_extractive_baseline,
        evaluate_retrieval,
        load_cases,
        load_corpus,
        load_recorded_rerank_scores,
    )

    retrieved = {"q1": ["a", "noise"], "q2": ["a"]}
    relevant = {"q1": {"a", "b"}, "q2": {"a"}}
    required = {"q1": {"a", "b"}, "q2": {"a"}}
    errors: list[str] = []
    if not math.isclose(recall_at_k(retrieved, relevant, k=2), 0.75):
        errors.append("retrieval Recall@k macro example mismatch")
    if not math.isclose(precision_at_k(retrieved, relevant, k=2), 0.75):
        errors.append("retrieval actual-returned Precision@k example mismatch")
    if not math.isclose(all_evidence_recall_at_k(retrieved, required, k=2), 0.5):
        errors.append("retrieval all-evidence query-rate example mismatch")

    project = ROOT / "projects" / "rag-foundations"
    pipeline = MarkdownBM25Pipeline(load_corpus(project / "sample_corpus.jsonl"))
    cases = load_cases(project / "sample_eval.jsonl")
    report = evaluate_retrieval(
        pipeline,
        cases,
        k=3,
    )
    answerable = report["answerable_metrics"]
    no_answer = report["no_answer_metrics"]
    if not (
        report["case_count"] == 5
        and answerable["case_count"] == 3
        and no_answer["case_count"] == 2
        and math.isclose(no_answer["zero_result_accuracy"], 0.5)
        and report["legacy_metric_scope"] == "answerable cases only"
    ):
        errors.append("RAG graded/no-answer fixture denominator mismatch")
    rerank_query = "RAG 为什么要先做 ACL 权限过滤"
    rerank_candidates = pipeline.retrieve(
        rerank_query,
        tenant_id="tenant-a",
        principals=("engineering",),
        top_k=3,
    )
    rerank_scorer = RecordedRerankScorer(
        load_recorded_rerank_scores(project / "reranker-scores.example.jsonl")
    )
    rerank_report = rerank_authorized_candidates(
        rerank_query,
        rerank_candidates,
        rerank_scorer,
        tenant_id="tenant-a",
        principals=("engineering",),
        top_k=2,
        scorer_identity=rerank_scorer.scorer_identity,
    )
    if not (
        [result.document.document_id for result in rerank_report.results]
        == ["chk_bd3e8a6757a7f05c07fdbcc4", "chk_8d8a68a02d85a198190fc293"]
        and rerank_report.authorized_candidate_count == 3
        and rerank_report.to_dict()["scope"]["relevance_quality_verified"] is False
    ):
        errors.append("RAG recorded rerank identity/order fixture mismatch")
    answers = load_recorded_answers(project / "sample_answers.jsonl")
    answer_report = evaluate_answers(
        pipeline,
        cases,
        answers,
    )
    if not (
        answer_report["answered_case_count"] == 3
        and math.isclose(answer_report["coverage"], 0.6)
        and math.isclose(answer_report["action_accuracy"], 1.0)
        and math.isclose(answer_report["recorded_gate_pass_rate"], 1.0)
        and answer_report["claim_verdict_counts"]["supported"] == 4
        and "supplied labels" in answer_report["scope_warning"]
    ):
        errors.append("RAG recorded-answer fixture scope/denominator mismatch")
    extractive_report = evaluate_extractive_baseline(
        pipeline,
        cases,
        candidate_k=20,
        budget_units=12000,
    )
    extractive_artifacts = {
        artifact["query_id"]: artifact for artifact in extractive_report["artifacts"]
    }
    extractive_answers = extractive_report["answer_evaluation"]
    multi_source_spans = extractive_artifacts["metrics-and-entailment"][
        "proposed_spans"
    ]
    if not (
        [
            extractive_artifacts[case.query_id]["action"]
            for case in cases
        ]
        == ["answer", "answer", "answer", "abstain", "abstain"]
        and {
            span["stable_source_id"] for span in multi_source_spans
        }
        == {"rag-security", "rag-evaluation"}
        and math.isclose(
            extractive_artifacts["topical-no-answer"]["coverage"], 2 / 9
        )
        and math.isclose(extractive_answers["coverage"], 0.6)
        and math.isclose(extractive_answers["action_accuracy"], 1.0)
        and math.isclose(extractive_answers["grounded_answer_pass_rate"], 1.0)
        and math.isclose(extractive_answers["recorded_gate_pass_rate"], 1.0)
        and extractive_artifacts["acl-before-ranking"][
            "artifact_fingerprint"
        ].startswith("sha256:")
        and "exact-substring"
        in extractive_artifacts["acl-before-ranking"]["scope_warning"]
        and "only after artifact generation"
        in extractive_report["generator_label_boundary"]
    ):
        errors.append("RAG extractive answer/action/scope fixture mismatch")
    trace_report = audit_rag_generation_traces(
        expected_cases={
            case.query_id: RAGTraceCaseBinding(
                query_sha256=(
                    "sha256:"
                    + hashlib.sha256(case.query.encode("utf-8")).hexdigest()
                ),
                tenant_id=case.tenant_id,
                principals=case.principals,
            )
            for case in cases
        },
        answers=answers,
        traces=load_rag_generation_traces(
            project / "generation-traces.example.jsonl"
        ),
        documents=pipeline.documents,
    )
    trace_payload = trace_report.to_dict()
    if not (
        trace_report.gate_passed
        and trace_report.trace_count == 5
        and trace_report.answer_count == 5
        and not trace_report.findings
        and trace_payload["scope"]["raw_output_claim_semantics_verified"] is False
        and trace_payload["scope"]["remote_model_execution_verified"] is False
    ):
        errors.append("RAG generation trace fixture binding/scope mismatch")

    first = SearchResult(
        Document("first", "short", "tenant", metadata={"source_id": "one"}),
        score=1,
        rank=1,
        source="accuracy-check",
    )
    oversized = SearchResult(
        Document("oversized", "x" * 2000, "tenant", metadata={"source_id": "two"}),
        score=0.5,
        rank=2,
        source="accuracy-check",
    )
    later = SearchResult(
        Document("later", "small", "tenant", metadata={"source_id": "three"}),
        score=0.25,
        rank=3,
        source="accuracy-check",
    )
    target_context = build_citation_context([first, later], tenant_id="tenant")
    byte_budget = utf8_byte_length(target_context.rendered)
    packed = pack_citation_context(
        [first, oversized, later],
        tenant_id="tenant",
        budget_units=byte_budget,
        cost_fn=utf8_byte_length,
        cost_unit="utf8_bytes",
    )
    if not (
        packed.selected_document_ids == ("first", "later")
        and packed.dropped_document_ids == ("oversized",)
        and packed.used_cost_units == byte_budget
        and [decision.reason.value for decision in packed.decisions]
        == ["selected", "budget", "selected"]
    ):
        errors.append("RAG prospective context packing example mismatch")
    non_monotonic = pack_citation_context(
        [first],
        tenant_id="tenant",
        budget_units=5,
        cost_fn=lambda context: 10 if not context else 5,
        cost_unit="synthetic_boundary_units",
    )
    if not (
        non_monotonic.base_cost_units == 10
        and non_monotonic.used_cost_units == 5
        and non_monotonic.selected_document_ids == ("first",)
    ):
        errors.append("RAG non-monotonic tokenizer-boundary example mismatch")
    rendered_user: list[str] = []

    def tokenize_chat(messages: tuple[dict[str, str], ...]) -> list[int]:
        rendered_user.append(messages[1]["content"])
        serialized = " ".join(message["content"] for message in messages)
        return list(range(len(serialized.split())))

    chat_cost = make_rag_chat_prompt_cost(
        system_prompt="follow evidence only",
        query="literal {context}",
        user_prompt_template="question {query} evidence {context}",
        tokenize_messages=tokenize_chat,
        reserved_output_tokens=7,
    )
    if not (
        chat_cost("literal {query}") == 16
        and rendered_user == [
            "question literal {context} evidence literal {query}"
        ]
    ):
        errors.append("RAG target-tokenizer full-chat/reservation example mismatch")
    rag_cli = (SRC / "about_llm" / "rag" / "cli.py").read_text(encoding="utf-8")
    for marker in (
        '"answer-extractive"',
        '"evaluate-extractive"',
        '"pack-tokenized"',
        '"audit-traces"',
        "tokenizer.apply_chat_template",
        '"final_prompt_token_ids"',
        '"model_context_window_verified": False',
        '"tokenizer_files_cryptographically_authenticated": False',
    ):
        if marker not in rag_cli:
            errors.append(f"RAG tokenized packing CLI missing marker: {marker}")
    system_prompt = (project / "system-prompt.example.txt").read_text(
        encoding="utf-8"
    )
    user_template = (project / "user-prompt-template.example.txt").read_text(
        encoding="utf-8"
    )
    if not (
        "不可信数据" in system_prompt
        and user_template.count("{query}") == 1
        and user_template.count("{context}") == 1
    ):
        errors.append("RAG tokenized packing prompt fixture mismatch")
    return errors


def check_conversation_memory_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.conversation import (
        ConversationMemoryLedger,
        MemoryKind,
        MemoryScope,
        MemoryStatus,
    )

    now = datetime(2026, 8, 6, tzinfo=timezone.utc)
    ledger = ConversationMemoryLedger()
    old = ledger.add_fact(
        fact_id="old",
        tenant_id="tenant-a",
        subject_id="user-1",
        key="language",
        value={"code": "zh-CN"},
        kind=MemoryKind.WORKING,
        scope=MemoryScope.SESSION,
        source_event_id="message-1",
        created_at=now,
        confidence=1,
        policy_version="memory-v1",
        expires_at=now + timedelta(hours=1),
    )
    new = ledger.correct_fact(
        previous_fact_id=old.fact_id,
        new_fact_id="new",
        tenant_id="tenant-a",
        subject_id="user-1",
        value={"code": "en"},
        source_event_id="message-2",
        created_at=now + timedelta(minutes=1),
        confidence=1,
    )
    observed_at = now + timedelta(minutes=2)
    errors: list[str] = []
    if ledger.active_facts(
        tenant_id="tenant-a", subject_id="user-1", now=observed_at
    ) != (new,):
        errors.append("conversation correction example must expose only the new fact")
    if ledger.status(
        fact_id=old.fact_id,
        tenant_id="tenant-a",
        subject_id="user-1",
        now=observed_at,
    ) is not MemoryStatus.SUPERSEDED:
        errors.append("conversation correction example must mark old fact superseded")
    if ledger.active_facts(
        tenant_id="tenant-b", subject_id="user-1", now=observed_at
    ):
        errors.append("conversation memory example leaked across tenants")
    ledger.retract_fact(
        retraction_id="retract-new",
        fact_id=new.fact_id,
        tenant_id="tenant-a",
        subject_id="user-1",
        source_event_id="message-3",
        reason="user correction",
        created_at=observed_at,
    )
    if ledger.active_facts(
        tenant_id="tenant-a", subject_id="user-1", now=observed_at
    ):
        errors.append("conversation retraction example must remove the active fact")
    return errors


def check_gpt_model_page(page: Path | None = None) -> list[str]:
    target = page or ROOT / "docs" / "models" / "gpt.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 220:
        errors.append(
            "GPT model page regressed to a product/API summary: "
            f"expected at least 220 non-empty lines, found {nonempty_line_count}"
        )
    required_markers = (
        "公开研究事实",
        "当前产品契约",
        "本地可执行证据",
        "GPT-5.6 Sol、Terra、Luna",
        "response/output item/content part",
        "response.output_text.delta",
        "response.function_call_arguments.done",
        "completed、incomplete 与 failed",
        "本地 replay 契约",
        "openai_responses_replay.py",
        "3,208",
        "12 input / 9 output / 21 total",
        "f2947212c1f67adf6f35bc976264db28c30abe1a32310daa284df42ca5a54686",
        "9cc5964da2517f2076a1c624c2636bd8ca75077b89f024c7710b1b720cbd713e",
        "c4829c19895dcb4013141da3d11b5dc9befee8189210a0901f0cb14c19942579",
        "不证明真实 OpenAI API",
        "不是完整 Responses API",
        "OpenAI SDK",
        "cost per successful task",
        "Create a response",
        "Streaming events",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(f"GPT model page missing research/API/scope marker(s): {missing}")
    return errors


def check_llama_model_page(page: Path | None = None) -> list[str]:
    target = page or ROOT / "docs" / "models" / "llama.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 400:
        errors.append(
            "Llama model page regressed to a family/config summary: "
            f"expected at least 400 non-empty lines, found {nonempty_line_count}"
        )

    required_markers = (
        "L0 前置标签 + L1\N{EN DASH}L5 五级证据阶梯",
        "checkpoint inventory",
        "25,416 bytes",
        "0e0b8c519242d5833d8c11bffc1232b77ad7f301",
        "cdc06052012c47654cfa49dc41a766cdb8801c4dfd469bee6d42774b058beb78",
        "be14f72e9cbf200abb9740acc3049b82dca717d4f1e1eb4a46a8ed439a3ceb99",
        "sha256:74166133716bfebddb444587e9f9a012b4beada923f5209482308ff61194953b",
        "sha256:40b3fe7b2a9c054ea6aa17e9e747d1831b8ae41ee3d55130c916f818acbe4638",
        "upstream_verified",
        "source_fragments_verified",
        "vendor-reported",
        "RMSNorm",
        "RoPE",
        "SwiGLU",
        "GQA",
        "P_{\\text{MLP}}=3dm",
        "M_{KV,\\text{ideal}}",
        "authored_standard_gqa",
        "不是任何 Llama checkpoint",
        "Base、Instruct 与 chat template",
        "P_{LoRA}=r(d_{in}+d_{out})",
        "QLoRA 不等于全流程 4-bit",
        "Transformers 与 vLLM",
        "128k 不等于有效上下文",
        "开放权重不等于 OSI 开源",
        "仓库未执行 Llama 权重",
        "面试与作品集验收",
        "不证明参数量、有效上下文、许可适用、GPU/runtime、质量、性能或生产安全",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "Llama model page missing architecture/release/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_qwen_model_page(page: Path | None = None) -> list[str]:
    target = page or ROOT / "docs" / "models" / "qwen.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 500:
        errors.append(
            "Qwen model page regressed to a family/control catalog: "
            f"expected at least 500 non-empty lines, found {nonempty_line_count}"
        )

    required_markers = (
        "L0 标签与 L1\N{EN DASH}L5 证据阶梯",
        "证据不可拼接原则",
        "Checkpoint inventory",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
        "659-byte",
        "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
        "sha256:ee6f9831a4c4729cf094af9a76a53dfe1dde8e34a8251889f527d2179c7d918d",
        "upstream_verified=false",
        "torch_dtype=bfloat16",
        "CPU FP32",
        "402,653,184 bytes",
        r"=384\ \text{MiB}",
        "999,586,347 bytes",
        "fdf756fa",
        "sha256:ddf41f2cff963bc2a8fc186c28369abba8a920b850152fc815e2b17c7d037876",
        "sha256:56528a3e02ed6ef9d205dcf83ba456658d639d7681ab6a1ad9eb110211edba62",
        "494,032,768",
        "[17,151645]",
        "3.719329833984375e-05",
        "run_qwen_weight_quantization_control.py",
        "802,816",
        "427,328",
        "7.514752134192002",
        "0.08513807180570929",
        "完整 low-bit checkpoint",
        "反量化 FP32",
        "sha256:3f8410f5c31666b1be4f83e343a5b849a0545b2f635f7d415da85a195eebb18c",
        "不是外部可信时间戳 preregistration",
        "generation_completed_before_sse_emission=true",
        "行为门禁 0/2",
        "sha256:00706d003921282625e7c8ad89291c64493d35c13faf4ad7e7553a1388f29ede",
        "framework/callback invocation 均为 0",
        "run_qwen_target_behavior_evaluation.py",
        "sha256:27ada9b1b16cebca8dd9135a5b875de11f412fc9a0f10c6acc462ff76b316201",
        "sha256:dd30a278cbc076c973c0b0babc9e752b1063d8bfb114c852b34ea42b2cd85c43",
        "literal exact",
        "normalized exact",
        "token F1",
        "**4/7**",
        "**5/7**",
        "**6/7**",
        "L5 仍未取得",
        "90 个监督 labels",
        "不执行 backward",
        "0.0038636348 / 0.5845565796",
        "1,093,728 bytes",
        "0.6931471825 / 0.3333517313",
        "0.54707717896",
        "held-out quality proven",
        "仓库当前没有目标 GPU 记录",
        "作品集与简历证据边界",
        "verify→loader reopen TOCTOU 未消除",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "Qwen model page missing release/runtime/training/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_deepseek_model_page(page: Path | None = None) -> list[str]:
    target = page or ROOT / "docs" / "models" / "deepseek.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 500:
        errors.append(
            "DeepSeek model page regressed to an architecture/control summary: "
            f"expected at least 500 non-empty lines, found {nonempty_line_count}"
        )

    required_markers = (
        "L0 标签与 L1\N{EN DASH}L5 证据阶梯",
        "通用机制证据是旁路\N{FULLWIDTH COMMA}不是升级台阶",
        "deepseek-ai/DeepSeek-V3",
        "e815299b0bcbac849fa540c768ef21845365c9eb",
        "1,660-byte",
        "cbf0b95dc614de208a109bb5fd4e7eed11385e9c68411d2c17db5319443035d9",
        "sha256:fed8c13b4637058cd68e600bd4bf7dc734bda4594dd583e3b49fa27c6e123cc6",
        "sha256:74166133716bfebddb444587e9f9a012b4beada923f5209482308ff61194953b",
        "sha256:40b3fe7b2a9c054ea6aa17e9e747d1831b8ae41ee3d55130c916f818acbe4638",
        "upstream_verified=false",
        "auto_map",
        "known_mla_markers_present=true",
        "known_moe_markers_present=true",
        "standard_kv_applicable: false",
        "estimate_refused: true",
        "standard_kv_estimates: []",
        "7168/128=56",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "n_routed_experts",
        "num_experts_per_tok",
        "44{,}040{,}192",
        "11{,}274{,}289{,}152",
        "不能把 `noaux_tc`",
        "不发布“576 elements”捷径",
        "AuthoredMLAMoECausalLM",
        "不是 DeepSeek-V2/V3/R1 配置快照",
        "quantization_config.quant_method = fp8",
        "FP8 不是 INT8",
        "max_position_embeddings = 163840",
        "不能把一个 config 数字写成“已实现多 token 解码加速”",
        "Group-relative policy optimization",
        "oracle@k",
        "仓库没有访问真实 DeepSeek 付费 endpoint",
        "从 L2 提升到 L3/L4",
        "作品集与简历证据边界",
        "没有下载或执行 DeepSeek weights/tokenizer/remote code",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "DeepSeek model page missing config/MLA/MoE/training/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_claude_model_page(page: Path | None = None) -> list[str]:
    target = page or ROOT / "docs" / "models" / "claude.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 500:
        errors.append(
            "Claude model page regressed to a research/API summary: "
            f"expected at least 500 non-empty lines, found {nonempty_line_count}"
        )

    required_markers = (
        "闭源 API 的 L0 标签与 L1\N{EN DASH}L5 证据阶梯",
        "保持未知",
        "2026-08-12",
        "L2 也不是 immutable byte evidence",
        "Canonical business model 与 wire model 分开",
        "system` 位于请求顶层",
        "x-api-key",
        "anthropic-version",
        "Response projection 的有损边界",
        "不会保真返回 tool/thinking/signature/citation/media/unknown blocks",
        "message_start",
        "content_block_start",
        "content_block_delta",
        "message_delta",
        "message_stop",
        "OpenAI `[DONE]`、Anthropic `message_stop` 与 Gemini finishReason+EOF",
        "SSE event/chunk 数不是 token 数",
        "没有工具流式解析证据",
        "工具调用要分 proposal、authorization 与 effect",
        "retryable?",
        "replay safe?",
        "outcome known?",
        "Streaming partial output 默认不自动 replay",
        "reserve 80 micro-USD",
        "settle 66 micro-USD",
        "逻辑调用合计 146",
        "每个 replay attempt 单独记账",
        "不是 Claude/Anthropic 价格、usage 或发票",
        "长上下文的三层上限",
        "仓库没有真实 prompt-caching request/response",
        "Evaluation unit 与分母",
        "生产 rollout / rollback bundle",
        "network_performed=false",
        "即使该 smoke 成功\N{FULLWIDTH COMMA}也只得到 L4 单请求协议证据",
        "作品集与简历证据边界",
        "未执行 Anthropic SDK、真实网络/账号/model",
        "不把论文结果外推为当前 Messages API 漏洞",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "Claude model page missing Messages/stream/tool/budget/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_gemini_model_page(page: Path | None = None) -> list[str]:
    target = page or ROOT / "docs" / "models" / "gemini.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 750:
        errors.append(
            "Gemini model page regressed to a platform/multimodal summary: "
            f"expected at least 750 non-empty lines, found {nonempty_line_count}"
        )

    required_markers = (
        "闭源多模态 API 的 L0\N{EN DASH}L5 证据阶梯",
        "L2 也不是 immutable byte evidence",
        "保持未知",
        "2026-08-15",
        "Canonical Core 与两套 wire model",
        "Interactions object graph",
        "`output_text` 是有损 projection",
        "previous_interaction_id",
        "必须原样保存并重发模型生成 steps",
        "Interactions streaming lifecycle",
        "interaction.created",
        "step.start",
        "step.delta",
        "step.stop",
        "interaction.completed",
        "event: done / data: [DONE]",
        "Background interaction 状态机",
        "`generateContent` 的完整对象图",
        "当前 response parser 的有损边界",
        "`streamGenerateContent` 与 Interactions stream 不可混写",
        "finishReason + EOF",
        "多模态评测\N{FULLWIDTH COLON}证明目标模态产生因果影响",
        "Thought/signature 是高风险 opaque artifact",
        "Safety surface 的版本漂移",
        "retryable?",
        "replay safe?",
        "outcome known?",
        "reserve 80 micro-USD",
        "settle 66 micro-USD",
        "合计 146 micro-USD",
        "不是 Gemini/Google 价格、usage 或发票",
        "Evaluation unit 与分母",
        "生产 rollout / rollback bundle",
        "network_performed=false",
        "即使该 smoke 成功\N{FULLWIDTH COMMA}也只得到 L4",
        "作品集与简历证据边界",
        "未实现 Interactions parser",
        "未执行 Google GenAI SDK",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "Gemini model page missing Interactions/generateContent/"
            f"multimodal/production/scope marker(s): {missing}"
        )
    return errors


def check_cloud_api_contracts_model_page(page: Path | None = None) -> list[str]:
    target = page or ROOT / "docs" / "models" / "cloud-api-contracts.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 330:
        errors.append(
            "cloud API model page regressed to a provider table/control summary: "
            f"expected at least 330 non-empty lines, found {nonempty_line_count}"
        )

    required_markers = (
        "Canonical Core",
        "Provider-specific Extensions",
        "协议分层",
        "canonical business model",
        "response → output item → content part",
        "Interactions API 已 GA",
        "Typed extension",
        "RequestSpec 是 wire identity",
        "provider completed 不表示业务任务成功",
        "三个独立问题",
        "retryable",
        "replay safe",
        "outcome known",
        "Retry-After",
        "arbitrary network byte chunk",
        "SSE framing event",
        "provider typed event",
        "OpenAI `[DONE]`、Anthropic `message_stop` 与 Gemini finishReason+EOF",
        "仅在非 2xx headers 阶段允许重试",
        "R_i=",
        "logical-call:attempt:1",
        "cost per successful task",
        "exact origin allowlist",
        "SQLite commit 不可能与远程 HTTP/provider billing 原子",
        "f2947212",
        "c4829c19",
        "不证明完整 Responses API",
        "不证明真实 provider 的当前错误、配额、幂等、计费或 endpoint 语义",
        "没有访问真实付费 endpoint",
        "Model catalog",
        "Messages API",
        "Gemini Interactions API",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "cloud API model page missing protocol/production/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_openai_responses_replay(fixture: Path | None = None) -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.integrations.openai_responses_replay import (
        replay_response_event_file,
    )

    target = (
        fixture
        or ROOT
        / "projects"
        / "cloud-api-contracts"
        / "openai-responses-events.example.jsonl"
    )
    errors: list[str] = []
    try:
        raw = target.read_bytes()
    except OSError as error:
        return [f"OpenAI Responses replay fixture unreadable: {error}"]

    expected_hash = "f2947212c1f67adf6f35bc976264db28c30abe1a32310daa284df42ca5a54686"
    if len(raw) != 3_208 or hashlib.sha256(raw).hexdigest() != expected_hash:
        errors.append(
            "OpenAI Responses replay input identity mismatch: expected "
            f"3208 bytes/sha256:{expected_hash}"
        )

    try:
        receipt = replay_response_event_file(target)
        payload = receipt.to_dict()
    except (OSError, TypeError, ValueError) as error:
        errors.append(f"OpenAI Responses replay failed: {error}")
        return errors

    calls = payload.get("function_calls")
    expected_call = {
        "item_id": "fc_authored_001",
        "call_id": "call_authored_001",
        "name": "lookup_weather",
        "arguments": '{"city":"上海"}',
        "arguments_is_strict_object": True,
    }
    expected_usage = {"input_tokens": 12, "output_tokens": 9, "total_tokens": 21}
    expected_scope = {
        "sdk_shaped_event_replay_executed": True,
        "strict_json_duplicate_nonfinite_unknown_event_field_rejection": True,
        "sequence_and_item_lifecycle_checked": True,
        "terminal_output_and_usage_reconciled": True,
        "http_sse_or_websocket_transport_executed": False,
        "openai_sdk_or_remote_api_executed": False,
        "model_output_quality_or_safety_proved": False,
        "provider_identity_usage_or_billing_authenticated": False,
        "complete_responses_api_surface_supported": False,
    }
    expected = (
        payload.get("schema_version") == "about-llm.openai-responses-event-replay.v1"
        and payload.get("response_id") == "resp_authored_001"
        and payload.get("model") == "gpt-reviewed-snapshot"
        and payload.get("terminal_status") == "completed"
        and payload.get("terminal_reason") is None
        and payload.get("output_text") == "天气\N{FULLWIDTH COLON}晴。"
        and payload.get("refusals") == []
        and calls == [expected_call]
        and payload.get("usage") == expected_usage
        and payload.get("event_count") == 15
        and payload.get("output_item_count") == 2
        and payload.get("event_projection_fingerprint")
        == "sha256:9cc5964da2517f2076a1c624c2636bd8ca75077b89f024c7710b1b720cbd713e"
        and payload.get("input")
        == {"size_bytes": 3_208, "sha256": f"sha256:{expected_hash}"}
        and payload.get("scope") == expected_scope
        and payload.get("receipt_fingerprint")
        == "sha256:c4829c19895dcb4013141da3d11b5dc9befee8189210a0901f0cb14c19942579"
    )
    if not expected:
        errors.append("OpenAI Responses fixed replay behavior/receipt mismatch")

    documentation_files = (
        ROOT / "docs" / "models" / "gpt.md",
        ROOT / "docs" / "models" / "cloud-api-contracts.md",
        ROOT / "docs" / "practice" / "projects" / "cloud-api-contracts.md",
        ROOT / "projects" / "cloud-api-contracts" / "README.md",
        ROOT / "docs" / "career" / "resume-projects.md",
    )
    documentation = "\n".join(
        path.read_text(encoding="utf-8") for path in documentation_files
    )
    scope_markers = (
        "openai_responses_replay.py",
        "12 input + 9 output = 21 total",
        "OpenAI SDK",
        "完整 Responses API",
        "生产可靠性",
    )
    missing = [marker for marker in scope_markers if marker not in documentation]
    if missing:
        errors.append(f"OpenAI Responses docs missing replay/scope marker(s): {missing}")
    return errors


def check_structured_evaluation_metrics() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.evaluation.cli import (
        METRIC_REVISIONS,
        METRICS,
        load_answers,
        score_answers,
    )
    from about_llm.evaluation.runner import load_cases

    project = ROOT / "projects" / "evaluation-gate"
    cases_path = project / "structured-metrics.cases.jsonl"
    answers_path = project / "structured-metrics.answers.jsonl"
    cases_raw = cases_path.read_bytes()
    answers_raw = answers_path.read_bytes()
    cases = load_cases(cases_path)
    answers = load_answers(answers_path)
    metric_names = (
        "literal_exact_match",
        "exact_match",
        "token_f1",
        "json_schema",
        "json_value_exact",
    )
    results = score_answers(
        cases,
        answers,
        {name: METRICS[name] for name in metric_names},
    )
    by_case = {result.case_id: result for result in results}

    errors: list[str] = []
    if not (
        len(cases_raw) == 1_474
        and hashlib.sha256(cases_raw).hexdigest()
        == "e29a1b1b07c9d1d96d80f15021a1001766a5e1eba17b7fb80bae5bc6b18259a2"
        and len(answers_raw) == 432
        and hashlib.sha256(answers_raw).hexdigest()
        == "e2a4007fb9ea9f3fe8ecf499eeb9814bc63d85c6efedad7df27a98ba931fb1f4"
    ):
        errors.append("Structured-evaluation fixture byte identity mismatch")
    if not (
        METRIC_REVISIONS.get("json_schema")
        == "about-llm.json-schema-metric.v2"
        and METRIC_REVISIONS.get("json_value_exact")
        == "about-llm.json-value-exact.v1"
        and [result.scores["json_schema"] for result in results]
        == [1.0, 1.0, 0.0, 0.0, 1.0]
        and [result.scores["json_value_exact"] for result in results]
        == [1.0, 0.0, 0.0, 0.0, 0.0]
        and by_case["json-object-order"].scores["literal_exact_match"] == 0.0
        and by_case["json-object-order"].scores["exact_match"] == 0.0
        and by_case["json-object-order"].scores["token_f1"] == 1.0
        and math.isclose(
            by_case["json-duplicate-key"].scores["token_f1"],
            2 / 3,
        )
        and by_case["json-array-order"].scores["token_f1"] == 1.0
        and all(result.latency_seconds == 0.0 for result in results)
    ):
        errors.append("Structured-evaluation metric matrix/revision mismatch")

    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "projects" / "evaluation-gate" / "README.md",
            ROOT / "docs" / "practice" / "projects" / "evaluation-gate.md",
            ROOT / "docs" / "quality" / "evaluation.md",
            ROOT / "docs" / "quality" / "evaluation-methodology.md",
            ROOT / "docs" / "applications" / "agents.md",
            ROOT / "docs" / "applications" / "rag-generation.md",
            ROOT / "docs" / "practice" / "labs.md",
            ROOT / "docs" / "career" / "interview-questions.md",
            ROOT / "docs" / "career" / "resume-projects.md",
            ROOT / "docs" / "reference" / "accuracy.md",
        )
    )
    required_markers = (
        "structured-metrics.cases.jsonl",
        "about-llm.json-schema-metric.v2",
        "about-llm.json-value-exact.v1",
        "duplicate object key",
        "NaN/Infinity",
        "local `$ref/$dynamicRef`",
        "`$id`",
        "format` 仍是 annotation",
        "object key order",
        "array order",
        "integer/float",
        "不等于业务语义",
        "latency_seconds=0.0",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "Structured-evaluation docs missing strict/value/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_citation_evidence_span_metric() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.evaluation.cli import (
        METRIC_REVISIONS,
        METRICS,
        load_answers,
        score_answers,
    )
    from about_llm.evaluation.runner import load_cases

    project = ROOT / "projects" / "evaluation-gate"
    cases_path = project / "citation-evidence-span.cases.jsonl"
    answers_path = project / "citation-evidence-span.answers.jsonl"
    cases_raw = cases_path.read_bytes()
    answers_raw = answers_path.read_bytes()
    cases = load_cases(cases_path)
    answers = load_answers(answers_path)
    results = score_answers(
        cases,
        answers,
        {"citation_evidence_span": METRICS["citation_evidence_span"]},
    )

    errors: list[str] = []
    if not (
        len(cases_raw) == 1_015
        and hashlib.sha256(cases_raw).hexdigest()
        == "ceb3ff9dc40ae973e07abbb0b5ba0eeb0a07876290a7dbe06021d0d48ff289e8"
        and len(answers_raw) == 1_138
        and hashlib.sha256(answers_raw).hexdigest()
        == "c61507ec49732a67167dc4bef0e57ee3a18c4359c9a91e7f607074959f712661"
    ):
        errors.append("Citation evidence-span fixture byte identity mismatch")
    if not (
        METRIC_REVISIONS.get("citation_evidence_span")
        == "about-llm.citation-evidence-span-metric.v1"
        and [result.scores["citation_evidence_span"] for result in results]
        == [1.0, 0.0, 0.0, 0.0, 1.0]
        and all(result.latency_seconds == 0.0 for result in results)
        and answers["span-semantic-boundary"].output.find("The moon is cheese.") >= 0
        and answers["span-semantic-boundary"].output.find('"quote":"Earth"') >= 0
    ):
        errors.append("Citation evidence-span metric matrix/revision mismatch")

    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "projects" / "evaluation-gate" / "README.md",
            ROOT / "docs" / "practice" / "projects" / "evaluation-gate.md",
            ROOT / "docs" / "quality" / "evaluation.md",
            ROOT / "docs" / "quality" / "evaluation-methodology.md",
            ROOT / "docs" / "applications" / "rag-generation.md",
            ROOT / "docs" / "practice" / "labs.md",
            ROOT / "docs" / "career" / "interview-questions.md",
            ROOT / "docs" / "career" / "resume-projects.md",
            ROOT / "docs" / "guide" / "knowledge-map.md",
            ROOT / "docs" / "practice" / "project-index.md",
            ROOT / "docs" / "reference" / "accuracy.md",
            ROOT / "CHANGELOG.md",
        )
    )
    required_markers = (
        "citation-evidence-span.cases.jsonl",
        "about-llm.citation-evidence-span-metric.v1",
        "[1,0,0,0,1]",
        "end-exclusive",
        "exact quote",
        "The moon is cheese.",
        "不证明 entailment",
        "latency_seconds=0.0",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "Citation evidence-span docs missing identity/semantic boundary marker(s): "
            f"{missing}"
        )
    return errors


def check_target_qwen_evaluation_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.evaluation.cli import METRIC_REVISIONS, METRICS
    from about_llm.evaluation.target_qwen_control import (
        TARGET_QWEN_EVALUATION_EVIDENCE_BOUNDARY,
        load_target_qwen_evaluation_spec,
        verify_recorded_target_qwen_evaluation_report,
    )

    project = ROOT / "projects" / "evaluation-gate"
    spec = load_target_qwen_evaluation_spec(
        project / "target-qwen-behavior-suite.control.json"
    )
    report = verify_recorded_target_qwen_evaluation_report(
        project / "target-qwen-behavior.recorded-report.json",
        spec,
    )
    aggregates = report.get("aggregates", {})
    generation = report.get("generation", {})
    results = report.get("results", [])
    scope = report.get("scope", {})
    by_case = {
        result.get("case_id"): result
        for result in results
        if isinstance(result, dict)
    }
    copy_case = next(case for case in spec.cases if case.case_id == "zh-copy")

    errors: list[str] = []
    if not (
        spec.suite_fingerprint
        == "sha256:27ada9b1b16cebca8dd9135a5b875de11f412fc9a0f10c6acc462ff76b316201"
        and report.get("suite_fingerprint") == spec.suite_fingerprint
        and report.get("report_fingerprint")
        == "sha256:dd30a278cbc076c973c0b0babc9e752b1063d8bfb114c852b34ea42b2cd85c43"
        and report.get("checkpoint_manifest_fingerprint")
        == "sha256:ddf41f2cff963bc2a8fc186c28369abba8a920b850152fc815e2b17c7d037876"
    ):
        errors.append("Target-Qwen behavior-evaluation identity binding mismatch")
    if not (
        len(results) == 7
        and by_case.get("en-arithmetic", {}).get("output") == "112"
        and by_case.get("zh-copy", {}).get("output") == "llm-2026"
        and by_case.get("zh-json", {}).get("output") == '{"answer": 42}'
        and aggregates.get("case_count") == 7
        and aggregates.get("literal_exact_match_pass_count") == 4
        and aggregates.get("exact_match_pass_count") == 5
        and sum(result.get("token_f1") == 1.0 for result in results) == 6
        and math.isclose(aggregates.get("token_f1_mean", math.nan), 6 / 7)
    ):
        errors.append("Target-Qwen behavior-evaluation output/metric mismatch")
    if not (
        METRIC_REVISIONS.get("literal_exact_match")
        == "about-llm.literal-exact-match.v1"
        and METRICS["literal_exact_match"](copy_case.as_evaluation_case(), "llm-2026")
        == 0.0
        and METRICS["exact_match"](copy_case.as_evaluation_case(), "llm-2026")
        == 1.0
    ):
        errors.append("Evaluation CLI literal/normalized exact scorer contract drift")
    if not (
        generation.get("framework") == "transformers.GenerationMixin.generate"
        and generation.get("batch_size") == 1
        and generation.get("do_sample") is False
        and generation.get("max_new_tokens") == 12
        and scope.get("target_checkpoint_weights_loaded") is True
        and scope.get("all_authored_cases_generated") is True
        and scope.get("framework_generate_executed") is True
        and scope.get("externally_preregistered_or_held_out_suite") is False
        and scope.get("representative_benchmark_or_quality_proven") is False
        and scope.get("performance_benchmark_performed") is False
        and scope.get("statistical_uncertainty_estimated") is False
        and scope.get("system_comparison_or_release_gate_executed") is False
        and "seven fixed authored cases"
        in TARGET_QWEN_EVALUATION_EVIDENCE_BOUNDARY
        and "not externally preregistered"
        in TARGET_QWEN_EVALUATION_EVIDENCE_BOUNDARY
        and "performance benchmark"
        in TARGET_QWEN_EVALUATION_EVIDENCE_BOUNDARY
    ):
        errors.append("Target-Qwen behavior-evaluation execution/scope drift")

    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "docs" / "models" / "qwen.md",
            ROOT / "docs" / "quality" / "evaluation.md",
            ROOT / "docs" / "quality" / "evaluation-methodology.md",
            ROOT / "docs" / "practice" / "project-index.md",
            ROOT / "docs" / "practice" / "labs.md",
            ROOT / "docs" / "career" / "interview-questions.md",
            ROOT / "docs" / "career" / "resume-projects.md",
            ROOT / "docs" / "guide" / "knowledge-map.md",
            ROOT / "docs" / "reference" / "accuracy.md",
        )
    )
    required_markers = (
        "literal exact",
        "normalized exact",
        "token F1",
        "LLM-2026",
        "llm-2026",
        '{"answer": 42}',
        "4/7",
        "5/7",
        "6/7",
        "未外部预注册",
        "非代表性",
        "不自动建立 construct validity",
        "Qwen 准确率 85.7%",
        "--metric literal_exact_match",
        "about-llm.literal-exact-match.v1",
        "避免静默改变",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "Target-Qwen behavior-evaluation docs missing metric/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_evaluation_gate_project_page(page: Path | None = None) -> list[str]:
    target = page or ROOT / "docs" / "practice" / "projects" / "evaluation-gate.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 150:
        errors.append(
            "evaluation-gate project page regressed to a navigation summary: "
            f"expected at least 150 non-empty lines, found {nonempty_line_count}"
        )

    required_markers = (
        "run_qwen_target_behavior_evaluation.py",
        "--metric literal_exact_match",
        "structured-metrics.cases.jsonl",
        "about-llm.json-schema-metric.v2",
        "about-llm.json-value-exact.v1",
        "duplicate object key",
        "NaN/Infinity",
        "| object key order/whitespace | 0 | 0 | 1 | 1 | 1 |",
        "| reversed array order | 0 | 0 | 1 | 1 | 0 |",
        "latency_seconds=0.0",
        "sha256:27ada9b1b16cebca8dd9135a5b875de11f412fc9a0f10c6acc462ff76b316201",
        "sha256:dd30a278cbc076c973c0b0babc9e752b1063d8bfb114c852b34ea42b2cd85c43",
        "literal exact=`4/7`",
        "normalized exact=`5/7`",
        "token F1=`6/7`",
        "不是外部预注册、独立抽样、held-out、代表性",
        "answers.baseline.example.jsonl",
        "answers.candidate.example.jsonl",
        "baseline.run-manifest.json",
        "candidate.run-manifest.json",
        "about-llm.evaluation-comparison.v2",
        "verification_scope: artifact_only",
        "referenced_manifests_revalidated: false",
        "statistics_recomputed: false",
        "verification_scope: full_local_recomputation",
        "artifact_only_render",
        "clustered_bootstrap_toy.py",
        "paired_randomization_toy.py",
        "clustered_randomization_toy.py",
        "holm_correction_toy.py",
        "sequential_peeking_toy.py",
        "authenticated_release_ledger_toy.py",
        "authenticated_chain=true",
        "referenced_artifacts_rehashed=true",
        "trusted_head_matched=true",
        "test_verify_evidence_rejects_recorded_answer_drift",
        "test_gate_threshold_tampering_invalidates_existing_fingerprint",
        "test_trusted_head_is_required_to_detect_valid_prefix_truncation",
        "不证明模型/provider 当时真实执行",
        "construct validity",
        "线上因果影响",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "evaluation-gate project page missing workflow/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_inference_serving_project_page(page: Path | None = None) -> list[str]:
    target = page or ROOT / "docs" / "practice" / "projects" / "inference-serving.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 150:
        errors.append(
            "inference-serving project page regressed to a control summary: "
            f"expected at least 150 non-empty lines, found {nonempty_line_count}"
        )

    required_markers = (
        "qwen2.5-0.5b-service.recorded-report.json",
        "incremental-streaming.recorded-report.json",
        "transformers-thread-cancellation.recorded-report.json",
        "sha256:63e566ca…617ddb",
        "sha256:25846822…2b5d00",
        "sha256:eadcab54…f62bc7",
        "about_llm.inference_analysis_cli",
        "offered_at = benchmark_started_at + scheduled_offset",
        "client-side coordinated omission",
        "SSE chunk 不是 token",
        "vllm serve Qwen/Qwen2.5-0.5B-Instruct",
        "benchmark_openai.py",
        "sampling_toy.py",
        "beam_search_toy.py",
        "constrained_decoding_toy.py",
        "stop_matching_toy.py",
        "speculative_decoding_toy.py",
        "continuous_batching_toy.py",
        "kv_preemption_batching_toy.py",
        "kv_block_allocator_toy.py",
        "prefix_cache_toy.py",
        "quantization_toy.py",
        "quantized_bundle_toy.py",
        "minigpt_checkpoint_toy.py",
        "kv_quantization_toy.py",
        "self_consistency_correlation_toy.py",
        "verifier_best_of_n_toy.py",
        "0.75349813248",
        "0.53896454244",
        "0.1852867601",
        "W=\\sum_i(P_i+O_i-1)=10",
        "4D/(D+4)",
        "test_cli_rejects_ambiguous_attempt_artifacts",
        "test_capacity_failure_is_atomic_before_mutating_an_exclusive_tail",
        "test_injected_hash_collision_never_bypasses_full_identity_or_token_comparison",
        "没有在本机执行 vLLM、CUDA、PagedAttention",
        "不得外推为 GPU/NCCL、目标模型或生产性能结论",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "inference-serving project page missing workflow/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_rag_foundations_project_page(page: Path | None = None) -> list[str]:
    target = page or ROOT / "docs" / "practice" / "projects" / "rag-foundations.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 210:
        errors.append(
            "rag-foundations project page regressed to a retrieval/control summary: "
            f"expected at least 210 non-empty lines, found {nonempty_line_count}"
        )

    required_markers = (
        "about_llm.rag.cli retrieve",
        "about_llm.rag.cli answer-extractive",
        "about_llm.rag.cli evaluate-extractive",
        "about_llm.rag.cli store-upsert",
        "about_llm.rag.cli store-retrieve",
        "about_llm.rag.cli store-delete",
        "about_llm.rag.cli store-backup",
        "about_llm.rag.cli store-verify-backup",
        "about_llm.rag.cli store-restore",
        "about_llm.rag.cli rerank-recorded",
        "about_llm.rag.cli evaluate",
        "about_llm.rag.cli pack-tokenized",
        "reserved-output-tokens 512",
        "about_llm.rag.cli evaluate-answers",
        "about_llm.rag.cli audit-traces",
        "rag_service_control.py",
        "run_qwen_rag_control.py --local-files-only",
        "qwen2.5-0.5b-rag.publication-policy-replay.json",
        "run_qwen_guarded_rag_control.py --local-files-only",
        "sha256:829663e2…e5b60",
        "sha256:ed4d16ad…b13239",
        "sha256:00706d00…f29ede",
        "行为 gate 是 **0/2**",
        "counterfactual replay",
        "test_database_trigger_failure_rolls_back_delete_and_version",
        "test_backup_verification_rejects_semantically_corrupted_rows_even_if_rehashed",
        "test_every_candidate_is_authorized_even_when_budget_would_drop_it",
        "test_body_cannot_self_report_security_context_and_auth_errors_are_closed",
        "test_report_cooperative_raw_output_rehash_cannot_hide_stale_local_audit",
        "claim-evidence entailment",
        "不得外推为生产安全、模型质量或性能结论",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "rag-foundations project page missing workflow/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_rag_framework_adapters_project_page(
    page: Path | None = None,
) -> list[str]:
    target = (
        page
        or ROOT
        / "docs"
        / "practice"
        / "projects"
        / "rag-framework-adapters.md"
    )
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 135:
        errors.append(
            "rag-framework-adapters project page regressed to an adapter summary: "
            f"expected at least 135 non-empty lines, found {nonempty_line_count}"
        )
    required_markers = (
        "Canonical-first",
        "ACL filter before scoring",
        "BaseRetriever.invoke()",
        "BaseRetriever.retrieve()",
        "excluded_embed_metadata_keys",
        "excluded_llm_metadata_keys",
        "document_id`、`tenant_id`、`acl`、`retrieval_score`",
        "连续 one-based rank",
        "`True` 不能冒充整数 1",
        "langchain-core==1.5.3",
        "llama-index-core==0.14.23",
        "b9c8cb77…e1e8e19c",
        "sha256:d1045446…48180cca",
        "16 个测试",
        "rank gap、duplicate ID、NaN/±Inf 与 bool score",
        "Supplied expected results",
        "native embedding/index/query engine",
        "当前 `answer_artifact_fingerprint` 来自确定性 extractive baseline",
        "tenant/principals 必须来自可信认证层",
        "不得让请求 body 自报安全身份",
        "不证明框架默认 ACL",
        "CPU 本地 authored fixture 也不得外推到目标向量库、模型、GPU",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "rag-framework-adapters project page missing workflow/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_rag_framework_adapters_project_readme(
    page: Path | None = None,
) -> list[str]:
    target = page or ROOT / "projects" / "rag-framework-adapters" / "README.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 300:
        errors.append(
            "rag-framework-adapters README regressed to a quickstart: "
            f"expected at least 300 non-empty lines, found {nonempty_line_count}"
        )
    required_markers = (
        "canonical-first",
        "ACL filter before scoring",
        "BaseRetriever.invoke()",
        "BaseRetriever.retrieve()",
        "excluded_embed_metadata_keys",
        "excluded_llm_metadata_keys",
        "连续 one-based rank",
        "bool 不能冒充整数/实数",
        "NaN、+Inf、-Inf score",
        "langchain-core==1.5.3",
        "llama-index-core==0.14.23",
        "b9c8cb77…e1e8e19c",
        "sha256:d1045446…48180cca",
        "当前 16 个测试覆盖",
        "Supplied expected results",
        "不能认证 expected 的来源",
        "native embedding/index/query engine",
        "当前 `answer_artifact_fingerprint` 来自确定性 extractive baseline",
        "tenant/principals 必须来自可信认证层",
        "不得让请求 body 自报安全身份",
        "框架默认 ACL",
        "CPU 本地 authored fixture 也不得外推到目标向量库、模型、GPU",
        "项目验收清单",
        "可以写进简历的结论",
        "不能写进简历的结论",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "rag-framework-adapters README missing run/evidence/production/"
            f"scope marker(s): {missing}"
        )
    return errors


def check_cloud_api_contracts_project_page(page: Path | None = None) -> list[str]:
    target = (
        page
        or ROOT / "docs" / "practice" / "projects" / "cloud-api-contracts.md"
    )
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 130:
        errors.append(
            "cloud-api-contracts project page regressed to a demo summary: "
            f"expected at least 130 non-empty lines, found {nonempty_line_count}"
        )

    required_markers = (
        "cloud_api_cli verify",
        "openai_responses_replay.py",
        "15 events",
        "12 input + 9 output = 21 total",
        "连续 `sequence_number`",
        "OpenAI SDK",
        "不证明完整 Responses API",
        "reasoning-replay-matrix",
        "unsafe_acceptance_demonstrated: true",
        "trajectory-release-gate",
        "provider_artifacts_interpreted: false",
        "secret_pii_scan_performed: false",
        "retry-matrix",
        "408/429/500/502/503/504",
        "501/505",
        "replay_safe=true",
        "outcome_uncertain=false",
        "execute_json_request",
        "max_response_bytes",
        "SSEDecoder",
        "2xx stream 一旦开始",
        "usage_budget_toy.py",
        "sqlite_usage_budget_demo.py",
        "budgeted_http_demo.py",
        "budgeted_retry_demo.py",
        "logical-call:attempt:N",
        "attempt 1 uncertain 80",
        "attempt 2 settled 66",
        "Hard limit=140",
        "test_context_bound_envelope_rejects_scope_drift",
        "test_stream_truncation_and_size_limit_are_terminal_and_close",
        "test_cancellation_after_reservation_never_fabricates_zero_usage",
        "test_retry_budget_gate_blocks_second_network_attempt",
        "不提供 exactly-once billing",
        "不得外推为生产 key custody",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "cloud-api-contracts project page missing workflow/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_safe_agent_project_page(page: Path | None = None) -> list[str]:
    target = page or ROOT / "docs" / "practice" / "projects" / "safe-agent.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 110:
        errors.append(
            "safe-agent project page regressed to a protocol-control summary: "
            f"expected at least 110 non-empty lines, found {nonempty_line_count}"
        )

    required_markers = (
        "about_llm.agents.cli run",
        "about_llm.agents.cli pending",
        "--resolution abandoned",
        "about_llm.agents.cli inspect",
        "about_llm.agents.cli loop",
        "about_llm.agents.cli pause-loop",
        "about_llm.agents.cli resume-loop",
        "model_planner_control.py",
        "Authored 62 tokens、0.03 cost",
        "framework_tool_adapter_control.py",
        "直接 `FunctionTool.call()`",
        "没有执行 LangGraph 或 LlamaIndex Agent loop",
        "framework_agent_loop_control.py",
        "`create_agent()`/LangGraph",
        "`FunctionAgent.run()`",
        "独立 verifier",
        "`InjectedToolCallId`",
        "可信 fixture case/action",
        "73 次 Pydantic deprecated-field warnings",
        "persistent checkpointer/resume",
        "没有真实模型/provider",
        "about_llm.agents.cli evaluate",
        "outbox_demo.py",
        "provider effect count=1",
        "at-least-once",
        "mcp_sdk_memory_control.py",
        "mcp_sdk_stdio_control.py",
        "mcp_sdk_streamable_http_control.py",
        "mcp_stdio_control.py",
        "mcp_streamable_http_control.py",
        "a2a_loopback_control.py --verify-official-schema",
        "不能互相借证据",
        "test_cached_replay_is_reauthorized_after_capability_revocation",
        "test_checkpoint_round_trip_restart_and_resume_without_double_usage",
        "test_crash_after_provider_success_causes_redelivery_not_exactly_once",
        "远端 completed 不能跳过本地 verifier",
        "不得外推为生产 Agent 安全",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "safe-agent project page missing workflow/scope marker(s): " f"{missing}"
        )
    return errors


def check_synthetic_data_audit_project_page(page: Path | None = None) -> list[str]:
    target = (
        page
        or ROOT / "docs" / "practice" / "projects" / "synthetic-data-audit.md"
    )
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 220:
        errors.append(
            "synthetic-data-audit project page regressed to a CLI summary: "
            f"expected at least 220 non-empty lines, found {nonempty_line_count}"
        )
    required_markers = (
        "about-llm.synthetic-data-audit.v2",
        "audit.example.json",
        "full_local_recomputation",
        "202d8db97b704c5542e8516c5bd0c945da1c1022100f6ecbfb828f2d2bb6f4cd",
        "1,457 bytes",
        "341 bytes",
        "candidate_count=4",
        "eligible_count=2",
        "eligible_unique_content_count=1",
        "lineage_cycle_record_ids",
        "nonmonotonic_parent_pairs",
        "duplicate JSON keys",
        "nfc_whitespace",
        "25% synthetic target",
        "预期消费 5 倍",
        "observed-token ledger",
        "unknown parent 不能静默解析",
        "输入 bytes 与外部 policy",
        "cooperative rehash",
        "exclusive-create",
        "directory entry durable",
        "40 个测试",
        "test_mixture_plan_uses_normalized_not_pre_normalized_weights",
        "Target mixture expectation 不得写成实际 token exposure",
        "eligibility 不得写成“高质量可用数据”",
        "Unkeyed hash 也不提供 publisher identity",
        "CPU/offline 结果不得外推到目标模型质量或生产系统可靠性",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "synthetic-data-audit project page missing workflow/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_single_gpu_finetuning_project_page(
    page: Path | None = None,
) -> list[str]:
    target = (
        page
        or ROOT
        / "docs"
        / "practice"
        / "projects"
        / "single-gpu-finetuning.md"
    )
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 230:
        errors.append(
            "single-gpu-finetuning project page regressed to a control catalog: "
            f"expected at least 230 non-empty lines, found {nonempty_line_count}"
        )
    required_markers = (
        "about_llm.finetuning_cli prepare-training",
        "train_trl_sft.py",
        "--data-preflight-only",
        "sft-training-readiness.json",
        "sft-template-mask-audit.json",
        "sft-final-label-audit.json",
        "assistant_only_loss=False",
        "qwen2.5-generation-aware-sft.jinja",
        "smoke_trl_sft.py",
        "smoke_peft.py",
        "qwen2.5-0.5b-sft-label.recorded-report.json",
        "qwen2.5-0.5b-lora.recorded-report.json",
        "qwen2.5-0.5b-dpo.recorded-report.json",
        "270,336",
        "loss 从约 0.003864 升到 0.584557",
        "reference replay max-abs drift=0.547077",
        "train_qlora.py",
        "--estimate-only",
        "不得写成“7B QLoRA 只需 6.18 GiB”",
        "`--qlora` 路径在本仓库当前环境未实跑",
        "训练身份\N{FULLWIDTH COLON}不可读 held-out 原文",
        "语义去重",
        "法律许可",
        "完整 PII/secret 检测",
        "base/Prompt/RAG/adapter",
        "CPU/Gloo/tiny/recorded 证据",
        "不同 control 不能拼接",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "single-gpu-finetuning project page missing workflow/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_jax_minigpt_project_page(page: Path | None = None) -> list[str]:
    target = page or ROOT / "docs" / "practice" / "projects" / "jax-minigpt.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 165:
        errors.append(
            "jax-minigpt project page regressed to a control summary: "
            f"expected at least 165 non-empty lines, found {nonempty_line_count}"
        )
    required_markers = (
        "四层证据不能合并",
        "632 个参数",
        "2.108591318130493",
        "0.0030041998252272606",
        "该脚本**不生成文本**",
        "loss.block_until_ready()",
        "enqueue latency",
        "20 个 unique parameters",
        "2.384185791015625e-07",
        "RMSNorm 反事实",
        "0.37747739627957344",
        "三步 AdamW trajectory parity",
        "native RNG equivalence",
        "0.06900620367377996",
        "`ALLMJAX1`",
        "13,476 bytes",
        "e9252e5dddfa4aa5",
        "wrong PRNG",
        "wrong cursor",
        "0.037261832505464554",
        "0.03700308472616598",
        "File `fsync` 不等于目录项已 durable",
        "Orbax/TensorStore",
        "CPU tiny 结果不能预测设备峰值",
        "可反序列化不等于 exact resume",
        "全 ignored batch 返回 0",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "jax-minigpt project page missing workflow/scope marker(s): "
            f"{missing}"
        )
    old_generation_claim = (
        "检查运行设备\N{IDEOGRAPHIC COMMA}初始/最终 loss 和生成结果"
    )
    if old_generation_claim in documentation:
        errors.append(
            "jax-minigpt project page incorrectly claims train_tiny emits generation"
        )
    return errors


def check_jax_minigpt_project_readme(page: Path | None = None) -> list[str]:
    target = page or ROOT / "projects" / "jax-minigpt" / "README.md"
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 250:
        errors.append(
            "jax-minigpt README regressed to a run/control summary: "
            f"expected at least 250 non-empty lines, found {nonempty_line_count}"
        )
    for error in check_jax_minigpt_project_page(target):
        if "regressed to a control summary" in error:
            continue
        errors.append(error.replace("jax-minigpt project page", "jax-minigpt README"))
    return errors


def check_transformers_basics_project_page(
    page: Path | None = None,
) -> list[str]:
    target = (
        page
        or ROOT
        / "docs"
        / "practice"
        / "projects"
        / "transformers-basics.md"
    )
    documentation = target.read_text(encoding="utf-8")
    errors: list[str] = []
    nonempty_line_count = sum(bool(line.strip()) for line in documentation.splitlines())
    if nonempty_line_count < 300:
        errors.append(
            "transformers-basics project page regressed to a control summary: "
            f"expected at least 300 non-empty lines, found {nonempty_line_count}"
        )
    required_markers = (
        "四层证据不能合并",
        "`actual_vocab_size=273`",
        "17 次 merge",
        "2.220446049250313e-16",
        "27,008 个参数",
        "3.4888949394226074→2.1879992485046387",
        "GenerationConfig EOS `{2,3}`",
        "provider 风格 finish reason",
        "`standard dense K/V formula must not be applied`",
        "sha256:74166133…53b",
        "Model card claim、config field、artifact identity 与 runtime observation",
        "999,586,347 bytes",
        "494,032,768",
        "3.719329833984375e-05",
        "hash→loader reopen 的 TOCTOU",
        "run_qwen_weight_quantization_control.py",
        "802,816",
        "427,328",
        "7.514752134192002",
        "0.08513807180570929",
        "完整 low-bit Qwen checkpoint",
        "反量化 FP32",
        "future-position 负对照",
        "1.000024/0.992244/0",
        "六条 control 的边界矩阵",
        "selected counts=`[4,0]`",
        "每 rank 五次 `all_to_all_single`",
        "416 logical tensor-payload bytes",
        "reverse-split backward",
        "20.78017329703821→19.41091750734501",
        "zero-assignment source rank",
        "15.253670387373656→14.530264380025987",
        "不同 control 不能拼接",
        "不重新运行约 1 GB 权重",
        "123 passed in 86.64s",
        "CPU/JAX/Gloo/loopback 结果不得外推 GPU、NCCL、目标模型或生产性能",
        "一步 loss 下降不等于收敛",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "transformers-basics project page missing workflow/scope marker(s): "
            f"{missing}"
        )
    return errors


def check_calibration_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.evaluation import (
        EvaluationCase,
        EvaluationReleaseLedger,
        EvaluationResult,
        ReleaseGate,
        binary_calibration,
        load_cases,
        load_evaluation_comparison_artifact,
        load_evaluation_release_ledger,
        load_evaluation_run_manifest,
        load_results,
        render_evaluation_comparison_html,
        risk_coverage_curve,
        validate_evaluation_run_manifest,
        verify_evaluation_release_ledger,
    )
    from about_llm.evaluation.cli import (
        _recompute_comparison_artifact,
        _verify_run_evidence,
        compare_results,
        load_answers,
    )

    calibration = binary_calibration(
        labels=[0, 1, 1, 0],
        probabilities=[0.1, 0.8, 0.6, 0.4],
        bins=2,
    )
    errors: list[str] = []
    if not math.isclose(calibration.brier_score, 0.0925):
        errors.append("binary Brier example mismatch")
    if not math.isclose(calibration.expected_calibration_error, 0.275):
        errors.append("equal-width ECE example mismatch")
    curve = risk_coverage_curve(
        correctness=[1, 0, 1, 0],
        confidence=[0.9, 0.8, 0.8, 0.1],
    )
    if [point.accepted_count for point in curve] != [1, 3, 4]:
        errors.append("risk-coverage ties must be accepted together")
    if not (
        math.isclose(curve[0].coverage, 0.25)
        and math.isclose(curve[1].risk, 1 / 3)
        and math.isclose(curve[-1].coverage, 1)
        and math.isclose(curve[-1].risk, 0.5)
    ):
        errors.append("risk-coverage example mismatch")
    project = ROOT / "projects" / "evaluation-gate"
    cases = load_cases(project / "cases.example.jsonl")
    for name in ("baseline", "candidate"):
        results = load_results(project / f"results.{name}.example.jsonl")
        manifest = load_evaluation_run_manifest(
            project / f"run.{name}.manifest.example.json"
        )
        try:
            validate_evaluation_run_manifest(
                manifest,
                cases=cases,
                results=results,
                required_metrics=("exact_match", "token_f1"),
            )
        except ValueError as error:
            errors.append(f"evaluation {name} run manifest mismatch: {error}")
        if manifest.system_id != f"authored-fixture-{name}@v1":
            errors.append(f"evaluation {name} fixture system identity mismatch")
    comparison = load_evaluation_comparison_artifact(project / "comparison.example.json")
    bootstrap = comparison.content["bootstrap"]
    if not (
        comparison.passed
        and comparison.baseline_system_id == "authored-fixture-baseline@v1"
        and comparison.candidate_system_id == "authored-fixture-candidate@v1"
        and comparison.content["comparison_version"]
        == "about-llm.evaluation-comparison.v2"
        and bootstrap["unit"] == "case"
        and bootstrap["cluster_metadata_key"] is None
        and comparison.comparison_fingerprint
        == "sha256:999e29b9d9fae5e37a3d8e680711e4cb79be222af859a35e9d41083ba587b18a"
    ):
        errors.append("evaluation comparison fixture identity/scope mismatch")
    html_report = render_evaluation_comparison_html(comparison)
    if not (
        html_report.startswith("<!doctype html>\n")
        and comparison.comparison_fingerprint in html_report
        and 'content="artifact_only_render"' in html_report
        and "default-src 'none'" in html_report
        and "<script" not in html_report.lower()
        and "https://" not in html_report
        and "未重新评分或重跑统计" in html_report
    ):
        errors.append("evaluation comparison HTML report identity/scope mismatch")
    baseline_results = load_results(project / "results.baseline.example.jsonl")
    candidate_results = load_results(project / "results.candidate.example.jsonl")
    baseline_manifest = load_evaluation_run_manifest(
        project / "run.baseline.manifest.example.json"
    )
    candidate_manifest = load_evaluation_run_manifest(
        project / "run.candidate.manifest.example.json"
    )
    try:
        _verify_run_evidence(
            label="baseline",
            cases=cases,
            answers=load_answers(project / "answers.baseline.example.jsonl"),
            results=baseline_results,
            manifest=baseline_manifest,
        )
        _verify_run_evidence(
            label="candidate",
            cases=cases,
            answers=load_answers(project / "answers.candidate.example.jsonl"),
            results=candidate_results,
            manifest=candidate_manifest,
        )
        rebuilt_comparison = _recompute_comparison_artifact(
            artifact=comparison,
            cases=cases,
            baseline_results=baseline_results,
            candidate_results=candidate_results,
            baseline_manifest=baseline_manifest,
            candidate_manifest=candidate_manifest,
        )
    except ValueError as error:
        errors.append(f"evaluation full evidence recomputation failed: {error}")
    else:
        if rebuilt_comparison.to_dict() != comparison.to_dict():
            errors.append("evaluation full evidence recomputation mismatch")

    release_ledger = load_evaluation_release_ledger(
        project / "release-ledger.example.json"
    )
    fixture_keys = {
        "fixture-hmac-2026-a": bytes.fromhex("11" * 32),
        "fixture-hmac-2026-b": bytes.fromhex("22" * 32),
    }
    artifact_paths = {
        "baseline-run-manifest": project / "run.baseline.manifest.example.json",
        "candidate-run-manifest": project / "run.candidate.manifest.example.json",
        "release-comparison": project / "comparison.example.json",
    }
    release_verification = verify_evaluation_release_ledger(
        release_ledger,
        key_resolver=fixture_keys,
        artifact_paths=artifact_paths,
        trusted_head=release_ledger.head,
    )
    if not (
        len(release_ledger.records) == 3
        and release_ledger.records[-1].key_id == "fixture-hmac-2026-b"
        and release_ledger.records[-1].decision == "approved"
        and release_ledger.head.record_mac
        == "hmac-sha256:c3cc197b76c5487d5304cb97ec20a5e5297c59902b582fd67cdefec917132c1c"
        and release_verification.referenced_artifacts_rehashed
        and release_verification.trusted_head_matched
    ):
        errors.append("evaluation authenticated release-ledger fixture mismatch")
    valid_prefix = EvaluationReleaseLedger(release_ledger.records[:2])
    prefix_verification = verify_evaluation_release_ledger(
        valid_prefix, key_resolver=fixture_keys
    )
    if prefix_verification.trusted_head_matched:
        errors.append("release-ledger unanchored prefix must not claim trusted head")
    try:
        verify_evaluation_release_ledger(
            valid_prefix,
            key_resolver=fixture_keys,
            trusted_head=release_ledger.head,
        )
    except ValueError:
        pass
    else:
        errors.append("release-ledger trusted head must reject tail truncation")

    cluster_cases = [
        EvaluationCase(
            case_id=f"cluster-case-{index}",
            input="input",
            expected="expected",
            metadata={"user_id": "A" if index < 5 else "B"},
        )
        for index in range(6)
    ]
    cluster_baseline = [
        EvaluationResult(
            case_id=case.case_id,
            output="",
            scores={"quality": 0.0 if index < 5 else 1.0},
            latency_seconds=0.1,
        )
        for index, case in enumerate(cluster_cases)
    ]
    cluster_candidate = [
        EvaluationResult(
            case_id=case.case_id,
            output="",
            scores={"quality": 1.0 if index < 5 else 0.0},
            latency_seconds=0.1,
        )
        for index, case in enumerate(cluster_cases)
    ]
    cluster_comparison = compare_results(
        cluster_cases,
        cluster_baseline,
        cluster_candidate,
        quality_metric="quality",
        safety_metric=None,
        confidence=0.95,
        bootstrap_samples=1000,
        seed=7,
        gate=ReleaseGate(minimum_quality_difference=-1),
        protected_slices=(),
        maximum_slice_regression=0,
        cluster_metadata_key="user_id",
        cluster_weighting="case",
        cluster_exact_max=6,
    )
    cluster_quality = cluster_comparison["quality"]
    if not (
        cluster_comparison["bootstrap"]["unit"] == "cluster"
        and cluster_quality["cluster_sizes"] == (5, 1)
        and cluster_quality["method"] == "exact"
        and cluster_quality["resamples_evaluated"] == 4
        and math.isclose(cluster_quality["mean_difference"], 2 / 3)
        and math.isclose(cluster_quality["confidence_low"], -0.875)
        and math.isclose(cluster_quality["confidence_high"], 0.975)
    ):
        errors.append("evaluation comparison v2 cluster integration mismatch")
    return errors


def check_paired_randomization_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.evaluation import paired_randomization_test

    errors: list[str] = []
    baseline = [0, 0, 0, 0, 1]
    candidate = [1, 1, 1, 1, 1]
    greater = paired_randomization_test(
        baseline, candidate, alternative="greater"
    )
    two_sided = paired_randomization_test(
        baseline, candidate, alternative="two-sided"
    )
    monte_carlo = paired_randomization_test(
        baseline,
        candidate,
        alternative="greater",
        exact_max_nonzero_pairs=2,
        monte_carlo_samples=1_000,
        seed=7,
    )
    if not (
        greater.pair_count == 5
        and greater.nonzero_pair_count == 4
        and greater.zero_difference_count == 1
        and math.isclose(greater.mean_difference, 0.8)
        and greater.assignments_evaluated == 16
        and greater.extreme_assignments == 1
        and math.isclose(greater.p_value, 1 / 16)
        and two_sided.extreme_assignments == 2
        and math.isclose(two_sided.p_value, 2 / 16)
        and monte_carlo.method == "monte_carlo"
        and monte_carlo.p_value > 0
        and math.isclose(monte_carlo.p_value_resolution, 1 / 1_001)
    ):
        errors.append("paired randomization exact/Monte Carlo fixture mismatch")

    demo = (
        ROOT / "projects" / "evaluation-gate" / "paired_randomization_toy.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"paired_case_sign_flip_distribution_executed": True',
        '"zero_difference_removed_from_sign_enumeration": True',
        '"exchangeability_or_random_assignment_established": False',
        '"cluster_dependence_modeled": False',
        '"multiple_comparison_correction_applied": False',
        '"causal_product_or_model_improvement_proved": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(
            f"paired-randomization toy missing scope marker(s): {missing_scope}"
        )
    return errors


def check_clustered_randomization_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.evaluation import (
        clustered_paired_randomization_test,
        paired_randomization_test,
    )

    errors: list[str] = []
    baseline = [0, 0, 0, 0, 0, 0]
    candidate = [1, 1, 1, 1, 1, -1]
    clusters = ["user-a"] * 5 + ["user-b"]
    naive = paired_randomization_test(
        baseline, candidate, alternative="greater"
    )
    case_weighted = clustered_paired_randomization_test(
        baseline,
        candidate,
        clusters,
        cluster_weighting="case",
        alternative="greater",
    )
    equal_cluster = clustered_paired_randomization_test(
        baseline,
        candidate,
        clusters,
        cluster_weighting="equal",
        alternative="two-sided",
    )
    if not (
        naive.assignments_evaluated == 64
        and naive.extreme_assignments == 7
        and math.isclose(naive.p_value, 7 / 64)
        and case_weighted.cluster_sizes == (5, 1)
        and case_weighted.assignments_evaluated == 4
        and case_weighted.extreme_assignments == 2
        and math.isclose(case_weighted.mean_difference, 4 / 6)
        and math.isclose(case_weighted.p_value, 2 / 4)
        and math.isclose(equal_cluster.mean_difference, 0)
        and math.isclose(equal_cluster.p_value, 1)
    ):
        errors.append("cluster-joint/weighting randomization fixture mismatch")

    demo = (
        ROOT / "projects" / "evaluation-gate" / "clustered_randomization_toy.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"cluster_joint_sign_flip_executed": True',
        '"within_cluster_case_independence_required": False',
        '"cluster_level_exchangeability_or_independence_established": False',
        '"estimand_or_cluster_definition_selected_without_outcome_looking": False',
        '"causal_or_general_model_improvement_proved": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(
            f"clustered-randomization toy missing scope marker(s): {missing_scope}"
        )
    return errors


def check_clustered_bootstrap_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.evaluation import clustered_paired_bootstrap

    errors: list[str] = []
    baseline = [0, 0, 0, 0, 0, 0]
    candidate = [1, 1, 1, 1, 1, -1]
    clusters = ["user-a"] * 5 + ["user-b"]
    case_weighted = clustered_paired_bootstrap(
        baseline, candidate, clusters, cluster_weighting="case"
    )
    equal_cluster = clustered_paired_bootstrap(
        baseline, candidate, clusters, cluster_weighting="equal"
    )
    if not (
        case_weighted.method == "exact"
        and case_weighted.resamples_evaluated == 4
        and case_weighted.cluster_sizes == (5, 1)
        and math.isclose(case_weighted.mean_difference, 4 / 6)
        and math.isclose(case_weighted.confidence_low, -0.875)
        and math.isclose(case_weighted.confidence_high, 0.975)
        and math.isclose(case_weighted.probability_of_improvement, 3 / 4)
        and math.isclose(equal_cluster.mean_difference, 0)
        and math.isclose(equal_cluster.confidence_low, -0.925)
        and math.isclose(equal_cluster.confidence_high, 0.925)
        and math.isclose(equal_cluster.probability_of_improvement, 1 / 4)
    ):
        errors.append("cluster-bootstrap estimand/interval fixture mismatch")

    demo = (
        ROOT / "projects" / "evaluation-gate" / "clustered_bootstrap_toy.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"ordered_cluster_resamples_enumerated": True',
        '"case_and_equal_weighting_treated_as_same_estimand": False',
        '"within_cluster_case_independence_required": False',
        '"representative_independent_clusters_established": False',
        '"bca_or_small_cluster_coverage_guarantee": False',
        '"causal_or_general_model_improvement_proved": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(
            f"clustered-bootstrap toy missing scope marker(s): {missing_scope}"
        )

    methodology = (
        ROOT / "docs" / "quality" / "evaluation-methodology.md"
    ).read_text(encoding="utf-8")
    required_inline_math = (
        r"cluster \(g\)",
        r"\(G^G\) 个 ordered resample",
        r"observed \(4/6\)",
    )
    missing_math = [
        marker for marker in required_inline_math if marker not in methodology
    ]
    if missing_math:
        errors.append(
            f"cluster inference docs missing inline-math marker(s): {missing_math}"
        )
    return errors


def check_holm_correction_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.evaluation import holm_bonferroni_correction

    errors: list[str] = []
    result = holm_bonferroni_correction([0.04, 0.01, 0.03, 0.20], alpha=0.05)
    ordered = result.ordered_hypotheses
    if not (
        result.family_size == 4
        and tuple(item.original_index for item in ordered) == (1, 2, 0, 3)
        and tuple(item.multiplier for item in ordered) == (4, 3, 2, 1)
        and all(
            math.isclose(actual, expected)
            for actual, expected in zip(
                (item.scaled_p_value for item in ordered),
                (0.04, 0.09, 0.08, 0.20),
                strict=True,
            )
        )
        and all(
            math.isclose(actual, expected)
            for actual, expected in zip(
                result.adjusted_p_values,
                (0.09, 0.04, 0.09, 0.20),
                strict=True,
            )
        )
        and result.rejected == (False, True, False, False)
    ):
        errors.append("Holm rank/running-maximum/input-remap fixture mismatch")

    tied = holm_bonferroni_correction([0.01, 0.01, 0.20])
    if not (
        tuple(item.original_index for item in tied.ordered_hypotheses) == (0, 1, 2)
        and all(
            math.isclose(actual, expected)
            for actual, expected in zip(
                tied.adjusted_p_values,
                (0.03, 0.03, 0.20),
                strict=True,
            )
        )
    ):
        errors.append("Holm stable-tie fixture mismatch")

    demo = (
        ROOT / "projects" / "evaluation-gate" / "holm_correction_toy.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"holm_rank_and_running_maximum_executed": True',
        '"arbitrary_dependence_fwer_control_requires_valid_input_p_values": True',
        '"family_prespecified_or_selection_bias_repaired": False',
        '"repeated_peeking_or_optional_stopping_repaired": False',
        '"effect_size_or_practical_importance_estimated": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(f"Holm toy missing scope marker(s): {missing_scope}")
    return errors


def check_sequential_peeking_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from fractions import Fraction

    from about_llm.evaluation import (
        analyze_repeated_two_sided_sign_tests,
        two_sided_sign_test_p_value,
    )

    errors: list[str] = []
    look_sample_counts = (10, 20, 30, 40, 50)
    naive = analyze_repeated_two_sided_sign_tests(
        look_sample_counts,
        per_look_alpha=Fraction(1, 20),
    )
    bonferroni = analyze_repeated_two_sided_sign_tests(
        look_sample_counts,
        per_look_alpha=Fraction(1, 100),
    )
    if not (
        two_sided_sign_test_p_value(positive_count=1, sample_count=10)
        == Fraction(11, 512)
        and two_sided_sign_test_p_value(positive_count=9, sample_count=10)
        == Fraction(11, 512)
        and naive.familywise_null_rejection_probability
        == Fraction(7_109_832_616_777, 70_368_744_177_664)
        and naive.looks[-1].fixed_look_null_rejection_probability
        == Fraction(18_486_790_962_201, 562_949_953_421_312)
        and naive.familywise_null_rejection_probability > Fraction(1, 20)
        and naive.logical_binary_sign_sequences == 2**50
        and naive.dynamic_programming_state_cells_evaluated == 685
        and bonferroni.familywise_null_rejection_probability
        == Fraction(2_142_139_082_367, 140_737_488_355_328)
        and bonferroni.familywise_null_rejection_probability < Fraction(1, 20)
        and bonferroni.union_bound == Fraction(1, 20)
    ):
        errors.append("sequential peeking exact/Bonferroni fixture mismatch")

    demo = (
        ROOT / "projects" / "evaluation-gate" / "sequential_peeking_toy.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"exact_fraction_dynamic_program_executed": True',
        '"logical_sign_sequence_enumeration_executed": False',
        '"look_schedule_and_thresholds_prespecified": True',
        '"confidence_sequence_or_always_valid_p_value_implemented": False',
        '"effect_size_power_or_sample_size_estimated": False',
        '"case_sampling_labels_clusters_or_exchangeability_validated": False',
        '"model_judge_provider_or_online_ab_test_executed": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(
            f"sequential-peeking toy missing scope marker(s): {missing_scope}"
        )

    methodology = (
        ROOT / "docs" / "quality" / "evaluation-methodology.md"
    ).read_text(encoding="utf-8")
    required_methodology = (
        "7109832616777/70368744177664",
        "18486790962201/562949953421312",
        "2142139082367/140737488355328",
        "不枚举 \\(2^{50}\\)",
        "不允许临时增加第六次 look",
        "不消除停止后 effect estimate 的选择偏差",
    )
    missing_methodology = [
        marker for marker in required_methodology if marker not in methodology
    ]
    if missing_methodology:
        errors.append(
            "sequential-peeking docs missing evidence/boundary marker(s): "
            f"{missing_methodology}"
        )
    return errors


def check_gradient_accumulation_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from fractions import Fraction

    from about_llm.finetuning import (
        CategoricalMicrobatch,
        CategoricalTokenRecord,
        analyze_default_ddp_gradient_accumulation,
        analyze_default_ddp_token_mean,
        analyze_masked_token_gradient_accumulation,
    )

    errors: list[str] = []
    microbatches = (
        CategoricalMicrobatch(
            "short",
            (
                CategoricalTokenRecord("short.valid", (9, 1), 0),
                CategoricalTokenRecord("short.padding-1", (1, 1), None),
                CategoricalTokenRecord("short.padding-2", (1, 1), None),
            ),
        ),
        CategoricalMicrobatch(
            "long",
            (
                CategoricalTokenRecord("long.valid-1", (4, 1), 1),
                CategoricalTokenRecord("long.valid-2", (4, 1), 1),
                CategoricalTokenRecord("long.valid-3", (4, 1), 1),
                CategoricalTokenRecord("long.padding", (1, 1), None),
            ),
        ),
    )
    analysis = analyze_masked_token_gradient_accumulation(microbatches)
    if not (
        analysis.valid_token_count == 4
        and analysis.ignored_token_count == 3
        and tuple(item.valid_token_count for item in analysis.microbatches) == (1, 3)
        and tuple(item.correct_global_weight for item in analysis.microbatches)
        == (Fraction(1, 4), Fraction(3, 4))
        and tuple(
            item.naive_equal_microbatch_weight for item in analysis.microbatches
        )
        == (Fraction(1, 2), Fraction(1, 2))
        and analysis.full_batch_class_aggregate_logit_gradient
        == (Fraction(23, 40), Fraction(-23, 40))
        and analysis.count_scaled_accumulated_class_aggregate_logit_gradient
        == analysis.full_batch_class_aggregate_logit_gradient
        and analysis.naive_equal_microbatch_class_aggregate_logit_gradient
        == (Fraction(7, 20), Fraction(-7, 20))
        and analysis.naive_minus_full_class_aggregate_logit_gradient
        == (Fraction(-9, 40), Fraction(9, 40))
    ):
        errors.append("masked-token gradient-accumulation exact fixture mismatch")

    ddp_analysis = analyze_default_ddp_token_mean(
        microbatches,
        data_parallel_world_size=2,
    )
    if not (
        ddp_analysis.data_parallel_world_size == 2
        and ddp_analysis.valid_token_count == 4
        and ddp_analysis.ignored_token_count == 3
        and ddp_analysis.valid_token_counts_by_rank == (1, 3)
        and ddp_analysis.correct_local_loss_sum_scale == Fraction(1, 2)
        and ddp_analysis.missing_world_size_local_loss_sum_scale
        == Fraction(1, 4)
        and ddp_analysis.rank_local_sum_class_aggregate_logit_gradients
        == (
            (Fraction(-1, 10), Fraction(1, 10)),
            (Fraction(12, 5), Fraction(-12, 5)),
        )
        and ddp_analysis.full_batch_class_aggregate_logit_gradient
        == (Fraction(23, 40), Fraction(-23, 40))
        and ddp_analysis.correctly_scaled_default_ddp_class_aggregate_logit_gradient
        == ddp_analysis.full_batch_class_aggregate_logit_gradient
        and ddp_analysis.missing_world_size_default_ddp_class_aggregate_logit_gradient
        == (Fraction(23, 80), Fraction(-23, 80))
        and ddp_analysis.equal_rank_local_mean_class_aggregate_logit_gradient
        == (Fraction(7, 20), Fraction(-7, 20))
    ):
        errors.append("default-DDP token-mean exact fixture mismatch")

    ddp_accumulation_windows = (
        (
            CategoricalMicrobatch(
                "rank-0.micro-0",
                (
                    CategoricalTokenRecord("r0.m0.valid", (9, 1), 0),
                    CategoricalTokenRecord("r0.m0.ignored", (1, 1), None),
                ),
            ),
            CategoricalMicrobatch(
                "rank-0.micro-1",
                (
                    CategoricalTokenRecord("r0.m1.valid-1", (4, 1), 1),
                    CategoricalTokenRecord("r0.m1.valid-2", (4, 1), 1),
                    CategoricalTokenRecord("r0.m1.ignored", (1, 1), None),
                ),
            ),
        ),
        (
            CategoricalMicrobatch(
                "rank-1.micro-0",
                (
                    CategoricalTokenRecord("r1.m0.valid-1", (4, 1), 1),
                    CategoricalTokenRecord("r1.m0.valid-2", (4, 1), 1),
                    CategoricalTokenRecord("r1.m0.valid-3", (4, 1), 1),
                    CategoricalTokenRecord("r1.m0.ignored", (1, 1), None),
                ),
            ),
            CategoricalMicrobatch(
                "rank-1.micro-1",
                (
                    CategoricalTokenRecord("r1.m1.valid", (9, 1), 0),
                    CategoricalTokenRecord("r1.m1.ignored", (1, 1), None),
                ),
            ),
        ),
    )
    ddp_accumulation = analyze_default_ddp_gradient_accumulation(
        ddp_accumulation_windows,
        data_parallel_world_size=2,
        unclipped_sgd_learning_rate=Fraction(7, 20),
    )
    if not (
        ddp_accumulation.accumulation_steps == 2
        and ddp_accumulation.valid_token_count == 7
        and ddp_accumulation.ignored_token_count == 4
        and ddp_accumulation.valid_token_counts_by_rank_and_microbatch
        == ((1, 2), (3, 1))
        and ddp_accumulation.valid_token_counts_by_rank == (3, 4)
        and ddp_accumulation.correct_local_loss_sum_scale == Fraction(2, 7)
        and ddp_accumulation.rank_accumulated_loss_sum_class_aggregate_logit_gradients
        == (
            (Fraction(3, 2), Fraction(-3, 2)),
            (Fraction(23, 10), Fraction(-23, 10)),
        )
        and ddp_accumulation.full_batch_class_aggregate_logit_gradient
        == (Fraction(19, 35), Fraction(-19, 35))
        and ddp_accumulation.one_sync_after_accumulation_class_aggregate_logit_gradient
        == ddp_accumulation.full_batch_class_aggregate_logit_gradient
        and ddp_accumulation.sync_every_microbatch_class_aggregate_logit_gradient
        == ddp_accumulation.full_batch_class_aggregate_logit_gradient
        and ddp_accumulation.unclipped_sgd_parameter_delta
        == (Fraction(-19, 100), Fraction(19, 100))
    ):
        errors.append("default-DDP accumulation exact fixture mismatch")

    demo = (
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "gradient_accumulation_toy.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"exact_fraction_logit_gradient_oracle_executed": True',
        '"pytorch_float64_cross_entropy_backward_executed": True',
        '"optimizer_step_or_parameter_update_executed": False',
        '"dropout_batchnorm_or_stochastic_model_equivalence_proved": False',
        '"ddp_fsdp_zero_collective_or_no_sync_executed": False',
        '"amp_cuda_gpu_memory_throughput_or_quality_measured": False',
        '"target_llm_tokenizer_dataset_or_training_run_executed": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(
            "gradient-accumulation toy missing scope marker(s): "
            f"{missing_scope}"
        )

    ddp_demo = (
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "ddp_token_mean_control.py"
    ).read_text(encoding="utf-8")
    required_ddp_execution = (
        'backend="gloo"',
        "dist.all_reduce(count, op=dist.ReduceOp.SUM)",
        "DistributedDataParallel(model)",
        '"real_two_process_same_host_gloo_process_group_executed": True',
        '"default_ddp_gradient_averaging_observed": True',
        '"global_valid_token_count_all_reduce_executed": True',
        '"temporary_file_store_rendezvous_executed": True',
    )
    missing_ddp_execution = [
        marker for marker in required_ddp_execution if marker not in ddp_demo
    ]
    if missing_ddp_execution:
        errors.append(
            "default-DDP control missing execution marker(s): "
            f"{missing_ddp_execution}"
        )
    required_ddp_boundaries = (
        '"optimizer_step_parameter_update_or_gradient_clipping_executed": False',
        '"gradient_accumulation_no_sync_amp_or_scaler_executed": False',
        '"fsdp_zero_tensor_pipeline_expert_parallel_executed": False',
        '"cuda_gpu_multi_node_or_remote_host_executed": False',
        '"target_llm_tokenizer_dataset_or_quality_evaluation_executed": False',
        '"bitwise_equivalence_across_hardware_or_world_sizes_proved": False',
        '"transport_security_packet_capture_or_fault_injection_executed": False',
    )
    missing_ddp_boundaries = [
        marker for marker in required_ddp_boundaries if marker not in ddp_demo
    ]
    if missing_ddp_boundaries:
        errors.append(
            "default-DDP control missing boundary marker(s): "
            f"{missing_ddp_boundaries}"
        )

    ddp_accumulation_demo = (
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "ddp_accumulation_no_sync_control.py"
    ).read_text(encoding="utf-8")
    required_accumulation_execution = (
        "with ddp_model.no_sync():",
        "ddp_model.register_comm_hook(hook_state, _counting_allreduce_hook)",
        "default_hooks.allreduce_hook(state.process_group, bucket)",
        "torch.nn.utils.clip_grad_norm_",
        "torch.optim.SGD",
        '"builtin_ddp_no_sync_forward_and_backward_scope_executed": True',
        '"pytorch_reference_allreduce_hook_counting_control_executed": True',
        '"backward_only_no_sync_negative_control_executed": True',
        '"gradient_clipping_after_synchronized_normalization_executed": True',
        '"plain_sgd_optimizer_step_and_parameter_update_executed": True',
    )
    missing_accumulation_execution = [
        marker
        for marker in required_accumulation_execution
        if marker not in ddp_accumulation_demo
    ]
    if missing_accumulation_execution:
        errors.append(
            "DDP accumulation control missing execution marker(s): "
            f"{missing_accumulation_execution}"
        )
    required_accumulation_boundaries = (
        '"builtin_reducer_collective_count_directly_instrumented": False',
        '"multiple_parameters_or_multiple_gradient_buckets_executed": False',
        '"dropout_batchnorm_or_stochastic_rng_equivalence_executed": False',
        '"amp_scaler_or_overflow_path_executed": False',
        '"fsdp_zero_tensor_pipeline_expert_parallel_executed": False',
        '"cuda_gpu_multi_node_or_remote_host_executed": False',
        '"target_llm_tokenizer_dataset_trainer_or_quality_evaluation_executed": (',
        '"optimizer_state_checkpoint_resume_or_failure_recovery_executed": False',
        '"throughput_latency_memory_or_communication_bytes_measured": False',
        '"bitwise_equivalence_across_hardware_or_world_sizes_proved": False',
        '"transport_security_packet_capture_or_fault_injection_executed": False',
    )
    missing_accumulation_boundaries = [
        marker
        for marker in required_accumulation_boundaries
        if marker not in ddp_accumulation_demo
    ]
    if missing_accumulation_boundaries:
        errors.append(
            "DDP accumulation control missing boundary marker(s): "
            f"{missing_accumulation_boundaries}"
        )

    docs = {
        "foundations": (
            ROOT / "docs" / "foundations" / "ml-dl.md"
        ).read_text(encoding="utf-8"),
        "pretraining": (
            ROOT / "docs" / "training" / "pretraining.md"
        ).read_text(encoding="utf-8"),
        "distributed": (
            ROOT / "docs" / "systems" / "distributed-training.md"
        ).read_text(encoding="utf-8"),
        "sft": (
            ROOT / "docs" / "training" / "sft-data-pipeline.md"
        ).read_text(encoding="utf-8"),
        "project": (
            ROOT / "projects" / "single-gpu-finetuning" / "README.md"
        ).read_text(encoding="utf-8"),
        "project_page": (
            ROOT
            / "docs"
            / "practice"
            / "projects"
            / "single-gpu-finetuning.md"
        ).read_text(encoding="utf-8"),
        "labs": (ROOT / "docs" / "practice" / "labs.md").read_text(
            encoding="utf-8"
        ),
        "interview": (
            ROOT / "docs" / "career" / "interview-questions.md"
        ).read_text(encoding="utf-8"),
        "accuracy": (
            ROOT / "docs" / "reference" / "accuracy.md"
        ).read_text(encoding="utf-8"),
        "production": (
            ROOT / "docs" / "practice" / "production-checklist.md"
        ).read_text(encoding="utf-8"),
    }
    required_docs = {
        "foundations": (
            "两个不同的 estimand",
            "1/(Mn_i)",
            "`D/N=1/2`",
            "只把 backward 放进 `no_sync` 已经太晚",
        ),
        "pretraining": (
            "可变 token 下的梯度累积",
            "DDP 默认 gradient averaging",
            "`(0.2875,-0.2875)`",
            "每 rank 两个 micro-batch",
            "只包 backward 时 2 次",
        ),
        "distributed": (
            "D/global_N",
            "temporary FileStore",
            "`(+23/80,-23/80)`",
            "gradient accumulation + `no_sync`",
            "(+19/35,-19/35)",
            "`default_hooks.allreduce_hook`",
            "一个两元素参数和一个 bucket",
        ),
        "sft": (
            "labels != -100",
            "Sequence mean",
            "不能替代真实 SFT 集成测试",
            "只包 backward 有两次",
        ),
        "project": (
            "(+23/40,-23/40)",
            "torch.equal",
            "双进程 CPU/Gloo DDP token-mean 控制",
            "跨硬件/world-size bitwise 等价也未证明",
            "DDP accumulation、`no_sync`、clip 与 SGD 控制",
            "built-in reducer 本身没有被直接插桩计数",
        ),
        "project_page": (
            "`D=2,N=4`",
            "`(+23/80,-23/80)`",
            "accumulation + `no_sync`",
            "counts `[[1,2],[3,1]]`",
            "负对照的数值在本线性 fixture 上仍正确",
        ),
        "labs": (
            "实验 4E",
            "真实双进程 DDP token mean",
            "实验 4F",
            "只包 backward 为 2 次",
            "`all_reduce`",
            "跨硬件/world-size bitwise 等价",
        ),
        "interview": (
            "loss/accumulation_steps",
            "world-size 因子",
            "为什么要乘 `D/N`",
            "为什么要同时包住 forward 与 backward",
            "通信没有省掉",
        ),
        "accuracy": (
            "class aggregation 不是目标模型参数梯度",
            "Default-DDP token-mean control",
            "gradient accumulation + `no_sync`",
            "DDP accumulation/`no_sync`/update control",
            "built-in reducer collective count 未直接插桩",
        ),
        "production": (
            "`no_sync` 必须包住非末批的 forward+backward",
            "单参数/单 bucket",
            "native distributed scaler",
        ),
    }
    for name, markers in required_docs.items():
        missing = [marker for marker in markers if marker not in docs[name]]
        if missing:
            errors.append(
                f"gradient-accumulation {name} docs missing marker(s): {missing}"
            )
    return errors


def check_amp_grad_scaler_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.finetuning.amp_scaler import run_cpu_amp_grad_scaler_control

    errors: list[str] = []
    report = run_cpu_amp_grad_scaler_control().to_dict()
    clip = report.get("clip_ordering", {})
    correct = clip.get("correct_unscale_then_clip", {})
    wrong = clip.get("wrong_clip_then_unscale", {})
    full = clip.get("full_batch_reference", {})
    resume = report.get("overflow_and_resume", {})
    initial = resume.get("initial_finite_adamw_step", {})
    overflows = resume.get("overflow_windows", [])
    checkpoint = resume.get("checkpoint", {})
    uninterrupted = resume.get("uninterrupted_after_checkpoint", {})
    restored = resume.get("restored_with_scaler_state", {})
    omitted = resume.get("restored_without_scaler_state", {})
    assertions = report.get("assertions", {})
    scope = report.get("scope", {})
    expected_true_scope = (
        "real_cpu_float16_autocast_executed",
        "real_cpu_grad_scaler_executed",
        "two_microbatch_scaled_accumulation_executed",
        "unscale_then_global_norm_clip_executed",
        "clip_before_unscale_negative_control_executed",
        "real_adamw_moments_and_step_executed",
        "intentional_nonfinite_accumulation_windows_executed",
        "in_memory_model_optimizer_scaler_resume_executed",
        "omitted_scaler_state_negative_control_executed",
    )
    expected_false_scope = (
        "cuda_or_gpu_kernel_executed",
        "file_checkpoint_or_process_restart_executed",
        "scheduler_rng_dataloader_or_distributed_state_executed",
        "target_model_trainer_tokenizer_or_dataset_executed",
        "convergence_quality_throughput_or_memory_proved",
    )
    overflow_scales = [
        (item.get("scale_before"), item.get("scale_after"))
        for item in overflows
        if isinstance(item, dict)
    ]
    overflow_skips = all(
        item.get("microbatch_count") == 2
        and item.get("scaled_gradient_is_finite") is False
        and item.get("scaled_gradient") is None
        and item.get("optimizer_step_executed") is False
        and item.get("parameter_before") == item.get("parameter_after")
        and item.get("optimizer_state_before") == item.get("optimizer_state_after")
        for item in overflows
        if isinstance(item, dict)
    )
    if not (
        report.get("implementation") == "about-llm.amp-grad-scaler-control.v1"
        and report.get("runtime", {}).get("device") == "cpu"
        and report.get("runtime", {}).get("autocast_dtype") == "torch.float16"
        and full.get("gradient_before_clip") == 3.0
        and correct.get("scaled_gradient_before_ordering") == 24.0
        and correct.get("clip_input_gradient") == 3.0
        and correct.get("reported_pre_clip_norm") == 3.0
        and math.isclose(
            correct.get("optimizer_gradient", math.inf),
            0.4999998211860657,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and correct.get("optimizer_gradient") == full.get("gradient_after_clip")
        and correct.get("parameter_after_step") == full.get("parameter_after_step")
        and wrong.get("scaled_gradient_before_ordering") == 24.0
        and wrong.get("clip_input_gradient") == 24.0
        and math.isclose(
            wrong.get("optimizer_gradient", math.inf),
            0.0624999962747097,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and initial.get("optimizer_step_executed") is True
        and initial.get("optimizer_state_after", {}).get("step") == 1
        and len(overflows) == 3
        and overflow_scales == [(8.0, 4.0), (4.0, 2.0), (2.0, 1.0)]
        and overflow_skips
        and checkpoint.get("grad_scaler_state", {}).get("scale") == 1.0
        and checkpoint.get("optimizer_state", {}).get("step") == 1
        and uninterrupted.get("optimizer_state_after")
        == restored.get("optimizer_state_after")
        and uninterrupted.get("parameter_after") == restored.get("parameter_after")
        and restored.get("scaled_gradient") == 10000.0
        and restored.get("optimizer_step_executed") is True
        and restored.get("optimizer_state_after", {}).get("step") == 2
        and omitted.get("scale_before") == 8.0
        and omitted.get("scale_after") == 4.0
        and omitted.get("scaled_gradient") is None
        and omitted.get("optimizer_step_executed") is False
        and omitted.get("optimizer_state_after", {}).get("step") == 1
        and assertions
        and all(assertions.values())
        and all(scope.get(field) is True for field in expected_true_scope)
        and all(scope.get(field) is False for field in expected_false_scope)
    ):
        errors.append(f"CPU AMP/GradScaler fixture mismatch: {report}")
    try:
        json.dumps(report, ensure_ascii=False, allow_nan=False)
    except ValueError as error:
        errors.append(f"CPU AMP/GradScaler report is not strict finite JSON: {error}")

    module = (
        ROOT / "src" / "about_llm" / "finetuning" / "amp_scaler.py"
    ).read_text(encoding="utf-8")
    script = (
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "amp_grad_scaler_control.py"
    ).read_text(encoding="utf-8")
    execution_markers = (
        'torch.amp.GradScaler(',
        'torch.amp.autocast(device_type="cpu", dtype=torch.float16)',
        "scaler.scale(loss).backward()",
        "scaler.unscale_(optimizer)",
        "torch.nn.utils.clip_grad_norm_",
        "torch.optim.AdamW",
        'loss_multipliers=(1.0, float("inf"))',
        "restore_scaler=False",
        '"file_checkpoint_or_process_restart_executed": False',
        '"cuda_or_gpu_kernel_executed": False',
    )
    missing_execution = [marker for marker in execution_markers if marker not in module]
    if missing_execution:
        errors.append(
            f"CPU AMP/GradScaler control missing execution marker(s): {missing_execution}"
        )
    script_markers = ("allow_nan=False", "run_cpu_amp_grad_scaler_control")
    missing_script = [marker for marker in script_markers if marker not in script]
    if missing_script:
        errors.append(f"CPU AMP/GradScaler CLI missing marker(s): {missing_script}")

    docs = {
        "foundations": ROOT / "docs" / "foundations" / "ml-dl.md",
        "pretraining": ROOT / "docs" / "training" / "pretraining.md",
        "sft": ROOT / "docs" / "training" / "sft-data-pipeline.md",
        "distributed": ROOT / "docs" / "systems" / "distributed-training.md",
        "finetuning": ROOT / "docs" / "training" / "finetuning.md",
        "peft": ROOT / "docs" / "training" / "peft-qlora-engineering.md",
        "project": ROOT / "projects" / "single-gpu-finetuning" / "README.md",
        "project_page": (
            ROOT
            / "docs"
            / "practice"
            / "projects"
            / "single-gpu-finetuning.md"
        ),
        "labs": ROOT / "docs" / "practice" / "labs.md",
        "interview": ROOT / "docs" / "career" / "interview-questions.md",
        "accuracy": ROOT / "docs" / "reference" / "accuracy.md",
        "production": ROOT / "docs" / "practice" / "production-checklist.md",
    }
    required_docs = {
        "foundations": ("optimizer 只看到约 0.0625", "scale `8→4→2→1`"),
        "pretraining": ("错误 `clip→unscale_`", "磁盘 checkpoint"),
        "sft": ("不能推进 scheduler", "scale=1/8"),
        "distributed": (
            "AMP scale、unscale、clip 与 skip control",
            "in-memory state replay",
        ),
        "finetuning": ("为什么 scaler 必须保存", "单参数、进程内 replay"),
        "peft": ("scaler omission", "不能补写成 PEFT exact resume 已完成"),
        "project": ("CPU AMP/GradScaler overflow", "finite=false,value=null"),
        "project_page": ("AMP state machine", "in-memory 恢复 scale=1"),
        "labs": ("实验 4G", "只加载 model+optimizer"),
        "interview": (
            "为什么必须先 unscale 再 clip",
            "不要用 `optimizer.step()` 的返回值判断成功",
        ),
        "accuracy": ("CPU AMP/GradScaler control", "allow_nan=False"),
        "production": ("scale-sensitive overflow/finite", "进程退出/重启"),
    }
    for name, path in docs.items():
        text = path.read_text(encoding="utf-8")
        missing = [marker for marker in required_docs[name] if marker not in text]
        if missing:
            errors.append(f"CPU AMP/GradScaler {name} docs missing marker(s): {missing}")
    return errors


def check_ddp_amp_overflow_consensus_control() -> list[str]:
    errors: list[str] = []
    control_path = (
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "ddp_amp_overflow_consensus_control.py"
    )
    test_path = ROOT / "tests" / "test_ddp_amp_overflow_consensus.py"
    control = control_path.read_text(encoding="utf-8")
    test = test_path.read_text(encoding="utf-8")
    execution_markers = (
        "DistributedDataParallel(model)",
        "with ddp_model.no_sync():",
        'torch.amp.GradScaler(',
        'torch.amp.autocast(device_type="cpu", dtype=torch.float16)',
        'model.weight.grad.fill_(float("inf"))',
        "scaler.unscale_(optimizer)",
        "dist.all_reduce(nonfinite_flag, op=dist.ReduceOp.MAX)",
        "scaler.update(new_scale=scale_before * BACKOFF_FACTOR)",
        "torch.optim.AdamW(",
        "torch.optim.lr_scheduler.StepLR(",
        '"builtin_default_ddp_reducer_executed": True',
        '"rank_local_scaler_divergence_negative_control_executed": True',
        '"global_nonfinite_max_all_reduce_gate_executed": True',
        '"common_manual_scaler_backoff_after_global_skip_executed": True',
        '"native_grad_scaler_found_inf_state_synchronized": False',
        '"multiple_parameters_or_multiple_gradient_buckets_executed": False',
        '"cuda_nccl_gpu_multi_node_or_remote_host_executed": False',
        "allow_nan=False",
    )
    missing_execution = [
        marker for marker in execution_markers if marker not in control
    ]
    if missing_execution:
        errors.append(
            "DDP AMP overflow-consensus control missing marker(s): "
            f"{missing_execution}"
        )

    test_markers = (
        "about-llm.ddp-amp-overflow-consensus-control.v1",
        'parse_constant=_reject_nonfinite_json',
        '"optimizer_step_executed"] for item in local] == [False, True]',
        '"global_nonfinite_after_max_all_reduce"] for item in gated',
        '"growth_tracker": 0',
        '"growth_tracker": 1',
        '"growth_tracker": 2',
        '"native_grad_scaler_found_inf_state_synchronized": False',
    )
    missing_tests = [marker for marker in test_markers if marker not in test]
    if missing_tests:
        errors.append(
            f"DDP AMP overflow-consensus tests missing marker(s): {missing_tests}"
        )

    docs = {
        "distributed": (
            ROOT / "docs" / "systems" / "distributed-training.md"
        ).read_text(encoding="utf-8"),
        "pretraining": (
            ROOT / "docs" / "training" / "pretraining.md"
        ).read_text(encoding="utf-8"),
        "finetuning": (
            ROOT / "docs" / "training" / "finetuning.md"
        ).read_text(encoding="utf-8"),
        "project": (
            ROOT / "projects" / "single-gpu-finetuning" / "README.md"
        ).read_text(encoding="utf-8"),
        "project_page": (
            ROOT
            / "docs"
            / "practice"
            / "projects"
            / "single-gpu-finetuning.md"
        ).read_text(encoding="utf-8"),
        "labs": (ROOT / "docs" / "practice" / "labs.md").read_text(
            encoding="utf-8"
        ),
        "interview": (
            ROOT / "docs" / "career" / "interview-questions.md"
        ).read_text(encoding="utf-8"),
        "accuracy": (
            ROOT / "docs" / "reference" / "accuracy.md"
        ).read_text(encoding="utf-8"),
        "production": (
            ROOT / "docs" / "practice" / "production-checklist.md"
        ).read_text(encoding="utf-8"),
    }
    required_docs = {
        "distributed": (
            "DDP + AMP overflow 共识控制",
            "rank 0 的 `unscale_` **之前**",
            "growth tracker 保持 1",
            "不是可直接复制到所有框架",
        ),
        "pretraining": (
            "`ddp_amp_overflow_consensus_control.py`",
            "rank 0 保持 step=1、rank 1 前进到 step=2",
            "显式 scale policy",
        ),
        "finetuning": (
            "多个 rank 是否对同一 update 作相同决定",
            "post-reduction 故障是 authored counterfactual",
            "分布式 exact resume 已完成",
        ),
        "project": (
            "双进程 DDP + AMP overflow 共识控制",
            "finite=false,value=null",
            "不能反向声称 vanilla DDP",
        ),
        "project_page": (
            "optimizer step 变成 `[1,2]`",
            "growth tracker 保持 1",
            "不能当通用 distributed scaler 实现",
        ),
        "labs": (
            "实验 4I",
            "local non-finite flags `[1,0]`",
            "不得通过写私有 found-inf 字段",
        ),
        "interview": (
            "一个 rank overflow",
            "不能脱离 overflow 发生位置回答",
            "step 后才发现 checksum 不同已经太晚",
        ),
        "accuracy": (
            "DDP + AMP overflow-consensus control",
            "post-reduction 损坏是 authored counterfactual",
            "collective count 未直接插桩",
        ),
        "production": (
            "任何 optimizer mutation 之前",
            "native distributed scaler",
            "post-reduction transform/故障",
        ),
    }
    for name, markers in required_docs.items():
        missing = [marker for marker in markers if marker not in docs[name]]
        if missing:
            errors.append(
                f"DDP AMP overflow-consensus {name} docs missing marker(s): {missing}"
            )

    return errors


def check_dataloader_prefetch_resume_control() -> list[str]:
    errors: list[str] = []
    control_path = (
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "dataloader_prefetch_resume_control.py"
    )
    test_path = ROOT / "tests" / "test_dataloader_prefetch_resume_control.py"
    control = control_path.read_text(encoding="utf-8")
    test = test_path.read_text(encoding="utf-8")
    execution_markers = (
        "DataLoader(",
        "num_workers=NUM_WORKERS",
        "prefetch_factor=PREFETCH_FACTOR",
        'multiprocessing_context="spawn"',
        "class TrackingOffsetSampler",
        "sampler.emitted_cursor",
        '"committed_cursor": SPLIT_COMMITTED_CURSOR',
        '"sampler_emitted_cursor": emitted_cursor',
        '"prefetched_but_uncommitted_sample_ids": skipped_ids',
        "worker_rng_max_abs_difference",
        "sample_keyed_max_abs_difference",
        '"real_dataloader_worker_processes_executed": True',
        '"worker_local_torch_rng_nonreplay_observed": True',
        '"sample_keyed_stateless_randomness_exact_replay_executed": True',
        '"private_dataloader_queue_fields_read_or_mutated": False',
        '"worker_local_rng_state_restored": False',
        '"sample_consumption_and_optimizer_commit_atomicity_proved": False',
        '"prefetch_depth_as_public_stable_api_contract_proved": False',
        "allow_nan=False",
    )
    missing_execution = [
        marker for marker in execution_markers if marker not in control
    ]
    if missing_execution:
        errors.append(
            "DataLoader prefetch-resume control missing marker(s): "
            f"{missing_execution}"
        )
    test_markers = (
        "about-llm.dataloader-prefetch-resume-control.v1",
        "duplicate JSON key",
        "non-finite JSON number",
        "canonical JSON",
        "prefetch-ahead",
        "refusing to overwrite",
        '"sampler_emitted_cursor_at_split": 7',
        '"prefetched_but_uncommitted_sample_ids": [7, 0, 9, 4]',
        '"sample_keyed_tail_max_abs_difference": 0.0',
        '"sample_consumption_and_optimizer_commit_atomicity_proved": False',
    )
    missing_tests = [marker for marker in test_markers if marker not in test]
    if missing_tests:
        errors.append(
            f"DataLoader prefetch-resume tests missing marker(s): {missing_tests}"
        )

    docs = {
        "finetuning": (
            ROOT / "docs" / "training" / "finetuning.md"
        ).read_text(encoding="utf-8"),
        "pretraining": (
            ROOT / "docs" / "training" / "pretraining.md"
        ).read_text(encoding="utf-8"),
        "sft": (
            ROOT / "docs" / "training" / "sft-data-pipeline.md"
        ).read_text(encoding="utf-8"),
        "peft": (
            ROOT / "docs" / "training" / "peft-qlora-engineering.md"
        ).read_text(encoding="utf-8"),
        "distributed": (
            ROOT / "docs" / "systems" / "distributed-training.md"
        ).read_text(encoding="utf-8"),
        "project": (
            ROOT / "projects" / "single-gpu-finetuning" / "README.md"
        ).read_text(encoding="utf-8"),
        "project_page": (
            ROOT
            / "docs"
            / "practice"
            / "projects"
            / "single-gpu-finetuning.md"
        ).read_text(encoding="utf-8"),
        "labs": (ROOT / "docs" / "practice" / "labs.md").read_text(
            encoding="utf-8"
        ),
        "interview": (
            ROOT / "docs" / "career" / "interview-questions.md"
        ).read_text(encoding="utf-8"),
        "accuracy": (
            ROOT / "docs" / "reference" / "accuracy.md"
        ).read_text(encoding="utf-8"),
        "production": (
            ROOT / "docs" / "practice" / "production-checklist.md"
        ).read_text(encoding="utf-8"),
        "knowledge_map": (
            ROOT / "docs" / "guide" / "knowledge-map.md"
        ).read_text(encoding="utf-8"),
        "repo_map": (ROOT / "docs" / "guide" / "repo-map.md").read_text(
            encoding="utf-8"
        ),
        "project_index": (
            ROOT / "docs" / "practice" / "project-index.md"
        ).read_text(encoding="utf-8"),
    }
    required_docs = {
        "finetuning": (
            "sampler emitted cursor=7",
            "worker-local `torch.rand` 最大差约 0.6544",
            "optimizer commit 与 sample consumption 原子",
        ),
        "pretraining": (
            "sampler 已产生 index",
            "从 7 恢复会跳过",
            "不能把具体 prefetch 深度当跨版本公开 API",
        ),
        "sft": (
            "DataLoader prefetch 与恢复 cursor",
            "emitted cursor 直接保存为“已训练位置”",
            "consumed 也不是 optimizer-committed",
        ),
        "peft": (
            "哪个 batch 已产生/已交付/已完成 backward/已执行 optimizer update",
            "不能把这个数据 control 借成 PEFT exact resume 证据",
        ),
        "distributed": (
            "sampler-emitted、main-loop-consumed 与 optimizer-committed cursor",
            "没有把 sample consumption 与 optimizer commit 做原子事务",
        ),
        "project": (
            "跨进程 DataLoader prefetch、cursor 与 worker RNG 控制",
            "[8,3,1,2,6,5]",
            "不能把它当任意版本/配置的公开 prefetch 深度保证",
        ),
        "project_page": (
            "真实 DataLoader worker/prefetch",
            "最大差约 0.654431",
            "consumed 不等于 optimizer committed",
        ),
        "labs": (
            "实验 4J",
            "sampler emitted、main-loop consumed、optimizer committed",
            "不得把它和 model checkpoint control 拼成完整 exact resume",
        ),
        "interview": (
            "sampler cursor 可能不能直接写进 checkpoint",
            "emitted=7、consumed=3",
            "worker-local RNG",
        ),
        "accuracy": (
            "DataLoader prefetch-resume control",
            "consumption—optimizer commit atomicity",
            "ahead=4 是当前 PyTorch 2.13.0+cpu fixture 的观察值",
        ),
        "production": (
            "sampler-emitted、main-loop-consumed 与 optimizer-committed cursor",
            "dataset/transform revision、epoch/visit、sample ID 的完整 stateless key",
        ),
        "knowledge_map": (
            "consumed=3 时 sampler emitted=7",
            "sample-ID-keyed tail exact",
            "仍未保存 queue/worker/Python/NumPy/CUDA RNG",
            "从 3 恢复也 bit-exact",
        ),
        "repo_map": (
            "2-worker prefetch control",
            "fresh worker RNG 不重放",
            "consumed=3/committed=2",
        ),
        "project_index": (
            "第六条 DataLoader control",
            "第七条扩展为六进程 main-process stochastic mask",
        ),
    }
    for name, markers in required_docs.items():
        missing = [marker for marker in markers if marker not in docs[name]]
        if missing:
            errors.append(
                f"DataLoader prefetch-resume {name} docs missing marker(s): {missing}"
            )
    return errors


def check_optimizer_commit_resume_control() -> list[str]:
    errors: list[str] = []
    control_path = (
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "optimizer_commit_resume_control.py"
    )
    test_path = ROOT / "tests" / "test_optimizer_commit_resume_control.py"
    control = control_path.read_text(encoding="utf-8")
    test = test_path.read_text(encoding="utf-8")
    execution_markers = (
        'CONTROL_VERSION = "about-llm.optimizer-commit-resume-control.v1"',
        'INFLIGHT_SIDECAR_VERSION = "about-llm.inflight-gradient-sidecar.v1"',
        'BUNDLE_MANIFEST_VERSION = "about-llm.optimizer-commit-bundle-manifest.v1"',
        "BUNDLE_MANIFEST_MAX_BYTES = 16 * 1024",
        "ACCUMULATION_STEPS = 2",
        "CRASH_AFTER_CONSUMED = 3",
        "COMMITTED_CURSOR_AT_CRASH = 2",
        "STOCHASTIC_MASK_SEED = 20260815",
        "DataLoader(",
        "torch.optim.SGD(",
        "torch.optim.lr_scheduler.StepLR(",
        "torch.rand_like(features)",
        "(loss / ACCUMULATION_STEPS).backward()",
        '"commit_boundary_torch_rng_state"',
        '"crash_observed_torch_rng_state"',
        "torch.set_rng_state(commit_rng)",
        "torch.set_rng_state(crash_rng)",
        '"in_flight_gradients_serialized": False',
        '"committed_cursor_resume_matches_uninterrupted_bit_exact"',
        '"negative_control_keeps_optimizer_step_count_but_diverges"',
        '"inflight_sidecar_resume_matches_uninterrupted_bit_exact"',
        '"wrong_rng_with_complete_gradients_isolated_negative_control"',
        '"rng_snapshots_select_commit_or_crash_boundary_as_declared"',
        '"inflight_sidecar_digest_and_base_binding_match_phase1"',
        '"bundle_manifest_matches_phase1_and_precedes_sidecar_load"',
        '"incomplete_and_tampered_bundle_snapshots_fail_closed"',
        '"in_flight_gradient_checkpoint_resume_executed": True',
        '"in_flight_sidecar_bound_to_base_checkpoint_digest": True',
        '"manifest_last_bundle_completeness_gate_executed": True',
        '"incomplete_and_tampered_bundle_fault_injection_executed": True',
        '"manifest_artifact_hashes_rechecked_at_payload_load": True',
        '"main_process_stochastic_mask_and_torch_cpu_rng_resume_executed": True',
        '"real_step_lr_advanced_after_optimizer_commit_executed": True',
        '"wrong_rng_with_complete_gradients_negative_control_executed": True',
        '"worker_rng_or_multi_epoch_policy_executed": False',
        '"grad_scaler_or_cuda_amp_executed": False',
        '"base_checkpoint_and_gradient_sidecar_atomic_publication_proved": False',
        '"concurrent_directory_replacement_or_storage_snapshot_proved": False',
        '"checkpoint_and_sample_commit_atomic_transaction_proved": False',
        '"power_loss_directory_fsync_or_storage_durability_proved": False',
        "loader contract values drifted",
        "in-flight sidecar base checkpoint digest drifted",
        "in-flight bundle manifest must use canonical JSON",
        "duplicate JSON key",
        "manifest is missing; publication is incomplete",
        "sidecar identity drifted",
        "optimizer momentum buffer must be finite",
        "weights_only=True",
        "io.BytesIO(encoded)",
        "os.replace(temporary_path, path)",
        "allow_nan=False",
    )
    missing_execution = [
        marker for marker in execution_markers if marker not in control
    ]
    if missing_execution:
        errors.append(
            "optimizer-commit resume control missing marker(s): "
            f"{missing_execution}"
        )

    test_markers = (
        "about-llm.optimizer-commit-resume-control.v1",
        "test_checkpoint_rejects_consumed_cursor_as_committed_boundary",
        "test_checkpoint_rejects_loader_contract_value_drift",
        "test_checkpoint_rejects_nonfinite_model_and_missing_momentum",
        "test_checkpoint_writer_refuses_overwrite",
        "test_inflight_sidecar_round_trip_and_base_digest_binding",
        "test_inflight_bundle_manifest_round_trip_and_fault_injection",
        "test_inflight_bundle_manifest_rejects_noncanonical_duplicate_and_unknown",
        "test_real_optimizer_commit_resume_control",
        '"about-llm.inflight-gradient-sidecar.v1"',
        '"about-llm.optimizer-commit-bundle-manifest.v1"',
        '"resume_inflight_gradient_pid"',
        '"resume_inflight_wrong_rng_negative_control_pid"',
        '"committed_resume_model_max_abs_difference"',
        '"consumed_resume_model_max_abs_difference"',
        '"inflight_resume_model_max_abs_difference"',
        '"wrong_rng_resume_model_max_abs_difference"',
        '"consumed_omission_terminal_torch_rng_sha256"',
        '"wrong_rng_terminal_torch_rng_sha256"',
        '"uninterrupted_optimizer_steps"',
        '"consumed_resume_optimizer_steps"',
        '"in_flight_gradient_checkpoint_resume_executed": True',
        '"in_flight_sidecar_bound_to_base_checkpoint_digest": True',
        '"manifest_last_bundle_completeness_gate_executed": True',
        '"incomplete_and_tampered_bundle_fault_injection_executed": True',
        '"manifest_artifact_hashes_rechecked_at_payload_load": True',
        '"main_process_stochastic_mask_and_torch_cpu_rng_resume_executed": True',
        '"real_step_lr_advanced_after_optimizer_commit_executed": True',
        '"wrong_rng_with_complete_gradients_negative_control_executed": True',
        '"worker_rng_or_multi_epoch_policy_executed": False',
        '"grad_scaler_or_cuda_amp_executed": False',
        '"base_checkpoint_and_gradient_sidecar_atomic_publication_proved": False',
        '"concurrent_directory_replacement_or_storage_snapshot_proved": False',
        '"checkpoint_and_sample_commit_atomic_transaction_proved": False',
    )
    missing_tests = [marker for marker in test_markers if marker not in test]
    if missing_tests:
        errors.append(
            f"optimizer-commit resume tests missing marker(s): {missing_tests}"
        )

    docs = {
        "finetuning": (
            ROOT / "docs" / "training" / "finetuning.md"
        ).read_text(encoding="utf-8"),
        "pretraining": (
            ROOT / "docs" / "training" / "pretraining.md"
        ).read_text(encoding="utf-8"),
        "sft": (
            ROOT / "docs" / "training" / "sft-data-pipeline.md"
        ).read_text(encoding="utf-8"),
        "peft": (
            ROOT / "docs" / "training" / "peft-qlora-engineering.md"
        ).read_text(encoding="utf-8"),
        "distributed": (
            ROOT / "docs" / "systems" / "distributed-training.md"
        ).read_text(encoding="utf-8"),
        "project": (
            ROOT / "projects" / "single-gpu-finetuning" / "README.md"
        ).read_text(encoding="utf-8"),
        "project_page": (
            ROOT
            / "docs"
            / "practice"
            / "projects"
            / "single-gpu-finetuning.md"
        ).read_text(encoding="utf-8"),
        "labs": (ROOT / "docs" / "practice" / "labs.md").read_text(
            encoding="utf-8"
        ),
        "interview": (
            ROOT / "docs" / "career" / "interview-questions.md"
        ).read_text(encoding="utf-8"),
        "accuracy": (
            ROOT / "docs" / "reference" / "accuracy.md"
        ).read_text(encoding="utf-8"),
        "production": (
            ROOT / "docs" / "practice" / "production-checklist.md"
        ).read_text(encoding="utf-8"),
        "knowledge_map": (
            ROOT / "docs" / "guide" / "knowledge-map.md"
        ).read_text(encoding="utf-8"),
        "repo_map": (ROOT / "docs" / "guide" / "repo-map.md").read_text(
            encoding="utf-8"
        ),
        "project_index": (
            ROOT / "docs" / "practice" / "project-index.md"
        ).read_text(encoding="utf-8"),
        "changelog": (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
    }
    required_docs = {
        "finetuning": (
            "optimizer_commit_resume_control.py",
            "optimizer-committed cursor=2",
            "六个独立顶层 PID",
            "当前 8,985-byte base checkpoint",
            "参数最大差仍为 `0.005767858566116724`",
            "参数最大差为 `0.017878893573032573`",
            "当前 7,905-byte sidecar",
            "最后发布的 827-byte 严格 canonical JSON manifest",
            "manifest-last 能识别当前进程故障留下的 incomplete/mismatched bundle",
        ),
        "pretraining": (
            "emitted/consumed/committed=`7/3/2`",
            "commit-boundary model/optimizer/scheduler/Torch RNG",
            "参数最大差仍为 `0.005767858566116724`",
            "参数最大差为 `0.017878893573032573`",
            "绑定 base digest 的 gradient sidecar",
            "最后发布的 canonical manifest",
            "base-only 并非坏 checkpoint",
        ),
        "sft": (
            "第三条 `[8,3,1]` 已被 main loop 消费并 stochastic backward",
            "同为 5 次 optimizer/StepLR step",
            "参数最大差仍约 `0.0057678586`",
            "参数最大差约 `0.0178788936`",
            "crash-observed Torch RNG",
            "sidecar 协议现在要求最后发布 strict canonical manifest",
            "base+sidecar+manifest 仍非 sample/optimizer 原子发布",
        ),
        "peft": (
            "从 2 恢复 RNG 并 replay 与 uninterrupted bit-exact",
            "`0.005767858566116724`",
            "`0.017878893573032573`",
            "pending `[1]`、position/divisor、逐参数 gradients 与 crash RNG",
            "最后发布的 strict canonical manifest",
            "manifest-last 只增加 completeness gate",
            "不能把 tiny control 借成 LoRA/QLoRA exact resume 证据",
        ),
        "distributed": (
            "committed=2 恢复 RNG并重放",
            "第五个 PID 加载绑定 base digest",
            "第六个 PID 保留 gradients/ledger",
            "参数仍漂移 `0.005767858566116724`",
            "参数漂移 `0.017878893573032573`",
            "post-manifest tamper 四种快照均 fail closed",
            "rank 间 ledger 共识",
            "base+sidecar+manifest 与所有 rank update 原子提交",
        ),
        "project": (
            "跨进程 consumed—optimizer-committed 崩溃窗口控制",
            "六个不同顶层 PID",
            "固定 seed `20260815`",
            "当前 8,985-byte `torch.save` base checkpoint",
            "从 consumed cursor 加载完整 sidecar 正确恢复",
            "参数最大差仍为 `0.005767858566116724`",
            "参数最大差为 `0.017878893573032573`",
            "sidecar→base digest",
            "父进程另构造四种 publication fault snapshots",
            "strict canonical JSON bundle manifest",
            "仅 base 仍可用于第一种 commit-boundary replay",
            "没有目录 `fsync`、断电/文件系统故障注入",
        ),
        "project_page": (
            "emitted/consumed/committed=`7/3/2`",
            "六个独立顶层 PID",
            "当前 8,985-byte base checkpoint",
            "最大参数差为 `0.005767858566116724`",
            "当前 7,905-byte sidecar",
            "当前 827-byte capped strict-canonical JSON manifest",
            "参数最大差为 `0.017878893573032573`",
            "base-only 仍可走 commit replay",
        ),
        "labs": (
            "实验 4K",
            "global step 和 RNG 相同仍不代表数据或训练轨迹相同",
            "8,985-byte base",
            "7,905-byte sidecar",
            "827-byte capped canonical manifest",
            "参数最大差 `0.017878893573032573`",
            "at-least-once replay",
            "base-only 仍可供 commit-boundary replay",
        ),
        "interview": (
            "31.4.2",
            "若像常见 model/optimizer `state_dict` 一样不含 parameter `.grad`",
            "六进程 2-worker/Float64/SGD/StepLR control",
            "参数最大差仍为 `0.005767858566116724`",
            "参数却漂移 `0.017878893573032573`",
            "若显式 sidecar 保存 pending `[1]`",
            "delivery semantics",
            "最后发布的 canonical manifest 为 complete",
            "manifest-last 是 completeness marker",
        ),
        "accuracy": (
            "Optimizer-commit resume control",
            "六个不同顶层 PID",
            "8,985-byte base",
            "7,905-byte sidecar",
            "827-byte strict canonical JSON manifest",
            "参数最大差仍为 `0.005767858566116724`",
            "参数最大差为 `0.017878893573032573`",
            "8 个测试另覆盖",
            "base+sidecar+manifest/checkpoint/sample/optimizer 原子事务",
        ),
        "production": (
            "必须从 commit boundary 重放",
            "commit-boundary RNG replay 与完整 gradients+crash-RNG sidecar resume 都可 bit-exact",
            "正确 RNG、相同 5 steps/LR 下漏 gradients/sample 并漂移",
            "gradients/ledger/steps/LR 完整时只因 RNG 错位而漂移",
            "manifest-last completeness gate",
            "sample—optimizer—base+sidecar+manifest 原子事务",
        ),
        "knowledge_map": (
            "emitted/consumed/committed=`7/3/2`",
            "seed `20260815`",
            "终态 RNG 相同下参数漂移 `0.005767858566116724`",
            "从 3 恢复也 bit-exact",
            "仍漂移 `0.017878893573032573`",
            "manifest-last completeness gate",
            "sample—optimizer—base+sidecar+manifest 原子事务",
        ),
        "repo_map": (
            "六进程 2-worker/stochastic-forward/backward/SGD/StepLR accumulation control",
            "完整 gradients+crash-RNG sidecar resume bit-exact",
            "正确 RNG、相同 steps/LR 仍会因漏 gradients/sample 漂移",
            "完整 ledger/gradients/steps/LR 仍会因错误 RNG 漂移",
            "manifest-last completeness",
            "原子 sample—optimizer—base+sidecar+manifest 事务",
        ),
        "project_index": (
            "第七条扩展为六进程 main-process stochastic mask",
            "相同 5 steps/LR 与终态 RNG 下仍漂移",
            "pending/position/divisor/gradients/crash-RNG sidecar",
            "完整 gradients/ledger/steps/LR 但使用错误 RNG",
            "最后发布 canonical manifest",
            "sample—optimizer—base+sidecar+manifest 原子事务",
        ),
        "changelog": (
            "六进程 stochastic RNG/StepLR 对照",
            "8,985-byte base checkpoint",
            "参数最大差仍为 `0.005767858566116724`",
            "7,905-byte sidecar 绑定 base SHA-256",
            "参数最大差为 `0.017878893573032573`",
            "strict canonical manifest",
            "base-only 仍可供 commit replay",
            "base+sidecar+manifest/sample/optimizer 并不原子",
        ),
    }
    for name, markers in required_docs.items():
        missing = [marker for marker in markers if marker not in docs[name]]
        if missing:
            errors.append(
                f"optimizer-commit resume {name} docs missing marker(s): {missing}"
            )
    return errors


def check_training_resume_process_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.finetuning.training_resume import (
        TRAINING_RESUME_CHECKPOINT_VERSION,
        TRAINING_RESUME_CONTROL_VERSION,
        run_training_resume_process_control,
    )

    errors: list[str] = []
    script_path = (
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "checkpoint_resume_control.py"
    )
    report = run_training_resume_process_control(script_path)
    fixture = report.get("fixture", {})
    processes = report.get("processes", {})
    checkpoint = report.get("checkpoint", {})
    uninterrupted = report.get("uninterrupted", {})
    split_resume = report.get("split_resume", {})
    negative = report.get("negative_controls", {})
    assertions = report.get("assertions", {})
    scope = report.get("scope", {})
    trace = uninterrupted.get("trace", [])
    phase1_trace = split_resume.get("phase1_trace", [])
    resumed_trace = split_resume.get("resumed_trace", [])
    terminal = uninterrupted.get("terminal", {})
    terminal_progress = terminal.get("progress", {})
    terminal_components = terminal.get("components", {})
    wrong_scheduler = negative.get("advance_scheduler_on_overflow", {}).get(
        "trace", []
    )
    omitted_scheduler = negative.get("omit_scheduler_state", {}).get("trace", [])
    omitted_scaler = negative.get("omit_grad_scaler_state", {}).get("trace", [])
    omitted_rng = negative.get("omit_rng_state", {}).get("trace", [])
    omitted_data = negative.get("omit_data_stream_state", {}).get("trace", [])
    expected_true_scope = (
        "real_independent_os_processes_executed",
        "phase1_process_exit_and_checkpoint_reopen_executed",
        "real_cpu_float16_autocast_executed",
        "real_cpu_grad_scaler_executed",
        "real_adamw_and_step_lr_executed",
        "intentional_nonfinite_optimizer_skip_executed",
        "scheduler_skip_and_wrong_advance_control_executed",
        "torch_cpu_and_python_rng_restored",
        "stateful_shuffle_generator_permutation_cursor_epoch_restored",
        "weights_only_checkpoint_load_executed",
        "scheduler_scaler_rng_and_data_omission_controls_executed",
    )
    expected_false_scope = (
        "cuda_or_gpu_kernel_executed",
        "dataloader_worker_prefetch_or_distributed_state_executed",
        "target_model_trainer_tokenizer_or_dataset_executed",
        "crash_power_loss_atomicity_or_remote_storage_proved",
        "checkpoint_origin_authentication_or_confidentiality_proved",
        "convergence_quality_throughput_or_memory_proved",
    )
    expected_components = {
        "model",
        "optimizer",
        "scheduler",
        "grad_scaler",
        "torch_cpu_rng",
        "python_rng",
        "data_stream",
    }
    overflow_trace = trace[1:4] if isinstance(trace, list) else []
    overflow_skip = all(
        isinstance(item, dict)
        and item.get("optimizer_step_executed") is False
        and item.get("optimizer_step_before") == item.get("optimizer_step_after") == 1
        and item.get("scheduler_last_epoch_before")
        == item.get("scheduler_last_epoch_after")
        == 1
        and item.get("scheduler_step_count_before")
        == item.get("scheduler_step_count_after")
        == 2
        and item.get("model_fingerprint_before")
        == item.get("model_fingerprint_after")
        for item in overflow_trace
    )
    omitted_rng_batches = (
        [item.get("batch_indices") for item in omitted_rng]
        if isinstance(omitted_rng, list)
        else []
    )
    resumed_batches = (
        [item.get("batch_indices") for item in resumed_trace]
        if isinstance(resumed_trace, list)
        else []
    )
    omitted_data_batches = (
        [item.get("batch_indices") for item in omitted_data]
        if isinstance(omitted_data, list)
        else []
    )
    if not (
        report.get("implementation") == TRAINING_RESUME_CONTROL_VERSION
        and report.get("runtime", {}).get("device") == "cpu"
        and report.get("runtime", {}).get("autocast_dtype") == "torch.float16"
        and fixture.get("total_attempts") == 8
        and fixture.get("split_after_attempts") == 4
        and fixture.get("intentional_overflow_attempts") == [1, 2, 3]
        and fixture.get("scale_sensitive_borderline_attempt") == 4
        and processes.get("phase1_process_exited_before_resume_launch") is True
        and processes.get("phase1_and_resume_pids_are_distinct") is True
        and processes.get("phase1_pid") != processes.get("resumed_pid")
        and checkpoint.get("schema_version") == TRAINING_RESUME_CHECKPOINT_VERSION
        and 1 <= checkpoint.get("bytes", 0) <= 16 * 1024 * 1024
        and checkpoint.get("loaded_with_torch_weights_only") is True
        and checkpoint.get("atomic_same_directory_temporary_replace_executed") is True
        and isinstance(trace, list)
        and len(trace) == 8
        and phase1_trace == trace[:4]
        and resumed_trace == trace[4:]
        and split_resume.get("terminal") == terminal
        and [item.get("scale_before") for item in trace[:5]]
        == [8.0, 8.0, 4.0, 2.0, 1.0]
        and [item.get("scale_after") for item in trace[:5]]
        == [8.0, 4.0, 2.0, 1.0, 1.0]
        and overflow_skip
        and terminal_progress
        == {
            "next_attempt_index": 8,
            "successful_updates": 5,
            "optimizer_step": 5,
            "scheduler_last_epoch": 5,
            "scheduler_step_count": 6,
            "learning_rate": 0.005,
            "grad_scaler_scale": 1.0,
            "data_epoch": 1,
            "data_cursor": 8,
        }
        and set(terminal_components) == expected_components
        and len(wrong_scheduler) == 8
        and [item.get("scheduler_last_epoch_after") for item in wrong_scheduler[1:4]]
        == [2, 3, 4]
        and len(omitted_scheduler) == 4
        and omitted_scheduler[0].get("scheduler_last_epoch_before") == 0
        and omitted_scheduler[0].get("learning_rate_after") == 0.02
        and resumed_trace[0].get("scheduler_last_epoch_before") == 1
        and resumed_trace[0].get("learning_rate_after") == 0.01
        and len(omitted_scaler) == 4
        and omitted_scaler[0].get("scale_before") == 8.0
        and omitted_scaler[0].get("scale_after") == 4.0
        and omitted_scaler[0].get("optimizer_step_executed") is False
        and resumed_trace[0].get("scale_before") == 1.0
        and resumed_trace[0].get("optimizer_step_executed") is True
        and omitted_rng_batches == resumed_batches
        and omitted_data_batches != resumed_batches
        and assertions
        and all(assertions.values())
        and all(scope.get(field) is True for field in expected_true_scope)
        and all(scope.get(field) is False for field in expected_false_scope)
    ):
        errors.append(
            "cross-process AMP training-resume fixture mismatch: "
            f"assertions={assertions}, terminal={terminal_progress}"
        )
    try:
        json.dumps(report, ensure_ascii=False, allow_nan=False)
    except ValueError as error:
        errors.append(f"training-resume report is not strict finite JSON: {error}")

    module = (
        ROOT / "src" / "about_llm" / "finetuning" / "training_resume.py"
    ).read_text(encoding="utf-8")
    script = script_path.read_text(encoding="utf-8")
    execution_markers = (
        'torch.amp.GradScaler(',
        'torch.amp.autocast(device_type="cpu", dtype=torch.float16)',
        "torch.optim.lr_scheduler.StepLR(",
        "torch.get_rng_state()",
        "random.getstate()",
        "torch.save(",
        "weights_only=True",
        "os.fsync(handle.fileno())",
        "os.replace(temporary, path)",
        "subprocess.Popen(",
        "ThreadPoolExecutor(max_workers=3)",
        'restore_scheduler=mode != "omit-scheduler"',
        'restore_scaler=mode != "omit-scaler"',
        'restore_rng=mode != "omit-rng"',
        'restore_data=mode != "omit-data"',
    )
    missing_execution = [marker for marker in execution_markers if marker not in module]
    if missing_execution:
        errors.append(
            "training-resume control missing execution marker(s): "
            f"{missing_execution}"
        )
    script_markers = (
        "allow_nan=False",
        "run_training_resume_process_control",
        "run_training_resume_worker",
    )
    missing_script = [marker for marker in script_markers if marker not in script]
    if missing_script:
        errors.append(f"training-resume CLI missing marker(s): {missing_script}")

    docs = {
        "finetuning": ROOT / "docs" / "training" / "finetuning.md",
        "pretraining": ROOT / "docs" / "training" / "pretraining.md",
        "project": ROOT / "projects" / "single-gpu-finetuning" / "README.md",
        "project_page": (
            ROOT
            / "docs"
            / "practice"
            / "projects"
            / "single-gpu-finetuning.md"
        ),
        "labs": ROOT / "docs" / "practice" / "labs.md",
        "interview": ROOT / "docs" / "career" / "interview-questions.md",
        "accuracy": ROOT / "docs" / "reference" / "accuracy.md",
        "production": ROOT / "docs" / "practice" / "production-checklist.md",
        "knowledge_map": ROOT / "docs" / "guide" / "knowledge-map.md",
        "repo_map": ROOT / "docs" / "guide" / "repo-map.md",
        "project_index": ROOT / "docs" / "practice" / "project-index.md",
    }
    required_docs = {
        "finetuning": ("真正统一的 CPU control", "不同 PID", "仍基于 pickle"),
        "pretraining": ("同一条真实 split-run", "漏 scheduler/scaler/RNG/data"),
        "project": ("跨进程 AMP checkpoint", "21,747-byte", "counterfactual"),
        "project_page": ("真正统一的跨进程 resume", "worker/prefetch"),
        "labs": ("实验 4H", "optimizer step=1", "weights_only=True"),
        "interview": ("统一的跨进程 CPU AMP 反例", "来源认证"),
        "accuracy": ("Cross-process AMP training-resume control", "21,747-byte"),
        "production": ("真实 update receipt", "通用 `did_step`"),
        "knowledge_map": (
            "真实跨 PID bit-exact control",
            "optimizer-commit control 再以 seed `20260815`",
        ),
        "repo_map": (
            "custom shuffle",
            "后续六进程 2-worker/stochastic-forward/backward/SGD/StepLR accumulation control",
        ),
        "project_index": (
            "第五条统一 CPU resume",
            "第七条扩展为六进程 main-process stochastic mask",
        ),
    }
    for name, path in docs.items():
        text = path.read_text(encoding="utf-8")
        missing = [marker for marker in required_docs[name] if marker not in text]
        if missing:
            errors.append(
                f"training-resume {name} docs missing marker(s): {missing}"
            )
    return errors


def check_synthetic_data_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.synthetic_data import (
        MixtureComponent,
        SourceKind,
        SyntheticRecord,
        VerificationResult,
        audit_synthetic_records,
        plan_mixture,
    )
    from about_llm.synthetic_data_cli import (
        SYNTHETIC_AUDIT_ARTIFACT_VERSION,
        verify_synthetic_audit_artifact,
    )

    shared = (
        VerificationResult("schema", "rules@v1", True),
        VerificationResult("grounding", "judge@v2", True),
    )
    records = [
        SyntheticRecord(
            "a",
            "same content",
            ("real-1",),
            "teacher@v1",
            "prompt@v1",
            1,
            shared,
            True,
        ),
        SyntheticRecord(
            "b",
            "same content",
            ("real-1",),
            "teacher@v1",
            "prompt@v1",
            1,
            (
                VerificationResult("schema", "rules@v1", True),
                VerificationResult("grounding", "teacher@v1", True),
            ),
        ),
        SyntheticRecord(
            "c",
            "missing gate",
            ("a",),
            "student@v2",
            "prompt@v2",
            2,
            (VerificationResult("schema", "rules@v1", True),),
        ),
    ]
    report = audit_synthetic_records(
        records,
        required_verifiers=("schema", "grounding"),
        known_parent_ids=("real-1",),
    )
    errors: list[str] = []
    if not (
        report.candidate_count == 3
        and report.eligible_count == 2
        and report.eligible_unique_content_count == 1
        and report.self_verified_record_ids == ("b",)
        and report.missing_verifier_record_ids == ("c",)
        and report.unresolved_parent_pairs == ()
        and report.nonmonotonic_parent_pairs == ()
        and report.lineage_cycle_record_ids == ()
    ):
        errors.append(f"synthetic-data audit example mismatch: {report}")

    plan = plan_mixture(
        [
            MixtureComponent("real", SourceKind.REAL, 800, 3),
            MixtureComponent("synthetic", SourceKind.SYNTHETIC, 100, 1, 1),
        ],
        total_consumed_tokens=2_000,
    )
    if not (
        math.isclose(plan.synthetic_fraction, 0.25)
        and math.isclose(plan.exposures[0].expected_consumed_tokens, 1_500)
        and math.isclose(plan.exposures[1].expected_repetition_factor, 5)
    ):
        errors.append(f"synthetic-data mixture example mismatch: {plan}")

    project = ROOT / "projects" / "synthetic-data-audit"
    artifact = verify_synthetic_audit_artifact(
        project / "audit.example.json",
        records_path=project / "records.example.jsonl",
        required_verifiers=("schema", "grounding"),
        known_parent_ids=("real-anchor-001",),
        mixture_path=project / "mixture.example.json",
    )
    artifact_audit = artifact.get("audit", {})
    artifact_scope = artifact.get("scope", {})
    if not (
        artifact.get("schema_version") == SYNTHETIC_AUDIT_ARTIFACT_VERSION
        and artifact.get("report_fingerprint")
        == "sha256:202d8db97b704c5542e8516c5bd0c945da1c1022100f6ecbfb828f2d2bb6f4cd"
        and isinstance(artifact_audit, dict)
        and artifact_audit.get("candidate_count") == 4
        and artifact_audit.get("eligible_count") == 2
        and artifact_audit.get("eligible_unique_content_count") == 1
        and isinstance(artifact_scope, dict)
        and artifact_scope.get("input_bytes_and_external_policy_bound") is True
        and artifact_scope.get("training_or_observed_token_ledger_executed")
        is False
    ):
        errors.append("synthetic-data recorded audit artifact mismatch")
    return errors


def check_sft_data_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from dataclasses import replace

    from about_llm.finetuning.data import (
        DataSplit,
        audit_sft_records,
        load_sft_records,
        validate_training_records,
        validate_training_subset,
    )
    from about_llm.finetuning.governance import (
        audit_sft_governance,
        load_sft_governance_policy,
    )
    from about_llm.finetuning.minhash_lsh import (
        MinHashLSHConfig,
        audit_minhash_lsh_recall,
        generate_minhash_lsh_candidates,
        lsh_candidate_probability,
    )
    from about_llm.finetuning.near_duplicate import (
        NearDuplicateProfile,
        NearDuplicateView,
        audit_sft_near_duplicates,
        character_ngrams,
        shingle_jaccard,
    )
    from about_llm.finetuning.readiness import SFTTrainingReadinessReport
    from about_llm.finetuning.template import audit_assistant_masks

    project = ROOT / "projects" / "single-gpu-finetuning"
    records = load_sft_records(project / "audit.example.jsonl")
    report = audit_sft_records(records)
    reversed_report = audit_sft_records(reversed(records))
    train_records = load_sft_records(project / "train.example.jsonl")
    train_report = validate_training_records(train_records)
    scope = report.to_dict()["scope"]
    errors: list[str] = []
    if not (
        report.gate_passed
        and report.record_count == 4
        and report.split_counts == {"test": 1, "train": 2, "validation": 1}
        and train_report.gate_passed
        and train_report.record_count == 2
        and all(set(record.to_training_row()) == {"messages"} for record in train_records)
        and report.ordered_dataset_fingerprint
        != reversed_report.ordered_dataset_fingerprint
        and report.unordered_dataset_fingerprint
        == reversed_report.unordered_dataset_fingerprint
        and report.manifest_fingerprint.startswith("sha256:")
        and scope["near_duplicate_detection"] is False
        and scope["license_legality_verified"] is False
        and scope["pii_or_secret_detection"] is False
        and scope["tokenizer_or_assistant_mask_verified"] is False
    ):
        errors.append(f"SFT example/identity/scope mismatch: {report}")

    leaked_group = replace(records[2], group_id=records[0].group_id)
    leaked_content = replace(records[3], messages=records[0].messages)
    leaked_report = audit_sft_records((records[0], leaked_group, leaked_content))
    if not (
        not leaked_report.gate_passed
        and len(leaked_report.cross_split_group_ids) == 1
        and len(leaked_report.cross_split_content) == 1
    ):
        errors.append(f"SFT cross-split gate mismatch: {leaked_report}")
    try:
        validate_training_records(records)
    except ValueError:
        pass
    else:
        errors.append("SFT trainer gate accepted validation/test records")
    if tuple(report.required_splits) != (
        DataSplit.TRAIN.value,
        DataSplit.VALIDATION.value,
        DataSplit.TEST.value,
    ):
        errors.append("SFT required split order mismatch")
    mask_report = audit_assistant_masks(
        train_records,
        render=lambda messages, tools: {
            "input_ids": [1, 2, 3, 4],
            "assistant_masks": [0, 0, 1, 1],
        },
        renderer_identity={"tokenizer": "accuracy-fixture@v1"},
        max_length=4,
    )
    mask_scope = mask_report.to_dict()["scope"]
    if not (
        mask_report.record_count == 2
        and mask_report.ordered_dataset_fingerprint
        == train_report.ordered_dataset_fingerprint
        and mask_report.input_token_count == 8
        and mask_report.assistant_token_count == 4
        and mask_scope["target_tokenizer_executed"] is True
        and mask_scope["collator_labels_verified"] is False
        and mask_scope["mask_semantics_independently_verified"] is False
    ):
        errors.append(f"SFT assistant-mask scope mismatch: {mask_report}")
    try:
        audit_assistant_masks(
            train_records,
            render=lambda messages, tools: {
                "input_ids": [1, 2, 3, 4],
                "assistant_masks": [0, 0, 1, 1],
            },
            renderer_identity={"tokenizer": "accuracy-fixture@v1"},
            max_length=3,
        )
    except ValueError:
        pass
    else:
        errors.append("SFT assistant-mask gate accepted silent right truncation")
    binding = validate_training_subset(train_records, records)
    if not (
        binding.training_report.ordered_dataset_fingerprint
        == train_report.ordered_dataset_fingerprint
        and binding.split_report.ordered_dataset_fingerprint
        == report.ordered_dataset_fingerprint
        and binding.binding_fingerprint.startswith("sha256:")
    ):
        errors.append("SFT train/combined subset binding mismatch")
    near_report = audit_sft_near_duplicates(
        records,
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=5,
        threshold=0.9,
    )
    near_scope = near_report.to_dict()["scope"]
    governance_report = audit_sft_governance(
        records,
        policy=load_sft_governance_policy(
            project / "governance-policy.example.json"
        ),
        evaluated_at=datetime(2026, 8, 6, 12, tzinfo=timezone.utc),
    )
    governance_scope = governance_report.to_dict()["scope"]
    readiness = SFTTrainingReadinessReport.from_reports(
        binding, near_report, governance_report
    )
    readiness_payload = readiness.to_dict()
    left = character_ngrams("abcd", size=2)
    right = character_ngrams("abce", size=2)
    similarity, intersection_size, union_size = shingle_jaccard(left, right)
    if not (
        near_report.gate_passed
        and near_report.record_pair_count == 5
        and near_report.comparison_count == 15
        and near_report.ordered_dataset_fingerprint
        == report.ordered_dataset_fingerprint
        and near_scope["semantic_equivalence_verified"] is False
        and near_scope["threshold_calibrated_for_caller_domain"] is False
        and near_scope["scalable_all_pairs_implementation"] is False
        and governance_report.gate_passed
        and governance_report.sensitive_candidates == ()
        and governance_scope["legal_permission_verified"] is False
        and governance_scope["comprehensive_pii_or_secret_detection"] is False
        and readiness.gate_passed
        and readiness.manifest_fingerprint.startswith("sha256:")
        and readiness_payload["scope"]["held_out_plaintext_embedded"] is False
        and readiness_payload["scope"]["trainer_needs_held_out_access"] is False
        and readiness_payload["scope"]["cryptographic_origin_authenticated"] is False
        and math.isclose(similarity, 0.5)
        and intersection_size == 2
        and union_size == 4
    ):
        errors.append(f"SFT near-duplicate math/scope mismatch: {near_report}")

    near_test = replace(
        records[3],
        messages=(
            replace(records[0].messages[1], content=records[0].messages[1].content + "!"),
            replace(records[0].messages[2], content=records[0].messages[2].content + "。"),
        ),
    )
    candidate_report = audit_sft_near_duplicates(
        (*records[:3], near_test),
        profile=NearDuplicateProfile.NFC_WHITESPACE,
        ngram_size=3,
        threshold=0.8,
    )
    if candidate_report.gate_passed or not {
        NearDuplicateView.USER_CONTENT,
        NearDuplicateView.ASSISTANT_CONTENT,
    }.issubset({finding.view for finding in candidate_report.findings}):
        errors.append("SFT lexical near-duplicate candidate example mismatch")

    minhash_namespace = runpy.run_path(project / "minhash_lsh_toy.py")
    minhash_toy = minhash_namespace["run_toy"](
        ngram_size=5,
        threshold=0.8,
        num_hashes=64,
        bands=16,
        seed=17,
    )
    minhash_candidates = minhash_toy["candidate_report"]
    minhash_recall = minhash_toy["exhaustive_recall_audit"]
    if not (
        minhash_candidates["item_count"] == 5
        and minhash_candidates["possible_pair_count"] == 10
        and minhash_candidates["candidate_pair_count"] == 3
        and math.isclose(minhash_candidates["candidate_fraction"], 0.3)
        and minhash_recall["exact_positive_pair_count"] == 1
        and minhash_recall["recovered_positive_pair_count"] == 1
        and minhash_recall["false_positive_candidate_count"] == 2
        and math.isclose(minhash_recall["candidate_recall"], 1)
        and math.isclose(minhash_recall["candidate_precision"], 1 / 3)
        and minhash_toy["scope"]["candidate_recall_guaranteed"] is False
        and minhash_toy["scope"]["semantic_or_translation_duplicate_detection"]
        is False
    ):
        errors.append("MinHash/LSH authored candidate/recall fixture mismatch")

    false_negative_items = {
        "a": frozenset({"a", "b", "c", "d", "e"}),
        "b": frozenset({"a", "b", "c", "d", "x"}),
        "c": frozenset({"other"}),
    }
    false_negative_candidates = generate_minhash_lsh_candidates(
        false_negative_items,
        config=MinHashLSHConfig(num_hashes=1, bands=1, seed=0),
    )
    false_negative_recall = audit_minhash_lsh_recall(
        false_negative_items,
        false_negative_candidates,
        threshold=2 / 3,
    )
    if not (
        false_negative_recall.exact_positive_pair_count == 1
        and false_negative_recall.recovered_positive_pair_count == 0
        and false_negative_recall.candidate_recall == 0
        and false_negative_recall.missed_exact_positive_pairs == (("a", "b"),)
        and math.isclose(
            lsh_candidate_probability(0.8, bands=20, rows_per_band=5),
            1 - (1 - 0.8**5) ** 20,
        )
    ):
        errors.append("MinHash/LSH fixed false-negative/probability fixture mismatch")
    return errors


def check_continual_learning_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.continual_learning import (
        reservoir_sample_indices,
        summarize_accuracy_matrix,
    )

    report = summarize_accuracy_matrix(
        (
            (0.80, 0.20, 0.30),
            (0.75, 0.70, 0.40),
            (0.90, 0.60, 0.85),
        ),
        pretraining_baseline=(0.50, 0.30, 0.25),
    )
    errors: list[str] = []
    if not (
        report.diagonal_accuracy == (0.80, 0.70, 0.85)
        and math.isclose(report.final_average_accuracy, 2.35 / 3)
        and math.isclose(report.backward_transfer, 0)
        and all(
            math.isclose(observed, expected)
            for observed, expected in zip(
                report.per_task_forgetting,
                (0.0, 0.10, 0.0),
                strict=True,
            )
        )
        and math.isclose(report.average_forgetting_old_tasks, 0.05)
        and math.isclose(report.forward_transfer, 0.025)
    ):
        errors.append(f"continual-learning accuracy-matrix example mismatch: {report}")
    if reservoir_sample_indices(20, 5, seed=7) != (3, 4, 14, 16, 17):
        errors.append("continual-learning reservoir fixture mismatch")
    return errors


def check_quantization_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import numpy as np

    from about_llm.inference import (
        quantization_error,
        quantize_kv_cache_int8,
        quantize_symmetric_groupwise,
        quantized_kv_grouped_query_attention,
        quantized_linear,
    )

    errors: list[str] = []
    weights = np.array([[0.0, 1.0, -1.0, 0.6, 0.0]], dtype=np.float32)
    quantized = quantize_symmetric_groupwise(
        weights, bit_width=4, group_size=2
    )
    storage_fixture = quantize_symmetric_groupwise(
        np.arange(15, dtype=np.float32).reshape(3, 5),
        bit_width=4,
        group_size=4,
    )
    packed_fixture = storage_fixture.pack()
    output = quantized_linear(np.ones((1, 5), dtype=np.float32), quantized)
    output_error = quantization_error(
        np.ones((1, 5), dtype=np.float32) @ weights.T,
        output,
    )
    if not (
        quantized.values.tolist() == [[0, 7, -7, 4, 0]]
        and np.allclose(quantized.scales, [[1 / 7, 1 / 7, 1]])
        and quantized.pack().packed_values.hex() == "e7b007"
        and storage_fixture.reference_fp32_weight_bytes == 60
        and storage_fixture.ideal_packed_weight_bytes == 8
        and storage_fixture.scale_metadata_bytes == 24
        and storage_fixture.ideal_total_bytes == 32
        and storage_fixture.unpacked_reference_bytes == 39
        and packed_fixture.packed_weight_bytes == 8
        and packed_fixture.raw_payload_bytes == 32
        and packed_fixture.padding_bits == 4
        and packed_fixture.serialized_artifact_bytes == 96
        and np.array_equal(packed_fixture.unpack().values, storage_fixture.values)
        and np.array_equal(packed_fixture.unpack().scales, storage_fixture.scales)
        and np.array_equal(
            type(packed_fixture)
            .from_bytes(packed_fixture.to_bytes())
            .unpack()
            .values,
            storage_fixture.values,
        )
        and packed_fixture.packed_values_sha256.startswith("sha256:")
        and packed_fixture.serialized_artifact_sha256.startswith("sha256:")
        and output_error.root_mean_squared_error >= 0
    ):
        errors.append("symmetric group-wise quantization fixture mismatch")

    key = np.array([[[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.5]]]])
    value = np.array([[[[0.0, 0.0], [2.0, -1.0]]]])
    kv_cache = quantize_kv_cache_int8(key, value)
    kv_output, kv_probabilities = quantized_kv_grouped_query_attention(
        np.ones((1, 2, 2, 4), dtype=np.float32), kv_cache
    )
    if not (
        kv_cache.key_codes.tolist()
        == [[[[0, 0, 0, 0], [0, 127, -127, 64]]]]
        and kv_cache.reference_fp32_bytes == 48
        and kv_cache.int8_code_bytes == 12
        and kv_cache.scale_metadata_bytes == 16
        and kv_cache.payload_bytes == 28
        and math.isclose(kv_cache.payload_compression_ratio, 48 / 28)
        and kv_output.shape == (1, 2, 2, 2)
        and np.all(kv_probabilities[..., 0, 1:] == 0)
        and np.allclose(np.sum(kv_probabilities, axis=-1), 1)
    ):
        errors.append("INT8 KV-cache quantization/GQA/storage fixture mismatch")

    demo = (
        ROOT / "projects" / "inference-serving" / "quantization_toy.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"actual_low_bit_packing_executed": True',
        '"self_contained_quantized_tensor_artifact_constructed": True',
        '"self_contained_model_artifact_written": False',
        '"fused_low_bit_kernel_executed": False',
        '"calibration_or_gptq_awq_executed": False',
        '"model_quality_or_latency_proved": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(f"quantization toy missing scope marker(s): {missing_scope}")

    bundle_namespace = runpy.run_path(
        str(ROOT / "projects" / "inference-serving" / "quantized_bundle_toy.py")
    )
    bundle_run_toy = bundle_namespace.get("run_toy")
    if not callable(bundle_run_toy):
        errors.append("quantized bundle toy does not expose callable run_toy")
    else:
        bundle_report = bundle_run_toy(
            seed=29,
            bit_width=4,
            group_size=4,
            artifact_path=None,
        )
        bundle_scope = bundle_report.get("scope", {})
        false_scope_fields = (
            "tokenizer_payload_embedded",
            "unquantized_parameter_kinds_supported",
            "model_forward_implementation_embedded",
            "runtime_specific_layout_or_kernel",
            "fused_low_bit_execution",
            "cryptographic_origin_authenticated",
            "full_llm_checkpoint",
        )
        if not (
            bundle_report.get("bundle_format_version") == 1
            and bundle_report.get("bundle_schema_version")
            == "about-llm.quantized-matrix-bundle.v1"
            and bundle_report.get("tensor_names")
            == ["layer.0.weight", "layer.1.weight"]
            and bundle_report.get("tensor_count") == 2
            and bundle_report.get("reference_fp32_weight_bytes") == 288
            and bundle_report.get("raw_quantized_payload_bytes") == 124
            and bundle_report.get("individual_tensor_artifact_bytes") == 252
            and bundle_report.get("bundle_artifact_bytes") == 987
            and bundle_report.get("bundle_container_overhead_bytes") == 735
            and bundle_report.get("exact_byte_round_trip") is True
            and bundle_report.get("exact_quantized_forward_round_trip") is True
            and bundle_report.get("disk_round_trip") is False
            and bundle_scope.get("multiple_named_quantized_matrices_embedded") is True
            and bundle_scope.get("architecture_and_revision_identity_embedded")
            is True
            and bundle_scope.get("tokenizer_identity_embedded") is True
            and all(bundle_scope.get(field) is False for field in false_scope_fields)
            and math.isclose(
                bundle_report["output_error_vs_fp32"]["root_mean_squared_error"],
                0.0368724678,
                rel_tol=1e-7,
                abs_tol=1e-9,
            )
            and math.isclose(
                bundle_report["output_error_vs_fp32"]["relative_l2_error"],
                0.1232720371,
                rel_tol=1e-7,
                abs_tol=1e-9,
            )
        ):
            errors.append(f"multi-matrix quantized bundle fixture mismatch: {bundle_report}")

    bundle_doc = (
        ROOT / "docs" / "systems" / "inference-optimization.md"
    ).read_text(encoding="utf-8")
    bundle_boundaries = (
        "tokenizer 只有 identity、没有 vocab/merges/chat template payload",
        "不是完整 LLM checkpoint",
        "没有证明 parent-directory fsync 或断电原子发布",
        "不是模型压缩率或质量结论",
    )
    missing_bundle_boundaries = [
        marker for marker in bundle_boundaries if marker not in bundle_doc
    ]
    if missing_bundle_boundaries:
        errors.append(
            "quantized bundle docs missing boundary marker(s): "
            f"{missing_bundle_boundaries}"
        )

    checkpoint_namespace = runpy.run_path(
        str(
            ROOT
            / "projects"
            / "inference-serving"
            / "minigpt_checkpoint_toy.py"
        )
    )
    checkpoint_run_toy = checkpoint_namespace.get("run_toy")
    if not callable(checkpoint_run_toy):
        errors.append("MiniGPT checkpoint toy does not expose callable run_toy")
    else:
        checkpoint_report = checkpoint_run_toy(
            seed=7,
            bit_width=4,
            group_size=4,
            prompt="abc abc",
            artifact_path=None,
        )
        storage = checkpoint_report.get("storage", {})
        forward = checkpoint_report.get("forward", {})
        tokenizer = checkpoint_report.get("tokenizer", {})
        checkpoint_scope = checkpoint_report.get("scope", {})
        true_scope_fields = (
            "byte_bpe_merge_payload_embedded",
            "architecture_config_and_revision_embedded",
            "all_unique_model_parameters_embedded",
            "quantized_matrices_and_float32_vectors_embedded",
            "tied_lm_head_restored",
            "repo_minigpt_forward_executed",
            "full_repo_native_minigpt_inference_checkpoint",
            "trusted_repo_loader_required",
        )
        false_scope_fields = (
            "forward_source_code_embedded",
            "normalizer_special_tokens_or_chat_template_supported",
            "optimizer_rng_or_training_resume_state_embedded",
            "gguf_safetensors_or_external_runtime_compatible",
            "packed_low_bit_kernel_executed",
            "pretrained_or_target_llm_quality_proved",
            "resident_vram_latency_or_speedup_measured",
            "cryptographic_origin_authenticated",
            "general_purpose_llm_checkpoint",
        )
        if not (
            checkpoint_report.get("schema_version") == 1
            and tokenizer.get("merges") == [[97, 98], [256, 99]]
            and tokenizer.get("prompt_token_ids") == [257, 32, 257]
            and tokenizer.get("round_trip_text") == "abc abc"
            and storage.get("unique_parameter_count") == 16
            and storage.get("header_parameter_count") == 16
            and storage.get("reference_fp32_parameter_bytes") == 10_976
            and storage.get("manifest_bytes") == 3_904
            and storage.get("parameter_payload_bytes") == 4_760
            and storage.get("checkpoint_artifact_bytes") == 8_720
            and storage.get("container_overhead_bytes") == 3_960
            and forward.get("logits_shape") == [1, 3, 258]
            and forward.get("exact_repeated_load_logits") is True
            and forward.get("greedy_generated_token_ids")
            == [257, 32, 257, 70, 106, 47]
            and math.isclose(
                forward["logit_rmse_vs_fp32"],
                0.0047727693,
                rel_tol=1e-7,
                abs_tol=1e-9,
            )
            and all(checkpoint_scope.get(field) is True for field in true_scope_fields)
            and all(
                checkpoint_scope.get(field) is False for field in false_scope_fields
            )
        ):
            errors.append(
                "repo-native quantized MiniGPT checkpoint fixture mismatch: "
                f"{checkpoint_report}"
            )

    checkpoint_doc = (
        ROOT / "docs" / "systems" / "inference-optimization.md"
    ).read_text(encoding="utf-8")
    checkpoint_boundaries = (
        "完整推理 checkpoint",
        "Forward 源码没有嵌入 artifact",
        "不等于 resident memory 更小或执行低位 kernel",
        "不是语言质量结论",
        "也不支持 normalization、special token、chat template、optimizer、RNG、训练 resume",
    )
    missing_checkpoint_boundaries = [
        marker for marker in checkpoint_boundaries if marker not in checkpoint_doc
    ]
    if missing_checkpoint_boundaries:
        errors.append(
            "MiniGPT checkpoint docs missing boundary marker(s): "
            f"{missing_checkpoint_boundaries}"
        )

    kv_demo = (
        ROOT / "projects" / "inference-serving" / "kv_quantization_toy.py"
    ).read_text(encoding="utf-8")
    kv_required_scope = (
        '"actual_int8_codes_and_fp32_scales_materialized": True',
        '"dequantized_gqa_attention_executed": True',
        '"attention_on_int8_codes_or_fused_kv_kernel_executed": False',
        '"paged_runtime_layout_or_resident_vram_measured": False',
        '"target_model_quality_or_speed_proved": False',
    )
    kv_missing_scope = [marker for marker in kv_required_scope if marker not in kv_demo]
    if kv_missing_scope:
        errors.append(f"KV quantization toy missing scope marker(s): {kv_missing_scope}")
    return errors


def check_minigpt_training_checkpoint_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    errors: list[str] = []
    namespace = runpy.run_path(
        str(
            ROOT
            / "projects"
            / "single-gpu-finetuning"
            / "minigpt_resume_toy.py"
        )
    )
    run_toy = namespace.get("run_toy")
    if not callable(run_toy):
        return ["MiniGPT training checkpoint toy does not expose callable run_toy"]

    report = run_toy(artifact_path=None)
    identity = report.get("identity", {})
    fixture = report.get("fixture", {})
    checkpoint = report.get("checkpoint", {})
    trajectory = report.get("trajectory", {})
    scope = report.get("scope", {})
    true_scope_fields = (
        "cpu_float32_minigpt_adamw_executed",
        "dropout_torch_cpu_rng_restored",
        "shuffle_generator_rng_restored",
        "optimizer_moments_and_step_restored",
        "linear_schedule_progress_restored",
        "dataset_content_identity_verified",
        "batch_permutation_cursor_and_epoch_restored",
        "uninterrupted_vs_split_run_bit_exact",
        "checkpoint_at_zero_grad_optimizer_boundary",
    )
    false_scope_fields = (
        "network_used",
        "python_numpy_or_cuda_rng_used_or_restored",
        "amp_scaler_or_gradient_accumulation_supported",
        "dataloader_workers_or_sampler_prefetch_supported",
        "distributed_sharded_training_supported",
        "dataset_payload_embedded",
        "target_checkpoint_or_cuda_executed",
        "loss_improvement_or_model_quality_proved",
        "cryptographic_origin_authenticated",
        "power_loss_atomic_publication_proved",
    )
    losses = trajectory.get("losses", [])
    losses_valid = (
        isinstance(losses, list)
        and len(losses) == 6
        and all(isinstance(loss, float) and math.isfinite(loss) and loss > 0 for loss in losses)
    )
    exact_trajectory_fields = (
        "first_segment_matches_uninterrupted",
        "resumed_segment_matches_uninterrupted",
        "state_exact_at_resume",
        "final_model_optimizer_stream_rng_exact",
        "external_torch_rng_unchanged",
    )
    if not (
        report.get("schema_version") == 1
        and identity
        == {
            "run_id": "authored-resume-control",
            "model_revision": "fixture-seed-13",
            "tokenizer_revision": "byte-v1",
            "data_revision": "fixture-seed-99",
        }
        and fixture
        == {
            "examples": 7,
            "sequence_tokens": 5,
            "batch_size": 2,
            "total_updates": 6,
            "split_after_updates": 3,
            "dropout": 0.2,
            "training_seed": 17,
            "data_seed": 19,
        }
        and checkpoint
        == {
            "manifest_bytes": 11_341,
            "tensor_count": 51,
            "payload_bytes": 42_520,
            "artifact_bytes": 53_917,
            "model_parameter_tensors": 16,
            "optimizer_moment_tensors": 32,
            "rng_tensors": 2,
            "permutation_tensors": 1,
            "global_step": 3,
            "epoch": 0,
            "cursor": 6,
        }
        and trajectory.get("batch_indices")
        == [[6, 5], [2, 1], [4, 0], [1, 0], [6, 5], [3, 4]]
        and trajectory.get("epochs") == [0, 0, 0, 1, 1, 1]
        and trajectory.get("learning_rates")
        == [0.003, 0.0026, 0.0022, 0.0018000000000000002, 0.0014, 0.001]
        and losses_valid
        and all(trajectory.get(field) is True for field in exact_trajectory_fields)
        and report.get("artifact_path") is None
        and report.get("disk_round_trip") is False
        and all(scope.get(field) is True for field in true_scope_fields)
        and all(scope.get(field) is False for field in false_scope_fields)
    ):
        errors.append(f"MiniGPT exact training-resume fixture mismatch: {report}")

    training_doc = (
        ROOT / "docs" / "training" / "finetuning.md"
    ).read_text(encoding="utf-8")
    project_doc = (
        ROOT / "projects" / "single-gpu-finetuning" / "README.md"
    ).read_text(encoding="utf-8")
    required_boundaries = (
        "artifact 不嵌入数据 payload",
        "没有保存 Python、NumPy 或 CUDA RNG",
        (
            "不支持 AMP scaler、gradient accumulation、DataLoader worker/prefetch、"
            "distributed/sharded state"
        ),
        "无密钥 SHA-256 不认证来源",
        "不证明断电原子发布",
        "loss 也不单调下降",
        "当前 CPU、PyTorch、FP32、MiniGPT architecture revision 和训练契约",
    )
    combined_docs = training_doc + "\n" + project_doc
    missing = [marker for marker in required_boundaries if marker not in combined_docs]
    if missing:
        errors.append(
            "MiniGPT training checkpoint docs missing boundary marker(s): "
            f"{missing}"
        )
    return errors


def check_peft_export_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    errors: list[str] = []
    namespace = runpy.run_path(
        str(
            ROOT
            / "projects"
            / "single-gpu-finetuning"
            / "smoke_peft.py"
        )
    )
    run_smoke = namespace.get("run_smoke")
    if not callable(run_smoke):
        return ["PEFT export control does not expose callable run_smoke"]

    report = run_smoke(steps=8, artifact_root=None)
    fixture = report.get("fixture", {})
    parameters = report.get("parameter_report", {})
    training = report.get("training", {})
    round_trip = report.get("round_trip", {})
    artifacts = report.get("artifacts", {})
    weights = artifacts.get("weights", {})
    verification = artifacts.get("strict_verification", {})
    scope = report.get("scope", {})
    true_scope_fields = (
        "cpu_random_tiny_gpt2_peft_training_executed",
        "frozen_base_unchanged",
        "base_full_safetensors_saved",
        "adapter_safetensors_saved_and_reloaded",
        "merge_and_unload_executed",
        "merged_full_safetensors_saved_and_reloaded",
        "strict_manifest_enforced_before_published_artifact_reload",
        "complete_directory_file_set_size_and_hash_bound",
        "safetensors_payloads_parsed_before_reload",
        "base_merged_config_payload_and_tensor_signature_match",
        "lora_target_a_b_tensor_coverage_validated",
        "tokenizer_and_chat_template_included_and_reloaded",
    )
    false_scope_fields = (
        "network_used",
        "peft_loader_itself_enforces_repo_manifest",
        "adapter_config_path_or_id_authenticates_base_content",
        "optimizer_scheduler_rng_or_training_resume_state_included",
        "quantized_base_or_qlora_merge_executed",
        "target_checkpoint_or_cuda_executed",
        "task_quality_or_production_compatibility_proved",
        "cryptographic_origin_authenticated",
        "atomic_or_power_loss_safe_publication_proved",
        "concurrent_mutation_or_verify_load_toctou_prevented",
    )
    expected_adapter_keys = [
        "base_model.model.transformer.h.0.attn.c_attn.lora_A.weight",
        "base_model.model.transformer.h.0.attn.c_attn.lora_B.weight",
        "base_model.model.transformer.h.1.attn.c_attn.lora_A.weight",
        "base_model.model.transformer.h.1.attn.c_attn.lora_B.weight",
    ]
    expected_weight_metadata = {
        "base": ("base/model.safetensors", 110_632),
        "adapter": ("adapter/adapter_model.safetensors", 4_608),
        "merged": ("merged/model.safetensors", 110_632),
    }
    expected_export_files = [
        "adapter/README.md",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "base/config.json",
        "base/generation_config.json",
        "base/model.safetensors",
        "merged/config.json",
        "merged/generation_config.json",
        "merged/model.safetensors",
        "tokenizer/chat_template.jinja",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ]
    weight_metadata_exact = all(
        weights.get(name, {}).get("relative_path") == relative_path
        and weights.get(name, {}).get("bytes") == size
        and isinstance(weights.get(name, {}).get("sha256"), str)
        and weights[name]["sha256"].startswith("sha256:")
        and len(weights[name]["sha256"]) == 71
        for name, (relative_path, size) in expected_weight_metadata.items()
    )
    if not (
        report.get("schema_version") == 2
        and fixture
        == {
            "seed": 31,
            "steps": 8,
            "input_ids": [[1, 5, 7, 9, 2], [1, 4, 6, 8, 2]],
        }
        and parameters
        == {
            "total_parameters": 28_032,
            "trainable_parameters": 1_024,
            "trainable_fraction": 0.0365296803652968,
            "parameter_storage_bytes": 112_128,
        }
        and training.get("initial_loss") == 3.5458385944366455
        and training.get("final_loss") == 3.408539056777954
        and training.get("base_parameters_unchanged") is True
        and training.get("adapter_tensor_count") == 4
        and training.get("adapter_tensor_keys") == expected_adapter_keys
        and round_trip.get("builder_adapter_matches_trained_maximum_logit_error")
        == 0.0
        and round_trip.get("verified_adapter_reload_maximum_logit_error") == 0.0
        and 0.0 <= round_trip.get("merge_maximum_logit_error", math.inf) < 1e-6
        and round_trip.get("verified_merged_reload_maximum_logit_error") == 0.0
        and round_trip.get("tokenizer_chat_template_token_ids") == [5, 7, 2, 9, 2]
        and round_trip.get("verified_tokenizer_chat_template_token_ids")
        == [5, 7, 2, 9, 2]
        and round_trip.get("tokenizer_chat_template_exact") is True
        and artifacts.get("persisted") is False
        and artifacts.get("root") is None
        and verification.get("identity")
        == {
            "artifact_id": "authored-peft-export-control",
            "architecture": "GPT2LMHeadModel",
            "base_model_id": "authored-random-gpt2",
            "base_revision": "fixture-seed-31",
            "tokenizer_revision": "authored-wordlevel-v1",
        }
        and verification.get("file_count") == 13
        and isinstance(verification.get("total_file_bytes"), int)
        and verification["total_file_bytes"] >= sum(
            size for _, size in expected_weight_metadata.values()
        )
        and isinstance(verification.get("manifest_bytes"), int)
        and verification["manifest_bytes"] > 0
        and verification.get("files") == expected_export_files
        and isinstance(verification.get("file_set_sha256"), str)
        and verification["file_set_sha256"].startswith("sha256:")
        and artifacts.get("adapter_config_base_reference_kind")
        == "immutable-id-string"
        and artifacts.get("adapter_config_base_model_id") == "authored-random-gpt2"
        and artifacts.get("pickle_weight_files") == []
        and weight_metadata_exact
        and all(scope.get(field) is True for field in true_scope_fields)
        and all(scope.get(field) is False for field in false_scope_fields)
    ):
        errors.append(f"PEFT adapter export fixture mismatch: {report}")

    project_doc = (
        ROOT / "projects" / "single-gpu-finetuning" / "README.md"
    ).read_text(encoding="utf-8")
    required_boundaries = (
        "路径或 identity string 不是内容认证",
        "PEFT 自身仍不会自动强制仓库 manifest",
        "WordLevel tokenizer、special tokens 和 chat template",
        "拒绝额外或缺失文件、symlink、路径穿越",
        "size/hash 漂移",
        "三个 safetensors 均可解析",
        "完整 config payload 与 tensor key/dtype/shape signature 一致",
        "LoRA A/B tensor",
        "没有 optimizer/scheduler/RNG/training-resume state",
        "未执行量化基座 merge、目标 checkpoint 或 CUDA",
        "任务质量、跨版本可移植性、来源认证或断电原子发布",
        "verify 与 load 之间的并发替换",
    )
    missing = [marker for marker in required_boundaries if marker not in project_doc]
    if missing:
        errors.append(f"PEFT export docs missing boundary marker(s): {missing}")
    return errors


def check_target_sft_label_control() -> list[str]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    from about_llm.finetuning.target_sft_label_control import (
        TARGET_SFT_LABEL_CONTROL_VERSION,
        TARGET_SFT_LABEL_REPORT_VERSION,
        load_target_sft_label_control_spec,
        verify_recorded_target_sft_label_report,
    )
    from about_llm.integrations.transformers_checkpoint_control import (
        load_checkpoint_control_spec,
    )
    from scripts.smoke_wheel import EXPECTED_TARGET_SFT_LABEL_VERSION_LINES

    errors: list[str] = []
    if EXPECTED_TARGET_SFT_LABEL_VERSION_LINES != (
        TARGET_SFT_LABEL_CONTROL_VERSION,
        TARGET_SFT_LABEL_REPORT_VERSION,
    ):
        errors.append(
            "wheel smoke target SFT version expectation drifted from exported contract"
        )
    project = ROOT / "projects" / "single-gpu-finetuning"
    checkpoint_control = (
        ROOT
        / "projects"
        / "transformers-basics"
        / "target-checkpoints"
        / "qwen2.5-0.5b-instruct.control.json"
    )
    checkpoint_report = checkpoint_control.with_name(
        "qwen2.5-0.5b-instruct.recorded-report.json"
    )
    checkpoint_spec = load_checkpoint_control_spec(checkpoint_control)
    spec = load_target_sft_label_control_spec(
        project / "qwen2.5-0.5b-sft-label.control.json",
        checkpoint_spec=checkpoint_spec,
    )
    report = verify_recorded_target_sft_label_report(
        project / "qwen2.5-0.5b-sft-label.recorded-report.json",
        spec=spec,
        checkpoint_spec=checkpoint_spec,
        checkpoint_report_path=checkpoint_report,
        training_path=project / "tool-sft.train.jsonl",
        readiness_path=project / "tool-sft-training-readiness.json",
        template_path=project / "qwen2.5-generation-aware-sft.jinja",
    )
    result = report.get("result", {})
    samples = result.get("samples", [])
    template = result.get("template", {})
    collator = result.get("collator", {})
    execution = result.get("execution", {})
    scope = report.get("scope", {})
    if not (
        spec.manifest_fingerprint
        == "sha256:b1c1a6b36db5a8671d8ccdda0669355e14c2efb21b012040ab1764b3e8c936e6"
        and report.get("report_fingerprint")
        == "sha256:8b61fa58ea8278444ce63ba5daa0fab88952c1c51e72bc7197b3a2678810421a"
        and [sample.get("input_token_count") for sample in samples]
        == [47, 301, 200]
        and [sample.get("assistant_token_count") for sample in samples]
        == [8, 51, 31]
        and all(
            sample.get("native_input_ids_equal") is True for sample in samples
        )
        and all(
            sample.get("assistant_generation_text_equal") is True
            for sample in samples
        )
        and template.get("checkpoint_native_generation_marker_present") is False
        and template.get("checkpoint_native_all_zero_assistant_mask_observed")
        is True
        and template.get("reviewed_generation_marker_present") is True
        and template.get("reviewed_render_matches_native_input_ids") is True
        and template.get(
            "reviewed_mask_matches_authored_assistant_generation_text"
        )
        is True
        and template.get("supported_roles")
        == ["system", "user", "assistant", "tool"]
        and template.get("tool_calls_supported_by_evidence") is True
        and template.get("multi_assistant_turns_supported_by_evidence") is True
        and template.get("record_tools_forwarded_to_chat_template") is True
        and template.get("arrow_pre_tokenization_executed") is True
        and template.get("raw_nested_records_passed_to_arrow") is False
        and collator.get("batch_shape") == [3, 301]
        and collator.get("attention_token_count") == 548
        and collator.get("padding_token_count") == 355
        and collator.get("supervised_label_count") == 90
        and collator.get("ignored_label_count") == 813
        and collator.get("assistant_labels_equal_input_ids") is True
        and collator.get("non_assistant_and_padding_labels_are_minus_100") is True
        and collator.get("silent_truncation_observed") is False
        and execution.get("forward_loss") == 1.251716136932373
        and execution.get("forward_loss_finite") is True
        and execution.get("target_forward_executed") is True
        and execution.get("backward_executed") is False
        and execution.get("optimizer_step_count") == 0
        and scope.get("target_checkpoint_weights_loaded") is True
        and scope.get("real_trl_sft_collator_executed") is True
        and scope.get("reviewed_template_exact_fixed_subset_executed") is True
        and scope.get("fixed_tool_calls_executed") is True
        and scope.get("fixed_multi_assistant_turns_executed") is True
        and scope.get("pre_arrow_tokenization_executed") is True
        and scope.get("target_forward_loss_executed") is True
        and scope.get("backward_or_optimizer_executed") is False
        and scope.get("qlora_cuda_or_vllm_executed") is False
        and scope.get("arbitrary_provider_tool_schemas_or_multimodal_proven")
        is False
        and scope.get("tool_execution_or_result_truth_proven")
        is False
        and scope.get("data_legality_or_semantic_quality_proven") is False
        and scope.get("convergence_generalization_or_safety_proven") is False
        and scope.get("verification_to_loader_reopen_toctou_eliminated") is False
    ):
        errors.append(f"target Qwen SFT final-label recorded evidence mismatch: {report}")

    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "README.md",
            ROOT / "CHANGELOG.md",
            project / "README.md",
            ROOT / "docs" / "training" / "sft-data-pipeline.md",
            ROOT / "docs" / "training" / "finetuning.md",
            ROOT / "docs" / "training" / "peft-qlora-engineering.md",
            ROOT / "docs" / "guide" / "knowledge-map.md",
            ROOT / "docs" / "guide" / "repo-map.md",
            ROOT / "docs" / "practice" / "labs.md",
            ROOT / "docs" / "practice" / "project-index.md",
            ROOT
            / "docs"
            / "practice"
            / "projects"
            / "single-gpu-finetuning.md",
            ROOT / "docs" / "career" / "interview-questions.md",
            ROOT / "docs" / "career" / "resume-projects.md",
            ROOT / "docs" / "reference" / "accuracy.md",
        )
    )
    required_markers = (
        "b1c1a6b3",
        "8b61fa58",
        "47 / 301 / 200",
        "8 / 51 / 31",
        "[3, 301]",
        "548 attention token",
        "355 padding token",
        "90 个监督 label",
        "813 个 `-100`",
        "1.251716",
        "全零 assistant mask",
        "{% generation %}",
        "<|im_end|>\\n",
        "并行 tool calls",
        "Arrow",
        "assistant_only_loss=False",
        "不执行 backward",
        "不证明数据合法性",
        "TOCTOU",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(
            "target Qwen SFT final-label docs missing boundary marker(s): "
            f"{missing}"
        )
    return errors


def check_target_lora_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    from about_llm.finetuning.target_lora_control import (
        load_recorded_target_lora_report,
        load_target_lora_control_spec,
    )
    from about_llm.integrations.transformers_checkpoint_control import (
        load_checkpoint_control_spec,
    )

    errors: list[str] = []
    project = ROOT / "projects" / "single-gpu-finetuning"
    checkpoint_control = (
        ROOT
        / "projects"
        / "transformers-basics"
        / "target-checkpoints"
        / "qwen2.5-0.5b-instruct.control.json"
    )
    checkpoint_report = checkpoint_control.with_name(
        "qwen2.5-0.5b-instruct.recorded-report.json"
    )
    checkpoint_spec = load_checkpoint_control_spec(checkpoint_control)
    spec = load_target_lora_control_spec(
        project / "qwen2.5-0.5b-lora.control.json",
        checkpoint_spec=checkpoint_spec,
    )
    report = load_recorded_target_lora_report(
        project / "qwen2.5-0.5b-lora.recorded-report.json",
        spec=spec,
        checkpoint_spec=checkpoint_spec,
        checkpoint_report_path=checkpoint_report,
        artifact_directory=(
            project / "target-adapters" / "qwen2.5-0.5b-instruct-step1"
        ),
    )
    model = report.get("model", {})
    sample = report.get("sample", {})
    execution = report.get("execution", {})
    artifact = report.get("adapter_artifact", {})
    round_trip = report.get("round_trip", {})
    scope = report.get("scope", {})
    files = artifact.get("files", [])
    if not (
        spec.manifest_fingerprint
        == "sha256:801b95fe6f35fdff9b0bef5db47d028e98ad0335e78055d5b33eeb48c3034885"
        and report.get("report_fingerprint")
        == "sha256:8a3897b10dbc2f55bb5ad3a8851fe659670e6951c19e58ae7fd269f9fb026230"
        and sample.get("prompt_token_count") == 41
        and sample.get("supervised_token_count") == 3
        and len(sample.get("input_token_ids", [])) == 44
        and model.get("base_parameter_count") == 494_032_768
        and model.get("adapter_parameter_count") == 270_336
        and model.get("total_parameter_count_with_adapter") == 494_303_104
        and execution.get("trainable_gradient_tensor_count") == 96
        and execution.get("finite_gradient_tensor_count") == 96
        and execution.get("frozen_base_gradient_tensor_count") == 0
        and execution.get("frozen_base_parameter_fingerprint_before")
        == "sha256:716454a96d2b6f34f4f846f458fd6a74bb5f4be2ac1bc9dd115e432808de7092"
        and execution.get("frozen_base_parameter_fingerprint_after")
        == execution.get("frozen_base_parameter_fingerprint_before")
        and execution.get("adapter_nonzero_b_tensor_count_after_step") == 48
        and execution.get("adapter_nonzero_b_element_count_after_step") == 98_304
        and execution.get("initial_loss") == 0.0038636347744613886
        and execution.get("post_step_loss") == 0.5845565795898438
        and artifact.get("manifest_fingerprint")
        == "sha256:ffab495858a3c81e9769f83f7c2c87d6ee7fc490d3a0d94eebda0d3da5c96c46"
        and artifact.get("manifest_bytes") == 1_488
        and artifact.get("tensor_count") == 96
        and artifact.get("total_tensor_numel") == 270_336
        and len(files) == 3
        and files[2].get("path") == "adapter_model.safetensors"
        and files[2].get("bytes") == 1_093_728
        and round_trip.get("maximum_logit_error") == 0.0
        and round_trip.get("trained_and_reloaded_logits_exact") is True
        and scope.get("target_checkpoint_backward_executed") is True
        and scope.get("peft_adapter_saved_and_reloaded") is True
        and scope.get("model_quality_or_convergence_proven") is False
        and scope.get("qlora_or_quantized_base_executed") is False
        and scope.get("cuda_executed") is False
        and scope.get("vllm_or_serving_runtime_executed") is False
    ):
        errors.append(f"target Qwen LoRA recorded evidence mismatch: {report}")

    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "README.md",
            project / "README.md",
            ROOT / "docs" / "training" / "finetuning.md",
            ROOT / "docs" / "training" / "peft-qlora-engineering.md",
            ROOT / "docs" / "practice" / "labs.md",
            ROOT / "docs" / "career" / "interview-questions.md",
            ROOT / "docs" / "career" / "resume-projects.md",
            ROOT / "docs" / "reference" / "accuracy.md",
        )
    )
    required_markers = (
        "270,336",
        "494,032,768",
        "1,093,728",
        "98,304",
        "0.003864",
        "0.584557",
        "8a3897b1",
        "ffab4958",
        "bit-exact",
        "不证明收敛",
        "QLoRA",
        "TOCTOU",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(f"target Qwen LoRA docs missing boundary marker(s): {missing}")
    return errors


def check_target_dpo_control() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    from about_llm.finetuning.target_dpo_control import (
        load_target_dpo_control_spec,
        verify_recorded_target_dpo_report,
    )
    from about_llm.integrations.transformers_checkpoint_control import (
        load_checkpoint_control_spec,
    )

    errors: list[str] = []
    project = ROOT / "projects" / "single-gpu-finetuning"
    checkpoint_control = (
        ROOT
        / "projects"
        / "transformers-basics"
        / "target-checkpoints"
        / "qwen2.5-0.5b-instruct.control.json"
    )
    checkpoint_report = checkpoint_control.with_name(
        "qwen2.5-0.5b-instruct.recorded-report.json"
    )
    checkpoint_spec = load_checkpoint_control_spec(checkpoint_control)
    spec = load_target_dpo_control_spec(
        project / "qwen2.5-0.5b-dpo.control.json",
        checkpoint_spec=checkpoint_spec,
    )
    report = verify_recorded_target_dpo_report(
        project / "qwen2.5-0.5b-dpo.recorded-report.json",
        spec=spec,
        checkpoint_spec=checkpoint_spec,
        checkpoint_report_path=checkpoint_report,
        training_path=project / "preference.train.example.jsonl",
        readiness_path=project / "preference-training-readiness.example.json",
    )
    result = report.get("result", {})
    tokenization = result.get("tokenization", {})
    model = result.get("model", {})
    execution = result.get("execution", {})
    scope = report.get("scope", {})
    initial_audit = execution.get("initial_adapter_disable_audit", {})
    final_audit = execution.get("final_adapter_disable_audit", {})
    if not (
        spec.manifest_fingerprint
        == "sha256:ebbf365523707c08d8c18c13a26551cf9af7420ce530274e9718bdc4f8d818b3"
        and report.get("report_fingerprint")
        == "sha256:3cafbade034045df61e09907185d6ae37a71e81075e96586bd9c46a3b549b7bc"
        and tokenization.get("collated_input_shape") == [4, 28]
        and tokenization.get("collated_attention_token_count") == 112
        and tokenization.get("completion_token_counts") == [5, 5, 5, 5]
        and model.get("base_parameter_count") == 494_032_768
        and model.get("adapter_parameter_count") == 270_336
        and execution.get("initial_trainer_loss") == 0.6931471824645996
        and execution.get("final_trainer_loss") == 0.333351731300354
        and execution.get("final_relative_margins")
        == [8.566291809082031, 10.01645278930664]
        and execution.get("trainable_gradient_tensor_count") == 96
        and execution.get("finite_gradient_tensor_count") == 96
        and execution.get("frozen_base_gradient_tensor_count") == 0
        and execution.get("frozen_base_parameter_fingerprint_before")
        == execution.get("frozen_base_parameter_fingerprint_after")
        and execution.get("frozen_non_adapter_state_fingerprint_before")
        == execution.get("frozen_non_adapter_state_fingerprint_after")
        and execution.get("normalized_model_config_fingerprint_before")
        == execution.get("normalized_model_config_fingerprint_after")
        and execution.get("normalized_generation_config_fingerprint_before")
        == execution.get("normalized_generation_config_fingerprint_after")
        and execution.get("reference_replay_max_abs_error_after_step")
        == 0.5470771789550781
        and execution.get("reference_replay_bitwise_equal") is False
        and execution.get(
            "reference_replay_drift_reported_not_equated_to_weight_drift"
        )
        is True
        and initial_audit.get("enabled_inside_reference") is False
        and final_audit.get("enabled_inside_reference") is False
        and execution.get("adapter_after", {}).get("nonzero_b_tensor_count") == 48
        and execution.get("adapter_after", {}).get("nonzero_b_element_count")
        == 98_304
        and scope.get("real_trl_dpo_trainer_executed") is True
        and scope.get("alignment_quality_or_generalization_proven") is False
        and scope.get("human_preference_or_annotator_evidence") is False
        and scope.get("qlora_cuda_or_vllm_executed") is False
    ):
        errors.append(f"target Qwen DPO recorded evidence mismatch: {report}")

    documentation = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "README.md",
            project / "README.md",
            ROOT / "docs" / "training" / "alignment.md",
            ROOT / "docs" / "training" / "finetuning.md",
            ROOT / "docs" / "guide" / "knowledge-map.md",
            ROOT / "docs" / "practice" / "labs.md",
            ROOT / "docs" / "practice" / "project-index.md",
            ROOT / "docs" / "career" / "interview-questions.md",
            ROOT / "docs" / "career" / "resume-projects.md",
            ROOT / "docs" / "reference" / "accuracy.md",
        )
    )
    required_markers = (
        "3cafbade",
        "0.693147",
        "0.333352",
        "8.566292",
        "10.016453",
        "0.547077",
        "reference replay",
        "non-adapter",
        "96",
        "98,304",
        "authored",
        "不证明人类偏好",
        "QLoRA",
    )
    missing = [marker for marker in required_markers if marker not in documentation]
    if missing:
        errors.append(f"target Qwen DPO docs missing boundary marker(s): {missing}")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    current_marker = "仓库另有固定 Qwen CPU FP32 DPO 单步 control"
    if current_marker not in changelog:
        errors.append("target Qwen DPO changelog is missing the current target-control boundary")
    stale_marker = "当前环境未验证目标模型或 CUDA"
    if stale_marker in changelog:
        errors.append(
            "target Qwen DPO changelog still claims the environment did not validate "
            "a target model"
        )
    return errors


def check_reward_model_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import numpy as np

    from about_llm.finetuning import (
        fit_linear_pairwise_reward_model,
        pairwise_reward_metrics,
    )

    errors: list[str] = []
    rejected = np.zeros((4, 2), dtype=np.float64)
    confounded_chosen = np.array(
        [[1, 1], [2, 2], [1, 1], [2, 2]], dtype=np.float64
    )
    counterfactual_chosen = np.array(
        [[1, -1], [2, -2], [1, -1], [2, -2]], dtype=np.float64
    )
    held_out_chosen = np.array([[1, -2], [2, -3]], dtype=np.float64)
    held_out_rejected = np.zeros_like(held_out_chosen)

    initial = pairwise_reward_metrics(confounded_chosen, rejected, [0, 0])
    confounded = fit_linear_pairwise_reward_model(
        confounded_chosen,
        rejected,
        steps=300,
        learning_rate=0.1,
    )
    confounded_held_out = pairwise_reward_metrics(
        held_out_chosen,
        held_out_rejected,
        confounded.weights,
    )
    balanced_chosen = np.concatenate(
        (confounded_chosen, counterfactual_chosen), axis=0
    )
    balanced = fit_linear_pairwise_reward_model(
        balanced_chosen,
        np.zeros_like(balanced_chosen),
        steps=300,
        learning_rate=0.1,
    )
    balanced_held_out = pairwise_reward_metrics(
        held_out_chosen,
        held_out_rejected,
        balanced.weights,
    )
    if not (
        math.isclose(initial.mean_loss, math.log(2))
        and initial.strict_pair_accuracy == 0
        and initial.tie_count == 4
        and confounded.final_metrics.strict_pair_accuracy == 1
        and confounded_held_out.strict_pair_accuracy == 0
        and math.isclose(confounded.weights[0], confounded.weights[1])
        and balanced.final_metrics.strict_pair_accuracy == 1
        and math.isclose(balanced.weights[1], 0, abs_tol=1e-12)
        and balanced_held_out.strict_pair_accuracy == 1
    ):
        errors.append("linear reward-model confounding/counterfactual fixture mismatch")

    toy_path = (
        ROOT / "projects" / "single-gpu-finetuning" / "reward_model_toy.py"
    )
    experiment = runpy.run_path(str(toy_path))["run_experiment"]()
    expected_scope = {
        "device": "CPU",
        "authored_numeric_features_and_preferences": True,
        "text_tokenizer_or_transformer_executed": False,
        "human_preference_quality_proved": False,
        "target_reward_model_quality_proved": False,
        "reward_hacking_or_policy_optimization_evaluated": False,
    }
    if not (
        experiment["schema_version"] == 1
        and experiment["feature_semantics"]
        == ["authored_quality_signal", "authored_length_proxy"]
        and experiment["confounded_training"]["final_metrics"]
        ["strict_pair_accuracy"]
        == 1
        and experiment["confounded_held_out"]["strict_pair_accuracy"] == 0
        and experiment["counterfactually_balanced_held_out"]
        ["strict_pair_accuracy"]
        == 1
        and experiment["scope"] == expected_scope
    ):
        errors.append("reward-model toy result/scope boundary mismatch")
    return errors


def check_ppo_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import numpy as np

    from about_llm.finetuning import (
        generalized_advantage_estimation,
        ppo_clipped_surrogate,
    )

    errors: list[str] = []
    gae = generalized_advantage_estimation(
        rewards=[0.0, 1.0, 999.0],
        values=[0.5, 0.25, 999.0],
        next_values=[0.25, 10.0, 999.0],
        valid_mask=[True, True, False],
        terminated=[False, True, False],
        truncated=[False, False, False],
        gamma=0.9,
        gae_lambda=0.8,
        bootstrap_truncated=True,
    )
    if not (
        np.allclose(gae.td_residuals, [-0.275, 0.75, 0.0])
        and np.allclose(gae.advantages, [0.265, 0.75, 0.0])
        and np.allclose(gae.returns, [0.765, 1.0, 0.0])
        and gae.bootstrap_mask.tolist() == [True, False, False]
        and gae.continuation_mask.tolist() == [True, False, False]
    ):
        errors.append("GAE terminated/padding analytic fixture mismatch")

    with_bootstrap = generalized_advantage_estimation(
        [1.0],
        [0.5],
        [2.0],
        valid_mask=[True],
        terminated=[False],
        truncated=[True],
        gamma=0.9,
        gae_lambda=0.8,
        bootstrap_truncated=True,
    )
    without_bootstrap = generalized_advantage_estimation(
        [1.0],
        [0.5],
        [2.0],
        valid_mask=[True],
        terminated=[False],
        truncated=[True],
        gamma=0.9,
        gae_lambda=0.8,
        bootstrap_truncated=False,
    )
    if not (
        math.isclose(with_bootstrap.advantages[0], 2.3)
        and math.isclose(without_bootstrap.advantages[0], 0.5)
        and not with_bootstrap.continuation_mask[0]
    ):
        errors.append("GAE truncation bootstrap/continuation fixture mismatch")

    ratios = np.array([1.5, 0.5, 1.0], dtype=np.float64)
    ppo = ppo_clipped_surrogate(
        np.log(ratios),
        np.zeros_like(ratios),
        [1.0, -1.0, 1.0],
        valid_mask=[True, True, True],
        clip_epsilon=0.2,
    )
    old_distribution = np.array([0.1, 0.45, 0.45], dtype=np.float64)
    new_distribution = np.array([0.1, 0.9 - 1e-12, 1e-12], dtype=np.float64)
    full_kl = float(
        np.sum(old_distribution * np.log(old_distribution / new_distribution))
    )
    sampled = ppo_clipped_surrogate(
        [math.log(new_distribution[0])],
        [math.log(old_distribution[0])],
        [1.0],
        valid_mask=[True],
        clip_epsilon=0.2,
    )
    if not (
        np.allclose(ppo.per_action_surrogate, [1.2, -0.8, 1.0])
        and math.isclose(ppo.clip_fraction, 2 / 3)
        and math.isclose(sampled.mean_probability_ratio, 1)
        and sampled.clip_fraction == 0
        and sampled.approximate_sampled_kl == 0
        and full_kl > 10
    ):
        errors.append("PPO clipping/sampled-ratio KL counterexample mismatch")

    toy_path = (
        ROOT / "projects" / "single-gpu-finetuning" / "ppo_objective_toy.py"
    )
    experiment = runpy.run_path(str(toy_path))["run_experiment"]()
    expected_scope = {
        "device": "CPU",
        "authored_rewards_values_and_distributions": True,
        "numpy_objectives_executed": True,
        "rollout_engine_or_language_model_executed": False,
        "reward_or_value_model_quality_proved": False,
        "full_distribution_kl_constrained": False,
        "stable_ppo_training_proved": False,
    }
    if experiment["scope"] != expected_scope:
        errors.append("PPO objective toy evidence scope mismatch")
    return errors


def check_torch_ppo_smoke_scope() -> list[str]:
    script = (
        ROOT / "projects" / "single-gpu-finetuning" / "smoke_torch_ppo.py"
    ).read_text(encoding="utf-8")
    test = (ROOT / "tests" / "test_torch_ppo_smoke.py").read_text(
        encoding="utf-8"
    )
    required_script_markers = (
        '"on_policy_categorical_sampling_executed": True',
        '"torch_policy_and_value_forward_executed": True',
        '"gae_and_minibatch_optimizer_executed": True',
        '"time_limit_truncation_executed": False',
        '"language_model_or_tokenizer_executed": False',
        '"reward_model_executed": False',
        '"reference_policy_kl_controller_executed": False',
        '"gpu_or_distributed_execution": False',
        '"target_model_quality_or_safety_proved": False',
        '"production_ppo_stability_proved": False',
        '"initial_exact_expected_return"',
        '"final_exact_expected_return"',
        '"all_stored_old_log_probabilities_unchanged"',
        '"all_snapshot_log_probability_errors_zero"',
        '"old_log_probabilities_require_grad"',
    )
    required_test_markers = (
        'report["initial_exact_expected_return"] == pytest.approx(1.0)',
        'report["final_exact_expected_return"] > 1.8',
        'report["total_optimizer_steps"] == 96',
        'report["policy_parameters_changed"] is True',
        'report["value_parameters_changed"] is True',
        'report["all_stored_old_log_probabilities_unchanged"] is True',
        'report["all_snapshot_log_probability_errors_zero"] is True',
        'iteration["old_log_probabilities_require_grad"] is False',
        'first_post_update["maximum_probability_ratio"] > 1.2',
        'first_post_update["minimum_probability_ratio"] < 0.8',
        "assert first == second",
    )
    missing_script = [
        marker for marker in required_script_markers if marker not in script
    ]
    missing_test = [marker for marker in required_test_markers if marker not in test]
    errors: list[str] = []
    if missing_script:
        errors.append(f"torch PPO smoke missing scope/metric marker(s): {missing_script}")
    if missing_test:
        errors.append(f"torch PPO smoke missing executable assertion(s): {missing_test}")
    return errors


def check_transformer_ppo_smoke_scope() -> list[str]:
    script = (
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "smoke_transformer_ppo.py"
    ).read_text(encoding="utf-8")
    test = (ROOT / "tests" / "test_transformer_ppo_smoke.py").read_text(
        encoding="utf-8"
    )
    required_script_markers = (
        '"integer_token_ids_without_tokenizer": True',
        '"random_tiny_gpt2_backbone_executed": True',
        '"autoregressive_token_sampling_executed": True',
        '"frozen_reference_forward_executed": True',
        '"sampled_reference_log_ratio_reward_executed": True',
        '"exact_two_step_task_reward_enumerated": True',
        '"gae_and_transformer_optimizer_executed": True',
        '"learned_reward_model_executed": False',
        '"natural_language_quality_proved": False',
        '"time_limit_truncation_executed": False',
        '"checkpoint_or_resume_executed": False',
        '"cuda_or_distributed_execution": False',
        '"target_llm_ppo_quality_or_safety_proved": False',
        '"initial_exact_expected_task_reward"',
        '"final_exact_expected_task_reward"',
        '"reference_parameters_unchanged"',
        '"maximum_snapshot_log_probability_error"',
    )
    required_test_markers = (
        'report["initial_exact_expected_task_reward"] == pytest.approx(1 / 3)',
        'report["final_exact_expected_task_reward"] > 1.8',
        'report["total_optimizer_steps"] == 36',
        'report["reference_parameters_unchanged"] is True',
        'report["backbone_parameters_changed"] is True',
        'report["policy_head_parameters_changed"] is True',
        'report["value_head_parameters_changed"] is True',
        'report["all_stored_old_log_probabilities_unchanged"] is True',
        'report["maximum_snapshot_log_probability_error"] <= 1e-7',
        'iterations[0]["post_update_maximum_ratio"] > 1.2',
        'iterations[0]["post_update_minimum_ratio"] < 0.8',
        "assert first == second",
    )
    missing_script = [
        marker for marker in required_script_markers if marker not in script
    ]
    missing_test = [marker for marker in required_test_markers if marker not in test]
    errors: list[str] = []
    if missing_script:
        errors.append(
            f"Transformer PPO smoke missing scope/metric marker(s): {missing_script}"
        )
    if missing_test:
        errors.append(
            f"Transformer PPO smoke missing executable assertion(s): {missing_test}"
        )
    return errors


def check_text_ppo_smoke_scope() -> list[str]:
    script = (
        ROOT / "projects" / "single-gpu-finetuning" / "smoke_text_ppo.py"
    ).read_text(encoding="utf-8")
    test = (ROOT / "tests" / "test_text_ppo_smoke.py").read_text(
        encoding="utf-8"
    )
    required_script_markers = (
        '"local_wordlevel_tokenizer_executed": True',
        '"chat_template_and_natural_language_prompt_executed": True',
        '"autoregressive_text_token_sampling_executed": True',
        '"eos_termination_executed": True',
        '"max_new_tokens_truncation_executed": True',
        '"padding_mask_executed": True',
        '"truncated_post_action_values_computed": True',
        '"truncated_transition_value_bootstrap_executed": bootstrap_truncated',
        '"finite_horizon_task_return_stops_at_generation_cap": True',
        '"optimizer_matches_reported_finite_horizon_objective"',
        '"exact_short_horizon_objective_enumerated": True',
        '"learned_reward_model_executed": False',
        '"human_preference_or_natural_language_quality_proved": False',
        '"target_checkpoint_executed": False',
        '"checkpoint_or_resume_executed": False',
        '"cuda_or_distributed_execution": False',
        '"initial_exact_objectives"',
        '"final_exact_objectives"',
        '"reference_parameters_unchanged"',
        '"maximum_snapshot_log_probability_error"',
    )
    required_test_markers = (
        'report["initial_exact_expected_task_reward"] == pytest.approx(25 / 169)',
        'report["initial_exact_good_then_eos_probability"] == pytest.approx(1 / 169)',
        'report["final_exact_expected_task_reward"] > 1.9',
        'report["final_exact_good_then_eos_probability"] > 0.95',
        'report["bootstrap_truncated_in_optimizer"] is False',
        'report["optimizer_matches_reported_finite_horizon_objective"] is True',
        'report["reference_parameters_unchanged"] is True',
        'report["value_backbone_parameters_changed"] is True',
        'report["all_stored_old_log_probabilities_unchanged"] is True',
        'report["maximum_snapshot_log_probability_error"] <= 1e-7',
        'first["terminated_transition_count"] > 0',
        'first["truncated_transition_count"] > 0',
        'first["padding_transition_count"] > 0',
        "assert first == second",
    )
    missing_script = [
        marker for marker in required_script_markers if marker not in script
    ]
    missing_test = [marker for marker in required_test_markers if marker not in test]
    errors: list[str] = []
    if missing_script:
        errors.append(f"text PPO smoke missing scope/metric marker(s): {missing_script}")
    if missing_test:
        errors.append(f"text PPO smoke missing executable assertion(s): {missing_test}")
    return errors


def check_learned_rm_ppo_smoke_scope() -> list[str]:
    script = (
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "smoke_learned_rm_ppo.py"
    ).read_text(encoding="utf-8")
    test = (ROOT / "tests" / "test_learned_rm_ppo_smoke.py").read_text(
        encoding="utf-8"
    )
    required_script_markers = (
        '"training_pair_count": 1',
        '"reachable_response_count": len(responses)',
        '"unseen_response_count": len(responses) - 2',
        '"score_centering": "subtract final training-pair midpoint"',
        '"target_response_rank_of_reachable": target_rank',
        '"reward_model_parameters_unchanged_during_ppo"',
        '"reference_parameters_unchanged_during_ppo"',
        '"local_wordlevel_tokenizer_and_chat_template_executed": True',
        '"generation_allowlist_bound_to_sampling_and_ppo_distribution": True',
        '"pairwise_transformer_reward_model_optimizer_executed": True',
        '"sparse_authored_preference_pair_not_human_labels": True',
        '"frozen_learned_sequence_reward_bound_to_terminal_action": True',
        '"all_reachable_two_token_responses_enumerated": True',
        '"ppo_optimizer_executed_against_learned_proxy": True',
        '"controlled_reward_hacking_counterexample_observed"',
        '"reward_model_quality_or_robustness_proved": False',
        '"human_preference_or_natural_language_quality_proved": False',
        '"target_checkpoint_executed": False',
        '"cuda_or_distributed_execution": False',
        '"production_ppo_stability_proved": False',
    )
    required_test_markers = (
        'reward_model["reachable_response_count"] == 57',
        'reward_model["unseen_response_count"] == 55',
        'reward_model["final_pairwise_loss"] < 0.005',
        'reward_model["target_response_rank_of_reachable"] > 1',
        'initial["authored_target_success_probability"] == pytest.approx(1 / 64)',
        'final["authored_target_success_probability"] < (',
        'report["exact_proxy_reward_improved"] is True',
        'report["exact_authored_dense_task_reward_improved"] is True',
        'report["exact_authored_target_success_improved"] is False',
        'report["reward_hacking_counterexample_observed"] is True',
        'report["reward_model_parameters_unchanged_during_ppo"] is True',
        'report["reference_parameters_unchanged_during_ppo"] is True',
        "assert first == second",
    )
    missing_script = [
        marker for marker in required_script_markers if marker not in script
    ]
    missing_test = [marker for marker in required_test_markers if marker not in test]
    errors: list[str] = []
    if missing_script:
        errors.append(
            f"learned-RM PPO smoke missing scope/metric marker(s): {missing_script}"
        )
    if missing_test:
        errors.append(
            f"learned-RM PPO smoke missing executable assertion(s): {missing_test}"
        )
    return errors


def check_transformer_reward_model_scope() -> list[str]:
    script = (
        ROOT
        / "projects"
        / "single-gpu-finetuning"
        / "smoke_transformer_reward_model.py"
    ).read_text(encoding="utf-8")
    test = (ROOT / "tests" / "test_transformer_reward_model_smoke.py").read_text(
        encoding="utf-8"
    )
    required_script_markers = (
        "GPT2ForSequenceClassification",
        "torch.nn.init.zeros_(model.score.weight)",
        "functional.softplus",
        '"train_only_tokenizer_vocabulary": True',
        '"training_process_without_held_out_access": True',
        '"actual_text_tokenization_executed": True',
        '"transformer_forward_and_optimizer_executed": True',
        '"full_prompt_and_response_scored": True',
        '"authored_preferences_not_human_labels": True',
        '"target_reward_model_quality_proved": False',
        '"broad_counterfactual_robustness_proved": False',
        '"reward_hacking_or_policy_optimization_evaluated": False',
        '"cuda_executed": False',
    )
    required_test_markers = (
        'report["initial_metrics"]["tie_count"] == 2',
        'report["final_metrics"]["strict_pair_accuracy"] == 1',
        'report["authored_counterfactual_metrics"]["strict_pair_accuracy"] == 0',
        'report["reward_head_parameters_changed"] is True',
        'report["transformer_backbone_parameters_changed"] is True',
        'assert not (tmp_path / "preference.example.jsonl").exists()',
        "manifest_fingerprint mismatch",
        "ordered fingerprint differs",
    )
    missing_script = [
        marker for marker in required_script_markers if marker not in script
    ]
    missing_test = [marker for marker in required_test_markers if marker not in test]
    errors: list[str] = []
    if missing_script:
        errors.append(
            f"Transformer reward-model smoke missing scope marker(s): {missing_script}"
        )
    if missing_test:
        errors.append(
            f"Transformer reward-model smoke missing executable assertion(s): {missing_test}"
        )
    return errors


def check_reward_model_training_entry_scope() -> list[str]:
    script = (
        ROOT / "projects" / "single-gpu-finetuning" / "train_reward_model.py"
    ).read_text(encoding="utf-8")
    test = (ROOT / "tests" / "test_reward_model_training_entry.py").read_text(
        encoding="utf-8"
    )
    required_script_markers = (
        "load_preference_training_readiness",
        "validate_preference_training_readiness",
        '"--data-preflight-only"',
        '"--tokenization-preflight-only"',
        "AutoModelForSequenceClassification",
        "num_labels=1",
        'task_type="SEQ_CLS"',
        'modules_to_save=modules_to_save',
        '"RewardTrainer filtered a pair after strict tokenization preflight"',
        '"held_out_dataset_passed_to_trainer": False',
        '"target_model_or_cuda_verified_by_repository": False',
        "len(prepared) != len(records)",
        '"trl_reward_trainer_executed": True',
        '"reward-model-train-result.json"',
        "trainer.state.global_step",
    )
    required_test_markers = (
        "must-not-download",
        'tokenization["scope"]["target_tokenizer_executed"] is True',
        '"manifest_fingerprint mismatch"',
        '"exceeds max_length"',
        'not (output / "reward-model-run-contract.json").exists()',
        'outcome["global_step"] == 1',
        '"lora_B" in name',
    )
    missing_script = [
        marker for marker in required_script_markers if marker not in script
    ]
    missing_test = [marker for marker in required_test_markers if marker not in test]
    errors: list[str] = []
    if missing_script:
        errors.append(
            f"reward-model training entry missing scope marker(s): {missing_script}"
        )
    if missing_test:
        errors.append(
            f"reward-model training entry missing executable assertion(s): {missing_test}"
        )
    return errors


def check_speculative_decoding_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.inference import (
        audit_speculative_distribution,
        speculative_sample_step,
        verify_speculative_block,
    )

    errors: list[str] = []
    audit = audit_speculative_distribution(
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4],
    )
    rejected = speculative_sample_step(
        [0.6, 0.4],
        [0.2, 0.8],
        draft_uniform=0.1,
        acceptance_uniform=0.5,
        correction_uniform=0,
    )
    block = verify_speculative_block(
        draft_tokens=(0, 0),
        draft_probabilities=((0.5, 0.5), (0.8, 0.2)),
        target_probabilities=((0.5, 0.5), (0.2, 0.8)),
        acceptance_uniforms=(0, 0.5),
        correction_uniforms=(0, 0),
        bonus_target_probabilities=(0.1, 0.9),
        bonus_uniform=0,
    )
    if not (
        all(
            math.isclose(observed, expected, abs_tol=1e-15)
            for observed, expected in zip(
                audit.theoretical_output_probabilities,
                audit.target_probabilities,
                strict=True,
            )
        )
        and math.isclose(audit.acceptance_probability, 0.6)
        and math.isclose(audit.rejection_probability, 0.4)
        and math.isclose(audit.total_variation_distance, 0.4)
        and not rejected.accepted
        and rejected.output_token == 1
        and rejected.correction_probabilities == (0.0, 1.0)
        and block.emitted_tokens == (0, 1)
        and block.accepted_draft_tokens == 1
        and block.first_rejection_index == 1
        and not block.used_bonus_target_token
    ):
        errors.append("speculative rejection-sampling fixture mismatch")

    demo = (
        ROOT
        / "projects"
        / "inference-serving"
        / "speculative_decoding_toy.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"authored_probability_vectors": True',
        '"monte_carlo_is_demonstration_not_proof": True',
        '"model_forward_or_tokenizer_executed": False',
        '"gpu_verification_kernel_executed": False',
        '"latency_or_speedup_proved": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(f"speculative toy missing scope marker(s): {missing_scope}")
    return errors


def check_self_consistency_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from fractions import Fraction
    from itertools import pairwise

    from about_llm.inference import (
        BinaryVoteRegime,
        analyze_latent_regime_binary_majority,
    )

    errors: list[str] = []
    independent = (BinaryVoteRegime("iid", 1, 3, 2),)
    correlated = (
        BinaryVoteRegime("easy", 1, 9, 1),
        BinaryVoteRegime("hard", 1, 3, 7),
    )
    expected_independent = {
        1: Fraction(3, 5),
        3: Fraction(81, 125),
        5: Fraction(2_133, 3_125),
        11: Fraction(36_791_901, 48_828_125),
    }
    expected_correlated = {
        1: Fraction(3, 5),
        3: Fraction(297, 500),
        5: Fraction(28_863, 50_000),
        11: Fraction(13_474_113_561, 25_000_000_000),
    }
    independent_analyses = tuple(
        analyze_latent_regime_binary_majority(
            independent,
            sample_count=sample_count,
        )
        for sample_count in expected_independent
    )
    correlated_analyses = tuple(
        analyze_latent_regime_binary_majority(
            correlated,
            sample_count=sample_count,
        )
        for sample_count in expected_correlated
    )
    for analysis in independent_analyses:
        if not (
            analysis.single_sample_success_probability == Fraction(3, 5)
            and analysis.majority_success_probability
            == expected_independent[analysis.sample_count]
            and analysis.pairwise_success_covariance == 0
            and analysis.pairwise_success_correlation == 0
            and analysis.logical_binary_vote_sequences == 2**analysis.sample_count
        ):
            errors.append(
                "independent binary self-consistency exact fixture mismatch at "
                f"N={analysis.sample_count}"
            )
    for analysis in correlated_analyses:
        if not (
            analysis.single_sample_success_probability == Fraction(3, 5)
            and analysis.majority_success_probability
            == expected_correlated[analysis.sample_count]
            and analysis.pairwise_success_covariance == Fraction(9, 100)
            and analysis.pairwise_success_correlation == Fraction(3, 8)
            and analysis.logical_binary_vote_sequences == 2**analysis.sample_count
        ):
            errors.append(
                "latent-correlated binary self-consistency exact fixture mismatch at "
                f"N={analysis.sample_count}"
            )
    independent_majorities = [
        analysis.majority_success_probability for analysis in independent_analyses
    ]
    correlated_majorities = [
        analysis.majority_success_probability for analysis in correlated_analyses
    ]
    if not (
        all(
            left < right
            for left, right in pairwise(independent_majorities)
        )
        and all(
            left > right
            for left, right in pairwise(correlated_majorities)
        )
    ):
        errors.append("binary self-consistency independence/correlation trend mismatch")

    toy_path = (
        ROOT
        / "projects"
        / "inference-serving"
        / "self_consistency_correlation_toy.py"
    )
    report = runpy.run_path(str(toy_path))["run_toy"]()
    expected_scope = {
        "authored_binary_answer_distribution": True,
        "one_latent_regime_drawn_per_question": True,
        "candidate_correctness_conditionally_iid_within_regime": True,
        "exact_fraction_binomial_tail_executed": True,
        "binary_vote_sequence_enumeration_executed": False,
        "multiclass_or_open_text_canonicalization_modeled": False,
        "model_tokenizer_dataset_or_judge_executed": False,
        "latency_cost_provider_or_target_quality_measured": False,
    }
    scenarios = report.get("scenarios", {})
    report_independent = scenarios.get("independent", [])
    report_correlated = scenarios.get("latent_correlated", [])
    if not (
        report.get("implementation")
        == "about-llm.self-consistency-correlation-toy.v1"
        and report.get("binary_answer_labels")
        == ["target_success", "target_failure"]
        and report.get("scope") == expected_scope
        and report.get("observations")
        == {
            "same_single_sample_success_probability": True,
            "independent_majority_strictly_increases": True,
            "correlated_majority_strictly_decreases": True,
            "independent_pairwise_correlation_is_zero": True,
            "latent_pairwise_correlation_is_three_eighths": True,
        }
        and [analysis["sample_count"] for analysis in report_independent]
        == [1, 3, 5, 11]
        and [analysis["sample_count"] for analysis in report_correlated]
        == [1, 3, 5, 11]
        and report_independent[-1]["majority_success_probability"]["numerator"]
        == 36_791_901
        and report_correlated[-1]["majority_success_probability"]["numerator"]
        == 13_474_113_561
    ):
        errors.append("binary self-consistency project report/scope mismatch")

    documentation_paths = (
        ROOT / "docs" / "frontier" / "reasoning-long-context-moe.md",
        ROOT / "projects" / "inference-serving" / "README.md",
        ROOT / "docs" / "practice" / "projects" / "inference-serving.md",
        ROOT / "docs" / "practice" / "project-index.md",
        ROOT / "docs" / "practice" / "labs.md",
        ROOT / "docs" / "career" / "interview-questions.md",
        ROOT / "docs" / "guide" / "knowledge-map.md",
        ROOT / "docs" / "reference" / "accuracy.md",
        ROOT / "docs" / "reference" / "glossary.md",
        ROOT / "CHANGELOG.md",
    )
    documentation = "\n".join(
        path.read_text(encoding="utf-8") for path in documentation_paths
    )
    required_markers = (
        "self-consistency",
        "conditional i.i.d.",
        "3/8",
        "0.75349813248",
        "0.53896454244",
        "2^11=2,048",
        "开放文本",
        "canonicalization",
        "model",
        "tokenizer",
        "dataset",
        "judge",
        "provider",
        "latency",
        "cost",
    )
    missing_markers = [
        marker for marker in required_markers if marker not in documentation
    ]
    if missing_markers:
        errors.append(
            "binary self-consistency docs missing assumption/scope marker(s): "
            f"{missing_markers}"
        )
    return errors


def check_verifier_selection_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from fractions import Fraction

    from about_llm.inference import (
        VerifierCandidate,
        analyze_verifier_guided_best_of_n,
    )

    errors: list[str] = []
    candidates = (
        VerifierCandidate("wrong", 5, 20, False),
        VerifierCandidate("correct", 4, 80, True),
        VerifierCandidate("verifier_hack", 1, 99, False),
    )
    analyses = {
        sample_count: analyze_verifier_guided_best_of_n(
            candidates,
            sample_count=sample_count,
        )
        for sample_count in (1, 4, 16)
    }
    expected = {
        1: (
            Fraction(2, 5),
            Fraction(2, 5),
            Fraction(0, 1),
            Fraction(519, 10),
            3,
        ),
        4: (
            Fraction(544, 625),
            Fraction(371, 625),
            Fraction(173, 625),
            Fraction(827_841, 10_000),
            81,
        ),
        16: (
            Fraction(152_544_843_904, 152_587_890_625),
            Fraction(28_951_056_265_019, 156_250_000_000_000),
            Fraction(127_254_863_892_677, 156_250_000_000_000),
            Fraction(954_783_461_138_377_521, 10_000_000_000_000_000),
            43_046_721,
        ),
    }
    for sample_count, expected_values in expected.items():
        analysis = analyses[sample_count]
        observed = (
            analysis.oracle_success_probability,
            analysis.selected_success_probability,
            analysis.oracle_selection_gap,
            analysis.expected_selected_verifier_score,
            analysis.logical_candidate_sequences,
        )
        if observed != expected_values:
            errors.append(
                f"verifier best-of-{sample_count} exact fraction mismatch: {observed}"
            )
        if sum(
            (
                selection.selection_probability
                for selection in analysis.selections
            ),
            start=Fraction(0, 1),
        ) != Fraction(1, 1):
            errors.append(
                f"verifier best-of-{sample_count} selection mass does not sum to one"
            )

    one, four, sixteen = (analyses[count] for count in (1, 4, 16))
    if not (
        one.oracle_success_probability
        < four.oracle_success_probability
        < sixteen.oracle_success_probability
        and one.expected_selected_verifier_score
        < four.expected_selected_verifier_score
        < sixteen.expected_selected_verifier_score
        and four.selected_success_probability > one.selected_success_probability
        and sixteen.selected_success_probability < one.selected_success_probability
    ):
        errors.append("verifier best-of-N oracle/proxy/selection trend mismatch")

    toy_path = (
        ROOT
        / "projects"
        / "inference-serving"
        / "verifier_best_of_n_toy.py"
    )
    report = runpy.run_path(str(toy_path))["run_toy"]()
    expected_scope = {
        "authored_finite_candidate_distribution": True,
        "iid_fixed_distribution_assumed": True,
        "closed_form_exact_fraction_analysis_executed": True,
        "candidate_sequence_enumeration_executed": False,
        "oracle_target_labels_authored": True,
        "model_tokenizer_or_prm_executed": False,
        "verifier_calibration_or_semantic_correctness_proved": False,
        "latency_cost_parallelism_or_target_quality_measured": False,
        "target_model_provider_or_gpu_behavior_proved": False,
    }
    report_analyses = {
        analysis["sample_count"]: analysis for analysis in report["analyses"]
    }
    if not (
        report.get("implementation")
        == "about-llm.verifier-best-of-n-toy.v1"
        and report.get("scope") == expected_scope
        and report.get("observations")
        == {
            "expected_verifier_score_strictly_increases": True,
            "selected_success_n4_above_n1": True,
            "selected_success_n16_below_n1": True,
            "oracle_success_strictly_increases": True,
        }
        and set(report_analyses) == {1, 4, 16}
        and report_analyses[16]["logical_candidate_sequences"] == 3**16
        and Fraction(
            report_analyses[16]["selected_success_probability"]["numerator"],
            report_analyses[16]["selected_success_probability"]["denominator"],
        )
        == expected[16][1]
    ):
        errors.append("verifier best-of-N project report/scope mismatch")

    documentation_paths = (
        ROOT / "docs" / "frontier" / "reasoning-long-context-moe.md",
        ROOT / "projects" / "inference-serving" / "README.md",
        ROOT / "docs" / "practice" / "projects" / "inference-serving.md",
        ROOT / "docs" / "practice" / "project-index.md",
        ROOT / "docs" / "practice" / "labs.md",
        ROOT / "docs" / "career" / "interview-questions.md",
        ROOT / "docs" / "guide" / "knowledge-map.md",
        ROOT / "docs" / "reference" / "accuracy.md",
        ROOT / "CHANGELOG.md",
    )
    documentation = "\n".join(
        path.read_text(encoding="utf-8") for path in documentation_paths
    )
    required_markers = (
        "oracle@N",
        "selected@N",
        "3^16=43,046,721",
        "wall-clock",
        "i.i.d.",
        "deterministic score",
        "model",
        "tokenizer",
        "PRM",
        "GPU",
        "provider",
        "verifier calibration",
        "语义正确",
        "目标模型质量",
    )
    missing_markers = [
        marker for marker in required_markers if marker not in documentation
    ]
    if missing_markers:
        errors.append(
            "verifier best-of-N docs missing assumption/scope marker(s): "
            f"{missing_markers}"
        )
    return errors


def check_sampling_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.inference import SamplingConfig, sample_next_token

    errors: list[str] = []
    step = sample_next_token(
        [math.log(0.4), math.log(0.3), math.log(0.2), math.log(0.1)],
        config=SamplingConfig(temperature=1, top_k=3, top_p=0.7),
        uniform=0.6,
    )
    penalty = sample_next_token(
        [2, -2, 0.5],
        config=SamplingConfig(repetition_penalty=2),
        prior_token_ids=(0, 1, 1),
        uniform=0,
    )
    if not (
        step.ranked_token_ids == (0, 1, 2, 3)
        and step.top_k_token_ids == (0, 1, 2)
        and step.top_p_token_ids == (0, 1)
        and step.support_token_ids == (0, 1)
        and all(
            math.isclose(observed, expected, rel_tol=1e-15, abs_tol=1e-15)
            for observed, expected in zip(
                step.probabilities,
                (4 / 7, 3 / 7, 0, 0),
                strict=True,
            )
        )
        and step.sampled_token_id == 1
        and tuple(penalty.repetition_adjusted_logits) == (1, -4, 0.5)
    ):
        errors.append("next-token sampling order/support/probability fixture mismatch")

    demo = (
        ROOT / "projects" / "inference-serving" / "sampling_toy.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"authored_finite_logits_processed": True',
        '"fixed_uniform_inverse_cdf_executed": True',
        '"processor_order_and_tie_break_fixed": True',
        '"model_forward_or_tokenizer_executed": False',
        '"multi_token_eos_stop_or_kv_modeled": False',
        '"runtime_default_equivalence_proved": False',
        '"generation_quality_latency_or_throughput_proved": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(f"sampling toy missing scope marker(s): {missing_scope}")
    return errors


def check_beam_search_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.inference import beam_search_from_probabilities

    errors: list[str] = []
    pruning_table = {
        (): [0.6, 0.4, 0.0],
        (0,): [0.49, 0.0, 0.51],
        (1,): [0.0, 0.0, 1.0],
    }
    narrow = beam_search_from_probabilities(
        pruning_table,
        vocabulary_size=3,
        eos_token_id=2,
        beam_width=1,
        max_new_tokens=2,
    )
    wide = beam_search_from_probabilities(
        pruning_table,
        vocabulary_size=3,
        eos_token_id=2,
        beam_width=2,
        max_new_tokens=2,
    )
    length_table = {
        (): [0.6, 0.4, 0.0, 0.0],
        (0,): [0.0, 0.0, 0.0, 1.0],
        (1,): [0.0, 0.0, 1.0, 0.0],
        (1, 2): [0.0, 0.0, 0.0, 1.0],
    }
    alpha_zero = beam_search_from_probabilities(
        length_table,
        vocabulary_size=4,
        eos_token_id=3,
        beam_width=2,
        max_new_tokens=3,
        length_penalty=0,
    )
    alpha_two = beam_search_from_probabilities(
        length_table,
        vocabulary_size=4,
        eos_token_id=3,
        beam_width=2,
        max_new_tokens=3,
        length_penalty=2,
    )
    if not (
        narrow.returned_sequences[0].token_ids == (0, 2)
        and math.isclose(
            math.exp(narrow.returned_sequences[0].cumulative_log_probability),
            0.306,
        )
        and wide.returned_sequences[0].token_ids == (1, 2)
        and math.isclose(
            math.exp(wide.returned_sequences[0].cumulative_log_probability),
            0.4,
        )
        and alpha_zero.returned_sequences[0].token_ids == (0, 3)
        and alpha_two.returned_sequences[0].token_ids == (1, 2, 3)
        and alpha_two.length_definition
        == "generated tokens only; emitted EOS included; prompt excluded"
        and all(
            2 not in prefix.token_ids
            for step in wide.steps
            for prefix in step.active_before
        )
    ):
        errors.append("beam pruning/EOS/length-penalty fixture mismatch")

    demo = (
        ROOT / "projects" / "inference-serving" / "beam_search_toy.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"beam_pruning_eos_and_length_finalization_executed": True',
        '"global_sequence_optimality_proved": False',
        '"length_penalty_includes_eos_and_excludes_prompt": True',
        '"model_tokenizer_kv_or_gpu_executed": False',
        '"runtime_or_provider_equivalence_claimed": False',
        '"text_quality_or_performance_proved": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(f"beam-search toy missing scope marker(s): {missing_scope}")
    return errors


def check_constrained_decoding_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.inference import (
        LiteralSetConstraint,
        constrained_greedy_from_probabilities,
    )

    constraint = LiteralSetConstraint.from_literals(('{"x":1}', '{"x":2}'))
    token_texts = ('{"x"', ":", "1}", "1]", "2}", None, "garbage")
    result = constrained_greedy_from_probabilities(
        {
            (): [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            (0,): [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.1],
            (0, 1): [0.0, 0.0, 0.25, 0.65, 0.10, 0.0, 0.0],
            (0, 1, 2): [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
        token_texts=token_texts,
        eos_token_id=5,
        constraint=constraint,
        max_new_tokens=4,
    )
    critical = result.steps[2]
    errors: list[str] = []
    if not (
        result.token_ids == (0, 1, 2, 5)
        and result.decoded_text == '{"x":1}'
        and result.finish_reason == "eos"
        and result.constraint_accepting
        and result.eos_emitted
        and critical.grammar_allowed_token_ids == (2, 4)
        and 3 in critical.grammar_blocked_token_ids
        and math.isclose(critical.raw_allowed_probability_mass, 0.35)
        and math.isclose(critical.normalized_probabilities[2], 5 / 7)
        and math.isclose(critical.normalized_probabilities[4], 2 / 7)
        and result.steps[-1].grammar_allowed_token_ids == (5,)
    ):
        errors.append("constrained full-token/mask/renormalization fixture mismatch")

    demo = (
        ROOT / "projects" / "inference-serving" / "constrained_decoding_toy.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"complete_multi_character_token_transition_checked": True',
        '"allowed_probability_mass_renormalized": True',
        '"eos_requires_accepting_state": True',
        '"finite_authored_literal_set_only": True',
        '"tokenizer_byte_state_or_normalization_executed": False',
        '"json_schema_cfg_or_provider_runtime_equivalence_proved": False',
        '"model_kv_gpu_quality_or_performance_executed": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(
            f"constrained-decoding toy missing scope marker(s): {missing_scope}"
        )
    return errors


def check_stop_matching_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.inference import IncrementalStopMatcher

    errors: list[str] = []
    payload = "甲🙂乙<END>尾".encode()
    matcher = IncrementalStopMatcher(("<END>", "STOP"))
    emitted: list[str] = []
    for chunk in (payload[:4], payload[4:7], payload[7:12], payload[12:]):
        update = matcher.feed(chunk)
        emitted.append(update.emitted_text)
        if update.stopped:
            break
    report = matcher.report()
    overlap = IncrementalStopMatcher(("BC", "ABC"))
    overlap_update = overlap.feed(b"ABCZ")
    if not (
        "".join(emitted) == "甲🙂乙"
        and report.stopped
        and report.matched_stop == "<END>"
        and report.decoded_characters == 9
        and report.emitted_characters == 3
        and report.held_characters == 0
        and report.discarded_after_stop_characters == 1
        and report.buffered_utf8_bytes == 0
        and overlap_update.matched_stop == "BC"
        and overlap_update.emitted_text == "A"
        and overlap_update.discarded_after_stop_characters == 1
    ):
        errors.append("incremental UTF-8 stop matching fixture mismatch")

    demo = (
        ROOT / "projects" / "inference-serving" / "stop_matching_toy.py"
    ).read_text(encoding="utf-8")
    required_scope = (
        '"strict_incremental_utf8_decoding_executed": True',
        '"partial_stop_withholding_executed": True',
        '"byte_chunk_independent_character_matching_executed": True',
        '"tokenizer_or_model_token_ids_decoded": False',
        '"provider_usage_or_finish_reason_equivalence_proved": False',
        '"server_cancellation_gpu_release_or_billing_proved": False',
        '"unicode_normalization_or_case_folding_performed": False',
    )
    missing_scope = [marker for marker in required_scope if marker not in demo]
    if missing_scope:
        errors.append(f"stop-matching toy missing scope marker(s): {missing_scope}")
    return errors


def check_inference_workload_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.inference import (
        ArrivalProcess,
        BatchingRequest,
        PrefixCache,
        PrefixCacheIdentity,
        WorkloadSLO,
        build_arrival_schedule,
        simulate_continuous_batching,
        simulate_kv_preemption_batching,
        summarize_attempts,
    )
    from about_llm.inference_analysis_cli import load_attempts

    attempts = load_attempts(
        ROOT / "projects" / "inference-serving" / "attempts.example.jsonl"
    )
    summary = summarize_attempts(
        attempts,
        benchmark_started_at=0,
        benchmark_completed_at=2,
    )
    errors: list[str] = []
    constant = build_arrival_schedule(
        4,
        process=ArrivalProcess.CONSTANT,
        requests_per_second=4,
    )
    poisson = build_arrival_schedule(
        8,
        process=ArrivalProcess.POISSON,
        requests_per_second=4,
        seed=7,
    )
    repeated_poisson = build_arrival_schedule(
        8,
        process=ArrivalProcess.POISSON,
        requests_per_second=4,
        seed=7,
    )
    batching = simulate_continuous_batching(
        [
            BatchingRequest("a", 0, 4, 3),
            BatchingRequest("b", 1, 2, 2),
            BatchingRequest("c", 1, 1, 1),
        ],
        max_batch_tokens=4,
        max_running_sequences=2,
        max_prefill_tokens_per_request=3,
    )
    kv_preemption = simulate_kv_preemption_batching(
        [
            BatchingRequest("a", 0, 4, 3),
            BatchingRequest("b", 1, 2, 2),
        ],
        total_blocks=3,
        block_size_tokens=2,
        max_batch_tokens=4,
        max_running_sequences=2,
        max_prefill_tokens_per_request=3,
    )
    prefix_identity = PrefixCacheIdentity(
        trusted_tenant_id="tenant-a",
        visibility_domain="private",
        authorization_revision="acl-v3",
        policy_revision="policy-v5",
        model_revision="model-sha-111",
        tokenizer_revision="tokenizer-sha-222",
        chat_template_revision="template-sha-333",
        adapter_revision="adapter-none",
        position_config_revision="rope-base-10000-max-8192",
        kv_dtype="float16",
    )
    other_prefix_identity = PrefixCacheIdentity(
        **{
            **prefix_identity.to_dict(),
            "trusted_tenant_id": "tenant-b",
        }
    )
    prefix_cache = PrefixCache(
        capacity_entries=3,
        fingerprint=lambda _identity, _tokens: "collision",
    )
    prefix_cache.store(prefix_identity, (11, 12))
    prefix_cache.store(prefix_identity, (11, 12, 13))
    prefix_cache.store(other_prefix_identity, (11, 12, 13))
    prefix_lease = prefix_cache.acquire_longest_prefix(
        prefix_identity, (11, 12, 13, 14)
    )
    cross_tenant_lease = prefix_cache.acquire_longest_prefix(
        PrefixCacheIdentity(
            **{
                **prefix_identity.to_dict(),
                "trusted_tenant_id": "tenant-c",
            }
        ),
        (11, 12, 13, 14),
    )
    if not (
        constant.offsets_seconds == (0.0, 0.25, 0.5, 0.75)
        and math.isclose(constant.realized_requests_per_second or -1, 4)
        and poisson == repeated_poisson
        and poisson.offsets_seconds[0] == 0
        and all(
            current >= previous
            for previous, current in zip(
                poisson.offsets_seconds,
                poisson.offsets_seconds[1:],
                strict=False,
            )
        )
    ):
        errors.append("inference finite arrival schedule example mismatch")
    if not (
        [step.used_token_slots for step in batching.steps] == [3, 3, 2, 2]
        and batching.prompt_tokens == 7
        and batching.output_tokens == 6
        and batching.modeled_forward_tokens == 10
        and batching.elapsed_token_capacity == 16
        and batching.requests[0].output_emitted_at_steps == (2, 3, 4)
        and batching.requests[2].queue_steps == 2
        and batching.requests[2].ttft_steps == 3
    ):
        errors.append("continuous-batching schedule/work accounting mismatch")
    batching_demo = (
        ROOT / "projects" / "inference-serving" / "continuous_batching_toy.py"
    ).read_text(encoding="utf-8")
    batching_scope = (
        '"deterministic_discrete_cpu_policy_simulated": True',
        '"prefill_last_position_emits_first_token": True',
        '"real_model_or_gpu_kernel_executed": False',
        '"vllm_scheduler_equivalence_proved": False',
        '"kv_capacity_preemption_or_prefix_cache_modeled": False',
        '"wall_clock_latency_throughput_or_slo_proved": False',
    )
    missing_batching_scope = [
        marker for marker in batching_scope if marker not in batching_demo
    ]
    if missing_batching_scope:
        errors.append(
            "continuous-batching toy missing scope marker(s): "
            f"{missing_batching_scope}"
        )
    if not (
        [step.used_token_slots for step in kv_preemption.steps]
        == [3, 3, 1, 1, 2, 1]
        and kv_preemption.logical_forward_positions == 9
        and kv_preemption.recomputed_positions == 2
        and kv_preemption.executed_forward_positions == 11
        and kv_preemption.preemption_count == 1
        and kv_preemption.peak_allocated_blocks == 3
        and kv_preemption.final_free_blocks == 3
        and kv_preemption.requests[0].output_emitted_at_steps == (2, 3, 4)
        and kv_preemption.requests[1].output_emitted_at_steps == (2, 6)
        and kv_preemption.requests[1].admission_steps == (1, 3)
    ):
        errors.append("KV-aware preemption/recompute work accounting mismatch")
    kv_preemption_demo = (
        ROOT
        / "projects"
        / "inference-serving"
        / "kv_preemption_batching_toy.py"
    ).read_text(encoding="utf-8")
    kv_preemption_scope = (
        '"metadata_only_paged_kv_and_scheduler_integrated": True',
        '"recompute_preemption_and_rebuild_executed": True',
        '"logical_and_executed_forward_work_separated": True',
        '"real_kv_tensor_values_or_gpu_kernel_executed": False',
        '"swap_prefix_cache_or_distributed_scheduler_modeled": False',
        '"vllm_scheduler_equivalence_proved": False',
        '"wall_clock_latency_throughput_vram_or_quality_proved": False',
    )
    missing_kv_preemption_scope = [
        marker
        for marker in kv_preemption_scope
        if marker not in kv_preemption_demo
    ]
    if missing_kv_preemption_scope:
        errors.append(
            "KV-aware preemption toy missing scope marker(s): "
            f"{missing_kv_preemption_scope}"
        )
    if prefix_lease is None:
        errors.append("prefix-cache longest exact-prefix fixture missed")
    else:
        prefix_cache.release(prefix_lease)
        prefix_report = prefix_cache.report()
        if not (
            prefix_lease.matched_token_ids == (11, 12, 13)
            and prefix_lease.matched_length == 3
            and cross_tenant_lease is None
            and prefix_report.resident_entries == 3
            and prefix_report.hits == 1
            and prefix_report.misses == 1
            and prefix_report.evictions == 0
            and prefix_report.active_leases == 0
        ):
            errors.append("prefix-cache collision/identity/lease fixture mismatch")
    prefix_demo = (
        ROOT / "projects" / "inference-serving" / "prefix_cache_toy.py"
    ).read_text(encoding="utf-8")
    prefix_scope = (
        '"full_identity_and_exact_token_comparison_executed": True',
        '"fingerprint_collision_injected": True',
        '"cross_tenant_reuse_observed": False',
        '"real_kv_tensors_or_gpu_runtime_executed": False',
        '"vram_latency_hit_rate_or_prefill_savings_proved": False',
        '"timing_channel_mitigation_proved": False',
        '"fingerprint_confidentiality_or_authorization_proved": False',
        '"vllm_prefix_cache_equivalence_proved": False',
    )
    missing_prefix_scope = [
        marker for marker in prefix_scope if marker not in prefix_demo
    ]
    if missing_prefix_scope:
        errors.append(
            "prefix-cache toy missing scope marker(s): "
            f"{missing_prefix_scope}"
        )
    if not (
        summary.attempted_requests == 4
        and summary.successful_requests == 3
        and math.isclose(summary.success_rate, 3 / 4)
        and summary.failure_counts == {"rate_limited": 1}
        and math.isclose(summary.successful_output_tokens_per_second, 7)
        and summary.offered_timing_attempt_count == 4
        and math.isclose(summary.client_queue_p95_seconds or -1, 0.185)
        and math.isclose(summary.successful_offered_ttft_p95_seconds or -1, 0.58)
        and math.isclose(summary.offered_to_terminal_p95_seconds or -1, 1.37)
    ):
        errors.append(f"inference attempt summary mismatch: {summary}")
    passed, reasons = WorkloadSLO(
        minimum_success_rate=0.75,
        maximum_ttft_p95_seconds=0.5,
        maximum_e2e_p95_seconds=1.5,
        maximum_tpot_p95_seconds=0.3,
        maximum_client_queue_p95_seconds=0.2,
        maximum_successful_offered_ttft_p95_seconds=0.6,
        maximum_offered_to_terminal_p95_seconds=1.5,
    ).evaluate(summary)
    if not passed or reasons:
        errors.append(f"inference offered/dispatch SLO example must pass: {reasons}")
    strict_passed, strict_reasons = WorkloadSLO(
        minimum_success_rate=1,
        maximum_ttft_p95_seconds=0.3,
    ).evaluate(summary)
    if strict_passed or len(strict_reasons) != 2:
        errors.append(
            "inference strict SLO must retain reliability and conditional latency: "
            f"{strict_reasons}"
        )
    return errors


def check_usage_budget_examples() -> list[str]:
    import asyncio

    import httpx

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.integrations.budgeted_cloud import (
        BudgetedCloudRetryError,
        BudgetedCloudRetryResult,
        execute_budgeted_json_request_with_retry,
    )
    from about_llm.integrations.cloud_api import (
        ChatMessage,
        build_openai_compatible_request,
        parse_openai_compatible_response,
    )
    from about_llm.integrations.cloud_http import HttpExecutorConfig
    from about_llm.integrations.retry import RetryPolicy
    from about_llm.integrations.sqlite_usage_budget import SQLiteUsageBudgetLedger
    from about_llm.integrations.usage_budget import (
        PostCallBudgetExceededError,
        TokenPricingSnapshot,
        UsageBudgetLedger,
        UsageBudgetLimits,
        cloud_request_budget_fingerprint,
    )

    errors: list[str] = []
    pricing = TokenPricingSnapshot(
        pricing_id="authored-provider/model@price-v1",
        provider="authored-provider",
        model="model",
        revision="price-v1",
        checked_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        input_microusd_per_million=1_000_000,
        output_microusd_per_million=2_000_000,
    )
    ledger = UsageBudgetLedger(
        limits=UsageBudgetLimits(
            max_input_tokens=100,
            max_output_tokens=20,
            max_estimated_microusd=140,
        ),
        pricing=pricing,
    )
    request = build_openai_compatible_request(
        base_url="https://provider.invalid",
        api_key="example-secret-not-real",
        model="model",
        messages=[ChatMessage("user", "authored request")],
        max_tokens=10,
    )
    request_fingerprint = cloud_request_budget_fingerprint(
        request, billing_scope="authored-account/project"
    )
    reservation = ledger.reserve_request(
        "call-1",
        request=request,
        billing_scope="authored-account/project",
        estimated_input_tokens=60,
    )
    settled = ledger.settle(
        "call-1",
        request_fingerprint=reservation.request_fingerprint,
        actual_input_tokens=58,
        actual_output_tokens=4,
    )
    second_request = build_openai_compatible_request(
        base_url="https://provider.invalid",
        api_key="rotated-example-secret-not-real",
        model="model",
        messages=[ChatMessage("user", "authored request")],
        max_tokens=5,
    )
    second_reservation = ledger.reserve_request(
        "call-2",
        request=second_request,
        billing_scope="authored-account/project",
        estimated_input_tokens=20,
    )
    uncertain = ledger.mark_usage_uncertain(
        "call-2", request_fingerprint=second_reservation.request_fingerprint
    )
    if not (
        reservation.maximum_estimated_microusd == 80
        and reservation.request_fingerprint == request_fingerprint
        and "example-secret-not-real" not in request_fingerprint
        and settled.pricing_id == "authored-provider/model@price-v1"
        and settled.max_estimated_microusd == 140
        and settled.committed_estimated_microusd == 66
        and settled.remaining_estimated_microusd == 74
        and uncertain.committed_input_tokens == 78
        and uncertain.committed_output_tokens == 9
        and uncertain.committed_estimated_microusd == 96
        and uncertain.uncertain_settlements == 1
    ):
        errors.append("cloud usage-budget reservation/settlement fixture mismatch")

    breached = UsageBudgetLedger(
        limits=UsageBudgetLimits(max_input_tokens=60), pricing=pricing
    )
    breached.reserve(
        "call",
        request_fingerprint=request_fingerprint,
        estimated_input_tokens=60,
        maximum_output_tokens=0,
    )
    try:
        breached.settle(
            "call",
            request_fingerprint=request_fingerprint,
            actual_input_tokens=61,
            actual_output_tokens=0,
        )
    except PostCallBudgetExceededError as error:
        if not error.snapshot.over_limit or error.snapshot.committed_input_tokens != 61:
            errors.append("cloud usage-budget post-call breach snapshot mismatch")
    else:
        errors.append("cloud usage-budget failed to reject post-call overrun")

    with tempfile.TemporaryDirectory() as temporary_directory:
        database = Path(temporary_directory) / "durable-budget.sqlite"
        durable = SQLiteUsageBudgetLedger(
            database,
            limits=UsageBudgetLimits(
                max_input_tokens=100,
                max_output_tokens=20,
                max_estimated_microusd=140,
            ),
            pricing=pricing,
        )
        durable_reservation = durable.reserve_request(
            "durable-call",
            request=request,
            billing_scope="authored-account/project",
            estimated_input_tokens=60,
        )
        reopened = SQLiteUsageBudgetLedger(
            database,
            limits=UsageBudgetLimits(
                max_input_tokens=100,
                max_output_tokens=20,
                max_estimated_microusd=140,
            ),
            pricing=pricing,
        )
        active = reopened.list_active()
        durable_snapshot = reopened.mark_usage_uncertain(
            "durable-call",
            request_fingerprint=durable_reservation.request_fingerprint,
        )
        event_types = tuple(
            event.event_type for event in reopened.events("durable-call")
        )
        if not (
            len(active) == 1
            and active[0].state == "active"
            and durable_snapshot.committed_input_tokens == 60
            and durable_snapshot.committed_output_tokens == 10
            and durable_snapshot.committed_estimated_microusd == 80
            and durable_snapshot.uncertain_settlements == 1
            and event_types == ("reserved", "uncertain")
        ):
            errors.append("durable cloud usage-budget reopen fixture mismatch")

    async def retry_fixture(
        *, hard_limit: int
    ) -> tuple[BudgetedCloudRetryResult | BudgetedCloudRetryError, int]:
        retry_ledger = UsageBudgetLedger(
            limits=UsageBudgetLimits(
                max_input_tokens=200,
                max_output_tokens=40,
                max_estimated_microusd=hard_limit,
            ),
            pricing=pricing,
        )
        calls = 0

        def handler(outbound: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    500, request=outbound, json={"error": "authored"}
                )
            return httpx.Response(
                200,
                request=outbound,
                headers={"content-type": "application/json"},
                json={
                    "model": "model",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "answer",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 58, "completion_tokens": 4},
                },
            )

        async def no_sleep(_delay: float) -> None:
            return None

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            try:
                result: BudgetedCloudRetryResult | BudgetedCloudRetryError = (
                    await execute_budgeted_json_request_with_retry(
                        ledger=retry_ledger,
                        logical_call_id="logical-call",
                        billing_scope="authored-account/project",
                        estimated_input_tokens=60,
                        client=client,
                        request=request,
                        parse_response=parse_openai_compatible_response,
                        policy=RetryPolicy(max_attempts=2, base_delay_seconds=0),
                        config=HttpExecutorConfig(
                            allowed_origins=frozenset({"https://provider.invalid"}),
                            deadline_seconds=5,
                            request_timeout_seconds=2,
                        ),
                        replay_safe=True,
                        sleep=no_sleep,
                        jitter=lambda: 0,
                    )
                )
            except BudgetedCloudRetryError as error:
                result = error
        return result, calls

    retry_result, retry_calls = asyncio.run(retry_fixture(hard_limit=200))
    if isinstance(retry_result, BudgetedCloudRetryError):
        errors.append(f"budgeted cloud retry fixture failed: {retry_result.reason}")
    else:
        retry_attempts = retry_result.attempts
        if not (
            retry_calls == 2
            and tuple(
                attempt.reservation.reservation_id for attempt in retry_attempts
            )
            == ("logical-call:attempt:1", "logical-call:attempt:2")
            and tuple(
                attempt.reconciliation_state for attempt in retry_attempts
            )
            == ("uncertain", "settled")
            and retry_result.budget_snapshot.committed_estimated_microusd == 146
        ):
            errors.append("budgeted cloud 500-to-200 retry fixture mismatch")

    gate_result, gate_calls = asyncio.run(retry_fixture(hard_limit=140))
    if not (
        isinstance(gate_result, BudgetedCloudRetryError)
        and gate_result.reason == "budget_reservation_rejected"
        and gate_calls == 1
        and len(gate_result.attempts) == 1
        and gate_result.attempts[0].reconciliation_state == "uncertain"
        and gate_result.budget_snapshot.committed_estimated_microusd == 80
    ):
        errors.append("budgeted cloud retry hard-gate fixture mismatch")

    project = (ROOT / "projects" / "cloud-api-contracts" / "README.md").read_text(
        encoding="utf-8"
    )
    required_boundaries = (
        "从实际 RequestSpec 提取最大输出 token",
        "同一 billing scope 换 key 不改变 identity",
        "只有能证明 transport 从未发送时才可释放",
        "按完整 reservation 保守入账",
        "策略估值单位，不是 provider 发票",  # noqa: RUF001
        "SHA-256 也不是签名/保密机制",
        "单进程内存对象，不 durable、不跨 worker",  # noqa: RUF001
        "Active reservation **不按 TTL 自动释放**",
        "SQLite 只提供单文件所在机器可达范围内的 durable atomic quota",
        "不证明 server cancellation、provider usage、invoice 或 exactly-once billing",
        "Config fingerprint 和 request SHA-256 都没有密钥",
        "任何 HTTP response 都证明 request 已越过客户端的“确定未发送”边界",
        "强制 `RetryPolicy(max_attempts=1)`",
        "每次 replay 都必须新建独立 reservation",
        "settled 66 micro-USD 与 uncertain 80 micro-USD",
        "execute_budgeted_json_request_with_retry",
        "before_attempt",
        "HTTP 500→200",
        "hard limit 设为 140",
        "这是 JSON-only reference",
        "不声称 HTTP 500 一定收费",
        "不证明 provider usage、invoice、server cancellation、idempotency 或 exactly-once billing",
    )
    missing = [marker for marker in required_boundaries if marker not in project]
    if missing:
        errors.append(f"cloud usage-budget docs missing boundary marker(s): {missing}")
    return errors


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    files = text_files()
    errors = (
        check_encoding(files)
        + check_ledger()
        + check_model_boundaries()
        + check_foundation_boundaries()
        + check_generation_boundaries()
        + check_core_boundaries()
        + check_training_boundaries()
        + check_quality_boundaries()
        + check_system_boundaries()
        + check_frontier_boundaries()
        + check_application_boundaries()
        + check_career_boundaries()
        + check_paper_boundaries()
        + check_stream_token_accounting()
        + check_tokenization_examples()
        + check_transformer_examples()
        + check_gpt_cross_framework_parity_control()
        + check_gpt_cross_framework_training_parity_control()
        + check_jax_training_resume_control()
        + check_model_config_examples()
        + check_model_release_evidence()
        + check_transformers_checkpoint_control()
        + check_target_weight_quantization_control()
        + check_structured_evaluation_metrics()
        + check_citation_evidence_span_metric()
        + check_target_qwen_evaluation_control()
        + check_target_activation_patching_control()
        + check_target_service_control()
        + check_incremental_streaming_control()
        + check_transformers_thread_cancellation_control()
        + check_rag_transformers_control()
        + check_rag_publication_policy()
        + check_rag_guarded_transformers_control()
        + check_generation_protocol_examples()
        + check_transformers_generation_runtime_control()
        + check_rag_framework_parity_control()
        + check_rag_service_asgi_control()
        + check_recorded_model_planner_control()
        + check_mcp_sdk_memory_control()
        + check_mcp_sdk_stdio_control()
        + check_mcp_sdk_streamable_http_control()
        + check_mcp_stdio_control()
        + check_mcp_streamable_http_control()
        + check_a2a_loopback_control()
        + check_moe_routing_examples()
        + check_distributed_moe_capacity_control()
        + check_moe_all_to_all_control()
        + check_moe_all_to_all_training_control()
        + check_moe_all_to_all_capacity_training_control()
        + check_kv_example()
        + check_kv_allocator_examples()
        + check_scaling_examples()
        + check_preference_examples()
        + check_roofline_example()
        + check_multimodal_examples()
        + check_llmops_examples()
        + check_agent_safety_examples()
        + check_agent_outbox_examples()
        + check_code_metric_examples()
        + check_retrieval_metric_examples()
        + check_conversation_memory_examples()
        + check_gpt_model_page()
        + check_llama_model_page()
        + check_qwen_model_page()
        + check_deepseek_model_page()
        + check_claude_model_page()
        + check_gemini_model_page()
        + check_cloud_api_contracts_model_page()
        + check_openai_responses_replay()
        + check_evaluation_gate_project_page()
        + check_inference_serving_project_page()
        + check_rag_foundations_project_page()
        + check_rag_framework_adapters_project_page()
        + check_rag_framework_adapters_project_readme()
        + check_cloud_api_contracts_project_page()
        + check_safe_agent_project_page()
        + check_synthetic_data_audit_project_page()
        + check_single_gpu_finetuning_project_page()
        + check_jax_minigpt_project_page()
        + check_jax_minigpt_project_readme()
        + check_transformers_basics_project_page()
        + check_calibration_examples()
        + check_paired_randomization_examples()
        + check_clustered_randomization_examples()
        + check_clustered_bootstrap_examples()
        + check_holm_correction_examples()
        + check_sequential_peeking_examples()
        + check_gradient_accumulation_examples()
        + check_amp_grad_scaler_control()
        + check_ddp_amp_overflow_consensus_control()
        + check_dataloader_prefetch_resume_control()
        + check_optimizer_commit_resume_control()
        + check_training_resume_process_control()
        + check_synthetic_data_examples()
        + check_sft_data_examples()
        + check_continual_learning_examples()
        + check_quantization_examples()
        + check_minigpt_training_checkpoint_examples()
        + check_peft_export_examples()
        + check_target_sft_label_control()
        + check_target_lora_control()
        + check_target_dpo_control()
        + check_reward_model_examples()
        + check_ppo_examples()
        + check_torch_ppo_smoke_scope()
        + check_transformer_ppo_smoke_scope()
        + check_text_ppo_smoke_scope()
        + check_learned_rm_ppo_smoke_scope()
        + check_transformer_reward_model_scope()
        + check_reward_model_training_entry_scope()
        + check_sampling_examples()
        + check_beam_search_examples()
        + check_constrained_decoding_examples()
        + check_stop_matching_examples()
        + check_speculative_decoding_examples()
        + check_self_consistency_examples()
        + check_verifier_selection_examples()
        + check_inference_workload_examples()
        + check_usage_budget_examples()
    )
    if errors:
        print("Content accuracy checks failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(
        f"OK: checked {len(files)} text files, {len(MODEL_BOUNDARIES)} model boundaries, "
        f"{len(FOUNDATION_BOUNDARIES)} foundation boundaries, "
        f"{len(GENERATION_BOUNDARIES)} generation boundaries, "
        f"{len(CORE_BOUNDARIES)} core boundaries, "
        f"{len(TRAINING_BOUNDARIES)} training boundaries, "
        f"{len(QUALITY_BOUNDARIES)} quality boundaries, "
        f"{len(SYSTEM_BOUNDARIES)} system boundaries, "
        f"{len(FRONTIER_BOUNDARIES)} frontier boundaries, "
        f"{len(APPLICATION_BOUNDARIES)} application boundaries, "
        f"{len(CAREER_BOUNDARIES)} career boundaries, "
        f"{len(PAPER_BOUNDARIES)} paper boundaries, "
        f"{len(OFFICIAL_URLS)} official sources, strict stream token accounting, "
        "byte-BPE identity/boundary math, RMSNorm/RoPE/GQA/cache and "
        "blockwise online-softmax math, PyTorch/JAX MiniGPT "
        "forward/backward/SGD plus stochastic AdamW trajectory parity "
        "and cross-process strict resume, "
        "strict decoder-config identity/standard-KV/MLA-refusal contracts, "
        "immutable Llama/Qwen/DeepSeek release-byte/semantic/projection contracts, "
        "immutable Qwen real-weight CPU forward/cache/generate control, "
        "target-Qwen selected-weight packed-INT4 artifact/dequantized-forward control, "
        "strict JSON-schema-v2/canonical-value evaluation metric control, "
        "strict citation source/evidence-span identity metric control, "
        "target-Qwen seven-case authored real-weight behavior-evaluation control, "
        "target-Qwen real-weight activation-patching/structural control, "
        "target-Qwen Transformers real-loopback HTTP service/report control, "
        "authored incremental-SSE real-loopback disconnect-cancellation control, "
        "tiny-Transformers cooperative generation-thread cancellation control, "
        "real-weight Qwen RAG retrieval/packing/greedy citation-abstention failure control, "
        "counterfactual fail-closed RAG publication-policy replay control, "
        "guarded real-weight Qwen RAG framework-generate invocation control, "
        "generation-protocol three-way special-token/bounds contracts, "
        "Transformers forced-token EOS/override/length runtime control, "
        "LangChain/LlamaIndex ACL-bound retriever/prompt/artifact parity control, "
        "persistent extractive RAG ASGI auth/ACL/timeout control, "
        "recorded strict-JSON model planner identity/runtime/verifier control, "
        "MCP 2025-11-25 official-SDK memory, real-stdio, and real-Streamable-HTTP "
        "schema/dispatch controls plus authored strict stdio and Streamable HTTP "
        "lifecycle/error controls, "
        "A2A 1.0 official-SDK JSON-RPC loopback/schema/error control, "
        "MoE top-k/capacity/drop/dispatch, trainable-router/MLP gradient, "
        "two-process Gloo collective-capacity, token-to-owner all-to-all, and "
        "all-to-all forward/backward + SGD plus capacity-aware kept-only "
        "training controls, "
        "KV formula/block-sharing/COW/fragmentation and "
        "prefix-cache identity/collision/lease math, "
        "scaling math, "
        "preference math/data/readiness/judgment identity, "
        "roofline math, multimodal math, "
        "artifact identity, Agent policy/approval/checkpoint/typed-loop/outbox gate, "
        "pass@k and retrieval/rerank/extractive-answer/target-tokenizer packing/"
        "trace binding math, "
        "typed conversation state, "
        "GPT research/Responses-object/typed-event page and fixed authored replay, "
        "Llama architecture/release-evidence/runtime-scope model page, "
        "Qwen config/weight/runtime/RAG/training-evidence model page, "
        "DeepSeek config/MLA/MoE/FP8/R1-scope model page, "
        "Claude Messages/block/stream/tool/budget-scope model page, "
        "Gemini Interactions/generateContent/multimodal/scope model page, "
        "cloud-API canonical/provider/transport/retry/stream/budget model page, "
        "evaluation-gate, inference-serving, RAG, RAG-framework-adapter, "
        "cloud-API, safe-Agent, "
        "synthetic-data, single-GPU-finetuning, JAX-MiniGPT, and "
        "Transformers-Basics project-page "
        "workflow/scope contracts, "
        "calibration/paired/cluster-bootstrap/cluster-randomization/Holm-FWER/"
        "sequential-peeking/"
        "masked-token-gradient-accumulation/CPU-AMP-GradScaler-overflow-state/"
        "DDP-AMP-overflow-consensus/"
        "DataLoader-prefetch-cursor-worker-RNG-resume/"
        "optimizer-commit-crash-replay/"
        "cross-process-scheduler-scaler-RNG-data-training-resume/"
        "evaluation-run-manifest/comparison-artifact/comparison-HTML-report/"
        "full-evidence-recomputation/"
        "authenticated-release-ledger/"
        "PPO-GAE math, "
        "tiny-torch-PPO scope/assertions, "
        "tiny-Transformer-PPO scope/assertions, "
        "local-text-PPO boundary/objective/scope assertions, "
        "learned-RM-PPO proxy-exploitation/scope assertions, "
        "synthetic-data audit math, strict SFT exact/binding/lexical/governance identity, "
        "continual-learning ACC/BWT/FWT/forgetting math, "
        "weight/KV quantization/bit-packing/tensor/bundle/MiniGPT-checkpoint/"
        "storage/error math, exact MiniGPT training-resume, PEFT adapter-export, and "
        "target-Qwen SFT final-label/collator/forward, LoRA backward/export/reload, "
        "and DPO state/scope assertions, "
        "linear Bradley-Terry RM optimization/shortcut/scope math, "
        "tiny Transformer RM text/optimizer/scope assertions, "
        "target RM readiness/tokenizer/trainer-boundary assertions, "
        "exact next-token processor/top-k/top-p/CDF, deterministic beam pruning/"
        "EOS/length ranking, full-token constrained mask/renormalization, "
        "incremental UTF-8 stop, and "
        "speculative rejection-sampling/TV/block math, exact binary "
        "self-consistency independence/correlation math, exact verifier-guided "
        "best-of-N oracle/selection/proxy math, "
        "and inference finite-arrival/continuous-batching/KV-preemption/strict-artifact/"
        "offered-dispatch/SLO math, plus cloud usage reservation/reconciliation math"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

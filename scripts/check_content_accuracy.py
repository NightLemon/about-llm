"""Check durable fact boundaries and executable numeric claims in the textbook."""

from __future__ import annotations

import hashlib
import math
import runpy
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ACCURACY_PAGE = ROOT / "docs" / "reference" / "accuracy.md"
TEXT_SUFFIXES = {".md", ".py", ".yml", ".yaml", ".toml", ".json", ".jsonl"}
IGNORED_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "site"}

OFFICIAL_URLS = {
    "https://developers.openai.com/api/docs/guides/structured-outputs",
    "https://platform.claude.com/docs/en/api/messages",
    "https://ai.google.dev/gemini-api/docs/interactions",
    "https://ai.google.dev/api/generate-content",
    "https://ai.google.dev/gemini-api/docs/text-generation",
    "https://modelcontextprotocol.io/docs/getting-started/intro",
    "https://modelcontextprotocol.io/specification/",
    "https://a2a-protocol.org/latest/specification/",
    "https://huggingface.co/docs/trl/en/sft_trainer",
    "https://docs.vllm.ai/en/stable/cli/",
    "https://huggingface.co/docs/transformers/en/chat_templating",
}

MODEL_BOUNDARIES = {
    "gpt.md": ("未披露", "时间敏感"),
    "llama.md": (
        "以所选 checkpoint",
        "config",
        "authored_standard_gqa",
        "不是任何 Llama checkpoint",
    ),
    "qwen.md": (
        "不能用一个架构",
        "检查 checkpoint",
        "authored_moe_gqa",
        "不对应任何 Qwen",
        "三方 special-token IDs",
    ),
    "deepseek.md": (
        "具体 checkpoint",
        "不能",
        "estimate_refused: true",
        "不是 DeepSeek-V2/V3/R1 配置快照",
        "三类 JSON 不能各看各的",
    ),
    "claude.md": ("保持未知", "不要"),
    "gemini.md": ("2026-08-06", "Interactions API", "generateContent"),
    "cloud-api-contracts.md": (
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
        "当前没有声称 NumPy、PyTorch 和 JAX 三套完整模型逐层或逐梯度等价",
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
        "当前仓库还没有对目标大模型执行",
    ),
}

TRAINING_BOUNDARIES = {
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
        "成功请求的 latency percentile 是条件统计",
        "429 可以是正确的过载保护",
        "不能把 429 从 availability 分母删除",
        "concurrency semaphore 也是队列",
        "Client queue 不等于服务端 queue",
        "快速 429",
        "scheduled timestamp 写成 `offered_at`",
        "不证明发生器实际按时执行",
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
        "没有目标长上下文 checkpoint",
        "active parameters 只包含当前 token 使用部分",
        "C=\\left\\lceil\\phi\\frac{Nk}{E}\\right\\rceil",
        "dropped assignment",
        "all-assignments-dropped token",
        "广义诊断",
        "不是所有论文/框架的 training loss",
        "不是 DeepSeek/Qwen 复现",
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
        "没有实现 MCP/A2A client/server",
        "核对日期 2026-08-11",
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
    ),
}


def text_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def check_encoding(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        if "\ufffd" in path.read_text(encoding="utf-8"):
            errors.append(f"{path.relative_to(ROOT)}: contains Unicode replacement character")
    return errors


def check_ledger() -> list[str]:
    if not ACCURACY_PAGE.exists():
        return ["docs/reference/accuracy.md: missing accuracy ledger"]
    text = ACCURACY_PAGE.read_text(encoding="utf-8")
    errors = [
        f"accuracy ledger missing official URL: {url}"
        for url in OFFICIAL_URLS
        if url not in text
    ]
    if "2026-08-06" not in text:
        errors.append("accuracy ledger missing checked_at date 2026-08-06")
    return errors


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


def check_moe_routing_examples() -> list[str]:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import numpy as np

    from about_llm.from_scratch import (
        route_topk_capacity,
        routed_linear_expert_forward,
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
        render=lambda messages: {
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
            render=lambda messages: {
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
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from about_llm.integrations.cloud_api import (
        ChatMessage,
        build_openai_compatible_request,
    )
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
        + check_stream_token_accounting()
        + check_tokenization_examples()
        + check_transformer_examples()
        + check_model_config_examples()
        + check_generation_protocol_examples()
        + check_transformers_generation_runtime_control()
        + check_rag_framework_parity_control()
        + check_rag_service_asgi_control()
        + check_recorded_model_planner_control()
        + check_moe_routing_examples()
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
        + check_calibration_examples()
        + check_paired_randomization_examples()
        + check_clustered_randomization_examples()
        + check_clustered_bootstrap_examples()
        + check_holm_correction_examples()
        + check_synthetic_data_examples()
        + check_sft_data_examples()
        + check_continual_learning_examples()
        + check_quantization_examples()
        + check_minigpt_training_checkpoint_examples()
        + check_peft_export_examples()
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
        f"{len(OFFICIAL_URLS)} official sources, strict stream token accounting, "
        "byte-BPE identity/boundary math, RMSNorm/RoPE/GQA/cache math, "
        "strict decoder-config identity/standard-KV/MLA-refusal contracts, "
        "generation-protocol three-way special-token/bounds contracts, "
        "Transformers forced-token EOS/override/length runtime control, "
        "LangChain/LlamaIndex ACL-bound retriever/prompt/artifact parity control, "
        "persistent extractive RAG ASGI auth/ACL/timeout control, "
        "recorded strict-JSON model planner identity/runtime/verifier control, "
        "MoE top-k/capacity/drop/dispatch diagnostics, "
        "KV formula/block-sharing/COW/fragmentation and "
        "prefix-cache identity/collision/lease math, "
        "scaling math, "
        "preference math/data/readiness/judgment identity, "
        "roofline math, multimodal math, "
        "artifact identity, Agent policy/approval/checkpoint/typed-loop/outbox gate, "
        "pass@k and retrieval/rerank/extractive-answer/target-tokenizer packing/"
        "trace binding math, "
        "typed conversation state, "
        "calibration/paired/cluster-bootstrap/cluster-randomization/Holm-FWER/"
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
        "storage/error math, exact MiniGPT training-resume and PEFT adapter-export "
        "state/scope assertions, "
        "linear Bradley-Terry RM optimization/shortcut/scope math, "
        "tiny Transformer RM text/optimizer/scope assertions, "
        "target RM readiness/tokenizer/trainer-boundary assertions, "
        "exact next-token processor/top-k/top-p/CDF, deterministic beam pruning/"
        "EOS/length ranking, full-token constrained mask/renormalization, "
        "incremental UTF-8 stop, and "
        "speculative rejection-sampling/TV/block math, "
        "and inference finite-arrival/continuous-batching/KV-preemption/strict-artifact/"
        "offered-dispatch/SLO math, plus cloud usage reservation/reconciliation math"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

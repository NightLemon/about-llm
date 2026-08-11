# 合成数据离线审计

这个项目验证 synthetic candidate 的显式 lineage、required verifier gate、generator/verifier revision overlap、exact identity、generation round 和 token mixture 暴露。它不调用模型或付费 API。

## 运行

~~~powershell
python -m about_llm.synthetic_data_cli `
  --records projects/synthetic-data-audit/records.example.jsonl `
  --required-verifier schema `
  --required-verifier grounding `
  --known-parent-id real-anchor-001 `
  --mixture projects/synthetic-data-audit/mixture.example.json `
  --output artifacts/synthetic-data/audit.json
~~~

安装包后也可使用 `about-llm-synthetic-audit`。

Fixture 有意包含：

- 两条 byte-exact duplicate；
- 一条 generator 与 grounding verifier revision 相同；
- 一条缺 required verifier；
- 一条 verifier failed；
- 第一代与第二代 parent chain；
- 25% target synthetic mixture，其 unique synthetic token 预期被消费 5 倍。

预期报告 `candidate_count=4`、`eligible_count=2`、`eligible_unique_content_count=1`。Eligibility 只表示 `schema` 与 `grounding` 两个声明的 gate 存在且通过；不证明语义正确、多样、安全、许可有效或不存在 model collapse。

## Record schema

每行 JSONL：

```json
{
  "record_id": "syn-001",
  "content": "...",
  "parent_ids": ["real-anchor-001"],
  "generator_revision": "teacher@v1",
  "prompt_revision": "expand@v3",
  "generation_round": 1,
  "verifications": [
    {"verifier_id": "schema", "revision": "schema-rules@v2", "passed": true}
  ],
  "human_reviewed": false
}
```

真实项目还需记录 raw response、sampling、时间、许可/consent、review rubric、split、数据 manifest 和 verifier 输入/输出 artifact；这个最小 CLI schema 不替代完整数据治理。

## Fingerprint profile

默认 `byte_exact`，任何 Unicode 或空白变化都会产生新 digest。可显式传 `--fingerprint-profile nfc_whitespace` 合并 NFC/空白 prose 变体，但不要对代码、Markdown 表格、Python 缩进或格式任务静默使用。

Exact duplicate 报告不会自动删除或选择 canonical item，因为保留哪条还取决于许可、parent、质量和时间。`eligible_count` 与 `eligible_unique_content_count` 分开报告，避免重复候选伪造 gate 规模。

## Mixture schema

`mixture.example.json` 指定 total consumed tokens、每个 component 的 unique tokens 与相对 weight。报告归一化 fraction、预期 consumed tokens 和 expected repetition factor。

这些是目标 sampler 的期望，不是训练日志。Packing、动态过滤、worker skew、读取失败和 curriculum 会让实际消耗偏离；生产训练必须用 observed-token ledger 对账。

## 证据等级

当前是 L2 CPU/offline artifact audit：

- 证明 JSON 契约、解析公式和错误边界；
- 不证明真实 teacher/student 质量；
- 不包含人工 verifier calibration；
- 不运行训练或多代反馈；
- 不验证许可、隐私或近似语义重复。

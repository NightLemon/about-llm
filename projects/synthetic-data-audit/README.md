# 合成数据离线审计

这个项目把 synthetic candidate 进入训练前最容易混淆的五件事拆开：

1. lineage 是否可解析；
2. required verifier 是否存在并通过；
3. generator/verifier revision 是否重叠；
4. exact content identity 与 eligible unique 分母；
5. target mixture expectation 与实际训练 exposure。

它不调用 teacher/student、judge API 或训练程序。当前交付是 `about-llm.synthetic-data-audit.v2` CPU 离线审计器：
解析时拒绝重复字段、非法数值和未知字段，随后执行确定性审计、绑定输入 bytes 与外部 policy、生成
self-fingerprint，并可从 caller-supplied 输入完整复算。

## 最短运行

~~~powershell
python -m about_llm.synthetic_data_cli `
  --records projects/synthetic-data-audit/records.example.jsonl `
  --required-verifier schema `
  --required-verifier grounding `
  --known-parent-id real-anchor-001 `
  --mixture projects/synthetic-data-audit/mixture.example.json `
  --output artifacts/synthetic-data/audit.json
~~~

`--output` 默认 exclusive-create，防止无意覆盖旧证据。确需替换时显式传 `--overwrite`。写入路径会执行 file `fsync`；这仍不等于 directory entry durable、断电原子发布或可信存储。

安装包后也可使用：

~~~powershell
about-llm-synthetic-audit --help
~~~

## 完整本地复算

不要只读报告里的 `report_fingerprint`。用同一份 caller-supplied records、mixture 和 policy 重建全部字段：

~~~powershell
python -m about_llm.synthetic_data_cli `
  --records projects/synthetic-data-audit/records.example.jsonl `
  --required-verifier schema `
  --required-verifier grounding `
  --known-parent-id real-anchor-001 `
  --mixture projects/synthetic-data-audit/mixture.example.json `
  --verify-report projects/synthetic-data-audit/audit.example.json
~~~

成功输出：

~~~json
{
  "report_fingerprint": "sha256:202d8db97b704c5542e8516c5bd0c945da1c1022100f6ecbfb828f2d2bb6f4cd",
  "schema_version": "about-llm.synthetic-data-audit.v2",
  "verification_scope": "full_local_recomputation",
  "verified": true
}
~~~

Verifier 不只是“检查 self-hash”，而是：

- 按同样规则解析现有 report；
- 重新读取 caller 指定的 records/mixture；
- 用 caller 指定的 required verifiers、known parents 和 fingerprint profile 重跑审计；
- 重算输入 size/SHA-256、audit、mixture、scope 与 report fingerprint；
- 对 canonical JSON 做完整相等比较。

所以攻击者即使同时篡改 `eligible_count` 并重算无密钥 self-hash，仍无法通过原输入与 policy 的 full-local-recomputation。反过来，如果 caller 也把输入/policy 换成攻击者版本，unkeyed hash 不能提供来源认证；可信 policy head、签名或受控发布渠道仍在项目范围外。

## 本例使用的固定输入

`records.example.jsonl` 有四条 candidates：

| ID | Round | Parent | Required verifier | 其他 finding |
|---|---:|---|---|---|
| `syn-001` | 1 | external real anchor | schema pass、grounding pass | human reviewed |
| `syn-002` | 1 | external real anchor | schema pass、grounding pass | 与 `syn-001` byte-exact duplicate；grounding revision 与 generator 相同 |
| `syn-003` | 1 | external real anchor | schema pass、grounding missing | 不 eligible |
| `syn-004` | 2 | `syn-001` | schema pass、grounding fail | 不 eligible |

`mixture.example.json` 声明 2,000,000 total consumed tokens：

- real anchor：800,000 unique tokens，weight 3；
- synthetic round 1：100,000 unique tokens，weight 1。

固定报告 `audit.example.json` 绑定：

| Artifact | Size | SHA-256 |
|---|---:|---|
| records JSONL | 1,457 bytes | `7b1fe328…8d8530` |
| mixture JSON | 341 bytes | `4bef57e8…669bc0` |
| report | schema v2 | `202d8db9…bb6f4cd` |

固定审计结果：

- `candidate_count=4`；
- `eligible_count=2`，eligibility rate 0.5；
- `eligible_unique_content_count=1`；
- self/revision-overlap record：`syn-002`；
- missing required verifier：`syn-003`；
- failed required verifier：`syn-004`；
- round 1 为 3 candidates / 2 eligible；
- round 2 为 1 candidate / 0 eligible；
- unresolved parent、nonmonotonic parent 和 lineage cycle 都为空。

## Record contract

每个 JSONL object 必须恰好使用以下字段：

~~~json
{
  "record_id": "syn-001",
  "content": "候选文本",
  "parent_ids": ["real-anchor-001"],
  "generator_revision": "teacher@v1",
  "prompt_revision": "expand@v3",
  "generation_round": 1,
  "verifications": [
    {
      "verifier_id": "schema",
      "revision": "schema-rules@v2",
      "passed": true
    }
  ],
  "human_reviewed": false
}
~~~

`human_reviewed` 可省略，默认 false；其他字段必需。遇到下列输入时，Loader 会在审计开始前停止：

- 拒绝 duplicate JSON keys、`NaN`/`Infinity`、invalid UTF-8；
- 拒绝 missing/unknown record、verification、mixture/component fields；
- `passed` 与 `human_reviewed` 必须是 JSON boolean，整数 0/1 不可冒充；
- `generation_round` 与 token budget 不接受 boolean；
- record ID、parent ID、revision、content 必须非空；
- parent IDs、verifier IDs、record IDs、known parent IDs 必须各自唯一；
- record 不可把自己列为 parent；
- input 有显式 byte cap。

严格 schema 只解决解析歧义，不证明字段值真实。

## Lineage graph

内部 parent 是同一 input 中的 `record_id`；external parent 只有在 caller 通过 `--known-parent-id` 明确声明时才算 resolved。未知 parent 保存在 `unresolved_parent_pairs`，不会被静默当作真实锚点。

对内部边 \(p\rightarrow c\)，期望：

\[
round(c)>round(p).
\]

违反项进入 `nonmonotonic_parent_pairs`。DFS 还单独报告真正处于环中的 `lineage_cycle_record_ids`，不会把仅指向环的后代误标成环成员。

这两类 finding 与 unresolved lineage 当前都不自动改变 verifier eligibility。原因是报告需要分开呈现“声明的质量 gate”与“来源治理 gate”，避免一个布尔值吞掉失败原因。生产 publication policy 应显式决定：

- unresolved/cycle/nonmonotonic 是否一票否决；
- external parent registry 如何认证；
- 删除 parent 时如何传播到派生数据、shard 与 checkpoint；
- 多 source parent 的 round 和许可如何合并。

## Required verifier gate

令 required verifier 集合为 \(V\)，record \(x\) 的结果为 \(r_v(x)\)。当前 eligibility 定义为：

\[
eligible(x)=
\bigwedge_{v\in V}
\left[present(v,x)\land passed(v,x)\right].
\]

“missing”和“present but failed”分别进入不同列表。额外 verifier 不参与该 policy，但仍保留在 record 中。

若任一 verifier 的 exact `revision` 等于 `generator_revision`，record 会进入 `self_verified_record_ids`。这个字段准确含义只是 **revision string overlap**：

- 它不能证明同一进程或同一权重真的生成并验证；
- 没有 overlap 也不能证明 judge 独立；
- 不同 revision 仍可能同模型族、同训练数据或同偏差；
- verifier pass 不等于 calibration、事实正确或不可被 reward hacking。

`human_reviewed_count` 也只是 authored boolean 计数，不认证 reviewer、rubric、盲评或一致性。

## Exact identity 与双分母

默认 `byte_exact`：

\[
id(x)=SHA256(UTF8(content_x)).
\]

任何空白或 Unicode 表示差异都会产生新 identity。可显式选择 `nfc_whitespace`：

1. Unicode NFC；
2. 按 Unicode whitespace split；
3. 用单个 ASCII space join；
4. 对 UTF-8 bytes 求 SHA-256。

~~~powershell
python -m about_llm.synthetic_data_cli `
  --records projects/synthetic-data-audit/records.example.jsonl `
  --required-verifier schema `
  --required-verifier grounding `
  --known-parent-id real-anchor-001 `
  --fingerprint-profile nfc_whitespace
~~~

该 profile 只适用于明确允许归一化的 prose。代码缩进、Markdown 表格、YAML、格式遵循任务可能因 whitespace folding 改变语义。

Duplicate finding 不自动删除记录，也不从 `eligible_count` 隐藏：

- candidate 分母回答“多少条进入 audit”；
- eligible 分母回答“多少条通过声明 gate”；
- eligible unique 回答“通过项中有多少 exact identity”。

保留哪条 duplicate 还取决于 parent、许可、时间、review、split 和 canonical-selection policy。本项目没有 embedding/MinHash/语义 near-duplicate detector，也不证明 train/held-out 无 paraphrase 污染。

## Mixture 与重复暴露

对 component \(i\)，relative weight \(w_i\) 先归一化：

\[
p_i=\frac{w_i}{\sum_jw_j}.
\]

给定 total consumed-token budget \(D\) 与 unique tokens \(n_i\)：

\[
E[C_i]=Dp_i,\qquad
E[repeat_i]=\frac{Dp_i}{n_i}.
\]

固定 fixture 得到：

| Component | Fraction | Expected consumed | Unique | Expected repetition |
|---|---:|---:|---:|---:|
| real | 0.75 | 1,500,000 | 800,000 | 1.875 |
| synthetic r1 | 0.25 | 500,000 | 100,000 | 5.0 |

Weight 3:1 是相对权重，不可直接把 `weight*D` 当 token 数。Synthetic component 必须声明正 generation round，real component 反而禁止声明 generation round。

这些都是 target sampler expectation。Packing、动态过滤、worker skew、读取失败、curriculum、sample replacement、checkpoint resume 与 early stop 会改变实际 exposure。生产必须另建 observed-token ledger，按 component/round/source/split 记录 committed tokens，并与 target expectation 分账。

## Artifact threat model

V2 artifact 绑定：

- exact records/mixture byte size 与 SHA-256；
- required verifier、known parent 和 fingerprint profile；
- 三个 finding 不影响 verifier eligibility 的 policy flags；
- 全部 audit/mixture 数值；
- 机器 scope 与 evidence boundary；
- canonical report fingerprint。

它防止 accidental drift，并在 caller 保持可信输入/policy 时拒绝协同重哈希的结果篡改。它没有：

- MAC/signature、publisher identity 或可信 timestamp；
- directory `fsync`、atomic rename 或 crash injection；
- verify 后交给另一个 consumer 的 verify-use TOCTOU 防护；
- 远程 object-store generation/lease；
- secret/PII/license scan；
- raw generator/verifier response 与计费；
- teacher/student execution、training 或质量评测。

## 测试与故意失败

~~~powershell
python -m pytest `
  tests/test_synthetic_data.py `
  tests/test_synthetic_data_cli.py -q
~~~

当前 40 个测试覆盖：

- eligible / missing / failed 分账；
- generator-verifier exact revision overlap；
- external/internal/unresolved parent；
- cycle 与 nonmonotonic generation round；
- byte-exact 与 NFC/whitespace identity；
- duplicate 不从 eligibility 分母消失；
- normalized mixture 与 repetition；
- duplicate/non-finite/unknown JSON；
- untyped boolean/enum、重复 ID、非法 weight/budget；
- report round trip、input byte drift、cooperative rehash；
- output exclusive-create 与显式 overwrite。

重点反例：

~~~powershell
python -m pytest `
  tests/test_synthetic_data.py::test_audit_reports_cycles_and_nonmonotonic_parent_rounds `
  tests/test_synthetic_data_cli.py::test_report_verifier_rejects_cooperative_rehash `
  tests/test_synthetic_data_cli.py::test_report_verifier_binds_exact_input_bytes `
  tests/test_synthetic_data_cli.py::test_output_is_exclusive_create_unless_overwrite_is_explicit `
  -q
~~~

## 接入真实流水线

至少再补：

- raw teacher request/response、sampling、seed、时间、cost 与 provider/model revision；
- prompt/template/tool schema 的 immutable bytes；
- source license、consent、PII/secret 与 deletion policy；
- verifier input/output、infra failure、calibration slice 和人工 adjudication；
- semantic near-duplicate、cluster 与 group-aware split；
- real-only untouched holdout；
- target/observed sampler ledger 和 resume state；
- 每代 dataset/model/eval manifest、rollback 与 stop rule。

Eligibility 不等于高质量可用数据；exact unique 不等于语义多样性；target mixture 不等于 observed exposure；full local recomputation 不等于 provenance authentication。

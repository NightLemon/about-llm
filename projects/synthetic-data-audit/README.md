# 合成数据审计：四条候选为什么只剩一条新内容

这个项目检查 synthetic data 在进入训练前发生了什么。固定样例有 4 条候选，其中 2 条通过必要的 verifier，
但这两条正文完全相同。因此最重要的结果不是一个笼统的“通过率”，而是三个不同分母：

```text
4 candidates → 2 eligible → 1 eligible unique content
```

第一次学习时，先预测每条记录会落入哪个集合，再复算仓库已经录制的报告。完整的 lineage、去重、mixture 公式和
威胁边界见 [Synthetic Data Audit 教学页](../../docs/practice/projects/synthetic-data-audit.md)。

## 第一次运行

从仓库根目录复核固定报告：

```powershell
python -m about_llm.synthetic_data_cli `
  --records projects/synthetic-data-audit/records.example.jsonl `
  --required-verifier schema `
  --required-verifier grounding `
  --known-parent-id real-anchor-001 `
  --mixture projects/synthetic-data-audit/mixture.example.json `
  --verify-report projects/synthetic-data-audit/audit.example.json
```

成功时会输出：

```json
{
  "report_fingerprint": "sha256:202d8db97b704c5542e8516c5bd0c945da1c1022100f6ecbfb828f2d2bb6f4cd",
  "schema_version": "about-llm.synthetic-data-audit.v2",
  "verification_scope": "full_local_recomputation",
  "verified": true
}
```

`verified=true` 表示程序用命令行提供的 records、mixture 和 policy 重新执行了全部计算，并与录制报告逐字段一致。
它不是只拿报告中的 hash 再验同一份报告。

## 先预测四条记录

| Record | 必要验证 | 其他信息 | 预期结果 |
|---|---|---|---|
| `syn-001` | Schema、grounding 均通过 | 人工复核标记为 true | Eligible |
| `syn-002` | Schema、grounding 均通过 | 正文与 `syn-001` 相同；grounding revision 与 generator 相同 | Eligible，但不增加 unique content |
| `syn-003` | 缺少 grounding | 第一轮候选 | Missing verifier，不 eligible |
| `syn-004` | Grounding 明确失败 | Parent 是 `syn-001` | Failed verifier，不 eligible |

打开报告后，先检查这些字段：

```text
audit.candidate_count                 = 4
audit.eligible_count                  = 2
audit.eligible_unique_content_count   = 1
audit.self_verified_record_ids        = ["syn-002"]
audit.missing_verifier_record_ids     = ["syn-003"]
audit.failed_verifier_record_ids      = ["syn-004"]
```

`self_verified_record_ids` 只表示 generator revision 与某个 verifier revision 字符串相同。它提醒你进一步调查独立性，
并不能证明两个调用真的来自同一个进程或同一份权重。

## 生成自己的报告

```powershell
python -m about_llm.synthetic_data_cli `
  --records projects/synthetic-data-audit/records.example.jsonl `
  --required-verifier schema `
  --required-verifier grounding `
  --known-parent-id real-anchor-001 `
  --mixture projects/synthetic-data-audit/mixture.example.json `
  --output artifacts/synthetic-data/audit.json
```

`--output` 默认不会覆盖已有文件。目标存在时，先比较旧报告；确实要替换才显式传入 `--overwrite`。
安装仓库后也可以运行 `about-llm-synthetic-audit --help` 查看全部参数。

## 一份报告包含什么

| 顶层字段 | 你要从中确认什么 |
|---|---|
| `inputs` | Records 与 mixture 的精确字节数和 SHA-256 |
| `policy` | Required verifiers、known parents 与 fingerprint profile |
| `audit` | Eligible、missing、failed、lineage、duplicate 与 round 账本 |
| `mixture` | 目标采样比例、预计消费 token 和预计重复次数 |
| `scope` | 这次实际运行和没有运行的能力 |
| `evidence_boundary` | 报告可以支持到哪一步 |
| `report_fingerprint` | 除本字段外整个 canonical payload 的 SHA-256 |

输入的换行或 JSON 格式只要改变，bytes identity 就会变化。审计结果可能仍然相同，但旧报告应当验证失败，因为它绑定的
已经不是同一个输入 artifact。

## 为什么 2 eligible 只剩 1 unique

Eligibility 只看命令行指定的 required verifiers 是否存在并通过。`syn-001` 和 `syn-002` 都满足这个规则，
所以分母仍然是 2。

默认的 `byte_exact` profile 会直接对正文的 UTF-8 字节求 SHA-256。两条记录正文相同，因此只算一份独立内容。
审计器会报告重复组，同时保留两条原始记录。最终保留哪一条，还要结合来源、许可、时间、人工复核和数据切分策略。

对于明确允许空白归一化的普通文本，可以改用：

```powershell
python -m about_llm.synthetic_data_cli `
  --records projects/synthetic-data-audit/records.example.jsonl `
  --required-verifier schema `
  --required-verifier grounding `
  --known-parent-id real-anchor-001 `
  --fingerprint-profile nfc_whitespace
```

这个 profile 会做 Unicode NFC 和空白折叠。代码、YAML、Markdown 表格或格式遵循任务中的空白可能承载语义，
所以不适合使用这套归一化。

两种 profile 都只能发现内容相同的记录。语义近重复和数据泄漏仍需要单独检测。

## Lineage 为什么单独记账

Parent 可以来自当前输入中的另一条记录，也可以是命令行明确声明的外部锚点。未解析的 parent、generation round 倒退和
cycle 会分别写入报告。

这些 finding 当前不会修改 verifier eligibility。这样你能看见“质量验证通过”和“来源治理失败”是两个问题。
生产发布策略通常会把严重 lineage 问题作为单独的拒绝条件，而不是继续复用一个含义模糊的 `eligible` 布尔值。

## Mixture 的 3:1 到底表示什么

固定样例计划消费 2,000,000 tokens。Real 与 synthetic 的相对权重为 3:1，因此目标比例是 0.75 和 0.25：

| Component | Expected consumed | Unique tokens | Expected repetition |
|---|---:|---:|---:|
| Real anchor | 1,500,000 | 800,000 | 1.875 |
| Synthetic round 1 | 500,000 | 100,000 | 5.0 |

按照这份采样计划，每个合成数据 token 平均会被消费 5 次。这不是训练过程的实际读取次数。动态过滤、worker 偏差、
读取失败、checkpoint 恢复和提前停止都可能改变结果，真实训练需要另建 token 消费账本。

## 为什么完整复算比 self-hash 更强

攻击者可以修改 `eligible_count`，再为篡改后的报告计算一条新的无密钥 hash。只检查 self-hash 时，这份报告仍然自洽。

`--verify-report` 会重新读取调用方指定的输入和 policy，从头生成预期报告，所以这种协同重哈希无法匹配原始输入。
但如果调用方同时接受了攻击者替换的 records、policy 和报告，无密钥 SHA-256 仍然无法认证发布者；这需要签名、MAC
或受控发布渠道解决。

## 主要文件

| 文件 | 用途 |
|---|---|
| `records.example.jsonl` | 四条固定候选及其 lineage、generator 与 verifier 信息 |
| `mixture.example.json` | Real/synthetic 目标采样权重与 token 预算 |
| `audit.example.json` | 可以完整复算的 v2 录制报告 |
| [`synthetic_data.py`](../../src/about_llm/synthetic_data.py) | Lineage、verifier gate、identity 与 mixture 核心计算 |
| [`synthetic_data_cli.py`](../../src/about_llm/synthetic_data_cli.py) | 无歧义 JSON loader、artifact 与 CLI |
| [教学页](../../docs/practice/projects/synthetic-data-audit.md) | 逐步解释公式、失败实验与生产扩展 |

## 常见故障

| 现象 | 先检查什么 |
|---|---|
| 报告一开始就无法解析 | Duplicate key、NaN/Infinity、unknown field、UTF-8 与 boolean 类型 |
| `candidate_count` 不符合预期 | JSONL 是否多行、少行，record ID 是否重复 |
| Eligible 数量不对 | Required verifier 名称、missing 与 failed 列表 |
| Unique 数量不对 | Fingerprint profile、正文空白与 Unicode 表示 |
| Lineage 出现 unresolved | `--known-parent-id`、parent 拼写和输入中是否存在对应 record |
| Mixture 数字放大数倍 | 是否先归一化 relative weights，再乘总 token budget |
| `full local recomputation` 失败 | 先比输入 bytes，再比 policy、audit、mixture，最后看 report fingerprint |
| 输出文件已存在 | 比较旧报告；确认替换后再使用 `--overwrite` |

## 运行专项测试

```powershell
python -m pytest `
  tests/test_synthetic_data.py `
  tests/test_synthetic_data_cli.py -q

python scripts/check_docs.py
python scripts/check_content_accuracy.py
```

当前 40 个测试覆盖三个分母、版本重叠、lineage graph、两种内容指纹规则、mixture 公式和 JSON 解析，
也会故意制造输入漂移、协同重哈希和输出文件冲突。

## 这个项目还没有做什么

当前实现只在 CPU 上离线读取本仓库准备的输入。接入生产流水线时，还要补充：

- Teacher、student 和 verifier model 的原始请求与响应；
- 模型、Prompt、许可和隐私检查记录；
- Verifier calibration、语义近重复和保留集污染评测；
- 实际 token 消费，以及跨代数据集、模型和评测 manifest。

因此 `eligible` 只表示声明的 verifier gate 通过；它不等于事实正确、高质量、语义多样或适合直接发布。

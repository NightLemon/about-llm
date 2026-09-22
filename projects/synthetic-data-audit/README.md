# 合成数据审计运行手册

本目录生成并复核合成数据审计报告。[项目教学页](../../docs/practice/projects/synthetic-data-audit.md)负责 4→2→1 主线、
约束顺序和结论边界；本页只维护安装、CLI、输入输出与排错。

## 第一次运行

从仓库根目录执行。首次使用时，先按[环境准备](../../docs/guide/environment.md)创建并激活虚拟环境，再安装本项目：

```powershell
python -m pip install -c constraints/ci.txt -e .
```

然后复核固定报告：

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

先看 `verification_scope="full_local_recomputation"` 与 `verified=true`。这表示程序用命令行提供的 records、mixture
和 policy 重新执行全部计算，并与录制报告逐字段一致。它不是只拿报告中的 hash 再验同一份报告；这个验证模式只输出
摘要、不写文件，重复运行会重算同一组本地输入。

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

`human_reviewed=true` 是输入中的声明，计数只能说明这份 artifact 带有标记；复核者、规则和实际裁决仍要由人审系统的
记录证明。

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

这条命令会在 `artifacts/synthetic-data/audit.json` 新建一份报告。`--output` 默认不会覆盖已有文件；目标存在时，先比较旧
报告，确实已审阅并需要替换才显式传入 `--overwrite`。安装仓库后也可以运行
`about-llm-synthetic-audit --help` 查看全部参数。

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

## 可选的内容归一化

默认 `byte_exact` 直接按正文 UTF-8 bytes 区分内容。只有任务明确允许空白归一化时才改用：

```powershell
python -m about_llm.synthetic_data_cli `
  --records projects/synthetic-data-audit/records.example.jsonl `
  --required-verifier schema `
  --required-verifier grounding `
  --known-parent-id real-anchor-001 `
  --fingerprint-profile nfc_whitespace
```

这个 profile 会做 Unicode NFC 和空白折叠。代码、YAML、Markdown 表格等空白可能承载语义，不适用它。两种 profile
都只检查 exact identity；lineage、mixture、完整复算与 self-hash 的解释见项目教学页。

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

这一步供修改项目的维护者使用。需要先在同一虚拟环境中安装测试依赖：

```powershell
python -m pip install -c constraints/ci.txt -e ".[dev]"
```

```powershell
python -m pytest `
  tests/test_synthetic_data.py `
  tests/test_synthetic_data_cli.py -q

python scripts/check_docs.py
python scripts/check_content_accuracy.py
```

当前 40 个测试覆盖记录与不同内容的计数、版本重叠、lineage graph、两种内容指纹规则、mixture 公式和 JSON 解析，
也会故意制造输入漂移、协同重哈希和输出文件冲突。

## 这个项目还没有做什么

当前实现只在 CPU 上离线读取本仓库准备的输入。接入生产流水线时，还要补充：

- Teacher、student 和 verifier model 的原始请求与响应；
- 模型、Prompt、许可和隐私检查记录；
- Verifier calibration、语义近重复和保留集污染评测；
- 实际 token 消费，以及跨代数据集、模型和评测 manifest。

因此 `eligible` 只表示声明的 verifier gate 通过；它不等于事实正确、高质量、语义多样或适合直接发布。

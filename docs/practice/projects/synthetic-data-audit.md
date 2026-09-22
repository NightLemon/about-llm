# Synthetic Data Audit：把候选记录变成可复算报告

**项目导航**：[项目索引](../project-index.md) · [合成数据教材](../../training/synthetic-data.md) ·
[数据工程](../../training/data.md) ·
[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/synthetic-data-audit/README.md) ·
[项目证据](../../evidence/project-controls.md)
{ .doc-nav }

教材解释生成、筛选、去重与 mixture；本页只说明怎样把它们装成一份可复算审计报告。README 维护 CLI、schema 和测试，
证据页维护固定结果。

## 先看清 4 → 2 → 1

固定输入有 4 条候选：2 条满足必需 verifier，但两条正文完全相同，所以报告必须同时保留三种数量：

```text
4 candidates → 2 eligible records → 1 eligible unique content
```

运行前为每条记录预测 eligible、missing verifier、failed verifier 或 duplicate。这里的 `eligible` 只表示声明的门禁通过，
不等于事实正确、来源独立、高质量或适合训练。

## 第一次运行 {#run}

```powershell
python -m about_llm.synthetic_data_cli `
  --records projects/synthetic-data-audit/records.example.jsonl `
  --required-verifier schema `
  --required-verifier grounding `
  --known-parent-id real-anchor-001 `
  --mixture projects/synthetic-data-audit/mixture.example.json `
  --verify-report projects/synthetic-data-audit/audit.example.json
```

先检查 `verification_scope=full_local_recomputation`，再核对 candidate、eligible、unique、missing、failed 与 lineage 计数。
完整安装、生成报告和 `--overwrite` 规则见[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/synthetic-data-audit/README.md)。

## 报告的四层职责

| 层 | 需要绑定 | 典型失败 |
|---|---|---|
| 输入 identity | Records/mixture 的原始 bytes 与 hash | 换行或内容变化后旧报告仍被复用 |
| Policy | Required verifiers、known parents、fingerprint profile | 看见结果后改筛选规则 |
| Audit | Eligibility、lineage、duplicates 与 generation round | 用一个布尔值吞掉不同失败 |
| Mixture | 归一化权重、token budget 与预计重复次数 | 把计划比例当作真实消费记录 |

Self-hash 只能说明报告对自身内容一致。完整本地复算会重新读取外部输入与 policy，能发现一起改写报告内部数字的篡改；
它仍不能认证输入发布者，签名或受控分发属于更外层的供应链。

## 按顺序增加约束

1. **严格解析**：拒绝重复字段、非法数值、错误类型和未知字段。
2. **Lineage**：区分输入内 parent 与明确声明的外部 anchor，拒绝 cycle 和 generation 倒退。
3. **Verifier gate**：把 missing、failed 与 passed 分开；同 revision 只触发独立性调查，不冒充证明。
4. **Content identity**：先用 byte-exact；只有任务允许时才启用 NFC/空白归一化。
5. **Mixture**：从相对权重得到目标 token 数和预计重复次数，但不冒充 trainer 的实际读取账本。
6. **Full recomputation**：重开 records、mixture 与 policy，逐字段重建报告。

每次只新增一层并保留前一层结果。出现失败时先找最早改变的输入或规则，不从最终 fingerprint 猜原因。

## 必做反例

- 删除必需 verifier，确认记录进入 missing 而非 eligible。
- 让 verifier 明确失败，确认它与缺失分开计数。
- 复制通过记录的正文，确认 eligible count 不变而 unique count 不增加。
- 制造 unresolved parent、round 倒退与 lineage cycle。
- 修改输入 bytes，再为报告重新计算 self-hash，确认 full recomputation 仍拒绝。
- 让输出路径已存在，确认默认不覆盖。

## 完成标准

- 能解释 candidate、eligible record 和 eligible unique content 的三个分母。
- 能指出 verifier eligibility 与 lineage finding 为什么是独立判断。
- 能说明 byte-exact 与 NFC/空白归一化分别适用于什么内容。
- 能把 mixture 计划与训练实际 consumption ledger 分开。
- 能说明 full recomputation 比 self-hash 多检查了什么、仍不能认证什么。
- 报告保留失败和未知项，并明确未运行 teacher、student、verifier 或真实训练。

项目完成后再到[合成数据教材](../../training/synthetic-data.md)讨论蒸馏与反馈环；不要把固定 4→2→1 样例外推成
真实数据质量结论。

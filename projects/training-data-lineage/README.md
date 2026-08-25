# Training Data Lineage：从一条帖子追到 checkpoint

这个小项目回答一个很实际的问题：如果训练语料中的 `thread-8841` 被要求删除，我们怎样知道它出现在哪些数据集、
shard、训练任务和 checkpoint 里？

仓库准备了一张很小的关系图。帖子有 `r2` 和 `r3` 两个版本，另一个网站保存了 `r3` 主帖的镜像。当前数据集使用
`r3` 的主帖和一条回复，它们位于 `train-00031.bin` 的两个 token 区间。训练任务
`pretrain-cn-run-42` 读过这个 shard，并产生了两个 checkpoint。

## 先运行一次

在仓库根目录执行：

~~~powershell
python projects/training-data-lineage/thread_lineage.py verify
~~~

看到 `"verified": true` 表示：程序重新读取关系图，从头计算报告，并确认结果与仓库保存的报告逐字段一致。它没有访问
论坛、删除文件或修改模型。

如果想看完整追踪过程：

~~~powershell
python projects/training-data-lineage/thread_lineage.py trace
~~~

重点观察三个位置：

1. `trace.known_revisions` 同时列出 `r2` 和 `r3`，但二者共享同一个稳定来源 ID；
2. `trace.shard_spans` 把主帖和回复定位到 `[4096, 4192)` 与 `[4192, 4256)`；
3. `deletion_impact` 找到需要重建的 shard、受影响的训练任务和两个 checkpoint。

## 镜像为什么不能自动顶替原帖

`thread-8841-main-r3` 这个去重簇包含两个成员：论坛原帖和镜像。当前 canonical item（真正进入数据集的代表项）是论坛
版本。删除请求到来后，镜像仍在簇中，但程序只会给出 `review-surviving-members`，不会自动把镜像提升为新代表项。

原因是“内容相同”没有回答“镜像是否有独立授权、是否仍应保留”。去重关系可以帮助定位候选，不能替代来源审查。

## 删除报告真正说明了什么

这个例子的删除范围是 `all_revisions`，所以 `r2` 与 `r3` 的解析和规范化产物都会受影响。当前数据集只使用 `r3`，
因此需要重建的是 `pretrain-cn-2026-08-v3` 和 `train-00031.bin`。

报告故意保留两个 `false`：

- `stored_data_deleted_by_this_analysis=false`：程序只计算影响范围，没有执行删除；
- `trained_weight_effect_removed_by_shard_rebuild=false`：重建数据不会让已经训练出的权重自动忘记这条内容。

这两个边界能防止把“找到了受影响对象”误写成“删除已经完成”，也能防止把“重建 shard”误写成“模型已经完成
unlearning”。

## 文件说明

| 文件 | 用途 |
|---|---|
| `thread-8841.lineage.json` | 来源版本、转换产物、去重簇、数据集位置和训练消费关系 |
| `thread-8841.recorded-report.json` | 从关系图计算出的示例报告 |
| `thread_lineage.py` | 查看完整追踪结果或复算示例报告 |
| `src/about_llm/training_data_lineage.py` | 解析、关系检查和删除影响计算 |
| `tests/test_training_data_lineage.py` | token 区间、canonical 选择、删除边界和输入错误的测试 |

输入文件不接受重复字段、`NaN`、未知字段或重叠 token 区间。每个 normalized item 必须恰好属于一个去重簇；只有簇的
canonical item 可以进入数据集；训练任务只能引用已声明的 shard。这些检查保证当前关系图能被明确解释，但不能证明
来源记录真实、授权有效或删除符合某个地区的法律要求。

## 主动制造一个失败

复制输入文件，把第二个 span 的 `start_token` 从 `4192` 改成 `4191`，再运行：

~~~powershell
python projects/training-data-lineage/thread_lineage.py trace `
  --spec path/to/changed.lineage.json
~~~

程序会报 `shard token spans must not overlap`。这个反例说明 span 账本不是装饰性元数据：如果两个 item 对同一位置提出
所有权，删除影响和 token 计数都会变得含糊。

也可以把第一条 placement 的 `item_id` 改成镜像成员。程序会拒绝它，因为这个成员不是当前去重簇的 canonical item。

## 当前例子没有覆盖什么

这是 CPU 上的离线关系图练习。它没有抓取真实论坛、运行 parser 或 tokenizer，也没有打开真实 shard、训练模型、执行
删除或验证 unlearning。接入真实流水线时，还要把来源系统的认证记录、构建任务、对象存储版本、真实 token ledger 和
删除执行回执接到同一条关系链上。

完整原理见[训练数据工程与治理](../../docs/training/data.md)。

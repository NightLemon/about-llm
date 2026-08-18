# 贡献与写作规范

## 内容原则

- **准确**：事实与结论可追溯；区分论文结论、社区经验和作者判断。
- **教学友好**：第一次出现的缩写必须展开；先解释直觉，再给公式和边界条件。
- **可复现**：实验写清模型、数据、软件版本、随机种子、硬件和评价指标。
- **不过时伪装**：产品价格、模型榜单、法律政策等必须标注查询日期。
- **尊重版权与许可**：不复制受限材料；代码、图片和数据注明来源与许可证。

## 章节模板

新章节可从 `docs/_templates/topic.md` 复制，通常包含：

1. 页面顶部的学习契约：适合读者、先修、首次阅读路径、完成信号和卡住时入口
2. 学习目标、一句话结论和核心直觉
3. 机制、公式与最小例子
4. 工程选择、权衡和失败模式
5. 自测题与延伸阅读

`models/`、`foundations/`、`core/`、`training/`、`systems/`、`applications/`、`quality/` 和 `frontier/` 下的每篇教学正文必须且只能包含一个 `<!-- learning-contract -->`。`python scripts/check_docs.py` 会验证标记、五个字段、链接和锚点。

同一检查还执行可读性棘轮：新教学页默认不得出现超过 200 字符的 prose 行，不得超过 45 个小节或 600 行。现有超标页记录在 `docs/reference/readability-baseline.json`；指标只能下降，改善正文时必须同步收紧对应 baseline。`docs/evidence/` 允许保存高密度台账，但开头必须把第一次阅读者引回教学入口。

## 状态标记

- `✅`：已有可独立学习的正文
- `🟡`：有骨架或简述，仍需扩充
- `⬜`：待编写

“完成”不表示主题永远不再更新，只表示已达到当前版本的最低教学标准。

## 引用

优先引用论文原文、官方文档、标准和可信的一手工程资料。论文用“作者，标题，会议/期刊或 arXiv，年份，链接”的格式。对存在争议的结论至少展示两方证据。

## 贡献许可

提交贡献即表示你有权提供相应内容，并同意按仓库的双许可证规则发布：源码、配置、测试和 Notebook 代码单元采用 [MIT License](LICENSE-CODE)；正文、图表和 Notebook Markdown 单元采用 [CC BY 4.0](LICENSE-DOCS)。第三方内容必须保留来源、许可证和必要的归属信息；无法确认授权范围时不要提交。

## 提交前检查

测试不是覆盖率竞赛。修改教材 claim 或对应实现时，先按[教材测试与证据策略](docs/reference/testing.md)写清 oracle 来源、实际证明范围和不能外推的结论；新增测试必须选择证据性质，跨组件或高成本测试还要标记运行属性。

```powershell
mkdocs build --strict
python scripts/check_docs.py
python scripts/check_content_accuracy.py
python -m pytest tests/test_check_docs.py -q
python -m pytest -m "not extended and not gpu and not network"
```

`docs/`、`mkdocs.yml` 与 `overrides/` 是站点源码；`site/` 是 `mkdocs build` 的生成物，不直接编辑或提交。部署前 `python scripts/check_built_site.py` 只检查 sitemap 和静态资源结构，不锁定教材原句。

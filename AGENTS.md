# AI 贡献入口

先按改动类型读取对应规则，不要把整套仓库说明复制进提示词：

- 教材正文、写作风格与提交前命令：`CONTRIBUTING.md`
- 目录职责、学习路径与实现约定：`docs/guide/repo-map.md`
- claim、证据边界与来源状态：`docs/reference/accuracy.md`
- 测试分层和证据等级：`docs/reference/testing.md`
- 治理决策及其理由：`docs/decisions/`

文档最小验证：

```powershell
python scripts/check_docs.py
python scripts/check_content_accuracy.py
python -m pytest tests/test_source_registry.py tests/test_source_probe.py tests/test_source_badges.py -q
mkdocs build --strict
```

网络探测不属于 PR 的确定性门禁。只有定时任务运行 `scripts/probe_sources.py`；探测失败或内容变化时保留构建，并将页面降级为待复核状态。

# Notebooks

Notebook 用于观察现象，不承载唯一实现。核心逻辑位于 src/about_llm/ 并由 tests/ 覆盖。

执行约定：

- 从仓库根目录启动 Jupyter。
- 先安装 README 中对应依赖组。
- 每本必须能 Restart & Run All。
- 默认不访问网络、不下载模型、不使用付费 API。
- 输出只展示小型结果，不提交大型二进制或模型权重。

当前顺序：

1. 01_attention_three_ways.ipynb：NumPy、PyTorch、JAX 的缩放点积注意力与因果 mask。
2. 02_minigpt_forward.ipynb：字节 tokenizer、PyTorch/JAX 微型 GPT、loss 与因果性。
3. 03_rag_retrieval_and_evaluation.ipynb：BM25、dense、RRF、ACL 与检索指标。

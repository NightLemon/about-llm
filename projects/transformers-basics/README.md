# Transformers：离线机制与真实 checkpoint

## 离线 smoke test

smoke_tiny.py 不下载任何模型。它从 GPT2Config 建立微型模型，在固定 batch 上训练 12 步并执行 greedy generation，用于验证：

- config 到模型的映射；
- labels 触发 causal LM shift/loss；
- optimizer 与 train/eval 状态；
- generate 的输入输出 shape；
- 参数量和纯权重存储量。

~~~powershell
python -m pip install "transformers>=4.48,<5"
python projects/transformers-basics/smoke_tiny.py
~~~

纯参数存储不是实际训练或推理显存；后者还包含梯度、optimizer、激活、KV cache 和工作区。

## 检查真实模型

inspect_checkpoint.py 只下载 config/tokenizer，不加载权重。必须给 revision，避免默认分支变化而无法复现：

~~~powershell
python projects/transformers-basics/inspect_checkpoint.py Qwen/Qwen2.5-0.5B-Instruct --revision <commit-hash>
~~~

脚本拒绝猜测 chat template，也默认 trust_remote_code=False。若某 checkpoint 必须执行远程代码，先审查对应 revision，再在隔离环境显式开启。

## 加载权重时的检查顺序

1. 阅读 model card、许可和用途限制；
2. 固定 model revision 与 tokenizer revision；
3. 核对 chat template、BOS/EOS/PAD 和 generation config；
4. 选择 torch_dtype，不用字符串“auto”掩盖实际 dtype；
5. 单张 GPU 先用 device_map 明确 placement；
6. 打印参数量、权重存储和实际峰值显存；
7. 用小型质量集确认量化/模板没有破坏输出；
8. 再做 batch、上下文和并发扫描。

## 家族差异

AutoModel 统一接口不代表模型协议相同。Llama、Qwen 和 DeepSeek checkpoint 可能在 GQA/MoE、RoPE、special tokens、tool template 和 generation config 上不同。GPT、Claude、Gemini 等云 API 也不能假设与 Transformers messages 一一等价。

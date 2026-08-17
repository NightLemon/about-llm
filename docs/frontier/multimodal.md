# 多模态模型：表示、对齐、评测与系统边界

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：视觉、音频、文档和多模态评测工程师。
- **先修**：[Transformer](../core/transformer.md)、数据治理和基础评测。
- **首次阅读**：输入输出契约 → 编码器 → 融合架构 → 评测 → 发布门禁。
- **完成信号**：能证明目标模态被使用，并按模态、语言和失败类型切片。
- **卡住时**：回到[模型选型](../models/landscape.md)，先固定目标任务与输入契约。

</div>

多模态系统把文本、图像、文档、音频、视频或传感器信号放入一个可交互任务。它不是“把像素转成几个 token 交给 LLM”这么简单：各模态的采样率、空间/时间结构、噪声、权限和正确性标准不同。

## 1. 先写清输入输出契约

对每种模态记录：

- raw format、codec、色彩/采样率与最大大小；
- resize/crop/normalization、OCR/ASR 和 metadata；
- encoder、projector/tokenizer revision；
- 模态插入顺序、special tokens 与位置/坐标系统；
- 输出是文本、框、mask、时间段、音频还是图像；
- 权限、来源、同意、保留与删除；
- 失败时是拒绝、降级到 OCR/ASR，还是请求更清晰输入。

“支持图片”没有说明数量、分辨率、动态范围、文字大小、动画、多页文档或计费 token，因此不是完整能力声明。

## 2. 视觉编码

### 2.1 Patch token

对 \(H\times W\) 图像，以 \(P\times P\) non-overlapping patch 切分，最简单情况下 patch 数为

\[
N=\frac{H}{P}\frac{W}{P},
\]

要求尺寸可整除或先 pad/resize。每个 patch 展平并投影为 visual embedding，再加 position information。

分辨率翻倍会让 patch 数约增至四倍；若 visual tokens 直接进入 full self-attention，相关计算/缓存会明显增加。实际模型可能用 pooling、resampler、token merging、局部 attention 或固定查询压缩，因此不能只按原始 patch 数推最终上下文成本。

### 2.2 Resize、crop 与坐标

Stretch 会改变形状，center crop 会丢边缘，letterbox 会加入 padding。若模型输出原图 bounding box，必须保存预处理变换并做 inverse mapping：

```text
original pixels -> resize/crop/pad -> model coordinates -> inverse transform
```

Normalized `[0,1]`、像素坐标、inclusive integer box 与 continuous `(x_min,y_min,x_max,y_max)` 的 IoU 不同。评测前统一坐标约定。

### 2.3 Dynamic resolution 与 tiling

高分辨率文档常用 thumbnail + tiles：缩略图保全局布局，局部块保细小文字。风险包括 tile 边界切断对象、重叠区域重复计数、tile 顺序混乱，以及 visual token 暴涨。需要报告 tile policy、最大块数和实际输入 token/成本。

## 3. 视觉语言架构

### 3.1 Dual encoder

图像编码器 \(f_I\) 和文本编码器 \(f_T\) 分别输出向量，以相似度训练匹配：

\[
s(I,T)=
\frac{f_I(I)^\top f_T(T)}
{\|f_I(I)\|\|f_T(T)\|}.
\]

适合 image-text retrieval 和 zero-shot classification。单向量压缩会损失精细布局，不天然支持开放式长文本生成。

### 3.2 Vision encoder + projector + LLM

视觉 encoder 输出 patch features，linear/MLP projector 映射到 LLM hidden dimension，作为“视觉前缀”与文本一起进入 decoder。Projector shape 对齐不意味着语义已对齐；需要图文训练。

### 3.3 Q-Former / resampler / cross-attention

一组 learned queries 从大量视觉特征提取固定/受限数量 token，降低 LLM 成本。压缩率越高，细字、计数和密集对象越可能丢失。Cross-attention 架构可让语言层读取独立视觉 memory，而不是把所有 visual tokens 拼入同一序列。

### 3.4 Unified/early-fusion token

图像可被离散 tokenizer/codec 编成 token，与文本一起 autoregressive modeling；也可让不同模态共享 Transformer 但保留 modality-specific encoder/decoder。所谓“统一 token”仍依赖不同 quantizer、loss 和输出 decoder，不等于所有 token 具有相同语义或成本。

## 4. 训练目标

### 4.1 Contrastive learning

InfoNCE 类目标提高 batch 内匹配 pair 相似度，降低负 pair。Batch negative 可能是假负例：同一场景的另一个正确 caption 被当成负样本。大 batch 提供更多 negatives，也改变优化与计算。

### 4.2 Captioning / conditional generation

给图像条件，最大化 caption/answer token likelihood。Alt text、网页邻近文本和自动 caption 噪声很大；模型可能学到网站模板而非视觉 grounding。

### 4.3 Image-text matching 与 grounding

Matching 判断整体是否匹配；grounding 把短语对齐到 box/mask/region。只有 caption loss 不保证模型学会精确坐标或计数。

### 4.4 Multimodal instruction tuning

对 OCR、VQA、定位、图表、文档和多轮对话做 SFT。Data mixture 决定能力：大量通用 caption 可能压过罕见图表/空间任务。合成 question/answer 会继承 teacher 的视觉错误。

### 4.5 冻结与端到端训练

冻结 vision encoder/LLM 只训练 projector 成本低，但对新视觉域适应有限；端到端训练更灵活，也更易破坏语言能力并需要更多数据。常见分阶段方案不能被理解为唯一正确流程。

## 5. 文档、OCR 与图表

### 5.1 OCR pipeline

典型：检测文本区域 → 识别 → 阅读顺序/布局 → table/form parser → LLM。优点是文字可检索、可引用、可单独校验；缺点是错误级联和视觉关系丢失。

### 5.2 Native vision

直接输入页面能利用布局、颜色、图形和手写信息，但细小字符、相似数字、小数点和复杂表格仍会出错。不能用“看起来懂页面”的回答替代逐字段 OCR/表格验证。

### 5.3 Hybrid

生产文档系统常把 image、OCR text、box、layout tree 和 source page 一起提供。对金额、日期、药物剂量等字段使用 deterministic parser/checksum/range validation，并把最终回答链接到 page/region。

### 5.4 图表

需要区分 title/legend/axis/scale、series 与 data point。常见失败：对数轴当线性轴、颜色/legend 错配、截断坐标轴、把趋势描述成精确数字。评测既要测问答，也要测 data extraction 与 visual grounding。

## 6. Audio 与语音

原始波形采样率很高，通常转 frame、spectrogram 或 learned feature，再下采样成 audio tokens。

### 6.1 ASR

- **CTC**：假设给定输入时 frame labels 条件独立，通过 blank 与 alignment 求和；解码可用 greedy/beam + LM。
- **RNN-T/Transducer**：结合 acoustic encoder 与 label-history predictor，适合 streaming。
- **Encoder-decoder**：audio encoder + autoregressive text decoder，能联合语言上下文。

WER/CER 依 normalization、分词、标点和语言。中文按 Unicode code point 的 CER 不等于按 grapheme/词评测。

### 6.2 TTS 与 speech-to-speech

TTS 生成 acoustic representation，再由 vocoder 输出波形；系统还要控制 speaker、prosody、latency 和 interruption。Speech-to-speech 可保留语气和低延迟，但安全文本 classifier 可能看不到中间音频语义，因此需要 audio-native control 或可信 ASR audit path。

### 6.3 Streaming

指标包括 first partial latency、finalization latency、endpointing、real-time factor、revision rate 和 interruption。过早 endpoint 截断，过晚 endpoint 增加延迟。Echo、噪声、重叠说话、口音和 codec 都应切片。

## 7. Video

视频是空间 × 时间 × 音频，token 成本远高于单图。常用：

- 均匀/随机 frame sampling；
- scene cut、motion、事件驱动 keyframe；
- spatial/temporal patch；
- hierarchical clip encoder；
- 长视频分段检索后精看。

抽帧会产生 sampling aliasing：瞬时事件、动作顺序或快速文字可能完全漏掉。只有字幕的 benchmark 可被文本模型取巧。

### 7.1 时间坐标

区分 seconds、milliseconds、frame index、variable frame rate 和 inclusive/exclusive endpoint。Temporal IoU 对连续区间计算时不加 `+1`；离散 inclusive frames 常用不同约定。

### 7.2 长视频任务

- event localization；
- temporal order/causality；
- cross-clip entity tracking；
- audio-visual synchronization；
- global summary 与细节 retrieval。

分别测短 clip 与长视频；把一小时视频只取八帧不能声称覆盖全局理解。

## 8. Image/Audio generation

图像生成可使用 autoregressive discrete tokens、diffusion、flow matching 或混合 decoder。语言模型负责 prompt/semantic plan，不代表像素 decoder 也是 autoregressive LLM。

评价包括 prompt adherence、视觉质量、多样性、文字渲染、人物一致性和安全。FID/CLIP-like score 是代理，不能单独评价版权、事实、审美或用户意图。

音频生成还需 speaker consent、voice similarity、intelligibility、prosody 和 watermark/provenance。Watermark 可能被变换破坏，不应作为唯一滥用防线。

## 9. Multimodal evaluation

### 9.1 OCR CER

\[
CER=\frac{S+D+I}{N_{reference}}.
\]

CER 可因大量 insertion 大于 1。必须固定 Unicode normalization、空白、大小写、标点和 reference unit。空 reference 的分母未定义，应单独处理而不是静默返回 0。

### 9.2 Bounding-box IoU

\[
IoU=\frac{|A\cap B|}{|A\cup B|}.
\]

先声明 pixel/normalized、continuous/inclusive coordinates。`(0,0,2,2)` 的 continuous area 为 4；若按 inclusive integer pixels 则定义不同。

### 9.3 Temporal IoU

对连续时间区间用 intersection duration / union duration。离散 frame index、variable FPS 与 timestamp rounding 必须另定义。

仓库实现明确口径：

```python
from about_llm.evaluation import box_iou, character_error_rate, temporal_iou

cer = character_error_rate("语言模型", "语言大模")  # 0.5
iou = box_iou((0, 0, 2, 2), (1, 1, 3, 3))  # 1/7
tiou = temporal_iou((0, 10), (5, 15))  # 1/3
```

这些函数只验证 metric convention，不运行任何多模态模型。

### 9.4 VQA 与开放回答

Exact match 对同义表达脆弱；LLM judge 又可能受长度、图像不可见或自身视觉能力限制。组合结构化答案、人工 rubric、证据 region 与 deterministic checks。

### 9.5 Modality-use test

为确认模型真的使用目标模态：

- image/audio/video masking；
- 替换成不匹配模态；
- 保留文本提示但改变关键视觉属性；
- 去掉字幕/alt text/metadata；
- counterfactual pair 只改变一个对象、数字或顺序；
- 比较 text-only baseline。

性能下降说明模态影响输出，但仍不证明精确内部机制；注意避免遮蔽造成极端 OOD。

## 10. Grounding 与 hallucination

Object hallucination 可能来自语言 prior、低分辨率、视觉压缩、训练 caption 噪声或问题暗示。评测：

- 图中存在/不存在对象；
- relation、count、attribute 和 OCR；
- 模型能否表示不可见/不确定；
- 输出 claim 是否链接 region/time span；
- conflicting text overlay 与真实图像内容。

模型说“根据图片”不是 grounding 证据；需要 region、OCR、可验证字段或反事实测试。

## 11. Serving 与成本

请求成本包括 media decode、resize/frame sampling、vision/audio encoder、projector、LLM prefill/decode 和缓存。不同 provider 的“image token”计数是服务契约，不等于 patch 数或可跨模型比较的物理单位。

优化：

- media hash + encoder feature cache（cache key 含 preprocessing/model revision/权限）；
- 限制像素、帧、时长、页数和并发；
- thumbnail + adaptive crop；
- batch vision encoder，但控制 tail latency；
- OCR/ASR 结果缓存与版本；
- 长视频先检索再精看。

Cache 可能保存人脸、文档和声音等敏感数据，必须带 tenant/ACL/TTL。

## 12. 安全与隐私

### 12.1 Cross-modal injection

图像文字、白字、二维码、PDF 隐藏层、音频指令、字幕和 metadata 都可能注入。OCR/ASR 结果仍是不可信数据，不能提升为 system instruction。

### 12.2 敏感属性

人脸、声音、生物特征、位置 metadata、医疗图像和家庭环境可能高度敏感。只因为模型能识别，不代表产品有权收集、推断或保存。做 purpose limitation、consent、access、TTL 和地域审查。

### 12.3 Deepfake 与 impersonation

声音/人像生成需要授权、identity protection、abuse monitoring 与 provenance。Detection/watermark 都有误报、漏报和鲁棒性限制，不能作为单一控制。

### 12.4 Adversarial media

压缩、裁剪、噪声、贴纸、不可见扰动或 frame insertion 可能改变输出。对现实 capture pipeline 做鲁棒性测试，而不只在干净 benchmark 上加数学噪声。

## 13. Data lineage

图文网页 pair、视频字幕和音频转写常存在弱对齐。保留 media hash、source/license、capture time、caption provenance、OCR/ASR revision、crop/frame selection 与 synthetic generator。

同一视频切出的多个 clip、同一文档多页和同一人物多图不能随机跨 train/test，否则 identity/content leakage。去重需要 media perceptual hash + text/metadata，阈值按变换校准。

## 14. 发布门禁

### 输入与预处理

- codec、尺寸、页数、时长和 decompression bomb 限制；
- EXIF/location 与敏感 metadata policy；
- resize/crop/tile/coordinate inverse 有测试；
- unsupported/blank/corrupt media 安全失败。

### 质量

- text-only 与 modality ablation baseline；
- OCR/CER、box/time localization 的口径固定；
- 小字、计数、图表、时间顺序和无答案切片；
- 多语言、设备、压缩、噪声和 accessibility；
- 输出 claim/region/source 可核查。

### 系统与安全

- media/feature cache 带 tenant、revision 和 TTL；
- cross-modal injection 不获得工具权限；
- 人脸/声音/位置有目的和授权；
- 真实 tool/action 仍由外部 policy 审批；
- 成本按实际像素/帧/时长/token 监控。

## 15. 当前仓库证据边界

仓库已有 CER、连续 box IoU 和 temporal IoU 的 12 个 CPU 单测，并有文本 RAG/安全协议可复用。但没有下载或运行目标视觉、音频、视频模型，也没有 GPU media encoder、真实 OCR 数据集或 provider 多模态 API 实测。因此本章证明 metric 与实验设计，不证明任何具体多模态模型能力或成本。

## 16. 常见错误结论

- **“图片被编码成 token，所以和文本 token 成本相同”**：encoder、压缩和服务计费各不相同。
- **“分辨率提高两倍，视觉 token 只提高两倍”**：固定 patch 下二维 token 数约提高四倍。
- **“模型回答正确，所以使用了图像”**：可能从题目、字幕或先验猜出。
- **“OCR CER 为 0 就理解了文档”**：阅读顺序、表格关系和任务推理仍可能错。
- **“IoU 实现都一样”**：continuous 与 inclusive integer 坐标不同。
- **“Speech-to-speech 不输出文本，所以更私密”**：音频仍含内容、身份和安全风险。
- **“加 watermark 就解决 deepfake”**：移除、漏标、误判和未采用系统仍存在。

## 自测与实践

1. 对 224→448、patch size 不变的 ViT 计算 patch 数变化。
2. 为 letterbox 图像写出 box inverse transform 并构造测试。
3. 为什么 CER 可以大于 1？空 reference 应怎样报告？
4. 设计验证视频模型不是只读字幕的 counterfactual set。
5. 比较 OCR+LLM、native vision 与 hybrid 在发票抽取上的错误链。
6. 列出 image/audio/video cache key 必须包含的权限和版本字段。

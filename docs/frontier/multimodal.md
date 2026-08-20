# 多模态模型：一张票据怎样变成可验证答案

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：想理解视觉、文档、音频或视频模型怎样进入真实系统的工程师。
- **先修**：[Transformer](../core/transformer.md)、基础数据治理与评测。
- **首次阅读**：先跟手机票据走完预处理、编码、回答与验证，再扩展到音频和视频。
- **完成信号**：能设计一项实验，证明模型使用了目标模态并指出失败发生在哪一层。
- **卡住时**：先忽略训练目标，只追踪像素怎样变成 visual tokens，又怎样映射回原图坐标。

</div>

用户上传一张手机拍摄的票据，问：“商户、日期和总金额是多少？请在原图上标出证据。”
图片是 3024×4032 JPEG，略有倾斜，金额位于右下角，背景还拍到了桌面和另一张小票。

模型最后回答“总金额 89.00 元”，只是这条任务的终点。系统在此前已经做了图片解码、方向修正、resize 或 tiling、
视觉编码、图文融合和生成；之后还要验证金额字符串、坐标、权限和证据。任何一步都可能让一句流畅答案失去依据。

这条票据任务会贯穿本章。音频与视频的信号形式不同，但仍要回答同一组问题：原始输入怎样采样，
模型看到了什么，输出怎样对回现实坐标，以及我们如何证明它真的使用了目标模态。

## 先固定任务契约

“支持图片”没有说明图片数量、分辨率、页数、动态范围、文字大小、动画或计费方式。
票据接口至少应固定：

```text
input:
  media_type = image/jpeg
  max_bytes / max_pixels / max_images
  orientation and color handling
  preprocessing revision

output:
  merchant: string | null
  date: ISO-8601 | null
  total: decimal + currency | null
  evidence: original-image bounding boxes
  status: complete | insufficient | conflict
```

同时记录 media hash、来源、授权、保留与删除策略，以及实际使用的 encoder、projector、tokenizer 和 template revision。
损坏图片、过小文字或不支持的 codec 要有明确终态；系统可以请求更清晰图片，也可以降级到 OCR 后转人工。

## 像素怎样变成视觉表示

Vision Transformer（ViT）常把 \(H\times W\) 图像切成 \(P\times P\) 的 non-overlapping patches。
在尺寸可整除的简单情形下，patch 数是：

\[
N=\frac{H}{P}\frac{W}{P}.
\]

每个 patch 展平后投影成 embedding，再加入位置表示。若 patch size 不变，把 224×224 输入提高到 448×448，
两个空间方向各变为两倍，patch 数约变为四倍。它会增加视觉 encoder 计算，也可能增加 LLM 收到的 visual tokens。

原始 patch 数并不总等于最终 token 数。实际模型可能使用 pooling、resampler、token merging、局部 attention，
或用一组固定查询压缩视觉特征。要估算上下文与显存，应读取目标模型的 processor、架构和真实 token 统计。

### Resize、crop 和 tiling 会改变证据坐标

票据可能先经过方向修正，再 letterbox 到模型尺寸。若模型返回 model-space box，系统必须保存变换并逆映射：

```text
original pixels
  -> orientation / resize / crop / pad
  -> model coordinates
  -> inverse transform
  -> original-image evidence box
```

Stretch 会改变形状，center crop 会丢掉边缘，letterbox 会加入 padding。高分辨率文档常采用 thumbnail + tiles：
缩略图保留全局布局，局部块保留小字，但 tile 边界可能切断对象，重叠区域也可能重复计数。

坐标约定要进入接口：pixel 还是 normalized、continuous 还是 inclusive integer，box 顺序是
`(x_min, y_min, x_max, y_max)` 还是别的形式。没有这些信息，同一个四元组会得到不同面积和 IoU。

## 视觉特征怎样接入语言模型

常见架构可以按“视觉信息在什么地方与文本相遇”理解。

### Dual encoder：两边各压成一个向量

图像编码器 \(f_I\) 和文本编码器 \(f_T\) 分别输出向量，并比较归一化相似度：

\[
s(I,T)=
\frac{f_I(I)^\top f_T(T)}
{\lVert f_I(I)\rVert\lVert f_T(T)\rVert}.
\]

这种结构适合 image-text retrieval 和 zero-shot classification。单向量把整张票据压得很紧，
不擅长保留每个金额的精细布局，也不会自然地产生长文本答案。

### Vision encoder + projector + LLM：把视觉前缀交给 decoder

视觉 encoder 产生 patch features，linear 或 MLP projector 将它们映射到 LLM hidden dimension，
再与文本 token 一起进入 decoder。Projector 的 shape 对上，只说明张量可以相加或拼接；
图文语义还要通过配对数据和训练目标建立。

Q-Former 或 resampler 会用 learned queries 从大量视觉特征中提取较少 token，降低 LLM 成本。
压缩过强时，小字、计数和密集对象会先丢失。Cross-attention 结构则让语言层读取独立视觉 memory，
不必把全部 visual tokens 当作普通前缀。

### Unified tokens：共享序列，不代表共享语义

图像也可以先经离散 codec 变成 token，与文本一起做 autoregressive modeling；另一类模型共享 Transformer，
但保留各模态的 encoder 或 decoder。即使都叫 token，图像与文本仍可能使用不同 quantizer、loss、采样率和输出 decoder，
成本也不能按“一个 token”直接等价比较。

## 模型通过哪些目标学会图文关系

不同训练目标留下不同能力：

| 目标 | 学到的主要关系 | 票据任务中的局限 |
|---|---|---|
| Contrastive / InfoNCE | 图像和文本整体是否匹配 | 很难给出精确金额位置 |
| Captioning / conditional generation | 根据图片生成文字 | 网页 alt text 噪声会教会模型猜模板 |
| Image-text matching | 整体 pair 是否相符 | 仍缺 region-level 对齐 |
| Grounding | 短语与 box / mask / region 对齐 | 依赖精确标注与坐标协议 |
| Multimodal SFT | 按指令完成 OCR、VQA、图表或多轮任务 | 能力取决于 mixture 和 teacher 质量 |

Batch contrastive learning 里的其他样本常被当作 negatives，其中可能存在同一场景的另一条正确描述。
Caption loss 能改善描述，却不保证精确计数和坐标。大量通用图片问答也可能压过稀有的文档、图表和空间任务。

冻结 vision encoder 与 LLM、只训练 projector，成本较低但新视觉域适应有限；端到端训练更灵活，
也更可能破坏已有语言能力并需要更多数据。分阶段训练是一种工程选择，不是所有架构的唯一顺序。

## 票据应该先 OCR，还是直接送入视觉模型

三条路径各有不同错误链。

### OCR pipeline

```text
text detection -> recognition -> reading order / layout
-> table or form parser -> LLM
```

优点是文字可检索、可逐字引用，OCR 和业务规则也能单独测试。缺点是上游识别和阅读顺序错误会逐层传播，
颜色、图形与空间关系可能在纯文本中丢失。

### Native vision

直接输入页面可以利用布局、颜色、logo 和手写标记。细小字符、相似数字、小数点和密集表格仍是高风险区域。
模型说“我看到了 89.00”不构成证据；至少还要检查原图 region、字符和业务范围。

### Hybrid

生产文档系统常把 image、OCR text、boxes、layout tree 与 source page 一起提供。
LLM 负责结合上下文，金额、日期或药物剂量等字段再由 deterministic parser、checksum、range rule 和人工流程验证。

图表任务也适合这种分层方法。Title、legend、axis、scale、series 和 data points 应分别抽取，
否则模型容易把对数轴当线性轴、把颜色映射错 series，或从趋势图报出并不存在的精确数字。

## 怎样证明模型真的看了图片

票据问题有时只靠语言先验就能猜中：用户问“总金额”，模型可能从文件名、OCR、历史对话或常见模板得到答案。
因此要加入 modality-use controls：

1. 遮蔽图片，保留同一文本问题；
2. 把票据替换为金额不同、布局相似的图片；
3. 只修改一个关键像素属性或数字，形成 counterfactual pair；
4. 去掉 alt text、OCR、文件名和 metadata；
5. 与 text-only baseline 比较；
6. 检查金额 claim 是否指向正确 region。

性能随图片变化而变化，说明该模态影响了输出。它还不能直接揭示内部神经机制，
而极端遮蔽也可能制造训练分布之外的输入，所以最好使用自然 counterfactual 和多种 ablation。

Object hallucination 常来自语言 prior、低分辨率、视觉压缩、caption 噪声和问题暗示。
评测应分开测对象是否存在、relation、count、attribute、OCR、无答案和不确定性表达。

## 三个指标先把坐标口径固定

### OCR character error rate

\[
CER=\frac{S+D+I}{N_{reference}}.
\]

Insertion 很多时 CER 可以大于 1。评测要固定 Unicode normalization、空白、大小写、标点与 reference unit；
空 reference 的分母未定义，应作为独立 case 处理。

### Bounding-box IoU

\[
IoU=\frac{|A\cap B|}{|A\cup B|}.
\]

Continuous coordinates 下，`(0,0,2,2)` 的面积是 4；inclusive integer pixels 使用另一套定义。
评测前统一 coordinate system，并验证预处理的 inverse transform。

### Temporal IoU

音频或视频的连续时间段同样使用 intersection duration / union duration。
Frame index、variable frame rate、timestamp rounding 与 inclusive endpoint 要另行约定。

仓库实现了这三种明确口径：

```python
from about_llm.evaluation import box_iou, character_error_rate, temporal_iou

cer = character_error_rate("语言模型", "语言大模")  # 0.5
iou = box_iou((0, 0, 2, 2), (1, 1, 3, 3))  # 1/7
tiou = temporal_iou((0, 10), (5, 15))  # 1/3
```

这些函数验证 metric convention，不会运行视觉、语音或视频模型。开放式 VQA 还需要结构化字段、人工 rubric、
证据 region 与 deterministic checks；exact match 对同义表达脆弱，LLM judge 也受自身视觉能力和长度偏好影响。

## 换成音频后，空间坐标变成时间

原始波形采样率很高，系统通常先转 frame、spectrogram 或 learned features，再下采样成 audio tokens。
三类常见自动语音识别（ASR）结构是：

- **CTC**：对含 blank 的 alignment 求和，解码可用 greedy 或 beam + LM；
- **RNN-T / Transducer**：结合 acoustic encoder 与 label-history predictor，适合 streaming；
- **Encoder-decoder**：audio encoder 加 autoregressive text decoder，能利用更强语言上下文。

WER/CER 依赖 normalization、分词和标点。中文按 Unicode code point 的 CER 与按词或 grapheme 评测不是同一口径。

Streaming 语音还要测 first partial、finalization、endpointing、real-time factor、revision rate 和 interruption。
过早 endpoint 会截断，过晚则增加延迟；echo、噪声、重叠说话、口音和 codec 都应单独切片。

TTS 先生成 acoustic representation，再由 vocoder 输出波形。系统除了 intelligibility，还要控制 speaker、prosody、latency 和打断。
Speech-to-speech 可以减少中间文本依赖，却仍处理内容、声音身份和敏感属性；安全系统需要 audio-native control 或受控 ASR audit path。

## 换成视频后，采样可能直接漏掉事件

视频同时包含空间、时间和音频，完整 token 成本通常远高于单图。系统会使用均匀或随机抽帧、scene cut、motion keyframe、
spatial/temporal patches、hierarchical clip encoder，或先检索长视频片段再精看。

抽帧会产生 sampling aliasing。一个瞬时动作、快速闪过的文字或动作先后顺序，可能恰好落在所有采样帧之间。
只有字幕的 benchmark 还可能被文本模型取巧，因此 modality-use test 要去掉字幕并改变关键画面。

长视频任务至少区分 event localization、temporal order、entity tracking、audio-visual synchronization、
global summary 和细节 retrieval。将一小时视频均匀取八帧，只验证这八帧对应的信息，不构成完整视频理解证据。

## 生成图像和音频是另一条输出链

图像生成可以使用 autoregressive discrete tokens、diffusion、flow matching 或混合 decoder。
语言模型可能负责 Prompt 理解和 semantic plan，像素 decoder 并不因此成为 autoregressive LLM。

评价需要组合 Prompt adherence、视觉质量、多样性、文字渲染、人物一致性和安全。
FID 或 CLIP-like score 是代理指标，无法单独判断版权、事实、审美和用户意图。

音频生成还要处理 speaker consent、voice similarity、intelligibility、prosody 与 provenance。
Watermark 可能被重编码、裁剪或变速破坏，应和授权、abuse monitoring、identity protection 及事件处置一起设计。

## Serving 成本从媒体解码就开始了

一次票据请求的成本包含 media decode、resize/tiling、vision encoder、projector、LLM prefill/decode、验证与缓存。
Provider 定义的“image token”是服务计费和容量契约，不能直接当作原始 patch 数，也不能跨模型比较。

常见优化包括：

- media hash 与 encoder feature cache；
- 像素、图片数、页数、帧数、音频时长和并发上限；
- thumbnail + adaptive crop；
- vision encoder batching，同时约束尾延迟；
- 版本化 OCR / ASR cache；
- 长视频先检索后精看。

Cache key 至少绑定 tenant/ACL、preprocessing、encoder/model revision 和 media identity，并设置 TTL 与删除机制。
图片、文档、人脸和声音本身可能是敏感数据，命中率优化不能越过访问与保留边界。

## 多模态输入扩大了攻击与隐私表面

图像文字、二维码、PDF 隐藏层、音频指令、字幕和 metadata 都可能携带 Prompt injection。
OCR/ASR 输出仍是外部数据，不能提升为 system instruction，也不能直接授予 tool 权限。

人脸、声音、生物特征、位置 metadata、医疗图像和家庭环境需要 purpose limitation、consent、access、TTL 与地域审查。
模型能识别某项属性，不代表产品有权收集、推断或保存。

压缩、裁剪、噪声、贴纸、frame insertion 和不可见扰动还可能改变输出。鲁棒性测试应尽量覆盖真实 capture pipeline，
而不仅是在干净 benchmark 上叠加数学噪声。Deepfake detection 和 watermark 都有误报、漏报与鲁棒性限制，
不能成为唯一控制。

训练数据同样保留 media hash、source/license、capture time、caption provenance、OCR/ASR revision、
crop/frame selection 与 synthetic generator。同一视频的 clips、同一文档的页面和同一人物的多张图片，
应按相关 group 切分，避免 identity 或 content leakage。

## 发布前跟票据再走一遍

| 环节 | 最低检查 |
|---|---|
| 输入 | codec、大小、页数、decompression bomb、metadata policy 与损坏终态 |
| 预处理 | resize/crop/tile 版本，coordinate inverse 和原图 region |
| 模态使用 | text-only baseline、自然 counterfactual、OCR/alt-text ablation |
| 质量 | CER、field accuracy、IoU、无答案、小字、计数、语言与噪声切片 |
| 安全 | injection 无工具权限，cache 隔离，人脸/声音/位置有授权 |
| 系统 | 实际像素/帧/时长/token、尾延迟、成本与失败终态 |

仓库现有 CPU 测试覆盖 CER、continuous box IoU 和 temporal IoU 的实现口径，
文本 RAG 与安全协议也可复用。仓库没有下载或运行目标视觉、音频和视频模型，
也没有真实 OCR 数据集、GPU media encoder 或 provider 多模态 API 实测。

因此当前证据能帮助你设计输入契约、指标和实验，不能代替目标模型在目标媒体分布上的能力与成本测试。

## 自测与实践

1. Patch size 不变时，224×224 提高到 448×448，patch 数为什么约变为四倍？
2. 为 letterbox 后的票据 box 写出 inverse transform，并构造一个边缘位置测试。
3. 模型正确回答金额时，哪些 counterfactual 能排除它只读文件名或语言先验？
4. 为什么 CER 可以大于 1？空 reference 应怎样进入结果表？
5. OCR + LLM、native vision 与 hybrid 的第一个可观察失败分别在哪里？
6. 设计一项视频实验，区分模型看到了关键动作还是只读取字幕。

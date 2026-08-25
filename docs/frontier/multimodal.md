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

模型最后回答“总金额 89.00 元”，只是这条任务的终点。在此之前，系统要解码图片、修正方向、调整尺寸，
再完成视觉编码、图文融合和文字生成。

回答出来以后，系统还要核对金额、坐标、权限和证据。

先把整条链放在一张图里：

| 层 | 它接收什么，又产出什么 | 常见依赖 | 票据可能怎样失败 |
|---|---|---|---|
| 媒体解码 | JPEG bytes → 像素 | Pillow、OpenCV 或服务端 codec | 图片损坏、方向错误、解压炸弹 |
| Processor | 像素 → 归一化 tensor、tile 和位置 metadata | 模型配套 image processor / `AutoProcessor` | 尺寸、crop、模板或坐标版本不一致 |
| Vision encoder | 像素 tensor → patch features | Transformers、timm 或模型自带实现 | 小字和局部细节在压缩中丢失 |
| Projector / resampler | 视觉特征 → LLM 可读取的 hidden states | Checkpoint 对应的 connector 模块 | Shape 对上，但语义没有正确对齐 |
| LLM decoder | 视觉状态 + 文本 → 输出 token | 模型类、tokenizer、chat template | 幻觉金额、忽略图片、输出不完整 |
| Serving runtime | 组织 batching、cache、流式输出和取消 | Transformers、vLLM、SGLang 等 | 新视觉路径未注册，或 batch/cost 语义错误 |
| Verifier | 字段与 box → 业务结果 | Schema、OCR、规则、数据库或人工 | JSON 合法，但金额、权限或证据错误 |

`safetensors` 只有权重，不能替代这条链中的 processor、模型计算图和 runtime 支持。某个库能加载 config，
也不表示它已经实现媒体 batching、视觉位置编码和完整服务路径。

这条票据任务会贯穿本章。音频与视频的信号形式不同，但仍要回答同一组问题：原始信号怎样采样，
模型实际看到了什么，输出怎样映射回现实坐标，以及如何证明它使用了目标模态。

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

除了输入输出字段，还要记录媒体 hash、来源、授权、保留与删除策略。实际使用的 encoder、projector、
tokenizer 和 template revision 也属于请求身份。

图片损坏、文字过小或 codec 不受支持时，请求要进入明确终态。系统可以要求用户上传更清晰图片，
也可以转到 OCR 或人工流程，不能悄悄返回猜测结果。

## 像素怎样变成视觉表示

Vision Transformer（ViT）常把 \(H\times W\) 图像切成互不重叠的 \(P\times P\) 小块，也就是 patches。
在尺寸可整除的简单情形下，patch 数是：

\[
N=\frac{H}{P}\frac{W}{P}.
\]

每个 patch 展平后投影成 embedding，再加入位置表示。如果 patch size 不变，把 224×224 输入提高到 448×448，
宽和高方向的块数都会翻倍，所以总 patch 数约变为四倍。

视觉 encoder 的计算随之增加。如果后续不做压缩，LLM 收到的视觉 token 也会更多。

原始 patch 数并不总等于最终视觉 token 数。模型可能使用 pooling、resampler、token merging 或局部 attention，
也可能用一组查询向量压缩视觉特征。

因此，不能只用图片尺寸和 patch size 推断上下文成本。应读取目标模型的 processor 与架构，
并记录实际生成了多少视觉 token。

### Resize、crop 和 tiling 会改变证据坐标

票据可能先经过方向修正，再用 letterbox 缩放到模型尺寸。如果模型返回的是模型坐标系中的 box，
系统必须保存每一步变换，最后做逆映射：

```text
original pixels
  -> orientation / resize / crop / pad
  -> model coordinates
  -> inverse transform
  -> original-image evidence box
```

直接拉伸（stretch）会改变形状，中心裁剪（center crop）会丢掉边缘，letterbox 则会加入 padding。

高分辨率文档常同时使用缩略图和局部 tiles。缩略图保留全局布局，局部块保留小字；代价是 tile 边界
可能切断对象，重叠区域也可能导致重复计数。

坐标约定必须写进接口：使用像素还是归一化数值，坐标表示连续边界还是包含端点的整数像素，box 顺序是否为
`(x_min, y_min, x_max, y_max)`。缺少这些信息，同一个四元组会得到不同面积和 IoU。

## 视觉特征怎样接入语言模型

常见架构可以按“视觉信息在什么地方与文本相遇”理解。

| 结构 | 视觉信息怎样进入任务 | Runtime 需要额外支持什么 | 更适合什么 |
|---|---|---|---|
| Dual encoder | 图像和文本各压成向量，再比较相似度 | 两个 encoder、向量索引或相似度计算 | 检索、匹配、zero-shot 分类 |
| Visual prefix | Projector 把 patch features 变成 LLM 前缀 | Processor、connector、视觉位置与联合 batching | VQA、文档问答、图文对话 |
| Cross-attention | 文本层读取独立视觉 memory | Encoder state 与每层 cross-attention 路径 | 保留独立视觉状态的生成任务 |
| Unified tokens | 图像和文本进入共享序列或共享主干 | 模态 tokenizer、位置规则和输出 decoder | 跨模态理解或联合生成 |

### Dual encoder：两边各压成一个向量

图像编码器 \(f_I\) 和文本编码器 \(f_T\) 分别输出向量，并比较归一化相似度：

\[
s(I,T)=
\frac{f_I(I)^\top f_T(T)}
{\lVert f_I(I)\rVert\lVert f_T(T)\rVert}.
\]

这种结构适合图文检索（image-text retrieval）和 zero-shot classification。整张票据被压成单个向量后，
很难保留每个金额的精细布局；相似度模型本身也不会自然生成长文本答案。

### Vision encoder + projector + LLM：把视觉前缀交给 decoder

视觉 encoder 先产生 patch features。Linear 或 MLP projector 再把特征映射到 LLM 的 hidden dimension。
完成映射后，视觉特征才能与文本 token 一起进入 decoder。

Projector 的 shape 对上，只证明张量能够进入计算图。模型是否把票据区域与“总金额”语义对应起来，
还取决于配对数据、训练目标和 checkpoint 权重。

Q-Former 或 resampler 使用 learned queries，从大量视觉特征中提取较少 token，以降低 LLM 成本。
压缩过强时，小字、计数和密集对象通常更容易丢失。

Cross-attention 采用另一种连接方式：语言层读取独立的视觉 memory，不必把全部视觉 token 放进普通前缀。

### Unified tokens：共享序列，不代表共享语义

图像也可以先经过离散 codec 变成 token，再与文本一起做 autoregressive modeling。
另一类模型共享 Transformer 主干，但保留各模态自己的 encoder 或 decoder。

即使都叫 token，图像与文本也可能使用不同 quantizer、loss、采样率和输出 decoder。
服务商定义的 image token 也有自己的计数规则，不能直接与文本 token 或另一个模型的 image token 等价比较。

## 模型通过哪些目标学会图文关系

不同训练目标留下不同能力：

| 目标 | 学到的主要关系 | 票据任务中的局限 |
|---|---|---|
| Contrastive / InfoNCE | 图像和文本整体是否匹配 | 很难给出精确金额位置 |
| Captioning / conditional generation | 根据图片生成文字 | 网页 alt text 噪声会教会模型猜模板 |
| Image-text matching | 整体 pair 是否相符 | 仍缺 region-level 对齐 |
| Grounding | 短语与 box / mask / region 对齐 | 依赖精确标注与坐标协议 |
| Multimodal SFT | 按指令完成 OCR、VQA、图表或多轮任务 | 能力取决于 mixture 和 teacher 质量 |

Batch contrastive learning 常把同批其他样本当作 negatives，但其中可能恰好有同一场景的另一条正确描述。
Caption loss 能改善描述流畅度，却不保证精确计数和坐标。训练数据若以通用图片问答为主，
稀有的文档、图表和空间任务也可能学得不足。

只训练 projector、冻结 vision encoder 与 LLM，成本较低，但适应新视觉域的空间有限。
端到端训练更灵活，也需要更多数据，并可能改变已有语言能力。分阶段训练只是工程选择，
不是所有多模态架构必须遵循的固定顺序。

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

生产文档系统常把原图、OCR 文本、boxes、layout tree 和 source page 一起提供。LLM 负责结合这些上下文。

金额、日期或药物剂量等字段，再由确定性 parser、checksum、范围规则或人工流程验证。

图表任务也适合分层处理。可以分别抽取标题、图例、坐标轴、刻度、数据序列和数据点，
再检查它们之间的关系。否则模型可能把对数轴当成线性轴、把颜色映射到错误序列，
或从趋势图中报出并不存在的精确数字。

## 怎样证明模型真的看了图片

票据问题有时只靠语言先验就能猜中。模型可能从文件名、OCR、历史对话或常见模板推测答案，
并没有读取目标图片。可以按由弱到强的方式加入对照：

1. **无图基线**：遮蔽图片，保留同一文本问题；
2. **消除旁路**：去掉 alt text、OCR、文件名和 metadata；
3. **自然反事实**：换成布局相似、金额不同的真实票据；
4. **局部反事实**：只修改一个关键数字，其他内容保持不变；
5. **证据定位**：检查金额 claim 是否指向原图中的正确 region；
6. **多样本复现**：在不同布局、语言、噪声和小字切片上重复上述实验。

输出随图片中的目标金额稳定变化，说明视觉模态影响了行为，但这还没有定位内部神经机制。
极端遮蔽也可能制造训练分布之外的输入。因此，自然反事实应与多种 ablation 配合使用。

对象幻觉（object hallucination）可能来自语言先验、低分辨率、视觉压缩、caption 噪声或问题暗示。
不要用一个总分混合所有错误。可以把“对象是否存在”、对象关系、计数、属性、OCR、无答案和
不确定性表达拆成不同切片。

## 三个指标先把坐标口径固定

### OCR character error rate

\[
CER=\frac{S+D+I}{N_{reference}}.
\]

插入错误很多时，\(S+D+I\) 可以大于 reference 长度，所以 CER 也可以大于 1。
评测前要固定 Unicode normalization、空白、大小写和标点规则。还要说明 reference 按 code point、
grapheme 还是词计数。

空 reference 的分母未定义，应作为单独 case 处理。

### Bounding-box IoU

\[
IoU=\frac{|A\cap B|}{|A\cup B|}.
\]

使用连续坐标时，`(0,0,2,2)` 的面积是 4。把坐标解释为包含端点的整数像素时，面积公式会不同。
评测前应统一坐标系，并用已知 box 检查预处理的 inverse transform。

### Temporal IoU

音频或视频的连续时间段也可以计算“交集时长 / 并集时长”。如果数据使用 frame index，
还要约定可变帧率、timestamp rounding 和端点是否包含在区间内。

仓库实现了这三种明确口径：

```python
from about_llm.evaluation import box_iou, character_error_rate, temporal_iou

cer = character_error_rate("语言模型", "语言大模")  # 0.5
iou = box_iou((0, 0, 2, 2), (1, 1, 3, 3))  # 1/7
tiou = temporal_iou((0, 10), (5, 15))  # 1/3
```

这些函数只实现上述 metric 口径，不会运行视觉、语音或视频模型。

开放式 VQA 还需要结构化字段、人工 rubric、证据 region 和确定性检查。Exact match 对同义表达很脆弱；
LLM judge 自身也受视觉能力和长度偏好影响。

## 换成音频后，空间坐标变成时间

原始波形采样率很高。系统通常先把它变成 frames、频谱图或 learned features，再下采样成 audio tokens。
三类常见自动语音识别（ASR）结构是：

- **CTC**：对含 blank 的 alignment 求和，解码可用 greedy 或 beam + LM；
- **RNN-T / Transducer**：结合 acoustic encoder 与 label-history predictor，适合 streaming；
- **Encoder-decoder**：audio encoder 加 autoregressive text decoder，能利用更强语言上下文。

WER/CER 会随 normalization、分词和标点规则变化。中文 CER 可以按 Unicode code point 计算，
也可以采用 grapheme 或词级单位；不同单位的结果不能直接比较。

流式语音要同时观察首个部分结果、最终结果延迟、endpointing、real-time factor、修订率和打断响应。
Endpoint 过早会截断话语，过晚会增加等待时间。

回声、噪声、重叠说话、口音和不同 codec 应分别切片，避免平均值掩盖失败场景。

TTS 先生成声学表示，再由 vocoder 输出波形。系统除了检查可懂度，还要控制说话人身份、韵律、延迟和打断。

Speech-to-speech 可以减少对中间文本的依赖，却仍然处理内容、声音身份和敏感属性。安全评测可以直接面向音频；
如果先转成文字，则 ASR 路径也要有明确权限并可审计。

## 换成视频后，采样可能直接漏掉事件

视频同时包含空间、时间和音频，完整 token 成本通常远高于单图。系统因此会先降低输入量，例如：

- 均匀或随机抽帧；
- 按 scene cut 或 motion keyframe 选帧；
- 使用时空 patches 或分层 clip encoder；
- 先检索长视频片段，再精看候选片段。

抽帧会产生 sampling aliasing。瞬时动作、快速闪过的文字或动作先后顺序，可能恰好落在所有采样帧之间。
如果 benchmark 的答案能从字幕直接得到，文本模型还可能取巧。模态使用实验应去掉字幕，并改变关键画面。

长视频任务可以拆成事件定位、时间顺序、实体跟踪、音画同步、全局摘要和细节检索。
把一小时视频均匀取八帧，只能验证模型看到了这八帧中的信息，不能代表它理解了完整视频。

## 生成图像和音频是另一条输出链

图像生成可以使用自回归离散 token、diffusion、flow matching 或混合 decoder。

语言模型可能只负责理解 Prompt 和制定语义计划，真正输出像素的是另一套 decoder。
描述架构时，应分别说明语言规划与像素生成由谁完成。

评价需要组合 Prompt adherence、视觉质量、多样性、文字渲染、人物一致性和安全。
FID 或 CLIP-like score 是代理指标，无法单独判断版权、事实、审美和用户意图。

音频生成还要处理说话人同意、声音相似度、可懂度、韵律和来源记录。
Watermark 可能被重编码、裁剪或变速破坏，所以还需要授权、滥用监控、身份保护和事件处置。

## Serving 成本从媒体解码就开始了

一次票据请求的成本从媒体解码就开始了。随后还有 resize/tiling、vision encoder、projector、
LLM prefill/decode、验证和缓存读写。优化时应分阶段测量，不能把全部时间归到“模型推理”。

服务商定义的“image token”是一项计费与容量契约。它可能已经包含 resize、tiling 或特征压缩，
所以既不等于原始 patch 数，也不能跨模型直接比较。

常见优化包括：

- media hash 与 encoder feature cache；
- 像素、图片数、页数、帧数、音频时长和并发上限；
- thumbnail + adaptive crop；
- vision encoder batching，同时约束尾延迟；
- 版本化 OCR / ASR cache；
- 长视频先检索后精看。

Cache key 至少要绑定 tenant/ACL、预处理版本、encoder/model revision 和媒体 identity。
缓存还要设置 TTL 与删除机制。

图片、文档、人脸和声音本身可能是敏感数据。提高命中率时，仍要遵守访问权限和数据保留边界。

## 多模态输入扩大了攻击与隐私表面

图像文字、二维码、PDF 隐藏层、音频指令、字幕和 metadata 都可能携带 Prompt injection。

OCR/ASR 输出仍然是不可信外部数据。系统应把它作为待处理内容，而不是 system instruction 或工具授权。

人脸、声音、生物特征、位置 metadata、医疗图像和家庭环境都可能包含敏感信息。
收集前应明确用途、同意方式、访问权限、TTL 和地域要求。

模型能够识别某项属性，只说明技术上可推断；产品是否有权收集、推断或保存，是另一项合规与授权判断。

压缩、裁剪、噪声、贴纸、插入帧和不可见扰动都可能改变输出。鲁棒性测试应覆盖真实 capture pipeline，
例如手机拍摄、平台转码和截图，而不只是给干净 benchmark 叠加数学噪声。

Deepfake detection 和 watermark 都会误报或漏报，也可能在媒体变换后失效。它们只能作为分层控制的一部分。

训练数据应记录媒体 hash、来源与许可、采集时间和 caption 来源。OCR/ASR revision、crop/frame 选择方式
与 synthetic generator 也会影响样本身份，应一并保存。

同一视频的多个 clips、同一文档的多页和同一人物的多张图片高度相关。数据切分应以这些 group 为单位，
避免身份或内容跨越 train/test。

## 发布前跟票据再走一遍

| 环节 | 最低检查 |
|---|---|
| 输入 | codec、大小、页数、decompression bomb、metadata policy 与损坏终态 |
| 预处理 | resize/crop/tile 版本，coordinate inverse 和原图 region |
| 模态使用 | text-only baseline、自然 counterfactual、OCR/alt-text ablation |
| 质量 | CER、field accuracy、IoU、无答案、小字、计数、语言与噪声切片 |
| 安全 | injection 无工具权限，cache 隔离，人脸/声音/位置有授权 |
| 系统 | 实际像素/帧/时长/token、尾延迟、成本与失败终态 |

仓库现有 CPU 测试覆盖 CER、连续坐标 box IoU 和 temporal IoU 的实现口径；文本 RAG 与安全协议也可复用。

当前仓库没有下载或运行目标视觉、音频和视频模型，也没有真实 OCR 数据集、GPU media encoder 或
多模态云 API 实测。因此，本章能够帮助你设计输入契约、指标和实验，但目标模型的能力与成本仍要在
实际媒体分布和目标硬件上测量。

## 自测与实践

1. Patch size 不变时，224×224 提高到 448×448，patch 数为什么约变为四倍？
2. 为 letterbox 后的票据 box 写出 inverse transform，并构造一个边缘位置测试。
3. 模型正确回答金额时，哪些 counterfactual 能排除它只读文件名或语言先验？
4. 为什么 CER 可以大于 1？空 reference 应怎样进入结果表？
5. OCR + LLM、native vision 与 hybrid 的第一个可观察失败分别在哪里？
6. 设计一项视频实验，区分模型看到了关键动作还是只读取字幕。

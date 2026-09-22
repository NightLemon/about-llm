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

“总金额 89.00 元”只是模型生成的一项声明。要把它交给业务，系统还要回答：模型读到的是哪一版图片，
金额来自原图哪个区域，这个区域能否支持答案，以及当前用户是否有权查看它。

先跟住这条请求，不必急着记库名：

```text
JPEG 文件
  -> 解码后的原始像素
  -> 缩放、补边后的模型输入
  -> 视觉表示
  -> 语言模型生成的字段与证据框
  -> 映射回原图的证据区域
  -> 通过验证的业务结果
```

每一步都由不同组件负责。Pillow、OpenCV 或服务端 codec 把文件解码成像素。

模型配套的 processor 再完成缩放、归一化和切块，同时留下坐标变换记录。视觉编码器与连接器把图片
变成语言模型能读取的表示。

推理 runtime 负责批处理、缓存、流式输出和取消。最后，schema、OCR、业务规则或人工流程共同核对答案。

模型权重文件（通常是 `safetensors`）只是其中一部分。能够读取配置和权重，也不等于运行库已经支持该模型的
图片预处理、视觉位置编码、批处理和服务接口。排查“模型能下载却不能推理”时，应逐层确认缺的是 processor、
模型实现、底层算子，还是服务 runtime 的多模态接入。

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

同一张票据至少有四个不能混写的状态：上传的 JPEG 字节、方向修正后解码出的像素、带有 resize/pad
记录的模型输入，以及可缓存的 encoder feature。文件 hash 只能标识第一项；模型输入还要绑定像素
解释和 preprocessing revision，feature 则还要绑定 encoder revision、租户和当前可见权限。用户删除原图、
撤回授权或权限变化时，服务不能只删下载链接：它要使相应的派生像素、feature 和 cache entry 失效，
否则下一次请求仍可能从旧特征还原出本不该交付的金额。

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

### 用票据算一次缩放、patch 和坐标

下面是一组为了方便复算而构造的数字，不代表某个具体模型的默认设置。假设系统已经按 EXIF 修正方向，
暂不做倾斜校正；processor 将长边缩放到 448 像素，再把短边两侧补到 448 像素：

1. 原图是 3024×4032，缩放比例为 \(448/4032=1/9\)；
2. 缩放后得到 336×448，左右各补 56 像素，模型输入变成 448×448；
3. 若视觉编码器使用 14×14 patch，输入网格有 \(32\times32=1024\) 个 patch 位置；
4. 原图中的金额框 `(2304, 3492, 2880, 3690)` 会映射为模型坐标 `(312, 388, 376, 410)`。

第四步只用了缩放与补边：

\[
x_{model}=x_{original}/9+56.
\]

\[
y_{model}=y_{original}/9.
\]

模型或后处理程序若返回 `(312, 388, 376, 410)`，逆变换就是：

\[
x_{original}=9(x_{model}-56).
\]

\[
y_{original}=9y_{model}.
\]

把这段代码复制到 Python 中，可以同时检查正向和逆向计算：

```python
scale = 1 / 9
pad_x = 56


def to_model(box: tuple[float, float, float, float]):
    x0, y0, x1, y1 = box
    return (x0 * scale + pad_x, y0 * scale, x1 * scale + pad_x, y1 * scale)


def to_original(box: tuple[float, float, float, float]):
    x0, y0, x1, y1 = box
    return ((x0 - pad_x) / scale, y0 / scale,
            (x1 - pad_x) / scale, y1 / scale)


original = (2304, 3492, 2880, 3690)
model = to_model(original)
assert model == (312, 388, 376, 410)
assert to_original(model) == original
```

它应当还原成原来的金额区域。真实系统若加入旋转、倾斜校正、裁剪或多个 tile，就要保存完整的变换链，
不能只记一个缩放比例。1024 也只是这组输入的原始 patch 数；经过 resampler、合并或池化后，语言模型实际
收到的视觉 token 可能更少。

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

- **双编码器（dual encoder）**把图像和文本各压成一个向量，最后比较相似度。运行时要执行两个编码器和
  相似度计算，常用于检索、匹配和零样本分类。
- **视觉前缀（visual prefix）**先把图片特征变成语言模型的输入前缀。运行时要补齐 processor、连接器、
  视觉位置和联合批处理，常用于文档问答与图文对话。
- **交叉注意力（cross-attention）**让语言层读取一份独立的视觉记忆。运行时要保存编码器状态，并执行
  语言层中的交叉注意力路径。
- **统一 token（unified tokens）**让图像和文本进入共享序列或共享主干。运行时仍要知道每种模态如何
  编码、使用什么位置规则，以及由哪个解码器还原输出。

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

视觉编码器先把每个图片块变成特征向量。连接器通常是一层线性映射或小型多层感知机（MLP），
负责把这些向量改成语言模型能够接收的宽度。完成映射后，视觉表示才能和问题中的文本 token 一起进入
语言模型。

Projector 的 shape 对上，只证明张量能够进入计算图。模型是否把票据区域与“总金额”语义对应起来，
还取决于配对数据、训练目标和 checkpoint 权重。

Q-Former 或 resampler 使用 learned queries，从大量视觉特征中提取较少 token，以降低 LLM 成本。
压缩过强时，小字、计数和密集对象通常更容易丢失。

Cross-attention 采用另一种连接方式：语言层读取独立的视觉 memory，不必把全部视觉 token 放进普通前缀。

### Unified tokens：共享序列，不代表共享语义

图像也可以先由离散编码器变成 token，再与文本放进同一条自回归序列。另一类模型共享 Transformer 主干，
但仍为图片、文字或音频保留各自的输入编码器或输出解码器。

“共享 token 序列”不表示不同模态使用相同词表或具有相同成本。图像可能经过自己的量化器，音频有自己的
采样率，生成图片还需要像素解码器。服务商给出的 image token 也有单独的计数规则，不能直接换算成文本
token，更不能拿两个模型的 image token 数直接比较。

### 架构决定运行库必须补齐什么

模型仓库里的配置告诉运行库“这是什么结构”，但真正执行还需要对应实现：

- Transformers 一类模型库要认识 processor、视觉编码器、连接器、语言模型和生成接口，并按 checkpoint
  约定装载每一组权重；
- `timm` 可以提供常见视觉骨干，却通常不负责多模态 chat template、视觉 token 插入位置或文本生成；
- vLLM、SGLang 等服务 runtime 还要实现图片请求解析、不同尺寸的批处理、视觉特征缓存、调度和显存估算；
- FlashAttention、Triton 或厂商算子可以加速某段计算，但不会自动补出缺失的 processor 或模型结构。

因此，判断“某个运行库是否支持这个模型”时，至少要检查三件事：能否生成正确的模型输入，能否执行完整计算图，
以及服务接口能否正确批处理和计量多模态请求。只看到模型名称出现在配置表中，还不足以回答这三个问题。

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

回到开头的票据，混合路径可以这样结束一次请求：

1. OCR 给出两个金额候选；
2. 视觉模型选择右下角的“89.00”，同时返回模型坐标中的证据框；
3. 系统把框逆变换到原图，再裁出该区域，确认数字旁边写的是“合计”而不是“优惠”；
4. OCR、视觉答案和业务规则若互相冲突，请求进入 `conflict`，交给更强模型或人工复核。

这里的关键不是让三个组件投票，而是让最终字段回到原始证据，并为冲突保留明确终态。

这里还要分开三个判断。视觉模型答对 `89.00`，只说明这个字段在该 case 恰好正确；框回到“合计 89.00”
区域，才支持它确实依据了目标区域；调用者仍须通过原图和该字段的访问检查，业务规则也要接受货币与金额。
这三项任一不成立，系统都不能把字段当作可交付结果。这样，正确答案、证据定位和授权交付不会被一个
`complete` 文本混成同一件事。

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

对象幻觉（object hallucination）可能来自语言先验，也可能来自图片没有被看清。低分辨率、视觉压缩、
错误的图片描述和带暗示的问题都可能改变答案。

评测时把对象存在性、关系、计数、属性、文字识别和无答案分别报告，再单独检查模型是否表达了不确定性。
这样才能区分“没有看清”和“看清后仍然编造”。

## 三个指标先把坐标口径固定

### OCR character error rate

\[
CER=\frac{S+D+I}{N_{reference}}.
\]

插入错误很多时，\(S+D+I\) 可以大于参考文本长度，所以 CER 也可以大于 1。
评测前要先统一 Unicode、空白、大小写和标点的处理方式，再说明分母按 Unicode 码点、用户感知字符
还是词来计数。采用不同文本单位得到的数字不能放在同一张表中直接比较。

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
用大模型评分（LLM-as-a-Judge）也会受模型视觉能力和长度偏好影响。

## 其他模态只看相对票据任务改变了什么

票据视觉任务是本页主线。换成音频、视频或生成任务时，不再重讲一套完整教程，只记录新增的坐标、失败与验收：

| 模态 | 表示或坐标变化 | 最容易漏掉的失败 | 额外验收 |
|---|---|---|---|
| 音频 | 波形变成 frame、频谱或 audio token；空间坐标变成时间 | endpoint 过早截断、过晚增加等待，噪声/口音被平均值掩盖 | WER/CER 清洗规则、首个与最终结果延迟、修订率、打断与说话人同意 |
| 视频 | 帧、clip 与音轨共同形成时空输入 | 抽帧漏掉瞬时事件，字幕让模型绕过视觉 | 事件定位、顺序、音画同步；去字幕和改变关键帧的反事实 |
| 图像/音频生成 | LLM 可能只规划语义，另一 decoder 产生像素或波形 | 代理分数掩盖版权、身份、事实或安全问题 | 明确每个生成组件、来源记录、人物/声音一致性与滥用控制 |

## 系统与治理只保留多模态增量

服务成本从媒体解码、缩放、切块或抽帧开始；供应商的“image token”不等于原始 patch，也不能跨模型直接比较。
缓存 identity 还要绑定租户权限、媒体内容、预处理和 encoder/model 版本，并随删除或撤权失效。通用队列、容量与发布
流程见[服务与可观测性](../systems/serving.md)。

图像文字、二维码、PDF 隐藏层、音频指令、字幕和 metadata 都是不可信输入，不能获得 system instruction 或工具权限。
人脸、声音、位置和环境信息的用途、同意、访问与保留由[隐私与公平](../quality/privacy-fairness.md)负责；训练数据 lineage
与 group-level 切分见[数据工程](../training/data.md)。本页只要求这些边界进入票据任务的输入契约与反事实测试。

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

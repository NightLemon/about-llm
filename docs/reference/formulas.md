# 公式速查

## 概率与损失

| 名称 | 公式 | 说明 |
|---|---|---|
| 链式法则 | \(p(x_{1:T})=\prod_t p(x_t\mid x_{<t})\) | 自回归分解 |
| Softmax | \(p_i=e^{z_i}/\sum_j e^{z_j}\) | logits → 概率 |
| 交叉熵 | \(H(p,q)=-\sum_x p(x)\log q(x)\) | 真实分布对模型的编码代价 |
| KL 散度 | \(D_{KL}(p\Vert q)=\sum_xp(x)\log\frac{p(x)}{q(x)}\) | 非对称，不是距离 |
| 困惑度 | \(PPL=e^{\text{mean NLL}}\) | 只在同 tokenization/数据上比较 |
| 熵 | \(H(p)=-\sum_xp(x)\log p(x)\) | 分布不确定性 |
| Temperature sampling | \(p_i(T)=\exp(z_i/T)/\sum_j\exp(z_j/T),\ T>0\) | `temperature=0` 是部分 API 的 greedy 特例，不能代入除法；temperature 会改变后续 top-p support |
| Top-p support | \(m=\min\{r:\sum_{i=1}^{r}p_{(i)}\ge p\}\) | 保留前 \(m\) 个，包含 crossing token；组合 top-k 时先明确在哪个 support 上归一化 |
| Sign-aware repetition（本仓库采用的规则） | \(z_i'=rz_i\ (z_i<0),\ z_i'=z_i/r\ (z_i\ge0)\) | 只处理历史中每个唯一 token 一次；不是 frequency penalty，也不是通用 provider 契约 |
| Beam final score（本仓库采用的规则） | \(s(x_{1:T})=\log p(x_{1:T})/T^\alpha\) | \(T\) 只计生成 token、包含已发出的 EOS、不含 prompt；active beam 仍按 raw cumulative log-prob 剪枝，此约定不等于所有 runtime |
| Constrained renormalization | \(\tilde p_i=p_i\mathbf1[i\in A(q)]/\sum_jp_j\mathbf1[j\in A(q)]\) | \(A(q)\) 由 token 完整片段对状态 \(q\) 的转移决定；分母为 0 必须显式失败，EOS 只在 accepting state 合法 |
| Paired sign-flip exact p | \(p=2^{-m}\sum_{s\in\{-1,1\}^m}\mathbf1[T(s\odot d)\text{ 至少同样极端}]\) | \(m\) 只计非零 pair；依赖 sharp-null label exchangeability，单/双侧须预先指定，不是 effect size 或 null posterior |
| Case-weighted cluster difference | \(\hat\Delta=N^{-1}\sum_g\sum_{i=1}^{n_g}d_{gi}\) | joint sign flip 用 cluster sum 作 contribution；大 cluster 权重更高 |
| Equal-cluster difference | \(\hat\Delta=G^{-1}\sum_g n_g^{-1}\sum_i d_{gi}\) | joint sign flip 用 cluster mean；与随机 case estimand 不同，须预指定 |
| Case-weighted cluster bootstrap | \(\hat\Delta^*=\frac{\sum_b\sum_i d_{g_b^*i}}{\sum_b n_{g_b^*}}\) | 有放回抽 \(G\) 个完整 cluster；每个 resample 的 ratio 分母会变 |
| Equal-cluster bootstrap | \(\hat\Delta^*=G^{-1}\sum_b n_{g_b^*}^{-1}\sum_i d_{g_b^*i}\) | 重采样 cluster means；不等于 case-weighted target |
| Holm adjusted p-value | \(\tilde p_{(i)}=\min(1,\max_{j\le i}(m-j+1)p_{(j)})\) | 对预定义 family 的有效 p-value 控制 FWER；running max 不可省，不修复 post-hoc selection/peeking |

## Transformer

| 名称 | 公式 | 说明 |
|---|---|---|
| 注意力 | \(\text{softmax}(QK^T/\sqrt{d_k}+M)V\) | M 含因果/填充约束 |
| LayerNorm | \(\gamma(x-\mu)/\sqrt{\sigma^2+\epsilon}+\beta\) | 按特征归一化 |
| RMSNorm | \(\gamma x/\sqrt{\text{mean}(x^2)+\epsilon}\) | 不减均值 |
| SwiGLU | \((\text{SiLU}(xW_g)\odot xW_u)W_d\) | 门控 MLP |
| LoRA | \(W'=W+(\alpha/r)BA\) | 低秩可训练增量 |
| Bradley–Terry RM margin/loss | \(m_i=(f_{w,i}-f_{l,i})^\top w,\ \ell_i=\operatorname{softplus}(-m_i)\) | \(P(y_w\succ y_l)=\sigma(m_i)\)；同 prompt 的共同 reward offset 不可识别 |
| 线性 RM 梯度 | \(\nabla J=-\frac1n\sum_i\sigma(-m_i)(f_{w,i}-f_{l,i})+\lambda w\) | \(J=\operatorname{mean}(\ell_i)+\lambda\lVert w\rVert^2/2\) |
| Reward centering（TRL 0.29 可选） | \(\lambda\operatorname{mean}[(r_w+r_l)^2]\) | 惩罚 pair midpoint；不要误写成 reward difference 或单边 score penalty |
| GAE TD residual | \(\delta_t=r_t+\gamma b_tV_{t+1}-V_t\) | \(b_t\) 控制 value bootstrap；terminated 为 0，truncated 必须按语义显式决定 |
| GAE recursion | \(A_t=\delta_t+\gamma\lambda c_tA_{t+1}\) | episode boundary/padding 令 \(c_t=0\)；truncated 即使 bootstrap 也不能跨轨迹递推 |
| PPO ratio | \(\rho_t=\exp(\log\pi_\theta(a_t\mid s_t)-\log\pi_{old}(a_t\mid s_t))\) | 只针对采样动作；不是完整 action/sequence distribution ratio |
| PPO clipped surrogate | \(\mathbb E[\min(\rho_tA_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)]\) | 正 advantage 裁上界、负 advantage 裁下界；不是 KL 硬约束 |
| PPO value loss（本仓库 toy） | \(L_V=\frac12\mathbb E[(V_\phi(s_t)-\hat R_t)^2]\) | 未做 value clipping；目标 return 在 rollout 后冻结 |
| PPO toy total loss | \(L=-L^{clip}+c_VL_V-c_H\mathcal H(\pi)\) | 工程实现还可能加入 reference KL、value clipping 等项；先核对具体库 |
| Sampled reference penalty | \(r_t^{shaped}=r_t^{task}-\beta[\log\pi_{old}(a_t\mid s_t)-\log\pi_{ref}(a_t\mid s_t)]\) | 单样本 log-ratio 可为负；对 \(a_t\sim\pi_{old}\) 的条件期望才是 categorical forward KL |
| 两 token 文本 PPO 精确目标 | \(J=p_0(g)+\sum_{a\ne e}p_0(a)p_1(e\mid a)\) | 目标序列概率为 \(p_0(g)p_1(e\mid g)\)；13 词表均匀初始值分别为 \(25/169\)、\(1/169\) |
| Learned-RM policy proxy | \(J_{RM}=\sum_{y\in\mathcal Y}\pi_\theta(y\mid x)[r_\phi(x,y)-c]\) | \(c\) 可固定 pairwise score offset；proxy 上升不推出独立 task/human utility 上升 |
| 训练-pair midpoint centering | \(c=[r_\phi(x,y_w)+r_\phi(x,y_l)]/2\) | 不改变 pair margin/ranking；不校准 OOD response，也不修复 reward hacking |

## 优化与规模

| 名称 | 公式 | 说明 |
|---|---|---|
| 梯度下降 | θₜ₊₁ = θₜ - η∇L(θₜ) | η 为学习率 |
| 全局 batch | \(B_{micro}\times accumulation\times DP\) | 不乘 TP/PP |
| 粗略训练 FLOPs | \(C\approx6ND\) | dense decoder 的预算近似 |
| Scaling law | \(L\approx L_\infty+aN^{-\alpha}+bD^{-\beta}\) | 参数依实验而变 |
| Compute-optimal 增长率 | \(N^*\propto C^{\beta/(\alpha+\beta)},\ D^*\propto C^{\alpha/(\alpha+\beta)}\) | 只适用于同一可分离幂律拟合与 \(C=kND\) 口径 |
| KV 元素/序列 | \(2LTH_{kv}D\) | 再乘 batch 与元素字节 |
| 单序列 KV block 数 | \(\lceil T/S\rceil\) | block size 为 \(S\)；tail 空 slot 为 \((S-T\bmod S)\bmod S\) |
| 物理 block 碎片 | \(N_{allocated}S-N_{physical\ token\ values}\) | 共享 prefix 的 logical tokens 会重复计数，不能代入物理项 |
| Prefix cache exact hit | \(I_e=I_q\ \land\ \tau_e=\tau_q[:|\tau_e|]\)，取最大 \(|\tau_e|\) | \(I\) 是完整安全/执行 identity，\(\tau\) 是 token ids；hash 只作候选索引，不能替代等值比较或授权 |
| Causal generation forward positions | \(W=\sum_i(P_i+O_i-1)\) | 无 prefix reuse/speculation/beam，且每请求输出 \(O_i\ge1\)；prefill 最后位置产生首 token 分布，不等于 API 计费或 padding/kernel work |
| MoE reference per-expert capacity | \(C=\lceil\phi Nk/E\rceil\) | \(N\) 为当前 group 有效 token、padding 不计，\(k\) 为 top-k；真实实现的 group/min/drop/reroute 语义可能不同 |
| MoE reference balance diagnostic | \(L_{bal}^{ref}=E\sum_ef_ep_e\) | \(f_e=n_e/(Nk)\) 为 pre-capacity assignment fraction，\(p_e\) 为平均 router probability；不是所有实现的 training loss |
| Router z-loss reference | \(L_z=N^{-1}\sum_i(\log\sum_e e^{r_{i,e}})^2\) | 约束 log-partition 尺度的一种常见形状；系数、group 与 reduction 依实现 |
| 对称分组量化 | \(s_g=\max_{i\in g}|w_i|/(2^{b-1}-1),\ q_i=\operatorname{clip}(\operatorname{round}(w_i/s_g),-q_{max},q_{max})\) | code-range/rounding/zero-group 约定必须显式 |
| 理想分组量化字节 | \(\lceil bRC/8\rceil+4R\lceil C/G\rceil\) | FP32 scale；不含 alignment、容器、zero point、未量化层和 workspace |
| 本仓库 packed code 映射 | \(u_i=q_i+q_{max}\in[0,2^b-2]\) | row-major、LSB-first；全 1 code 非法，末 byte 高 padding bit 为 0；不是通用 runtime 格式 |
| 本仓库单矩阵 v1 artifact | \(M_{file}=32+\lceil bRC/8\rceil+4R\lceil C/G\rceil+32\) bytes | fixed header + code + little-endian FP32 scales + unkeyed SHA-256；不含文件系统开销，不是整模型格式 |
| Per-token INT8 KV payload | \(M=2BH_{kv}T(D+4),\ \rho_{FP32/INT8}=4D/(D+4)\) | K/V head dim 同为 D，各自一个 FP32 scale/token/head；不含 block/alignment/workspace |
| Speculative 接受率 | \(\alpha(x)=\min(1,p(x)/q(x))\) | \(x\sim q\)，故被采到时 \(q(x)>0\) |
| Speculative 拒绝分布 | \(r(i)=(p(i)-q(i))_+/\sum_j(p(j)-q(j))_+\) | 拒绝概率 \(=TV(p,q)\)，一步总输出边际恢复为 \(p\) |

## 检索与评价

| 名称 | 公式 | 说明 |
|---|---|---|
| 余弦相似度 | \(x^Ty/(\lVert x\rVert\lVert y\rVert)\) | 向量方向相似 |
| RRF | \(\sum_i1/(k+\text{rank}_i(d))\) | 融合多个排序 |
| Precision | \(TP/(TP+FP)\) | 预测为正中有多少正确 |
| Recall | \(TP/(TP+FN)\) | 真实为正中找回多少 |
| F1 | \(2PR/(P+R)\) | P/R 调和平均 |

公式前先确认符号、shape、对数底、归一化单位与 mask。不同论文同一字母可能含义完全不同。

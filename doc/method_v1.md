# method_v1 — 论文 Novelty + 方法实施 (MIA, 骨科, 无曲线)
日期 2026-06-24 创建; R29 claim ladder; R30 五项; R31 P3定稿=L_smooth; R32 四微裂缝+两工程钉子; R33 八组深修(L_ST detach/W₂残差图/Sobolev精确Gram/投影末位/bake-off全投影器/RMSNorm/S_order撤阻断/非劣效/采样三错/口径)。 **本文件 = 干净权威方法主体, 头脑风暴在此改。** 问题/雷区/先发威胁/钉子日志 → novelty_study_v4.md。
口径: 用户主导设计, 我做方法学守门。 **curve-annotation-free / curve-supervision-free**(无外部曲线输入/标注/监督/Cobb-loss/change-map/分割伪标签; 内部 x_c(y) 仅术前图像估的无监督 soft axial corridor, 非绝对 curve-free)。 **举证逻辑: 不预先写满, 用可降级 claim ladder, 措辞+模型命名由反事实实验挣得。**

## 0. 问题定义 (orthopedic-from-start)
从术前正位脊柱 X 光预测术后正位脊柱 X 光; 协调多节段形变(非任意局部编辑)。 核心: 术前条件随生成区间变化时, 只沿脊柱头尾有序链协调传播, 同时保留患者局部骨性特征。
分工: MeanFlow=当前区间应完成多少影像变化; SC-PGA=条件变化如何沿有序脊柱区域协调传播。
统一叙事: 有限区间割线 D_{r,t}[f]=(f(t)-f(r))/(t-r) 同驱 影像 u*=D[z]; 条件势函数 c̄=D[P]。

## 1. 三个核心贡献 (已对齐先发占位)
**让功对象(锁定)**: FMM(2406.07507, two-time flow map+可变步数, 统一 consistency); SplitMeanFlow(2507.16884, 定积分加法→interval-splitting 一致性, MeanFlow 微分恒等式=无穷小分割极限); LBM(2503.07535, ICCV25, latent 单步快速 I2I)。
1. **Source-anchored compositional MeanFlow (系统化改造, 非理论发现)**: 任意区间平均运输适配到患者术前锚定的配对骨科预测, 与同区间势函数条件摘要+脊柱链受限动态条件**共同训练**。 不 claim "首次可组合区间/首次发现分区一致性"; L_comp/L_roll = 采用 SplitMeanFlow 速度侧一致性(让功); 可变步数归 FMM/SplitMeanFlow。
2. **Spinal-chain potential conditioning (条件侧构造)**: 势函数把时间变化条件转为**与平均运输共享区间索引的条件割线**, 具精确加权分区加法+局部极限。 **构造性质非新积分定理**; 与 SplitMeanFlow 关系写 "inspired by the same finite-interval secant algebra", **不写 dual formulation**。
3. **Bandlimited partition-sensitive adaptation (最强骨科, 分级)**: 随 flow interval 变化的条件模态硬限有序脊柱区低图频, 正交局部残差保持静态患者条件。 **v1(固定 path-graph)≈一维 DCT, 不 claim 图谱新意, 只 claim ordered-token bandlimiting**; patient-specific chain 只属 v2 且须打过 DCT(见 §4)。
剩余独有空间 = interval-indexed condition potential + dynamic-only ordered/patient-specific spectral restriction + orthopedic paired task。

## 2. 方法实施

### 2.1 几何保持 AE (前置, 承重墙)
单一共享 image-only 几何保持 AE(弱正则 RAE), MeanFlow 前冻结。 L_rec+L_grad **+ 潜路径平滑正则 L_smooth(见 §6, R30: 用解码加速度平滑替代 blend-target, 防训练鬼影)**。 发车前过 **P3 latent 桥体检(§6)**。

### 2.2 Source-Anchored Compositional MeanFlow
z_pre=E(x_pre), z_post=E(x_post), δ=z_post-z_pre。 z_t=z_pre+α_γ(t)δ+σ_m sin²(πt)ε, t∈[0,1], z_1=z_pre, z_0=z_post。
α_γ(t)=(e^{γ(1-t)}-1)/(e^γ-1): α(0)=1,α(1)=0; γ 固定。 σ=σ_m sin²(πt): σ(0)=σ(1)=σ̇(0)=σ̇(1)=0; σ_s=0。 v*_t=α̇_γ(t)δ+σ_m π sin(2πt)ε。
单 u 头(删 v 头) u_θ(z_t,r,t|h^SC_{r,t}):
- L_span=|u_θ-(z_t-z_r)/(t-r)|² (同病例同ε同轨 r<t) 主监督。
- L_ST(**R33-1 detach 目标, 硬阻塞#1**): q_t=sg(v*_t-(t-r)D_t u_θ), L_ST=|u_θ-q_t|²; D_t u_θ 由 F(t)=u_θ(z_t,r,t|h^SC_{r,t}) 一次前向 JVP 得, **目标 detach** → 反向不含混合二阶导(“单次 JVP”只描述前向调用数, 不代表一阶训练成本)。 论文写 "one forward-mode JVP with a detached JVP-derived target"。 消融: no-ST / ST-detach(主) / ST-full-gradient(小规模成本对照)。 λ_ST 小权重。
- 组合(只压图像, 采用 SplitMeanFlow): T_{t→r}(z_t)=z_t-(t-r)u_θ; L_comp=|T_{s→r}(T_{t→s}(z_t))-sg(T^{EMA}_{t→r}(z_t))|₁; L_roll=|T_{s→r}(T_{t→s}(z_t))-z_r|₁。 **R32 微裂缝四(comp↔roll 拉扯)**: 早期单步 T_{t→r} 糊图, L_comp 把组合路径往糊图拽, 与 L_roll 拉向真值 z_r 冲突(梯度夹角>90°震荡)。 修: (1) L_comp 目标用 **EMA teacher** T^{EMA} 非 live sg; (2) λ_comp 从 0 ramp(S4 进场); (3) 早期靠 L_roll(z_r 沿规定路径解析可得)主导, L_comp 渐入。

### 2.3 SC-PGA (主条件机制, Base PGA=Π=I 退化)
**(a) 有序轴向脊柱 token (无标注)**: 冻结 AE 特征 F_pre; J 个纵向有序软区域 μ_1<…<μ_J(J=8/12, 非椎体)。 π_jhw ∝ exp( cos(q_j,W_f F_hw) - β(y_h-μ_j)² - η(x_w-x_c(y_h))² )。 内容项 cosine 归一化(**降低对特征幅值捷径的敏感; 注: 不能排除方向相似的金属/文字伪影**); 横向中心=弯曲中线 x_c(y_h)(**≥三阶多项式**, 极小 head/显著图无监督; η 中等, 跟随顶椎与留脊柱解耦; **R32 微裂缝三: 低阶在重度 S 弯左右抵消→中线穿纵隔, 故须 ≥cubic 表达一个拐点 + Huber/RANSAC 鲁棒; 不退化为纯逐带局部中心否则约束失牙**); 纵向高斯保序。 命名 ordered axial spinal-region tokens。 得 B=[b_1..b_J]。

**(b) 链图+带限投影 Π_spine (三版, 钉子1 梯度断路修正)**
> 不可同立 {可学 W, eig 全 stop-grad, 图端到端学}: 对 Π 整体 stop-grad ⟹ ∂L/∂W=0。
- **v1(稳定默认)**: 通用固定 path-graph, L_path 特征向量解析(DCT 类)预计算, Π_spine=U_low U_low^T(前 K_g)。 无 eig, 患者无关, 只 g_k 可学; 仅 claim ordered-token bandlimited。
- **v2-Frozen(正式主候选)**: **图构造零可训练参数**, eig 每患者确定性预计算(可缓存), Π_spine 硬 rank-K, 只 g_k 可学。 诚实名 patient-specific feature-weighted spinal chain, **非 learned graph metric**。 与 v1 唯一差别=Π 来源 → **G_graph 干净 A/B**。
  - **边权构造(R33-2 残差W₂, 硬阻塞#2)**: 禁高维 cosine + 禁 JS(支撑脱节 JS≡ln2 坍缩); 且**裸 2D W₂ 仍可能退化**——π_j 由不同 μ_j 约束, W₂ 主要测固定纵向带间距, 患者特异横移只是小扰动→w≈常数 v2 再退 v1。 **改去名义轴距的残差距离**: m̃_j=[x̄_j/s_x, (ȳ_j-μ_j)/s_y], d²_j=||m̃_{j+1}-m̃_j||²+λ_Σ d²_Bures(Σ̃_j,Σ̃_{j+1}), **w_{j,j+1}=w_min+(1-w_min)exp(-d²_j/τ), w_min>0**(防 apex 处链被切断)。 只反映横向弯曲/相对局部偏移/形状变化, 不反映固定头尾轴距。 **方向性预注册**(否则 v2 失败无法区分图无效 vs 边权语义取反): distance-decay / uniform / bounded-decay / 反向或置信度型权重。
- **v2-Learned(后续, 非第一篇)**: W 真可学 → 软谱滤波 Π_β=exp(-βL) 或多项式 Σa_m T_m(L̃), 梯度穿 W; **代价 放弃严格 rank-K 硬投影(变 soft low-pass)**。
实现一律用投影器 Π_spine(基不变), 写 Π_spine g_k(B), 不用 U_low a_k。 **第一篇主线=v2-Frozen**。

**(c) 条件场 (动态硬限低图频)**: c(τ)=B+A_0(B)+Σ_{k=1}^{K_t} ℓ_k(τ) Π_spine g_k(B), **K_t=2 固定(R33-3, 不再写2~3)**; ℓ_k=shifted Legendre 零均值。 B_chain=Π_spine B, B_local=(I-Π_spine)B; 静态=患者身份/局部骨性/全跨度, 动态=子区间调节。 硬性质 (I-Π_spine)∂c/∂τ=0; TV 界(R33-7 索引修正) tr(C_dyn^T L C_dyn) ≤ λ_{K_g-1}||C_dyn||²_F(谱索引 0..K_g-1)。 **关键区分(vs FGDM/低通)**: 只有随区间变化的条件自由度被限低图频; **静态患者条件不受限**。 **R32 微裂缝二(时间鞭打)**: Π_spine 只限空间(图谱)带宽未限时间 τ 剧烈度→大 g_k 致条件沿 τ 大幅摆动(K_t=2 抛物线单次激越; 真双向鞭打需 K_t≥3, 但幅度问题 K_t=2 已在)。 补**时间 Sobolev 正则(R33-3 精确 Gram, 硬阻塞#3)**: 导数 Gram R_kl=∫_0^1 ℓ'_k ℓ'_l dτ, A_k=Π_spine g_k(B), L_time=Σ_{k,l} R_kl⟨A_k,A_l⟩_F = **精确** ∫_0^1|∂_τ c_dyn|²(我 R32 的 Σk(k+1) 比例对但系数错; 导数基一般不正交故须 Gram)。 K_t=2: R=diag(4,12) → L_time=4||A_1||²+12||A_2||²。 与空间 TV 界对称。

**(d) 条件势函数+趋势窗 (钉子2 解析除差)**: P(t)=∫_0^t c; c̄_{r,t}=(P(t)-P(r))/(t-r) **用对多项式解析可约除差闭式(可约奇点解析消, fp32, Δ_min 仅安全带)**。 嵌套自动: 精确加法/局部极限/dc̄/dt=(c_t-c̄)/Δ。 全跨度 c̄_{0,1}=B_local+B_chain+A_0。 趋势 d^c_{r,t}=(1-Δ)(c(t)-c(r))/Δ → d^c_{0,1}=0; Δ→0→ċ(p)。 h^SC_{r,t}=[c̄_{r,t}, d^c_{r,t}, e(p,Δ)]; h_{0,1}=[c_static,0,e(1/2,1)]。

**(e) 单一注入路 — 投影末位 (R33-4 硬阻塞#4, 公平, 无 retain/couple)**: 硬带限须传到实际调制——M_ψ 非线性后 M_ψ(Πx)∉range(Π), 故须 **projection-last**: m_dyn=Π_spine M_dyn(c̄_dyn,d_dyn,e), m_static=M_static(c_static,e), m=m_static+m_dyn→(shift,scale,gate)→AdaLN ⟹ 严格 (I-Π_spine)m_dyn=0。 这是**同一注入器内的静态-动态分解**(非恢复 retain/couple 双注入)。 J 个 ordered token 经**固定轴向插值**作用到 2D latent patch(axially conditioned AdaLN), **不全局平均**(否则 ordered-chain 局部意义被削)。

### 2.4 总目标
L=L_span+λ_end L_end+λ_ST L_ST+λ_comp L_comp+λ_roll L_roll; L_end=|T_{1→0}(z_pre)-z_post|₁(latent, 概率ρ整区间)。 端监督仅 latent。 无: 曲线/Cobb-loss/change-map/L_cond-mean/cocycle/Γ/w_ψ/v 头/retain-couple。

### 2.5 推理 + 1-NFE Trade-off Guard (R30, 漏洞1)
T_{1→0}=1-NFE; few-step=T 组合(2/4-NFE)。 静态链条件+A_0 服务 1-NFE; 动态低图频模态服务 few-step。
**1-NFE Accuracy-Speed Trade-off Guard(预注册)**: SC-PGA 对 1-NFE **不承诺提升, 仅承诺不显著退化**(退化边界 ≤ Base PGA 1-NFE 的 95% CI; 动态分支全跨度为 0, 仅防附加计算/归一化统计扰动静态流)。 **核心收益锚定 2/4-NFE 组合自洽性 E_DC**。 **R33-5 修正**: 不能说"测试时关动态模态即退回 Base"——动态模态即使 full-span descriptor 为零, 训练中仍经共享参数间接影响; Base PGA 必须**独立训练**对照。 诚实表述: "非常数模态对 full-span descriptor 无直接推理贡献; 任何 1-NFE 差异均属间接训练效应"。

## 3. 实验矩阵 (新意=判据; 钉子2 + R30 缺陷1)

### 3.1 主竞争 = J×J 算子 bake-off (同 J token/同秩 K_g/同 adapter/同 AdaLN, 仅换 Π)
主门 G_graph 统一为 **rank-K_g 正交投影器**(Π^T=Π, Π²=Π, rank=K_g; 满秩 soft 滤波不进主门, R33-二):
| 投影器 | range(Π) | 作用 |
|---|---|---|
| Identity (Base PGA) | I 全通 | 上界/是否需限制 |
| Random-subspace | 随机正交子空间 | G_struct 对照 |
| DCT | 前 K_g 个 DCT 模态 | **v1 真身**, G_graph 关键对照 |
| Gaussian-subspace | Gaussian 核 top-K_g 特征空间 | 平滑子空间 |
| Toeplitz-subspace | 冻结 Toeplitz top-K_g 特征空间 | 平移不变子空间 |
| SC-v1-path | path-Laplacian top-K_g | 应≈DCT(预期 sanity, 不指望打过) |
| SC-v2-frozen | patient-weighted Laplacian top-K_g(W₂残差边权) | **唯一能 claim patient-specific chain** |

补充 soft-filter 组(**不进 G_graph 主门**): 普通 Token-Gaussian 平滑 / 联合学习 Toeplitz(满秩, 仅软滤波对照)。

### 3.2 公平协议 (R30 缺陷1: 禁能量重缩放, 改共享 Norm)
**禁止全局 RMS 标量重缩放 a_Π**(per-operator 固定标量=有效学习率混淆+重放大低频→抵消带限)。 **R33-三 修正 Norm 定义与顺序**: "无 affine LayerNorm 纯除标准差"自相矛盾(LayerNorm 要减均值)——改 **无 affine、沿特征通道 RMSNorm**, 顺序 **Norm→Π**: g̃_k=RMSNorm_d(g_k(B)), A_k=Π g̃_k。 **投影后绝不再做 tokenwise Norm**(否则抵消投影能量衰减/重生高图频/破坏硬子空间)。 全算子共享同一 RMSNorm。 其余公平: 匹配 rank/adapter 参数量/训练预算。

### 3.3 频谱思想边界对照 (降级, 非主对手)
Gaussian-2D/Fourier-FFT/Wavelet/FGDM-inspired(动态限 2D 图像低频): 仅"图谱 vs 图像谱"边界。

## 4. 诚实闸门 → 统计判据 + claim/命名阶梯 (钉子3 + C + R30)
### 4.1 主机制指标=E_DC; 外部效用单列
E_DC=||T_{1→0}-T_{1/2→0}∘T_{1→1/2}||(+E_NFE, 1/2/4-NFE 方差)=机制主指标。 **R33-7 主门用相对误差**: E_DC^rel=||z_0^dir-z_0^comp||_1 / max(||z_post-z_pre||_1, q_10), 同时报 absolute E_DC; bootstrap **同时重采训练 seed 与患者**。 **E_DC 只测自洽可被 gaming → 必配独立结构指标(Cobb/CR 或 endpoint structural error)方向一致, 不混门**。
### 4.2 三道门 (统计)
- **G_struct 拆 v1/v2(R33-7)**: G_struct^v1=E_DC^Random-E_DC^v1; G_struct^v2=E_DC^Random-E_DC^v2。 G_graph=min_{b∈{DCT,Gaussian-sub,Toeplitz-sub,v1}}E_DC^b-E_DC^v2(取min保守, 全为 rank-K_g 投影器)。 S_order=E_DC^permuted-E_DC^correct。
- 通过=**LCB_95%(G)>δ_min**(非仅 p<0.05; δ_min 由 pilot 方差/可接受相对改善定); 第二层强制 ≥1 独立结构指标方向一致。
- **1-NFE 非劣效检验(R33-5, 改配对)**: 患者级配对差 Δ_1NFE=E^SC_1NFE-E^Base_1NFE(**Base 独立训练**), 预注册非劣效界 m_NI, 通过=**UCB_95%(Δ_1NFE)<m_NI**(非"≤Base 自身 CI")。
### 4.3 claim+命名阶梯 (预注册; 名字随门变)
| 门结果 | 命名/claim |
|---|---|
| S_order≈0 | 禁用 ordered/orthopedic mechanism claim |
| G_struct 失败 | SC 移出主方法/仅负结果 |
| G_struct✓ G_graph✗ S_order✓ | **ordered-token bandlimited PGA**(v1≈DCT 为预期 sanity 非失败) |
| 三门✓ | **patient-specific spinal-chain spectral conditioning**(满档) |
### 4.4 S_order 提纯 (R30 漏洞2: 只乱拓扑且阻断绝对位置泄漏)
**R33-4 撤销 R30/R32 阻断绝对位置的决定。** 主 S_order = **只换投影器** Π→Q^T Π Q, 同时 **B/μ_j/位置编码/tokenizer/adapter 全不变**(真正 topology-only 反事实)。 若模型经绝对位置等绕过错误拓扑致 S_order≈0, 这**正说明链拓扑非 load-bearing**——不得人为删绕过路径把 S_order 做大("同步置换 μ_j"会改 token 空间含义, 不再 topology-only)。 分层: **主 S_order=只换 projector(最纯=测试时 projector intervention)**; 辅助 sensitivity=去绝对位置后再测; 次级=从头训错误拓扑; 内容 shuffle=补充压力测试。

## 5. Related Work 防守靶 (两层)
- **Tier-1 主防守**: FMM(two-time map+可变步数) · SplitMeanFlow(速度侧 interval-splitting algebra) · consistency/任意区间 flow-map · LBM(latent fast I2I, +I2SB/DSB) · time-varying/interval-conditioned adapter · **ordered-token 一维低通(=Token-DCT 基线, 最危险近邻)**。 让功固定: FMM→two-time+变步; SplitMeanFlow→速度侧区间代数; LBM→latent 快速 I2I。
- **Tier-2 频谱边界**: FGDM(图像 Fourier 零样本翻译) · DRIFT-Net(PDE 神经算子图像-频谱双分支) · GF-NODE(分子图态谱域连续演化)。

## 6. 预注册硬门 (发车前必过)
- **P3 (雷区一, 发车前置硬门) — R30 修正版**:
  - **方法门已定稿 (R31, 用户授权自主判断): 采用下方 L_smooth, 弃用 blend-target。** 被弃原式留痕: L_interp=E_t‖D((1-t)z_pre+t z_post) - ((1-t)x_pre+t x_post)‖₁ 加 AE 预训练。 方法门反对: 目标 (1-t)x_pre+t x_post 对未配准图(NCC≈0.40, CASE2 preop 裁切)=alpha 混合=鬼影; 监督 decoder 复现 = 训练 AE 在中间 t 产鬼影, 与雷区一防鬼影正相反(ACAI 共识: naive blend-target 须避免)。
  - **关键事实**: 推理 few-NFE **不解码中间态**(端监督仅 latent, 输出只在 t=0=z_post) → 不严格需可信中间图。
  - **定稿采用(R33-7 正式)**: L_smooth=E_{t∈[h,1-h]}‖(D(z_{t+h})-2D(z_t)+D(z_{t-h}))/h²‖_Charb (**decoded-path curvature regularization**; **不声称"保证匀速几何形变"**——像素二阶差小也可由模糊实现), **AE 预训练阶段**加。 P3 判据=端点重建好 + 解码路径平滑有界无撕裂(**非**"单一可信脊柱")。 Cobb>75° 仍撕裂 → 学习型 bridge(弃解析 α_γ, 学 ODE 路径)。
  - **数值安全边界(R32 工程钉子1, 预注册)**: λ_smooth **pilot 候选区间 0.01~0.05(pilot 后冻结单一值; 不同损失归一化下该数无普遍意义)**, 比 L_rec 小 2 数量级防全局线性化损端点重建; AE 阶段 z_t=(1-t)z_pre+t z_post 线性插值, **采样 t∈[0.1,0.9] 禁近 0/1**(防外推崩溃), Δ 固定小值(如0.1); 出模糊**直接关 L_smooth 退标准 AE**(不伤 MeanFlow, L_span 仍正则潜速度)。
  - (历史备选) 日后若改回 blend-target 须重评训练鬼影风险; 当前定稿=L_smooth。
- **fp32 解析除差单元测试(雷区二)**: 已知多项式核验除差闭式=解析值, Δ∈[1e-4,1] 无 NaN, 相对误差<1e-5。
- **distractor 鲁棒性+弯曲走廊消融(雷区三)**: 合成肺结节/金属伪影测 token CoM 偏离脊柱率; 全局 x_c vs x_c(y_h)。

## 7. 诚实边界 (不变)
不解决不可辨识性(术后依赖未观测手术方案; 端点确定⟹每患者唯一输出; 去曲线后暴露更大)。 同区间时间平均条件预测区间平均速度=设计非 identity 必然; 矩=零阶平均+一阶趋势; L_ST=样本路径正则非边缘场恒等式; 非常数模态不改善 full-span 1-NFE(故 §2.5 Guard)。 临床措辞克制(weak orthopedic prior for multilevel structural coupling; 非生物力学/非椎体运动/非手术传播)。 UQ(σ_s>0)与 agent 待方法闭环+条件离散度审计后。

## 8. Stage 流程 + 受影响文件
S1: 图像 canonicalization(相似变换, robust 主轴+共同可见区间, CASE2) → 训冻几何保持 AE(+L_smooth) → **P3 体检**。 S2: L_span+λ_end L_end。 S3: +L_ST。 S4: +L_comp+L_roll。 S5: bake-off(3.1)+三门(4)。 **区间采样分布(R33-6 修正三错, 写死抄进 config)**: ① **先采 Δ 再采 r~U(0,1-Δ)**(原 r+Δ 会 t>1); ② **L_span 须覆盖宽跨度**(原只小Δ 削弱 2/4-NFE): 40% local 小Δ(Δ~Gamma(2,0.15)截断<0.3, 主供 ST)/40% broad Δ~U(Δ_min,1)或Beta/20% full-span Δ=1; L_ST 主用 local; ③ **三元组防极小子区间**: t-r≥2Δ_min, s~U(r+Δ_min, t-Δ_min); ④ L_end=**以概率 ρ=0.25 启用 full-span(0,1) endpoint loss**(L_end 定义即 full-span, 无"随机区间 L_end")。
文件: models/dit.py(FGA→SC-PGA: 有序token+Π_spine(v1/v2-Frozen, W₂残差边权)+势函数+Legendre+趋势窗→共享Norm→AdaLN); meanflow.py(source-anchored 路径+5损失+单JVP L_ST, 删v头); mydataset.py(跨扩展名 stem+240×480, 不载曲线); config.yaml(γ/σ_m/K_t/K_g/J/β/η/τ/λ_*/区间采样/Δ_min/算子开关); 新增 image-only AE(+L_smooth) + canonicalization 工具 + x_c(y) 中线 head + bake-off 算子库 + 共享RMSNorm(Norm→Π) + 投影末位注入器 + 轴向条件AdaLN。

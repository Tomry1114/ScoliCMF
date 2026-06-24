# novelty_study_v4 (FINAL, curve-free) — Source-Anchored Compositional MeanFlow + Potential Gating Adapter (PGA)
日期 2026-06-22。 用户主导设计, 我做方法学守门。 **曲线已整体移除**(无曲线输入/监督/Cobb-loss/change-map/分割伪标签/手术动作变量)。

## 统一叙事
有限区间割线算子 D_{r,t}[f]=(f(t)-f(r))/(t-r) 同时驱动两路: 影像学割线 u*_{r,t}=D[z]; 条件势函数割线 c̄_{r,t}=D[P]。
MeanFlow 管"区间内走多远(平均速度)"; PGA 管"该区间整体用什么条件 + 条件朝哪变(零阶平均+一阶趋势)"。
诚实边界(写论文必须守): 以同区间时间平均条件预测区间平均速度, 是有限区间一致性"设计", 不是 MeanFlow identity 必然推出; 矩=零阶平均+一阶趋势, 非精确前两阶矩; L_ST 是样本路径割线-切线正则, 非边缘场严格恒等式; 非常数模态作用于 few-step/composed, 不改善 full-span 1-NFE conditioning(1-NFE 若改善只能归因 source-anchored path / 全局 A0 / ST 稳定性)。

## 1. Source-Anchored 路径 (latent)
z_pre=E(x_pre), z_post=E(x_post), δ=z_post-z_pre。
z_t=z_pre+α_γ(t)δ+σ_m sin^2(πt)ε,  t∈[0,1], z_1=z_pre, z_0=z_post。
α_γ(t)=(e^{γ(1-t)}-1)/(e^γ-1): α(0)=1,α(1)=0, 两端导数≠0; σ=σ_m sin^2(πt): σ(0)=σ(1)=0 且 σ̇(0)=σ̇(1)=0 → 端点速度纯结构、无噪声。 γ 第一版固定, 不可学。
瞬时样本速度 v*_t=α̇_γ(t)δ+σ_m π sin(2πt)ε。

## 2. PGA (替换原 FGA 的 sigmoid 通道门)
术前条件 token c_pre=E_c(x_pre)(保留空间 token, 不全局池化)。
瞬时条件场(正交时间模态, shifted Legendre 零均值 ∫_0^1 ℓ_k=0, k≥1):
  c(τ)=c_pre + A_0(c_pre) + Σ_{k=1}^K A_k(c_pre) ℓ_k(τ),  ℓ_1=2τ-1, ℓ_2=6τ^2-6τ+1, ...  A_k 低秩 adapter, K=2~3。
势函数 P(t)=∫_0^t c(τ)dτ; 区间平均条件 c̄_{r,t}=D_{r,t}[P]=(P(t)-P(r))/(t-r)。
全跨度严格 c̄_{0,1}=c_pre+A_0 (Legendre 零均值之故); A_0=全跨度静态适配, A_{k≥1}=分区敏感模态(仅子区间起效)。
区间条件描述符 h_{r,t}=[ c̄_{r,t}, (c_t-c_r)/(t-r), e(p,Δ) ], p=(r+t)/2, Δ=t-r。
  第二项=一阶趋势 =ċ(p)+O(Δ^2)(只用 P' 端点, 不建二阶原函数); 设 Δ_min, Δ小时用 ċ(t) 代差商; 两条件分支各 LayerNorm; 网络显式收 Δ。
嵌套性质(自动): 势函数端点差 ⟹ 精确区间加法 (t-r)c̄_{r,t}=(s-r)c̄_{r,s}+(t-s)c̄_{s,t} (故无需 L_cond-mean); 局部极限 lim_{r→t}c̄=c_t; 闭式导数 dc̄/dt=(c_t-c̄)/Δ。
消融轴: 静态(K=0, 含 A_0) vs 分区敏感(K≥1)。 实现: PGA 输出 P_ψ → 推 h_{r,t} → M_ψ(h) → AdaLN (shift,scale,gate)。 AdaLN 只是实现, 不是创新主体。

## 3. 单 u 头 + 割线-切线正则 (删 v 头)
u_θ(z_t,r,t|h_{r,t}); 主监督 L_span=|u_θ-(z_t-z_r)/(t-r)|_2^2 (同病例同ε同轨采 r<t)。
L_ST=|v*_t-[u_θ+(t-r)D_t u_θ]|_2^2; D_t u_θ 由对 F(t)=u_θ(z_t,r,t|h_{r,t}) 做单次 JVP 得到(自动含 z_t/c̄/趋势/time-emb/势参数全部导数 → 杜绝"训练一种条件、JVP 对另一种求导"偏差)。 λ_ST 小权重当正则。
(可选)左端点 L_ST-left 每 iter 随机选左/右, 控成本=一次 JVP。

## 4. 组合一致性(只压图像算子)
T_{t→r}(z_t)=z_t-(t-r)u_θ(z_t,r,t|h_{r,t})。
ẑ_r^dir=T_{t→r}(z_t); ẑ_s=T_{t→s}(z_t); ẑ_r^comp=T_{s→r}(ẑ_s)。
L_comp=|ẑ_r^comp-sg(ẑ_r^dir)|_1; L_roll=|ẑ_r^comp-z_r|_1 (真值锚防错路自洽)。

## 5. 总目标
L = L_span + λ_end L_end + λ_ST L_ST + λ_comp L_comp + λ_roll L_roll,
  L_end=|T_{1→0}(z_pre)-z_post|_1 (latent, 概率ρ强制整区间)。
无: 曲线/Cobb-loss/change-map/L_cond-mean/L_cond-cocycle/Γ/w_ψ/v 头。

## Headline 实验 & 成败判据
- Dist(T_{1→0}, T_{1/2→0}∘T_{1→1/2}) + 少步组合一致性 + NFE 鲁棒性(方法内指标)。
- MAKE-OR-BREAK 消融: 静态(K=0) vs 分区敏感(K≥1) — 分区矩到底买不买账(一行开关, 落地后先测)。
- Cobb/CR 当测试指标(需独立曲线提取器在生成图上量; GT 用数据术后曲线, 仅评测不进训练), 对着条件离散度下限解读。

## Stage 流程
S1: 图像内容 canonicalization(相似变换, 见 stage0_report CASE2) → 训冻 image-only 几何保持 AE(L_rec+L_grad) → P3 latent 桥体检(去中间态解码后门槛放松, 只需端点重建好+latent 大致线性)。
S2: L_span+λend L_end (验 MeanFlow 成立)。 S3: +L_ST。 S4: +L_comp+L_roll。 S5: 消融(K, 各损失)。
需定 (r,s,t) 区间采样分布(小Δ喂ST/局部, 整跨喂end, 三元组喂comp)。

## 受影响文件
models/dit.py(FGA→PGA: 势函数+Legendre模态+h_{r,t}→AdaLN); meanflow.py(source-anchored路径+5损失+单JVP的L_ST, 删v头); mydataset.py(跨扩展名stem+240x480, 不再载曲线); config.yaml(γ/σ_m/K/λ_*/区间采样/Δ_min); 新增 image-only AE + 图像 canonicalization 工具。

## 诚实边界(不变)
不解决不可辨识性(L_end 对单一已实现术后, 端点确定⟹每患者唯一输出, δ 依赖未观测手术方案); 去曲线后此暴露更大(失去"哪里该变"结构先验)。贡献钉在方法自洽/少步一致性; UQ(开 σ_s>0)与 agent 均待方法闭环+条件离散度审计后。

---

# 扩展 §6. SC-PGA — Spinal-Chain Potential Gating Adapter (可开关骨科扩展; 默认 off = base PGA)
定位: base PGA 之上的骨科归纳偏置, 经开关(链 on/off, K_g) **严格嵌套**到 base PGA。 新意 = 成败判据 = 同一判别性消融(链图谱 vs 图像频率)。 默认 off; 仅当消融证明其价值才作为贡献上 paper, 否则退回 base PGA。

## 三轴 FGDM 区分 (必须这么定位, 否则被 FGDM/TMI2023 合并)
FGDM 在图像 Fourier 域做频带分解+重建(低频=强度/语义, 高频=解剖细节, 并生成缺失中频)。SC-PGA 完全不同:
- (对象) 作用于**条件轨迹**, 不对输入图像做频带分解/重建;
- (约束) 限制的是条件**随 flow interval 变化的自由度**, 不是图像频带;
- (域) 带限子空间来自**患者特异有序链图谱**, 不是二维像素 Fourier 网格。

## 有序轴向脊柱 token (无标注)
冻结 AE 空间特征 F_pre 沿头尾轴软池化 → J 个有序 token B=[b_1..b_J](固定/轻学中心 μ_j + 横向内容注意力)。
命名严格用 ordered axial spine tokens / ordered spinal-region tokens, **不叫 vertebral tokens**(无椎体标注)。 只编码弱可靠先验 b_1→…→b_J 头侧到尾侧有序。
CASE 2 注: 术前被裁 ⟹ token 是 canonical 轴向带, 非解剖椎体; 顺序效用只能 claim **轴向有序**, 不能 claim 椎体对应。

## 链图 + 带限投影 Π_low
链图(仅相邻): A_{j,j+1}=exp(-||W b_j - W b_{j+1}||^2/τ), |i-j|>1 → 0; 归一化 L=I-D^{-1/2}AD^{-1/2}=UΛU^T。 取前 K_g 低图频 U_low, 投影 Π_low=U_low U_low^T, Π_local=I-Π_low。
- **v1(稳定, 默认)**: 固定 path-graph(无权/固定权), Laplacian 特征向量解析(DCT 类) ⟹ Π_low **无需 eig、完全稳定**。
- **v2(可学权, 消融)**: 特征相似边权图; eig 走 **stop-gradient**(每病例谱当固定基, 只对学习场反传), 规避近简并 eig 反传爆 + 特征向量符号/旋转歧义。
实现一律用**投影器 Π_low(基不变)**, 动态部分写 Π_low g_k(B), **不用 U_low a_k**(依赖具体基有歧义)。

## 条件场 (动态模态硬限低图频)
B_chain=Π_low B, B_local=Π_local B。
c(τ)=B_local + B_chain + A_0(B) + Σ_{k=1}^{K_t} ℓ_k(τ) Π_low g_k(B)。
B_local 始终静态(患者局部信息, 不可被区间动态改写); A_0=全跨度静态适配; ℓ_k=零均值 Legendre; 动态项 ∈ span(Π_low)。
**硬性质** Π_local ∂c/∂τ = 0。 **TV 界** c_dyn^T L c_dyn ≤ λ_{K_g}||c_dyn||^2 (c_dyn=Σℓ_k Π_low g_k) ⟹ 区间动态条件只能表达"沿相邻有序脊柱区平缓传播", 不能任意逐节段跳变。

## 势函数性质保留 (投影时间无关)
P(t)=∫_0^t c; c̄_{r,t}=(P(t)-P(r))/(t-r)。 加法/局部极限/dc̄/dt=(c_t-c̄)/Δ 全保留。 全跨度 Legendre 零均值 ⟹ c̄_{0,1}=B_local+B_chain+A_0。
定位: 1-NFE 用静态链条件+A_0; 2/4-NFE 与组合路径才用动态低图频模态; SC-PGA 主目标=改善 partition/组合一致性, **不单独声称提升 full-span 1-NFE**。

## 注入 (retain/couple 双支, 单一主路)
R_local^l=CrossAttn(H_t^l, B_local); R_chain^l=CrossAttn(H_t^l,[c̄^chain_{r,t}, d^{c,chain}_{r,t}])。
[G_retain,G_couple]=Softmax(f_l[H_t^l,R_local,R_chain,e(r,t)]); H_t^l += G_retain⊙R_local + G_couple⊙R_chain。
命名 retain/couple(**非 keep/change**, 无空间变化监督): retain=节段特异术前信息; couple=沿链协调传播的区间条件。
MeanFlow 输入 h_{r,t}=[c̄^chain_{r,t}, d^{c,chain}_{r,t}, B_local, e(r,t)]; 单次 JVP 对 F(t)=u_θ(z_t,r,t|h) 自动穿全部导数(投影器在反向是常量)。

## 嵌套
链 off 或 K_g=J ⟹ Π_low=I ⟹ 退化为 base PGA(严格嵌套开关)。

## 成败实验 (新意=判据)
Group1 频域竞争(**同 backbone 同参数量**, 仅动态模态作用空间不同): Base PGA(无结构限) / Gaussian-PGA / Fourier-PGA / Wavelet-PGA / GraphHeat-PGA / Bandlimited SC-PGA。 **必含 FGDM-inspired 基线 = 动态模态限到 2D 图像低频** → 唯一变量=图谱 vs 图像谱。
Group2 链真实性: correct vs **shuffled(最关键)** / ring / full / uniform-weight / dynamic-on-all-freq / low-freq-only-no-local。
指标三层: L1 区间法(E_DC=||T_{1→0}-T_{1/2→0}∘T_{1→1/2}||, E_NFE, 1/2/4NFE 方差); L2 骨科(Cobb/CR + 高曲率/长弯子集, **曲线仅测试**); L3 机制(R_low 硬投影下≈1 **仅 sanity 非机制证据**; **S_order=E_DC^shuffle-E_DC^correct, ≈0 则链先验无效 ← 真正机制证据**)。

## 必引/必区分
FGDM(IEEE TMI 2023, arXiv:2304.02742); DRIFT-Net(arXiv:2509.24868); GF-NODE(arXiv:2411.01600); Kadoury(MedIA 2011 + MICCAI 2018 arXiv:1806.02285); GraphHeat/AGC; ConsisID/MeDUET(arXiv:2602.17901)。

## 临床表述 (克制)
"Low graph-frequency modes encode smoothly coordinated variation across adjacent ordered spinal regions, serving as a weak orthopedic prior for multilevel structural coupling." 非生物力学 / 非已识别椎体运动 / 非手术作用传播模型。

## 边界
不解决不可辨识性(SC 约束的是条件随区间变化的自由度, 不改变"术后依赖未观测方案")。 默认 off, 仅当 Group1 显示链低图频打过图像频率低通 + S_order>0 时才上 paper。

---

# 决策 (2026-06-22): SC-PGA = 核心骨科机制 (COMMITTED, 作废 §6 "默认 off" 定位)
用户拍板: 这篇是骨科方法论文, SC-PGA 是核心机制, 不是可选扩展。
- 方法 = Source-Anchored Compositional MeanFlow + SC-PGA(骨科 identity)。
- base PGA / 图像频率变体(Gaussian/Fourier/Wavelet/GraphHeat/FGDM-inspired) 由"退路"改为**内部 baseline/ablation**, 作用=证明链结构 load-bearing(功劳钉在链结构而非通用平滑), 不再是"可能退回 base"。实验矩阵不变, 定位从 hedge → proof。
- **诚实闸门 = S_order**(乱序链): S_order>0 → 骨科 claim 成立(用了轴向有序耦合); S_order≈0 → 链先验未起作用, 此时不得坚持"骨科"措辞(避免 overclaim)。commit 的是方向与投入; 可写进 paper 的"骨科"强度由 S_order 裁决。
- 临床措辞仍守克制版(weak orthopedic prior for multilevel structural coupling; 非生物力学/非椎体运动/非手术传播)。

---

# 终版修订 (2026-06-22, orthopedic-from-start, 单一注入; 作废 §6 retain/couple 双支)
论文核心问题直接定为骨科版: "如何让术前条件随生成区间变化时, 只沿脊柱头尾有序链协调传播, 同时保留患者局部骨性特征?" 分工: MeanFlow=当前区间应完成多少影像变化; SC-PGA=这些条件变化如何沿有序脊柱区域协调传播。

## 主模型条件场 (SC-PGA 即主方法, Base PGA 只是 Π=I 退化)
c(τ)=B + A_0(B) + Σ_{k=1}^{K_t} ℓ_k(τ) Π_spine g_k(B);  动态项满足 (I-Π_spine) ∂c/∂τ = 0。
等价分解 c(τ)=B_local + B_chain + A_0(B) + Σ ℓ_k(τ) Π_spine g_k(B), B_chain=Π_spine B, B_local=(I-Π_spine)B。
静态部分(B_local+B_chain+A_0)=患者身份/局部骨性/全跨度条件/术前整体上下文; 动态部分=子区间条件调节, 沿头尾有序区域协调变化, one/few-step 划分适配。
关键: **只有"随区间变化"的条件自由度被限到有序脊柱链低图频; 静态患者条件不受限** —— 这是与 FGDM/普通 Fourier 低通的最重要区别。

## v1 链图 (固定 path-graph, 无 eig)
固定路径图 1-2-…-J, L_path=I-D^{-1/2}AD^{-1/2}, 特征向量解析(DCT 类)预计算, Π_spine=U_low U_low^T。
注意: v1 的 Π_spine **患者无关**(沿有序 token 轴低频), 患者特异只来自 B 与 g_k(B) → **v1 只 claim "ordered-token bandlimited", 不 claim "patient-specific chain"(=v2 特征权图)**。
v1 头号机制证据 = S_order(乱序), 因 v1 相对"有序语义 token 上的粗 1D 低通"的新意边际几乎全压在 S_order + token 语义。

## 有序轴向脊柱 token
冻结 AE 特征 F_pre; J 个纵向有序软区域 μ_1<…<μ_J; π_jhw ∝ exp(q_j^T W_f F_hw - β(y_h-μ_j)^2 - η(x_w-x_c)^2)。
纵向高斯保头尾序; 横向项=弱中央走廊先验(η 必须**小**, 否则重度侧弯顶椎跑出走廊 → 在高 Cobb 子集验 token 贴 apex); 内容注意力允许明显侧弯。 J=8 或 12(不对应真实椎体数)。 命名 ordered axial spinal-region tokens。

## 趋势窗 (保证动态只服务 few-step 严格成立)
d^c_{r,t}=(1-Δ)(c(t)-c(r))/Δ, Δ=t-r → d^c_{0,1}=0; Δ→0 时 →ċ(p)。
条件描述符 h_{r,t}=[c̄_{r,t}, (1-Δ)(c(t)-c(r))/Δ, e(p,Δ)], p=(r+t)/2。 全跨度 h_{0,1}=[c_static, 0, e(1/2,1)]。
势函数闭环不变(投影时间无关): 加法/局部极限/dc̄/dt=(c_t-c̄)/Δ 全保留。

## 单一注入路 (公平性, 作废双支)
h_{r,t} → M_ψ → (shift,scale,gate) → AdaLN。 与 Base PGA 唯一区别 = g_k(B) vs Π_spine g_k(B)。
消融完全公平: Base(Π=I)/Fourier(Π_FFT)/Wavelet(Π_wavelet)/Gaussian(2D平滑)/SC(Π_spine)。 **不再加 retain/couple 双支**(否则无法判断增益来自链结构还是额外注入器)。

## MeanFlow 主体不变
z_t=z_pre+α_γ(t)δ+σ_m sin^2(πt)ε; u_θ(z_t,r,t|h_{r,t}); L_span=|u_θ-(z_t-z_r)/(t-r)|_2^2; L_ST 对 F(t)=u_θ(z_t,r,t|h^SC_{r,t}) 单次 JVP(穿 latent/Legendre/势函数/c̄/趋势窗/链投影动态模态/time-emb); 组合损失只压图像算子。

## 三个核心贡献 (定稿)
1. Source-anchored compositional MeanFlow: 术前→术后表示为从患者术前 latent 出发的有限区间运输, 支持全跨度与子区间组合推理。(必要基础, 不过claim理论)
2. Spinal-chain potential conditioning: 无标注有序轴向脊柱条件表示, 经条件势函数产生具精确分区加法与局部极限的有限区间条件。
3. Bandlimited partition-sensitive adaptation: 把随 flow interval 变化的条件模态硬限在脊柱链低图频子空间, 正交局部残差保持静态患者条件 → 多节段协调变化的骨科归纳偏置。**(最强骨科贡献)**

---

# 雷区审查 (2026-06-24): 三处最易爆点 + 修法 + 预注册测试
用户提三处致命隐患, 方法门逐个判定。 结论: #1 最致命且无法靠路径参数化绕过(只能用 P3 裁), #2 仅朴素实现才真(势函数设计本就是解药), #3 真且威胁骨科 claim 诚实性(有清洁设计修法 + S_order 兜底)。

## 雷区一: 潜空间测地线幻觉 (Latent Linearity Trap) — 最致命
- 隐患: z_t=z_pre+α_γ(t)δ 是潜空间欧氏直线弦; 弱正则 AE 潜流形在大形变下非平坦 ⟹ 直线弦映回图像为非测地线; 高 Cobb(>75°)中间态 t=0.5 解码可能椎体"质壁分离"/半透明鬼影, 近 t=0 才突然凝固。
- 判定: **真, 且最致命。**
- 关键纠正: **α_γ(t) 治不了此病** —— 它是指数型时间重参, 只决定在 z_pre—z_post 直线弦上跑到哪/多快, 路线仍是直线段, 不朝流形弯; σ_m sin²(πt)ε 只把弦吹成各向同性管子, 同样不朝流形弯。 ⟹ 现有路径项任何参数化都绕不过, 只能被检验裁决。
- 修法 (升级为发车前置硬门 = P3):
  1. P3 必跑: 在 Cobb>75° 最坏子集上解码线性插值 (1-s)z_pre+s·z_post, 看是否单一可信脊柱。
  2. P3 通过 → 路径假设成立, 发车。
  3. P3 不通过 → 升级顺序: ① **AE 训练阶段直接加线性插值可信度正则**(让潜空间沿形变方向尽量平; 不需中间态 GT, 最该先试); ② 放弃"全程可解码", 只保端点质量 + 抬 NFE; ③ 最后才考虑学习型 bridge / 流形测地路径。
- 含义: "几何保持 AE" 是承重墙, 不可当配角随便冻; 选型/训练目标必须把"线性插值可信"写进去。

## 雷区二: 趋势项 Δ→0 数值深渊
- 隐患: d^c_{r,t}=(1-Δ)(c(t)-c(r))/Δ; Δ→0 时若做浮点有限差分, 分子微小差/分母近零 → 灾难性相消, BF16/FP16 下 NaN/Inf 头号嫌疑。
- 判定: **半真 —— 仅当朴素有限差分实现才成立。**
- 解药 (势函数设计本身): MeanFlow 的 JVP 是前向模式**解析方向导数, 非有限差分**; 深渊只在把 h 在两时间点算出再相减时出现。 正确实现:
  1. h^SC 作为 t 的**解析闭式**(Legendre + 势函数解析除差)喂进 JVP, 让 autodiff 取 dh/dt; c̄、d^c 全走多项式除差闭式 —— (P(t)-P(r))/(t-r) 对多项式 P 是解析可约的除差(可约奇点解析消掉), 不写浮点相减/除小数。
  2. 该小块计算强制 fp32(混合精度下 cast); Δ_min 仅做安全带。
- 落地这条 ⟹ 深渊不存在。 预注册: fp32 解析除差单元测试(对已知多项式核验除差闭式 = 解析值, Δ∈[1e-4,1] 无 NaN/相对误差<1e-5)。

## 雷区三: 软 Token 侧向漂移 (η 困境) — 威胁骨科诚实性
- 隐患: π_jhw ∝ exp(q_j^T W_f F_hw - β(y_h-μ_j)² - η(x_w-x_c)²); η 小则横向束缚垮 → 内容点积 q_j^T W_f F_hw 主导 ⟹ 重度侧弯+肺高密度结节/体外金属支架时, token 被高幅值特征勾引漂出脊柱, 锁死肋骨/病灶, "有序脊柱链"变"脊柱-肋骨随机漫步"。
- 判定: **真, 中等, 且威胁 contribution-3 骨科诚实性。**
- 根因: x_c 是单一**全局竖直**走廊 —— "跟随顶椎横移"与"别跑出脊柱"被强行耦合在一个 η 上。
- 修法 (解耦):
  1. x_c 从全局常数 → **随高度弯曲的中线 x_c(y_h)**(低阶多项式, 由极小 head 从 preop 预测, 或脊柱显著图逐行横向质心**无监督**估 —— 保住"无标注"招牌); η 保持中等, 约束中心随顶椎弯 ⟹ "跟随顶椎"与"留在脊柱"解耦。
  2. 内容项改 **cosine**(归一化 F, 杀肺结节/支架高幅值诱惑)。
  3. S_order 洗链检验保留为**探测器**(漂到肋骨 → 洗链不变 → 被抓), 但目标是**预防**而非仅事后检出。
- 预注册: distractor 鲁棒性测试(合成肺结节/金属伪影注入, 测 token CoM 偏离脊柱率) + 弯曲走廊消融(全局 x_c vs x_c(y))。

## 母题与门禁意见
三条共享母题: 方法优雅性假设了重病例+成像伪影会破坏的"平坦/光滑/局部"。 各钉一个预注册测试: #1→P3(发车前置硬门), #2→fp32 解析除差单元测试, #3→distractor 鲁棒性 + 弯曲走廊消融。 清洁方法主体 → doc/method_v1.md; 本文件 (novelty_study_v4.md) 自此为**问题/雷区/审查日志**。

---

# 先发占位威胁 + 三技术钉子 (2026-06-24, R29 审查日志)
prior-art scout 锁定让功对象 + 用户提三钉子。 清洁方法已迁 doc/method_v1.md(claim ladder 版); 本节为威胁/钉子留痕。

## 先发占位威胁 (让功对象, ID 锁定)
- **FMM (arXiv 2406.07507)**: 学底层 ODE 的 two-time flow map, 支持推理选 one/few-step, 统一多类 consistency。 ⟹ 贡献1 不得 claim "首次可组合区间运输/可变步数"。
- **SplitMeanFlow (arXiv 2507.16884)**: 定积分加法导出 interval-splitting 一致性, MeanFlow 微分恒等式=无穷小分割极限。 ⟹ **图像速度侧区间代数明确让功**; L_comp/L_roll = 采用其一致性, 不 claim 发现。
- **LBM (arXiv 2503.07535, ICCV2025)**: latent 单步快速 I2I。 ⟹ "latent bridge + fast I2I" 不能作主新意。
- **最大机制隐患 = ordered-token 一维低通(非 FGDM)**: 固定均匀 path-graph 低图频 ≈ 一维余弦谱; 审稿"这不就是沿纵轴一维低通?为何叫 spinal-chain graph?"。 ⟹ 收编为 Token-DCT 主基线并打过(或诚实退 v1 档)。 FGDM/DRIFT-Net/GF-NODE 降为"频谱思想边界"非主对手。

## 钉子1: v2 stop-gradient eig 梯度断路 (最重要剩余问题)
{可学边权 W + eig 全 stop-grad + 图端到端学} **三者不可同立**: 对 Π 整体 stop-grad ⟹ ∂L/∂W=0(经 Π 路径)。 修: 拆 **v2-Frozen**(冻结 AE 特征定边权, 图构造零可训练参数, 硬 rank-K 保留, 仅 g_k 可学; 名 patient-specific feature-weighted chain, **非 learned graph metric**)与 **v2-Learned**(软谱滤波 exp(-βL)/多项式, 梯度穿 W 但变 soft low-pass, 放弃硬 rank-K)。 **第一篇主线 = v2-Frozen**。 已写入 method_v1 §2.3(b)。

## 钉子2: Energy-matched = 公平协议非模型行
"范数/秩/参数量匹配"无唯一 J×J 算子。 改为统一协议: 所有低秩算子匹配 rank/adapter 参数/动态输出 RMS/训练预算; 标量校准 a_Π=sqrt(E||Y_ref||²/(E||Y_Π||²+ε)), **训练集先验固定非逐样本**。 主表保留 7 行(Identity/Random/Token-DCT/Token-Gaussian/Learned-Toeplitz/v1/v2-frozen)。 已写入 method_v1 §3.2。

## 钉子3: 三门 符号→统计判据
主机制指标=E_DC(自洽, 可被 gaming → 必配独立结构指标 Cobb/CR/endpoint structural error 方向一致, 不混门)。 G_struct=E_DC^Random-E_DC^SC; G_graph=min_{b∈{DCT,Gaussian,Toeplitz,v1}}E_DC^b-E_DC^v2(取min故意保守); S_order=E_DC^permuted-E_DC^correct。 通过=**LCB_95%(G)>δ_min**(非仅 p<0.05)。 **模型命名随门变**(见 method_v1 §4.3)。 S_order 提纯: 主版 Π_perm=Q^T Π Q 只乱拓扑保内容/位置/rank/谱能量; 分 train-from-scratch 与 test-time 两协议; 内容 shuffle 降补充压力测试。 已写入 method_v1 §4。

## A–D 评价收口
A 正确(镜像仅叙事, 不写 dual formulation); B 决定性(v1≈DCT 是预期 sanity, 不指望打过 → 写"近乎等价"更可信); C 最成熟(名字本身随门变); D 正确(三让功位置固定; 剩余独有空间=interval-indexed condition potential + dynamic-only ordered/patient-specific spectral restriction + orthopedic paired task)。

---

# R30 审查 (2026-06-24): 两致命缺陷 + 两叙事漏洞 + 一工程雷点
用户提五项; 方法门四接受一部分反对。 清洁版已迁 method_v1。

## 缺陷1 (真): Bake-off 能量匹配逻辑自杀
a_Π=sqrt(E||Y_ref||²/E||Y_Π||²) 把 Πg_k(B) 重缩放回 Base RMS = 抵消带限物理意义(把残存低频振幅拔高, 频谱扭曲); 且 per-operator 固定标量 → 各组有效学习率不同, G_graph 不可信。 修: **禁全局 RMS 重缩放; 改 g_k(B) 后接无 affine LayerNorm(纯除标准差), 全算子(含 Identity)共享** → Π 能量衰减真实穿前向, 隔离"哪些模态被保留"非幅度。 已入 method_v1 §3.2。

## 缺陷2 (真): v2-Frozen 特征图被噪声主导
原始高维 AE 特征 cosine → 噪声/软组织/金属主导边权, v2 退化为一维随机图, G_graph 必败。 修: **边权改 相邻 token 空间注意力图 JS 散度(感受野重叠) w=exp(-JS/τ)** → 患者特异=空间重叠非纹理, 天然无噪, 合轴向有序解剖区直觉; 备选 空间池化+逐样本 PCA 白化(前8主成分)。 已入 method_v1 §2.3(b)。

## 漏洞1 (真): 1-NFE 必赢叙事陷阱
动态模态对 1-NFE 中立甚至有害(附加计算/归一化扰动静态流); 若 SC 1-NFE<Base 审稿质疑"为何复杂结构"。 修: **§2.5 加 1-NFE Trade-off Guard** —— 不承诺提升只承诺不显著退化(≤Base 95%CI), 收益锚 2/4-NFE E_DC, 可关动态退回 Base → "不改善"=无害可插拔。 §4.2 加 1-NFE 非退化 Guard。 已入 method_v1。

## 漏洞2 (真, 纠正上版错): S_order 绝对位置泄漏
上版 R29 竟写"保绝对位置编码不变"=正是泄漏口: Π_perm=Q^T Π Q 乱拓扑后, 模型用固定 μ_j 绝对位置绕过错误邻接恢复顺序。 修: **去正弦绝对位置编码只留相对序, 或置换时同步置换 μ_j**。 已入 method_v1 §4.4。

## 雷点 (方法门部分反对): P3 线性插值正则
用户原式 L_interp 目标=((1-t)x_pre+t x_post)=未配准图 alpha 混合=**鬼影**; 监督 decoder 复现=训练 AE 产鬼影, 与雷区一防鬼影**正相反**(ACAI 共识)。 关键: few-NFE **不解码中间态** → 不需可信中间图。 **方法门默认改 L_smooth=E_t||D(z_{t+Δ})-2D(z_t)+D(z_{t-Δ})||₁(解码加速度平滑)**, P3 判据=端点重建好+解码平滑有界无撕裂(非单一可信脊柱); Cobb>75° 仍撕裂→学习型 bridge。 **方法门定稿 L_smooth (R31, 用户已授权自主判断); blend-target 留痕为被弃备选**。 已入 method_v1 §6(含两版留痕)。

---

# R32 审查 (2026-06-24): 四微裂缝 + 两工程钉子
方法门逐条核实(微裂缝一为数学铁证, 二/三精修表述/fix)。 清洁版已入 method_v1。

## 微裂缝一 (真, 数学铁证=WGAN 动机): JS 零重叠饱和坍缩
JS(P‖Q) 在支撑集不重叠时 ≡ln2≈0.693 死常数。 token 高斯方差小→相邻 π_j 支撑脱节→所有边权=exp(-ln2/τ) 同一常数→v2 患者特异图坍缩为均匀链=退回 v1。 修: 弃 JS, 改 π_j 近似 2D 高斯 + 闭式 2-Wasserstein W₂, w=exp(-W₂²/τ)(不饱和, 随顶椎侧移平滑增长, 直脊柱≈均匀链/侧弯分化=患者特异)。 入 method_v1 §2.3(b)。

## 微裂缝二 (真, 表述精修): 时间轴勒让德鞭打
Π_spine 限空间图谱带宽, 未限时间 τ; 大 g_k → 条件沿 τ 大幅摆动(空间不乱扭但时间灾难鞭打)。 方法门精修: K_t=2 抛物线只能单次激越, 真双向鞭打需 K_t≥3; 但幅度问题 K_t=2 已在。 修: 补时间 Sobolev 正则 Σ_k k(k+1)||Π g_k||²(界 ∫|∂_τ c_dyn|², 与空间 TV 对称)+K_t 取小。 入 §2.3(c)。

## 微裂缝三 (真, fix 精修): 重度 S 弯中线海市蜃楼
低阶多项式拟合逐行质心: S 弯左右凸起求和抵消→中线笔直穿纵隔/心影→token 被 η 拽向软组织弃顶椎。 方法门 fix: x_c 须 **≥三阶**(cubic 才表达 S 一个拐点)+ Huber/RANSAC 鲁棒; **不退化纯逐带局部中心**(否则约束失牙回 η 困境)。 入 §2.3(a)。

## 微裂缝四 (真): comp↔roll 早期动量撕扯
L_comp 拽组合路径向 live 糊图单步 T_{t→r}, L_roll 拽向真值 z_r, 早期梯度夹角>90°震荡。 修: L_comp 目标改 **EMA teacher**(非 live sg)+λ_comp 从0 ramp(S4)+早期 L_roll(z_r 解析可得)主导。 入 §2.2。

## 工程钉子1 (接受): L_smooth 数值安全边界
λ_smooth 0.01~0.05(比 L_rec 小2数量级); t∈[0.1,0.9] 禁近 0/1(防外推崩溃); 模糊则关退标准 AE(L_span 仍正则潜速度)。 入 §6。

## 工程钉子2 (接受+补): 区间采样写死
L_span/ST: r=U(0,0.8),Δ~Gamma(2,0.15)截断<0.3; L_end: 固定(0,1) ρ=0.25; L_comp/roll: 有序 r<s<t(t~U(0.1,1],r~U[0,t-Δmin],s~U(r,t))。 用户原述缺 r, 已补。 入 §8。

---

# R33 审查 (2026-06-24): 八组深修(4 硬阻塞 + bake-off同类性 + Norm + S_order撤销 + 非劣效 + 采样三错 + 数学小修 + 口径)
方法门逐条核实, 全部接受(§四为对 R30/R32 决定的主动撤销, 见下)。 清洁版已入 method_v1。

## 一、四项硬阻塞
1. **L_ST 反向二阶导**: 真。 梯度穿 D_t u_θ → 反向含混合二阶导, "单次 JVP"只算前向调用。 改 detach 目标 q_t=sg(v*_t-Δ D_t u_θ), L_ST=|u_θ-q_t|²; 论文 "one forward-mode JVP with detached JVP-derived target"; 消融 no-ST/ST-detach(主)/ST-full-grad。 入 §2.2。
2. **W₂ 图仍可能退化**: 真。 π_j 由不同 μ_j 约束 → 裸 2D W₂ 主测固定纵向带间距, 患者特异横移成小扰动 → w≈常数 v2 再退 v1。 改**去名义轴距残差距离** m̃_j=[x̄/s_x,(ȳ-μ_j)/s_y]+Bures(Σ), w=w_min+(1-w_min)exp(-d²/τ), **w_min>0** 防 apex 切链; 方向性(distance-decay/uniform/bounded/反向)预注册。 入 §2.3(b)。
3. **时间 Sobolev 不精确**: 真。 Σk(k+1) 比例对系数错(导数基一般不正交)。 改精确导数 Gram R_kl=∫ℓ'_kℓ'_l, L_time=Σ R_kl⟨A_k,A_l⟩; K_t=2 → R=diag(4,12) → 4||A_1||²+12||A_2||²。 **K_t 固定 2**。 入 §2.3(c)。
4. **硬带限未传到 AdaLN**: 真。 M_ψ(Πx)∉range(Π), 当前只能 claim 调制器前带限。 改 **projection-last**: m_dyn=Π M_dyn(...), m=m_static+m_dyn → (I-Π)m_dyn=0; 同一注入器内静态-动态分解(非双注入); J token 经固定轴向插值作用 2D patch(轴向 AdaLN, 不全局平均)。 入 §2.3(e)。

## 二、bake-off 同类性: 主 G_graph 全统一为 rank-K_g 正交投影器(Π^T=Π,Π²=Π,rank=K_g)
普通 Gaussian/Learned-Toeplitz 满秩, 不满足投影器性质, 共享 Norm 修不了。 主表改: Random-subspace/DCT/Gaussian-subspace(top-K_g eig)/Toeplitz-subspace(top-K_g eig)/v1 path-Lap/v2 patient-weighted-Lap。 满秩 soft 滤波移补充组不进主门。 修陈旧残留: "JS 重叠权图"/"JS 边权"→W₂残差图。 入 §3.1+§8。

## 三、共享 Norm: LayerNorm→RMSNorm, 顺序 Norm→Π
"无 affine LayerNorm 纯除标准差"自相矛盾(LN 减均值)。 改无 affine 沿通道 RMSNorm, g̃_k=RMSNorm(g_k), A_k=Π g̃_k。 **投影后不再 tokenwise Norm**(否则抵消能量衰减/重生高图频/破坏硬子空间)。 入 §3.2。

## 四、撤销 R30/R32 "阻断绝对位置"(方法门主动改判)
R33 论点更强且我同意: 主 S_order 应**只换投影器** Q^T Π Q, B/μ_j/位置编码/tokenizer/adapter 全不变(真 topology-only)。 若经绝对位置绕过致 S_order≈0, **正说明链非 load-bearing**, 不得删绕过路径人为做大; "同步置换 μ_j" 改 token 空间含义不再 topology-only。 分层: 主=只换 projector(最纯=测试时 intervention); 辅助=去绝对位置 sensitivity; 次级=从头训错误拓扑。 入 §4.4。 **(此条覆盖 R30 漏洞2 + R32 漏洞2 的处理)**

## 五、1-NFE 改配对非劣效
"≤Base 自身 95%CI" 非规范非劣效。 改患者级配对差 Δ=E^SC-E^Base(Base 独立训练), 预注册 m_NI, UCB_95(Δ)<m_NI。 且"关动态退回 Base"不准(动态模态训练中经共享参数间接影响) → Base 必独立训练; 诚实表述"非常数模态对 full-span descriptor 无直接推理贡献, 1-NFE 差异属间接训练效应"。 入 §2.5+§4.2。

## 六、区间采样三工程错
① t 可能>1 → 先采 Δ 再 r~U(0,1-Δ); ② L_span 只小Δ 削弱中跨度 → 40% local/40% broad/20% full-span(L_ST 主用 local); ③ 三元组极小子区间 → t-r≥2Δ_min, s~U(r+Δ_min,t-Δ_min); ④ L_end 应"以 ρ=0.25 启用 full-span"(L_end 定义即 full-span, 无"随机区间 L_end")。 入 §8。

## 七、数学小修
L_smooth 正式 Charbonnier+/h², 称 decoded-path curvature regularization(不声称匀速形变); λ_smooth 仅 pilot 区间后冻结。 TV 界索引 λ_{K_g-1}+tr(C^T L C)+||·||_F。 G_struct 拆 v1/v2。 E_DC 主门用相对误差 + bootstrap 同重采 seed+患者。 入 §6/§2.3(c)/§4。

## 八、口径残留
"曲线整体移除"有语义漏洞(仍有 x_c(y))。 改 **curve-annotation-free / curve-supervision-free**(内部 x_c 仅无监督 soft axial corridor)。 "cosine 杀高幅值诱惑"→"降低对幅值捷径敏感(不排除方向相似伪影)"。 入 §0口径+§2.3(a)。

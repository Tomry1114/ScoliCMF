# Correction-Grounded SCM/SHMM — 方法重定义 v2（主计划）

> **取代旧"加性条件"SCM/SHMM 的方法含义**（旧设计已 code-verified 终结，见 CHANGELOG R53/R54）。
> 残差架构 doc/residual_correction_v1.md **是本计划的第 4 步**，不是起点。核心转变：
> **旧 SCM 的问题 = "割线没有真实对象"；旧 SHMM 的问题 = "谐波约束错了对象"。要把二者共同改成"对术后矫形量 ΔB 建模"。**

## 0. 为什么旧设计从方法定义上就不成立（诊断，已验证）
1. **旧 SCM = (B,p,Δ) 的时间重参数化，无新信息**（code-verified，legendre.py，R54）：
   令 p=(r+t)/2, Δ=t−r。一阶 secant mean `(1/Δ)∫ℓ₁ = ℓ₁(p)`（与 midpoint point 完全相同，误差 0.0）；二阶 `(1/Δ)∫ℓ₂ = ℓ₂(p)+Δ²/2`（误差 3e-7）。故 `c̄_{r,t}=ℓ₁(p)A₁(B)+(ℓ₂(p)+Δ²/2)A₂(B)` 是 (A₁(B),A₂(B),p,Δ) 的确定函数；p,Δ 已输入网络 → SCM 没观察到新信息。`static≈point≈secant` 是必然。
2. **secant 加法性在非线性 M_dyn 后消失**：c̄ 满足 (t−r)c̄_{r,t}=(s−r)c̄_{r,s}+(t−s)c̄_{s,t}，但注入的是 m=ΠM_dyn(c̄,d)，MLP 后一般不再满足 → "Secant-Coupled" 只是 MLP 前中间量的叙事，非模型承重性质。
3. **全区间动态项恒零**（code-verified：potential_dd(0,1)=[0,0]、trend(0,1)=[0,0]）：1-NFE（术前→术后整段，最该被患者条件描述的区间）SCM 动态贡献为零 → 更像 step-count 相关的时间编码器，非术后变化条件。
4. **trend 的 (1−Δ) 因子**只为强制全区间趋势为零，非从 MeanFlow 恒等式或数据推出；secant_mean+trend 看似两量，实际都只是 (A_k(B),r,t) 的确定组合，无两个独立信息源。
5. **旧 SHMM 图错对象**：v2 用术前 token 的空间残差/方差/邻差构图，描述"术前相邻纵向区域外观连续性"，而任务是"术后该怎么变"。D3 的 +1.5pp 只说明略贴合当前特征差分，不证明捕获真实矫形规律。
6. **static/dynamic 不可辨识**：两支都来自同一术前 token、联合训练、无监督规定归属。m_static+q / m_dyn−q 任意搬运对训练目标等价 → 残差架构能提杠杆，但不能保证 dynamic 表达的就是"区间矫形量"。
7. **低秩 ≠ 低频**：D3 E_top4(ΔB)=0.99 只说明真实变化本征低秩，但其主方向未必是固定脊柱链低频方向（局部器械/融合边界/旋转可低秩但属链上高频）。故"rank-4 不够就改 rank-6"是错的方向。

→ **残差架构（强迫支路成唯一通路）即使让 dyn_off 后指标大降，也只证明"架构使其不可绕过"，不证明"条件表达合理"。必须先把条件表达本身改对。**

## 1. 新 SCM —— Correction Potential SCM（outcome-grounded）
从术前预测患者总矫形表示 `a₀ = f_corr(B_pre)`；训练时用配对术后构造目标 `ΔB = sg(B_post − B_pre)`，令 a₀ 能预测/重建 ΔB。
累计矫形势函数（源锚定，t=1 为 source 端、t=0 为 target 端，与 Bridge α 约定一致）：
  `A_φ(B,t) = (1−t)·a₀ + t(1−t)·Σ_{k=1..K} a_k(B)·ψ_k(t)`，满足 `A_φ(B,1)=0`、`A_φ(B,0)=a₀`。
区间割线：`ā_{r,t} = (A_φ(B,r) − A_φ(B,t)) / (t−r)`。
**关键**：全区间 `ā_{0,1} = a₀ ≠ 0` → 1-NFE 不再为零，且势函数有真实含义（模型从术前预测的累计矫形内容），而非为造"割线"而设的零均值时间函数。

## 2. 新 SHMM —— Correction-Aware Spinal Harmonic Basis
由术前 token 预测患者特异正交基 `Q_φ(B_pre) ∈ R^{J×K}`，`QᵀQ=I`。训练监督其覆盖真实配对变化：
  `L_sub = ||(I − QQᵀ)ΔB||²_F / (||ΔB||²_F + ε)`（最小化漏到子空间外的真实矫形量）。
保留骨科有序结构先验（而非先假定"低频一定对"）：`L_harm = tr(Qᵀ L_path Q)`。
**软谐波调制**（不再硬删高频）：`m_out = QQᵀ m_raw + γ_{r,t}·(I − QQᵀ) m_raw`，0≤γ≤1（γ=0 纯低频 / γ=1 不限制 / 中间保留必要局部变化）。γ≈1 也是有效结论（数据不支持低频限制）。

## 3. 统一机制（SCM 与 SHMM 语义闭环，不再各拼各的）
  `B_pre → (Q_φ, a_φ)[预测患者矫形空间+内容] → ā_{r,t}[区间割线] → Q_φ ā_{r,t} + γ(I−Q_φQ_φᵀ)·[…] → u_corr`
SCM 答"该区间应完成多少患者特异矫形内容"；SHMM 答"这些内容沿脊柱如何协调分布"；二者共同受真实术前—术后差分 ΔB 监督。

## 4. 实施顺序（前置门优先；不先给旧模块决定权）
**Step 1（前置门，最关键）**：验证**术前能否预测有意义的 ΔB**。frozen/EMA encoder 取 B_pre/B_post（同术前 pi 池化），ΔB=sg(B_post−B_pre)；训小 f_corr(B_pre)→ΔB̂，val 上比"群体均值 ΔB̄"基线的解释方差 EV=1−||ΔB−f||²/||ΔB−ΔB̄||²。
  - EV≈0/负 → 术前无患者特异 ΔB 信号（identifiability kill 主导：术后取决于不可观测手术方案）→ **任何 correction-grounded 模块都不可能 work → 诚实转 Bridge-only**。
  - EV 显著>0 → 重定义可行，进 Step 2。
**Step 2**：验证 learned basis Q_φ 是否明显优于 DCT 和 Identity（用 Step 1 的 ΔB，比 L_sub 覆盖率）。
**Step 3**：加入区间势函数 A_φ，比较 point vs secant（此时 secant 才有真实对象）。
**Step 4**：接入 Frozen Bridge 残差分支（doc/residual_correction_v1.md：乘性 head、c_dyn=0⇒u_corr=0、残差速度损失），跑 Pilot A/B/C 验收门。

总损失（最终）：`L = L_MeanFlow + λ_corr L_corr + λ_sub L_sub + λ_harm L_harm`，初始 λ_sub=0.1, λ_harm=0.01。

## 5. 非泄漏纪律
所有模块输入推理时**只有 x_pre**；x_post 仅进训练 loss；ΔB 需 stop-grad；target encoder 冻结/EMA（防双方共同塌缩）。这是"从术前预测 correction-aware 表示"的标准监督学习，非测试泄漏。

## 6. 命名/主张收敛
- **保留**：Pre-to-Post MeanFlow Bridge（已被实验支持）。
- **重设计**：SCM → outcome-grounded correction potential（全区间非零、受 ΔB 监督）；SHMM → correction-aware learned basis + soft harmonic modulation。
- 在 Step 1 门通过前，**不把当前三个名称当已完成贡献**。

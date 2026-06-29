# 残差校正阶段 v1 — Frozen Bridge + SCM/SHMM 残差分支

> **新实验阶段（独立于已封板的旧"加性条件"SHMM/SCM 设计，见 CHANGELOG R53）。**
> 核心思想：**冻结已验证有效的 Pre-to-Post MeanFlow Bridge，让 SCM/SHMM 专门预测 Bridge 尚未解释的残差**——从"加性条件旁路"改成"真正的残差校正分支"。

## 为什么（诊断依据 R53 D1–D3）
旧架构 `m = m_static + m_dyn` 一起作 AdaLN 条件进所有 block，主干还直读 `[z_t, blur(x_pre)]`。
- D1：dct≈v1≈v2≈identity，只 random 伤 → 模型要结构化子空间但不在乎哪个平滑基；患者特异 v2 零增益。
- D2：C_dyn=0.305 有因果通路，但 dyn_off 后 SSIM 仅掉 0.012 → 动态支路对终点只值 1.2%，杠杆太小。
- D3：E_v2=0.769>E_dct=0.754（+1.5pp，假设方向对），但都丢 ~25% 高频术后变化。
→ SHMM 没机会兑现是因为它在和强主干竞争。改为**只解释 Bridge 残差**，不与主干竞争。

## 两阶段结构
**阶段一（冻结）**：保留并锁定已验证的纯 Bridge —— patch embed / DiT blocks / base conditioning / 原 velocity head。产出 `u_base = f_bridge(z_t, x_pre, r, t)`。SCM/SHMM 不得改动它。
**阶段二（学残差）**：
- 目标平均速度 `u* = (z_t - z_r)/(t - r)`。
- Bridge 未解释部分 `e* = u* - sg(u_base)`（sg=stop-grad）。
- 校正分支 `u_corr = f_corr(h_base, m_dyn)`；最终 `u_final = u_base + u_corr`。
- 损失：`L_corr = ||u_corr - sg(e*)||`，`L_full = ||u_final - u*||`；`loss = λ_corr·L_corr + λ_full·L_full`，初始 λ_corr=λ_full=1.0。
- endpoint 仍只监督最终 x̂_post^full；**不要**给校正分支单独的自由图像生成目标。

## 代码拆分
### 1. sc_pga.py：新增 `DynamicCorrectionConditioner`（只出动态条件）
- forward(x_pre,r,t,t_emb,r_emb) → (c_dyn_patch, aux)。
- 纯粹来自 preop tokens + interval secant + harmonic modulation。
- **删除** `m_static`、删除末尾 `c_patch = c_patch + e[:,None,:]`（时间已通过 r,t 的割线系数进入 SCM，不再另开旁路）。否则 dyn_off 仍残留静态/时间信号。

### 2. models/sc_dit.py：基础模型暴露隐藏特征
- 拆 `forward_features(z_t,r,t,x_pre) -> (h, c_base)`；`forward = head(forward_features)`。
- 新增包装 `ResidualScoliCMF(bridge, dyn_cond, corr_head)`：bridge 全参 requires_grad=False；
  forward 中 `with no_grad: h_base,c_base=bridge.forward_features(...); u_base=bridge.head(...)`；
  `c_dyn,aux=dyn_cond(...)`；`u_corr=corr_head(h_base.detach(), c_dyn)`；`u=u_base+u_corr`；aux 存 u_base/u_corr。

### 3. correction head：必须防止绕过动态条件（乘性校正）
- `h_corr = W_h(h_base) ⊙ tanh(W_g(c_dyn)) + W_c(c_dyn)`，再 ResidualConvHead 解码。
- 要求：所有 bias 关闭；**c_dyn=0 时严格 u_corr=0**；residual decoder 不再接收 x_pre/z_t/静态条件。
- 这样 dyn_off 精确退化为冻结 Bridge：`u_dyn_off = u_base`（干净因果消融）。

### 4. meanflow_sa.py：残差监督
- `u,aux=model(...,return_aux=True)`；`target=(z_t-z_r)/(t4-r4).clamp_min(1e-6)`；
  `corr_target=(target-aux["u_base"]).detach()`；`l_corr=adaptive_l2_loss(u_corr-corr_target)`；
  `l_full=adaptive_l2_loss(u-target.detach())`；`loss=λ_corr·l_corr+λ_full·l_full`。

## 执行顺序（逐 Pilot，门不过即止）
**Pilot A — 动态残差校正**：只改 冻结Bridge + 独立 dynamic conditioner + 乘性 residual head + residual velocity loss。**用当前 v2 projector，不改 projector**。若无法优于 Bridge → 立即停、删 SCM/SHMM。
**Pilot B — correction-aware basis**（仅 A 有效后）：learned orthonormal basis + L_sub + path smoothness prior；比较 DCT/旧v2/learned。
**Pilot C — soft harmonic modulation**（仅 B 有表示优势但硬 rank-4 降质后）：加高频 gate。

### Pilot A 验收门（同 checkpoint，paired bootstrap）
- 条件一 完整模型优于冻结 Bridge：ΔSSIM>0 且 ΔLPIPS<0，95%CI 至少一个排除 0、另一方向一致。
- 条件二 dyn-off 丢掉大部分新增收益：设 G=SSIM_full−SSIM_bridge，应 SSIM_full−SSIM_dyn_off ≈ G（收益确来自动态校正）。
- 条件三 动态分支非纯噪声：corr(u_corr, u*−u_base) 或 residual MAE；连 Bridge 残差都拟合不了则不进入下一步。

## Pilot B — correction-aware 子空间（不直接对 eigh 开梯度）
- 预测 rank-K 正交基：`basis_logits=basis_net(Btok)` (B,J,K) → `Q,_=torch.linalg.qr(basis_logits,mode="reduced")` → `Pi=Q@Q^T`。
- 术后变化监督（冻结/EMA target encoder，防共同塌缩）：no_grad 下取 F_pre/F_post，用 pi.detach() 池化得 B_pre/B_post，`delta_B=B_post-B_pre`。
- `L_sub = ||(I-Π_θ)ΔB||_F^2 / (||ΔB||_F^2 + ε)`（最小化漏到子空间外的真实变化）。
- 脊柱有序平滑先验：`L_harm = (1/K)·tr(Q_θ^T L_path Q_θ)`。
- 总损失 `L = L_MeanFlow + λ_corr·L_corr + λ_sub·L_sub + λ_harm·L_harm`，初始 λ_sub=0.1, λ_harm=0.01。
- **非泄漏**：projector 输入仍只有 x_pre；x_post 只进训练 loss；推理不用术后；delta_B 需 stop-grad；target encoder 冻结/EMA。

## Pilot C — 软谐波调制（替代硬 rank-4）
- `m_dyn = Π·m_raw + γ_{r,t,x}·(I-Π)·m_raw`，0≤γ≤1（γ=0 纯低频、γ=1 Identity）。
- 实现：`low=proj_apply(Pi,raw); high=raw-low; gamma=sigmoid(highfreq_gate(interval_summary)).view(B,1,1); m_dyn=low+gamma*high`。
- 不强迫 γ 小；若学到 γ≈1 → 数据不支持低频限制，这本身是有效结论。

## 最终目标结构
`u_final = u_bridge + u_corr`，`u_corr = ResidualHead[ h_bridge ⊙ G( SHMM( SCM(x_pre,r,t) ) ) ]`，
`SHMM(X) = Π_θ X + γ(I-Π_θ)X`。同时解决：动态条件可绕过旁路 / 患者图与真实矫形量错位 / rank-4 硬投影丢局部高频。

# Changelog

> 每一步改动都追加在最上面（倒序，最新在前）。

## 2026-06-29 — 第 58 轮：Step 4 secant gate pilot 未通过 —— 表示层赢但端点不兑现（过拟合/数据受限）

- **gate pilot（res_secant，debug，2000 步，cond_mode=secant）**：
  | step | FULL SSIM4 | DYN-OFF | dSSIM | FULL LPIPS4 |
  |---|---|---|---|---|
  | baseline 冻结Bridge | — | 0.2490 | — | 0.5090 |
  | 400 | 0.2489 | 0.2490 | −0.0001 | 0.5089 |
  | 800 | 0.2480 | 0.2490 | −0.0011 | 0.5117 |
  | 1200 | 0.2463 | 0.2490 | −0.0027 | 0.5168 |
  | 1600 | 0.2443 | 0.2490 | −0.0048 | 0.5226 |
  | 2000 | 0.2417 | 0.2490 | −0.0073 | 0.5283 |
- **判定：FAILED**。FULL 端点在任何 step **都没超过 baseline**(最好 step400 = 0.2489 ≈ baseline,之后**单调下滑**);DYN-OFF 恒等于 0.2490(不变量再次确认)。**验收条件一(ΔSSIM>0 且 ΔLPIPS<0)不满足。**
- **机制(关键)**：训练损失同时下降(l_full 0.232→0.182、l_corr 同、l_sub cov=0.938、γ 0.52→0.12)——**correction 分支确实在拟合训练残差/ΔB,但完全不兑现到端点,反而单调伤害 val SSIM/LPIPS**。训练↓+val端点↑ = 典型**过拟合**(5.35M 参数 / 432 训练对);误差在 4-NFE rollout 上累积。
- **与全局一致**：Step1(EV0.358)、Step2(cov0.937)、本轮(拟合残差)——**模型在表示/特征层反复能拟合,但没有一次转化为端点增益**。瓶颈不是表示,是**数据规模(432)+ identifiability floor**,额外容量只过拟合。
- **下一步候选**：① 强正则小容量 retry(基+小 head <0.5M、重 wd、dropout、早停/数据增强)确认是否纯过拟合;② 诚实 Bridge-only。待用户拍板。
- **产物**：runs/res_secant(5 ckpt+eval)。

## 2026-06-29 — 第 57 轮：Step 4 残差校正 pilot 实现 + 冒烟通过

- **实现（doc/correction_grounded_v2.md 第 4 步）**：
  - models/sc_dit.py：SCDiT 拆出 forward_features（出 h tokens + c + aux）/ head_forward，forward 重构复用之（行为不变）。
  - residual_model.py（新）：① DynamicCorrectionConditioner = 术前 token → correction potential A_φ(B,t)=(1−t)a₀+t(1−t)Σa_kℓ_k(t)（全区间割线 ā_{0,1}=a₀≠0；cond_mode=secant/point/static）→ learned 正交基 Q_φ(B_pre)（QR）→ 软谐波 m_dyn=QQᵀā+γ_{r,t}(I−QQᵀ)ā → 2D mass-preserving 回投到 patch；**无 m_static、无 time-emb 旁路**。② DynamicResidualHead = 乘性 h_corr=W_h(h)⊙tanh(W_g(c))+W_c(c)，全 bias-free + ResidualConvHead bias-free → **c_dyn=0 ⇒ u_corr=0 严格成立**。③ ResidualScoliCMF = 冻结 Bridge(no_grad)+dyn_cond+corr_head，u=u_base+u_corr，dyn_off 标志。
  - train_residual.py（新）：加载冻结 s2_base，损失 L=λ_full·L_full+λ_corr·L_corr+λ_sub·L_sub+λ_harm·L_harm（L_corr=‖u_corr−sg(u*−u_base)‖，L_sub=‖(I−Π)ΔB‖²/‖ΔB‖²，ΔB=sg(B_post−B_pre)，L_harm=tr(QᵀL_path Q)）；EMA；周期评估 FULL vs DYN-OFF。
- **冒烟（debug 卡，secant，60 步）通过**：5.35M 可训练参数 A40 不 OOM；**INVARIANT c_dyn=0→|u_corr|max=0.00e+00**；baseline 冻结 Bridge val SSIM4=0.2490（=s2_base，加载正确）；4 项损失齐（full/corr/sub/harm），cov=0.951、γ=0.522；DYN-OFF 精确回到 0.2490。
- **下一步**：跑 secant 正式 gate pilot（debug 卡,周期 eval）→ 看 FULL 是否显著超 baseline 0.2490 且 dyn-off 丢掉增益。过 → 跑 point/static 消融 + 长训;不过 → 诚实 Bridge-only。
- **产物**：models/sc_dit.py(改)/residual_model.py/train_residual.py;runs/res_smoke(冒烟)。

## 2026-06-29 — 第 56 轮：Step 2 通过（强）—— learned correction-aware 基远胜固定低频基

- **探针（step2_basis_probe.py，debug 卡）**：复用 Step1 的 ΔB（冻结 stem+术前 pi），比较 rank-K 子空间对 val ΔB 的覆盖率 cov=‖ΠΔB‖²/‖ΔB‖²；learned Q_φ(B_pre)=小网络→J×K logits→QR→Π=QQᵀ，训练最小化 ‖(I−Π)ΔB‖²/‖ΔB‖²（x_post 仅训练用，输入仍只 x_pre）。
- **结果（val 覆盖率）**：
  | K | DCT | v1 | v2(旧) | random | learnedQ(val)[train] | oracle(per-sample SVD topK) |
  |---|---|---|---|---|---|---|
  | **4** | 0.754 | 0.742 | 0.769 | 0.337 | **0.937** [0.950] | 0.989 |
  | 6 | 0.932 | 0.902 | 0.919 | 0.583 | 0.989 [0.992] | 0.999 |
  | 8 | 0.953 | 0.939 | 0.954 | 0.667 | 0.996 [0.997] | 1.000 |
- **判定：GREEN（强）**。rank-4 处 learned Q_φ val=0.937 ≫ DCT 0.754（+18.3pp，几乎补回 D3 丢掉的 ~25%），train-val gap 仅 0.012（真泛化）；远胜 random 0.337；逼近 oracle 0.989。
- **证实用户第 8 点（低秩≠低频）**：真实变化本质 rank-4（oracle 0.989），但主方向**不是固定脊柱低频方向**（DCT 仅 0.754）；从术前预测的 learned 基对齐真实子空间并补回。learned 基**秩效率高**：K=4 即达 0.937 ≈ DCT 需 K=6（0.932）。
- **诚实边界**：这是**表示层**对 ΔB 的覆盖，**非端点 SSIM/LPIPS**；"覆盖 token 变化更多" ≠ "生成术后图更好"——端点收益要 Step 4（接 Frozen Bridge 残差）才能判。stem OEED 套 x_post 仍压低绝对值。
- **结论**：Step1（术前可预测 ΔB，EV0.358）+ Step2（learned 基覆盖 0.937）**两道表示层门都过**，correction-aware basis 是对的对象。SHMM 重设计在表示层成立。
- **产物**：step2_basis_probe.py；step2.out（gitignored）。下一步候选：Step3（加区间势函数 A_φ 比 point vs secant）或直接 Step4 整合 pilot（端点判定）。待用户拍板。

## 2026-06-29 — 第 55 轮：Step 1 前置门通过 —— 术前可预测患者特异 ΔB（重定义可行）

- **探针（step1_dB_probe.py，debug 卡，eval+小训练）**：冻结 shmm_v2 的 cond.stem + 术前 pi 池化取 B_pre/B_post（同一 pi），ΔB=sg(B_post−B_pre)，train 432/val 54；训 f_corr(B_pre)→ΔB̂，验收 val 解释方差 EV=1−‖ΔB−f‖²/‖ΔB−ΔB̄‖²（ΔB̄=训练集群体均值）。
- **结果**：
  - frac patient-specific (val) ‖ΔB−ΔB̄‖²/‖ΔB‖² = **0.478**（48% 患者特异，52% 共有均值矫形）。
  - **MLP best EV_val = 0.358**（EV_train@best 0.474，gap 小 → 真信号非过拟合）。
  - 线性探针欠拟合（巨型 J·D 线性层+重 wd 优化太慢，EV_val 负、单调上升未收敛）→ 无结论，不影响判定。
- **判定：GREEN**。术前携带有意义、可泛化的患者特异 ΔB 信号（EV_val 0.358 ≫ 0.1 阈值）→ identifiability kill 未完全主导，correction potential / correction-aware basis 有可学对象 → **进 Step 2**。
- **诚实边界**：① stem OOD 套 x_post 取 ΔB 会压低 EV → 0.358 是保守下界（换冻结/EMA target encoder 应更高）；② 52% 共有均值是任何模块的平凡基线，SCM/SHMM 要挣的是 48% 患者特异部分（术前可预测其中 ~36%）= 模块收益的现实天花板。
- **产物**：step1_dB_probe.py；step1.out（gitignored）。下一步 Step 2：验证 learned 正交基 Q_φ 是否明显优于 DCT/Identity（用本轮 ΔB，比 L_sub 覆盖率）。待用户拍板。

## 2026-06-29 — 第 54 轮：旧 SCM = 时间重参数化（code-verified）→ 方法重定义为 correction-grounded

> 用户深度方法批判 + 我用 legendre.py **数值验证**：旧 SCM/SHMM 的**方法定义本身**有根本问题，残差架构（R53/0b23d49）只解决"无决定权"、解决不了"信息是否成立"。**残差架构降级为 4 步计划的第 4 步。**
- **code-verified（legendre.py，CPU）旧 SCM = (B,p,Δ) 时间重参数化，无新患者信息**：令 p=(r+t)/2,Δ=t−r。
  - `max|potential_dd₁(r,t) − ℓ₁(p)| = 0.0`（一阶 secant mean ≡ midpoint point condition）。
  - `max|potential_dd₂(r,t) − (ℓ₂(p)+Δ²/2)| = 3.1e-7`（二阶 = ℓ₂(p)+Δ²/2）。
  - `potential_dd(0,1)=[0,0]`、`trend(0,1)=[0,0]`（全区间/1-NFE 动态项恒零）。
  → `c̄_{r,t}=ℓ₁(p)A₁(B)+(ℓ₂(p)+Δ²/2)A₂(B)` 是 (B,p,Δ) 确定函数，p,Δ 已入网 → static≈point≈secant 是数学必然，非训练不足。
- **其余方法缺陷（认同）**：secant 加法性经非线性 M_dyn 后消失（叙事非承重）；trend 的 (1−Δ) 只为强制全区间零；旧 SHMM v2 图描述术前外观非术后矫形（D3 +1.5pp 只贴合特征差分）；static/dynamic 不可辨识（同源+联合训练，q 可任意搬运）；低秩≠低频（E_top4=0.99 不支持"低频足够"）。
- **方法重定义（doc/correction_grounded_v2.md）**：① 新 SCM=Correction Potential（a₀=f_corr(B_pre) 受 ΔB=sg(B_post−B_pre) 监督；A_φ(B,t)=(1−t)a₀+t(1−t)Σa_kψ_k(t)，A_φ(B,1)=0、A_φ(B,0)=a₀；全区间割线 ā_{0,1}=a₀≠0）；② 新 SHMM=Correction-Aware learned 正交基 Q_φ(B_pre)（L_sub=||(I−QQᵀ)ΔB||²/(||ΔB||²+ε)）+ 软谐波 m_out=QQᵀm_raw+γ(I−QQᵀ)m_raw；③ 统一链 B_pre→(Q_φ,a_φ)→ā_{r,t}→Q_φā+γ(I−Q_φQ_φᵀ)…→u_corr。
- **实施顺序（前置门优先）**：Step1 验证术前能否预测有意义 ΔB（EV vs 群体均值基线；EV≈0 → identifiability kill 主导 → 诚实 Bridge-only）→ Step2 learned basis 是否胜 DCT/Identity → Step3 加 A_φ 比 point vs secant → Step4 接 Frozen Bridge 残差分支。**不先给旧模块决定权。**
- **产物**：doc/correction_grounded_v2.md（主计划）；doc/residual_correction_v1.md 加头注降级为 Step4。
- **待用户拍板**：是否运行 Step1 前置门探针（debug 卡，小 f_corr，验收 EV）。

## 2026-06-29 — 第 53 轮：D1–D3 同-checkpoint 干预 → 旧「加性条件」SHMM/SCM 设计正式终结

> 本轮是**旧 SHMM/SCM 设计的正式终点**。新残差架构（doc/residual_correction_v1.md）为独立新阶段，**不得覆盖/改写**此处的干净阴性结论。

- **动机**：R52 的「SHMM 真 null」依赖独立重训比较，存在第 5 点混淆（每个模型把 g_k/M_dyn 重参数化适配固定 Π）。改用**同一 v2 checkpoint、eval-only**的三个受控干预（diag_shmm_causal.py，debug 卡）。
- **D1 投影器替换（同权重，无重训；在 v2-trained 特征上换 Π）**：
  - v2 0.2511/LPIPS0.4657 | dct 0.2497(dOut0.022) | v1 0.2500(0.017) | identity 0.2497(0.029) | permuted 0.2500(0.042) | **random 0.2409/LPIPS0.5121(dOut0.119)**。
  - → 消除独立重训混淆后**结论不变**：dct≈v1≈v2≈identity（含满秩无限制），输出几乎不动；只有 random 明显恶化。即**模型需要"某个结构化动态子空间"，但不在乎是哪个平滑基**，患者特异 v2 相对固定 dct 零增益。投影分支**非惰性**（random 伤害），但患者特异性无用。
- **D2 动态分支因果（同权重）**：C_dyn=0.305（关 m_dyn → u 变 ~30%，**有真实因果通路**）；C_stat=0.067；endpoint SSIM4 full 0.2511 → dyn_off 0.2392（**Δ 仅 0.012**）。
  - → 动态分支改变输出却几乎不改变**正确性**；SHMM 优化的支路对终点只值 1.2% SSIM —— **杠杆太小**。
- **D3 真实术后变化 ΔB=B_post−B_pre 的子空间可表示性（能量比，修正范数→能量）**：E_top4(ΔB)=0.989（真实变化**本质 rank-4**）；E_dct=0.7543 / E_v1=0.7417 / **E_v2=0.7693**。
  - → v2 比 dct **略好 +1.5pp**（核心假设方向对、**未被数据否定**），但三者都只装 ~75%，**共有 ~25% 真实变化（局部高频术后改变）落在任意 rank-4 低频子空间外**。
- **综合（旧设计终判）**：旧「加性条件」SHMM/SCM 在终点层 = 确认阴性。机制真实、方向正确，但 ① 杠杆太小（D2）② 模型对平滑基选择不敏感（D1）③ rank-4 硬投影对 dct/v1/v2 共丢 ~25% 高频（D3）。**单调 L_tokdiv/τ/K_g 注定无效**——根因是架构（加性旁路 + 主干直读 z_t/blur(x_pre)），非超参。
- **决策**：旧设计到此封板。新阶段 = **冻结 Bridge + 残差校正分支**（Pilot A/B/C，见 doc/residual_correction_v1.md）。
- **产物**：diag_shmm_causal.py（D1/D2/D3 三合一，eval-only）；diag_causal.out（gitignored）。

## 2026-06-29 — 第 52 轮：撤回 SCM「+23%」+ SHMM 阴性定锤（L_tokdiv 排除 collapse 混淆）

- **重大修正（撤回 R51 的 SCM-secant +23%）**：该正面结果是 **P0-1 评测污染假象**。根因：`eval_gates.build_model()` 未把 `cond_mode` 透传给 SCPGA（cond_mode 不在 state_dict 里），导致 point/static 的 ckpt 被一律当 `secant_full` 评测 → 三条 cond_mode 其实跑的是同一前向，差异是噪声。
  - **修复**：统一 `build_scpga(cfg,H,W)` 工厂为唯一入口（sc_pga / train_sa / eval_gates / eval_ablation 同源传 cond_mode/proj/tau/w_min），并用旧码快照 `sc_pga_ce0b042.py` + 正确 cond_mode 对原 ckpt 重评（`eval_scm_oldcode.py`）。
  - **修正后 SCM（val SSIM4 / PSNR4 / LPIPS4，best-val=step5000）**：static 0.2562[.2433,.2682] / point 0.2503[.2376,.2623] / secant 0.2529[.2401,.2651]。
    → **SCM = NULL**：secant ≈ point ≈ static，CI 完全重叠。R51 的「secant vs point +23% / +0.93dB」**作废**。
- **SHMM 阴性定锤（排除 token-collapse 混淆）**：R51 的 SHMM 阴性当时无法区分「真阴性」还是「token 塌缩使任何低通投影都惰性」。本轮加 `L_tokdiv=0.1` 重训 shmm_dct/v1/v2 到 5000 步后复测：
  - **诊断（best ckpt）**：tok_cos 0.993→**0.285**（塌缩已破）、R_removed 0.026→**0.31~0.33**（投影器已激活，剥掉 ~10% 动态能量）。
  - **指标（val SSIM4 / PSNR4 / LPIPS4）**：dct 0.2518[.2395,.2636] / v1 0.2514[.2390,.2632] / v2 0.2511[.2389,.2629]。
    → **SHMM = 真 NULL**：collapse 破了、投影器在工作，dct≈v1≈v2 仍噪声内。患者特异图（v2）相对固定 DCT 基零增益，不是 bug 而是机制无效。
- **三创新点最终裁决**：
  - **① Pre-to-Post MeanFlow Bridge ✅**：源锚定（s2_base 0.249@4）大幅胜 noise→image（orig 0.194@4，PSNR 11.4），1NFE 差距更大。**唯一有效创新。**
  - **② Secant Conditioning Module (SCM) ❌ NULL**。
  - **③ Spinal Harmonic Modulation Module (SHMM) ❌ NULL**（已排除 collapse 混淆）。
  - 整个 SC-PGA(SCM+SHMM) ≈ 纯 Bridge(s2_base) → **价值全在 Bridge**。
- **文件/产物**：`eval_scm_oldcode.py` / `sc_pga_ce0b042.py` / `diag_shmm_new.py` / `run_eval_shmm_new.sh` / `scripts/run_{scm,shmm,shmm2}.sh` / `scripts/run_ablation_v2.sh` / `configs/verify_light.yaml`；runs/{shmm_dct,shmm_v1,shmm_v2}（L_tokdiv 重训）、runs/scm_*_pre_tokdiv。
- **遗留**：实验阶段到此收口（用户指示停跑）；下一步=据此重写论文叙事（以 Bridge 为主，SCM/SHMM 报为诚实阴性或重新设计）。

## 2026-06-28 — 第 51 轮：修正后消融表完成（干净单因子，val 4-NFE）

- **改动**：用修复后代码(P0/issue3/5/6 + lambda_time=0 + 同秩 SHMM)重训 7 变体并评测,出修正消融表。
- **结果(val SSIM4 / PSNR4 / LPIPS4,best-val=step5000,reflect-pad SSIM)**：
  - Bridge+组合：s2_base 0.249/13.55/0.509 → s3_st 0.246(L_ST 中性) → **s4_comp 0.258/13.83/0.478(组合 ✅,P0修后)**。
  - SCM(同 SCPGA 骨架,Π=I,只切 cond_mode)：static 0.229 / point 0.205/13.00/**0.417** / **secant 0.253/13.93/0.458**。
    → **SCM 割线 ✅**:secant vs point **SSIM4 +23% / PSNR4 +0.93dB**;secant>static。单点 LPIPS 最优=fidelity↔perceptual 权衡。
  - SHMM(同 rank-4,secant_full)：dct 0.253 / v1 0.254 / **v2 0.255**,LPIPS4 0.458/0.461/0.467。
    → **SHMM 患者特异 ✗ 阴性**:v2≈v1≈dct 噪声内,即便去退化(‖Pi_v2-Pi_v1‖0.15)也无增益,脊柱限制中性。
- **结论**:两个确认贡献(Composition、SCM-secant),一个诚实阴性(SHMM)。SCM-secant 是最强干净正面证据。
- **遗留/建议**:① 单 seed n=54,headline(SCM secant vs point)需多 seed 固化 CI;② SHMM 要么报为诚实阴性、要么重新设计(当前患者特异图机制无效);③ s2/s3 复用旧 ckpt 已用新 SSIM 重评对齐。
- **产物**：runs/{s4_comp,scm_*,shmm_*};runs/eval2_{s4,scm,shmm,base}.out。


## 2026-06-27 — 第 50 轮：深度 code review → 修复 5 个根因 + 修正消融重训

- **改动(用户 code review 找出,逐条修)**：
  - **P0 共享噪声 bug**：`_l_comp_roll` 给 z_t/z_r 各采独立 eps(sigma_m=0.1)→ rollout 追踪另一条随机桥,监督冲突。修：同一 eps。**s4_comp 及后续必须重训。**
  - **issue-3 时间 embedding 旁路**：动态分支 `M_dyn(cat[cbar,trend,e])` 含时间 e → full-span(cbar=trend=0)m_dyn 仍非零 = 脊柱投影的时间 MLP,非患者割线。修：M_dyn 无 bias、输入仅 [cbar,trend];e 只进 m_static+c_patch。验证 secant full-span m_dyn_rms==0。
  - **issue-6 SCM 不公平消融**：s4_comp(base 全局池化) vs identity(scpga 逐patch) 同时改了架构。修：同一 SCPGA 骨架加 `cond_mode={static,point,secant_mean,secant_full}`,只切 point↔secant。
  - **issue-5 v2 图退化**：beta=40 锁 token → 残差极小 → exp(-d/tau=1)≈1 → v2≈固定 path graph(实测 ||Pi_v2-Pi_v1||/||Pi_v1||=0.0003)。修：per-sample 中位数尺度 → 0.0003→0.1539(真患者特异)。
  - **issue-4/7/8**：lambda_time=0(L_time 对 Identity≈常数、对低秩有钻零空间捷径,不公平);SHMM 改同秩 dct/v1/v2(非 Identity 全秩);SSIM 改 reflect padding;LPIPS/eval_ablation 入库(本次新加,预处理 gray→3ch+[0,1]→[-1,1])。
  - GAN(discriminator+lambda_adv/perc)加了但**休眠**(默认0,推理纯 MeanFlow)——真问题是上述 bug,不是缺对抗。
- **文件**：meanflow_sa.py / sc_pga.py / metrics_img.py / train_sa.py / tools/gen_configs.py / models/discriminator.py / eval_ablation.py;commits d948663, ce0b042。
- **结论修正**：之前"病态均值回归天花板/要上 GAN"的判断**撤回** —— Bridge+composition 基本 work,平表主因是上述 SCM/SHMM 实现 bug + 噪声 bug + 不公平消融。
- **修正后实验表(9 config,patch8/batch4/5000/lambda_time=0)**：composition 阶梯 s2/s3/s4 + SCM(identity×{static,point,secant}) + SHMM(secant_full×{dct,v1,v2})。s2_base/s3_st 不受影响复用旧 ckpt;重训 7 个(job 9918875)。
- **评测口径修正**：SCM/SHMM 主要在 2/4-NFE 比(1-NFE 下 secant 结构性为零,只能验 Bridge+静态条件)。


## 2026-06-26 — 第 49 轮：原始 ScoliCMF vs 创新版 对照（创新点验证）

- **改动**：① 停掉过拟合的 leak-fix 长训练(job 9912392，s5b val 在 step4k 见顶后单调下滑)；② 忠实复刻原始 ScoliCMF(`/share/ScoliCMF-main` = BraTS conditional MeanFlow，noise→image + FGA)为匹配 harness baseline；③ 同数据/同 split/同 480×240/同 regime(wd0.02/aug/EMA0.999/batch8/lr1e-4)/同 metrics_img，只留"方法"一个变量；④ 训练 16k 步并扫全 ckpt 找各自 best-val。
- **文件**：`baseline_orig/{baseline_orig.py,train_orig.py,eval_orig.py,sweep_orig.py}`(新增，复用 models.sc_dit timm-free 原语 + dataset_sa + metrics_img)；`runs/orig_baseline/`(33.79M，16 ckpt)。
- **原因**：用户口径——"指标用 SSIM 这些生成指标；问题是代码/方法不是训练技术；要比最初版本 ScoliCMF 和现在改的版本，看创新点有没有用"。
- **结果（各取 best-val，val 集 n=54）**：
  | val | 原始(step16k,33.79M) | s5b(step4k,19.7M) | Δ |
  |---|---|---|---|
  | SSIM@1NFE | 0.130 | **0.258** | +98% |
  | SSIM@4NFE | 0.194 | **0.236** | +22% |
  | best-SSIM | 0.191@20NFE | **0.258@1NFE** | +35% |
  | PSNR@1NFE | 11.41 | **13.97** | +2.6dB |
  | L1@1NFE | 0.217 | **0.156** | -28% |
  - **创新点确有用**：源锚定+SC-PGA 用 1/20 NFE、6 成参数，val SSIM 高 35–98%，每个匹配 NFE 都赢。
  - **train 侧也赢**：原始 train≈val≈0.19(欠拟合/方法天花板低)；s5b train 在 best-val 点 0.27、16k 到 0.46 → 拟合+泛化双赢。之前"train 不如崩溃版"是和泄漏作弊版(train0.65 假象)比，与正确对照(原始)比 s5b 全面胜。
  - **诚实边界**：绝对 SSIM 仍 0.19–0.26(432 样本小)；原始 16k 仍极缓升(8k→16k +0.008，基本走平)；s5b 过 4k 过拟合。相对优势在所有 NFE 一致且大。


## 2026-06-21 — 第 2 轮：创新点调研 + 方案锁定

- **改动**：调研 CVPR2026 生成最新工作；锁定 MIA 期刊版主创新；产出方法设计文档。
- **文件**：`doc/novelty_study.md`（新增）、`STATUS.md`（更新主线）
- **原因**：审稿人(R1/R2)质疑创新弱/像 adaptation、FGA trivial；需做面向骨科的真创新。
- **结果**：定方向① **术前出发的桥接式 Mean-Flow**(source 改术前图,借鉴 Better Source/LBM)+ ④ **曲线感知 FGA 重构**(替代全局均值门 + 可微 Cobb)。形变场路线(②)因无逐椎骨标注暂搁置。逐条对应 4 条创新质疑,含消融矩阵与风险。未动代码。

## 2026-06-21 — 第 1 轮：搭建仓库 + 下载并解压 ScoliSurg 数据

- **改动**：① 从 Google Drive 下载 ScoliSurg(3 个加密 rar,密码 HKU_Med);② 装用户态 unrar 7.12 到 `~/bin/`;③ 解压;④ Pillow 做数据 profile。
- **文件**：`data/Spine生成_Miccai数据集/{train,data2.0,Pre_case_curve_new}/`；`~/bin/unrar`
- **原因**：改投 MIA、只做脊柱,需真实 ScoliSurg 数据。
- **结果**：解压成功。train=540 对、data2.0=92 对(stem 541–632,合计 632=论文口径)、Pre_case_new=296(仅术前)。配对 stem 100% 对齐,240×480;standardized=RGB,curve=L。

## 2026-06-21 — 第 3 轮：创新方案对抗审稿（5 审稿人）

- **改动**：对 novelty_study.md 跑 5 视角内部审稿 + 主编综合。
- **文件**：`doc/REVIEW_novelty_2026-06-21.md`（新增）
- **原因**：投代码前验证创新是否真能逃出原拒稿理由。
- **结果**：判定**当前设计仍会被拒**（创新=适配、近线性论证自相矛盾、σ_t 破坏 JVP、单步退化为回归器、任务缺手术方案而病态、Cobb 损失循环、幻觉植入物、快速推理无价值）。收敛出重构方向：方案条件化 + 不确定性 + 受约束桥算子 + 几何为主输出。**方案待重构，未动代码。**

## 2026-06-21 — 第 4 轮：深挖数据现实 + 创新余量(文献),改写方向

- **改动**：数据 inspect(元数据扫描/配准重影定量/曲线掩膜格式)+ 两路文献深挖(桥上 mean-flow 数学是否开放、prior art 贴近度)。
- **文件**：`data/_inspect/montage_*.png`(术前|术后|0.5混合|术前曲线|术后曲线)
- **原因**：用户「先深挖再定」,投代码前查清方向是否成立。
- **结果**：① 数据无任何手术方案元数据(纯图);术前/术后未配准,0.5混合严重重影→像素桥出局。② 文献:桥上 MeanFlow 已发表(FMM/CDBM/Stochastic Interpolants),原 σ_t 数学错(需 score 修正 v̄−γ̇γ∇logp);唯一算子级空白=「微分同胚形变桥上的平均速度 flow-map」(无需逐椎骨标注)。③ plan-conditioned 因无方案数据不可行;不确定性/分布是几乎全空且数据够的次要白区。竞品 TraceTrans(2510.22379)待核。**候选新主线:骨形变 diffeomorphic 桥 + 平均速度 + 分布输出;未拍板,未动代码。**

## 2026-06-21 — 第 5 轮：竞品核查(TraceTrans/配准领域)→ 方向绿灯

- **改动**：精读 TraceTrans + 核微分同胚速度场配准是否占用算子空白。
- **原因**：用户「先核竞品再拍板」。
- **结果**：TraceTrans=确定性形变+合成,医美/脑 MRI,非脊柱/非mean-flow/非桥/无分布 → 不重叠但须引。LDDMM/SVF/神经速度场=配准(两图都给、确定性、不生成),与"只给术前、生成术后形变分布"不同。三元组{平均速度少步flow-map}∩{微分同胚形变}∩{术前→术后生成桥+分布}确认无人占。诚实定级:机理驱动新组合+一处新数学(微分同胚桥score修正平均速度)+数据集/临床贡献,对MIA足够、非范式级。决策:mean-flow价值重述为数学+分布采样(非速度);mask内固定物只预测骨形态。**方向绿灯,待写完整方法设计。**

## 2026-06-21 — 第 6 轮：v2 创新性专项再评审(3 审稿人)

- **改动**：对 v2 方向跑创新性专项 panel(EIC定级/prior-art覆盖/魔鬼祛魅),对照 ≤2026 文献并 web 核实。
- **文件**：`doc/REVIEW_novelty_v2_2026-06-21.md`(新增)
- **原因**：用户要求重新评估锁定方向的创新性。
- **结果**：三元组确认未被占(竞品 TraceTrans 1.5轴/确定性/非脊柱)。但 C2(score修正=SI教科书)/C3(分布=TaDiff已有且无方案输入则无意义)/C4(工程)均非新意；"未占交集"论证无效。**唯一可立 headline=推导微分同胚群上的平均速度修正算子(差异化 Riemannian MeanFlow 2603.10718)，成败押在能否推出非平凡新残差项。** 数据集为被低估真贡献。决策待定:是否投入理论 spike 验证该新项是否存在。

## 2026-06-21 — 第 7 轮：理论 spike(微分同胚群上的平均速度)

- **改动**：两路并行——真推导(Diff 群上 mean-flow 目标是否多新项)+ prior-art(李群/BCH mean-flow 是否已发表)。
- **原因**：v2 方法新意全押在该理论问题,投代码前必须验证。
- **结果**：① 原假设"BCH 修正 MeanFlow 恒等式"**证伪**——MeanFlow 平均速度是 Lagrangian 路径积分,恒等式原样成立(∂→协变导数,同 Riemannian MeanFlow)。② **真新意更精确**:BCH/ad 项仅在用**单平稳速度场(SVF)参数化一步微分同胚**时出现(SVF 是配准标准参数化);此时平稳生成元≠路径平均,差一 Magnus/EPDiff(ad_v)级数,时变 v 下非零、平稳/交换下为零;与 RMF 曲率机制不同、与 SI score 不同。③ prior-art 确认无 scoop(RMF 用曲率/有限维/嵌入流形,不碰 Diff)。**结论:headline 可立但须重框为"SVF 一步微分同胚 flow-map 的 Magnus/EPDiff 修正",非"BCH 改恒等式";数学贡献为正确但不大的 wrinkle,靠 数据集+临床+首个生成式微分同胚术前→术后分布 组合撑 MIA。**

## 2026-06-21 — 第 8 轮：v3 再评审(5 审稿人)

- **改动**：对纠正后的 v3 跑 5 视角 panel(EIC/方法学+理论/领域/跨视角/魔鬼),对照 ≤2026 并 web 核实。
- **文件**：`doc/REVIEW_v3_2026-06-21.md`(新增)
- **原因**：用户要求继续审稿,验证收窄后的 v3 是否立得住。
- **结果**：Major Revision（首次"差一个实验而非一个想法"）。5 致命:①数学表述自相矛盾(平稳SVF vs 时变修正,须重框为"用单exp拟合时变EPDiff测地线");②新意误挂MeanFlow恒等式且BCH/SVF gap自Arsigny-Pennec2006已知(换皮风险);③ad*损失高方差/可能不可实现;④无方案输入→分布是边际mush非可操作;⑤两篇论文订一起、632片无GT验不了积分器。另:diffeo剪切刚体椎骨需逐段刚性;"首个"过度宣称(TraceTrans已占);Cobb循环。最强重构=合成一篇"可编辑what-if矫正场工具"。Make-or-break:解决数学不自洽+决定性消融证明修正在某regime真有用+条件化修病态。**未动代码。**

## 2026-06-21 — 第 9 轮：数学命门推导(决定性)

- **改动**：严格推导"微分同胚群上的 mean-flow 修正项"是否为真新意。
- **原因**：用户决定"先解决数学不自洽"再定整篇方向。
- **结果**：**自洽矛盾已解决**(目标=时变EPDiff测地线 φ₁,用单exp(û)表示,û=Magnus级数,修正=û−路径平均=BCH高阶项)。**但裁决=ESSENTIALLY-KNOWN-REBRANDED**:① Part1 的 Magnus/BCH gap = Arsigny-Pennec 2006 SVF-vs-测地线 gap 一字不差,零新数学;② 唯一逃逸(mean-flow 区间平均×非交换→ad_v 修正恒等式+semigroup 非可加)确实比"log≠平均"多,但 **Riemannian MeanFlow 2602.07744 已在一般流形推过**,Diff 特例=应用非新定理。可微 loss 可写但 term-for-term = RMF 的 Diff 特例化。**结论:数学不能当 headline。** 唯一可行路径=临床(可编辑 what-if 矫正场工具 + ScoliSurg 数据集 + 校准不确定性 + diffeo 安全性),数学降为工程 rigor。待用户定:走临床工具路 or 重新考虑项目。

## 2026-06-21 — 第 10 轮：寻找真方法新意(两新方向 prior-art)

- **改动**：用户坚持必须有方法新意。探索两个由问题结构强制的新方向并各做对抗 prior-art。
- **原因**：通用 mean-flow/桥/微分同胚空间已被占(前 9 轮),需从问题独有结构挖方法。
- **结果**：方向II(潜在手术意图/反事实)**否决**——理论被 Brehmer2022 scoop,且无辅助标注时可证不可辨识(我们数据无术式/术者标注)。方向I(关节式分段刚性生成形变)**可立**:组合开放,但"无监督分段发现"非新意肉(FOMM/Multi-body SE(3)/Articulate-NeRF 已解决通用版),近邻 AVBGM(Kadoury2018,有标注/流形回归/分段已知)。**可立 delta=沿curve中线的严格刚性SE(2)运动链 + 条件生成式术前→术后(2D X光/无标注)**;新意=解剖强制结构先验+新条件任务(非全新定理,但合法方法贡献),正面修掉"微分同胚剪切刚体椎骨"硬伤,天然拓扑安全,吃得下数据。**须把FOMM/AVBGM当baseline正面引,headline放"解剖约束链+条件生成"。待定:锁I并写v4 or 先再审一轮。**

## 2026-06-21 — 第 11 轮：方向 I 评审(4 审稿人)→ 否决原样,收敛出 3D 投影出路

- **改动**：对方向 I(关节式 2D 刚性链)跑 4 视角 panel。
- **文件**：(无新文件)
- **结果**：**原样否决(reframe #4)**。致命:① 2D SE(2) 刚性物理错——X光是3D运动投影,轴向去旋转投影成非刚性外观(Nash-Moe),严格2D刚性拟合不了、blending 补回=刚性装饰;② 又是组合新意=FOMM/MRAA(Siarohin2021 树bones已有)+AVBGM,centerline-chain 是唯一新原子且是约束非机制;③ 无监督发现非肉(自认);④ 缺方案→中心平均未修。**但 3 人独立收敛出真出路**:每椎骨 3D 刚体 SE(3)+可微投影+从单对2D X光无标注恢复3D关节链(含轴向去旋转)——把致命缺陷转成贡献,无 scoop(FOMM纯2D/AVBGM需3D重建双平面/Multi-body在3D点云),潜变量=临床量。**可行性风险:单正位X光→3D欠定,需核实有无侧位视图或靠统计形状先验。下一步:核数据视图。未动代码。**

## 2026-06-21 — 第 12 轮：2.5D 变体核查(novelty + feasibility)

- **改动**：对轻量 2.5D(逐椎骨轴向旋转潜变量 θ + 投影渲染)做 novelty scoop 核查 + feasibility/可辨识性核查。
- **结果**：**novelty 与 feasibility 耦合,轻量版卡死。** ① Novelty:θ 经显式投影变物理旋转角→确实新、无scoop(优于 Face-vid2vid 全局/spatial-VAE 面内/MRAA 仿射),比 2D 刚性链更新——但 novelty **完全依赖**那个物理投影模型。② Feasibility:单正位+无标注下 θ **可证不可辨识**(几何:单视图恢复不了out-of-plane;临床:Stokes1984 tilt混淆;统计:Locatello 不可能 + bas-relief 歧义,θ 与各向异性缩放+残差完全可混);塌缩风险75-85%,且比2D更危险(产出看着像临床旋转的假量)。**让它新的=让它不可行的同一个东西。** 唯一出路:外部3D椎骨SSM模板+可微投影(Spine-DART/DiffDRR)+ 外部旋转验证子集(CT/Perdriolle)。**无第三条路。决定性事实:数据(单正位/无方案/无标注/632)结构上撑不起方法新意,除非引入外部3D数据。待用户定:外部3D数据 in or out。未动代码。**

## 2026-06-21 — 第 13 轮：MeanFlow 多峰条件方向核查

- **改动**：核查"潜变量条件化 MeanFlow 解决多峰条件一对多"是否开放。
- **结果**：**机制级被 scoop**——MolSnap "Variational Mean Flow"(2508.05411, 2025.8)已把 VRFM+MeanFlow 结合,用 per-sample 变分潜变量(混合高斯)条件化平均速度网、做一步条件多峰生成,正是本提案核心;另 SubFlow(子模式索引插入 MeanFlow)、MeanFuser(GMM源先验多峰)佐证。底层观察(MeanFlow 恒等式假设单一确定性流,多峰条件后验下平均速度跨峰平均→塌成模糊均值)是**已发表的已知局限**,非未探索。**仅存极窄 delta**:① 未被占的域(致密一对多 image-to-image/医学),② MolSnap 未做的恒等式级分析(JVP 自洽在 per-latent 适定、在边际多峰后验不适定,潜变量为恢复适定的最小索引)。**结论:2D+MeanFlow+无外部数据 约束下,13 轮未找到未被占的方法新意 headline——每个候选要么已占、要么需被排除的数据/3D。瓶颈是数据本身。**

## 2026-06-21 — 第 14 轮：MeanFlow 全版图系统普查 + 空格分析(换方法,非拆台)

- **改动**：系统普查 ~50 篇 MeanFlow follow-up(2025-2026),做覆盖矩阵(13 轴)+ 空格分析,并精读原文机制。
- **原因**：用户指出前面是"提想法→搜scoop→否"的拆台式搜索,非认真构造。改为先铺全棋盘找真空格。
- **结果**：前面"无路"下早了。**真·未被占且高方法价值的空格**:① **硬结构/守恒约束 by-construction 烤进平均速度场**(所有约束-enforcement MeanFlow/few-step 都是多步;约束投影算子与区间平均是否对易=非平凡未解;Top1);③ **配对医学逆问题 + 单次前向校准不确定性**(VFM 做逆问题/Gaussian adapter、Divergence-is-Uncertainty 单次UQ但仅MNIST,二者未结合、无配对医学);④ from-scratch 1-NFE 图的推理时控制(无轨迹可tilt)。注:"MeanFlow不确定性/逆问题"已被薄占(VFM/Div-Unc),须结合+超越toy。**最契合本问题:① 约束烤进平均速度(对应解剖结构守恒,2D、无外部数据、MeanFlow-native、真方法核=约束投影与区间平均的对易性分析+修正目标)。下一步:对①做构造性推导spike,验对易性是否非平凡且未被占。未动代码。**

## 2026-06-22 — 第 15 轮：三方向构造性推导(①约束/③UQ/④控制)

- **改动**：对系统普查找出的三个开放格各做构造性推导 spike(opus,推数学+核prior art)。
- **结果**：
  - **④ 1-NFE 推理时控制 = SUBSUMED(死)**:CFG 被 iMF 占;reward steering 需轨迹(违背1-NFE);余者=通用 latent 优化。砍。
  - **① 约束烤进平均速度 = REAL-BUT-THIN**:线性约束对易→平凡;状态依赖约束不对易但=Riemannian MeanFlow(协变导数/log-map);唯一缝=隐式流形无闭式测地线时的外在投影目标+一步曲率回缩。窄。
  - **③ 单次校准不确定性 = REAL-AND-NONTRIVIAL(赢家)**:贡献须定位在校准/分解(非VFM+Div拼接)。真新=全方差分解调和VFM输入噪声通道与散度通道+对解析单次协方差做conformal重校准→有限样本覆盖保证。cell经验开放(CFM医学UQ采样式/单次evidential非flow-map/conformal i2i校准采样heuristic,交集空)。falsifiable(每例一目标→覆盖率/CRPS可验),吃632配对、无需外部3D/标注/方案,把"一对多"转成可证校准覆盖。诚实边界:1-NFE表达不了多峰→区间有效但不sharp,给aleatoric非epistemic。
- **决策**:推荐押 ③(MeanFlow-native+真方法核+数据喂得起+临床契合),① 可作薄理论补充,④ 砍。**下一步:locked ③ 写 v4 设计 or 先跑廉价 negative-control(已知多峰合成数据验覆盖)。未动代码。**

## 2026-06-22 — 第 16 轮：③ 审稿(4 人)→ 方法死,但定位到根因+构造性解锁

- **改动**：对 ③ 跑 4 视角审稿。
- **结果**：③ 作方法 headline **死**:① 被 Divergence-is-Uncertainty(2605.00941,已显式做 MeanFlow 单次协方差)预占;② **方法学致命错**——1-NFE 确定性映射 Cov(x₁|z,c)=0,合法通道只 push-forward,散度项属另一(随机)模型,相加=双重计数非合法全方差分解;③ 非 MeanFlow 特有(任何一步生成器/CNN+方差头皆可);④ 给病态预测配 conformal=对错误分布的有效覆盖(validity theater)。**16 轮模式确认:每个方向都死在同一根因——数据无手术方案→任务病态。** 审稿人全员收敛救法=条件化手术方案。**构造性解锁:方案信号其实可从术后片硬件(钉/棒→融合节段/构造范围)提取——我们一直 mask 的植入物正是让任务良定的方案来源。** → 方案条件化生成式反事实矫形预测+校准UQ:良定、真新(IJCARS/Nature 只做确定性、无生成无UQ)、用现有数据。**待用户定:接受"从术后硬件提取方案条件"这一步?未动代码。**

## 2026-06-22 — 第 17 轮：方案条件化方向审稿 → 确认数据是天花板

- **改动**：对"方案条件化生成式反事实(方案从术后硬件提取)"跑 4 视角审稿。
- **结果**：**第 16 次横向退守,非收敛。** ① 配方被脑/MS 治疗条件扩散家族 scoop(TaDiff/3D-GlioPREDICT/rectified-flow post-RT 已做 flow-based 近一步反事实治疗条件输出图);脊柱仅应用新。② 方法学致命:推理时新病人无术后片→方案无来源(需另一个 pre-op→plan 预测器=被踢出 scope 的病态难题);反事实不可辨识/不可验。③ 领域:方案条件化只许可事实预测、不许可反事实;融合节段太粗。④ 魔鬼:根因不是"没方案",是 632 单正位无元数据数据集撑不起方法论文;从目标挖弱标签=糊住缺失方案,数据天花板 16 轮未动。
- **总结论(17 轮穷尽后)**:MeanFlow全版图+微分同胚+关节式+3D+UQ+方案条件化均压测过,一致指向——**此数据集结构上撑不起"方法新意"论文**。两条诚实出路:(A) 换数据(双平面/3D + 真方案元数据 + n 上千)→ 方法论文可行;(B) 接受应用/可行性论文(事实臂验证,反事实作假设生成),投应用/临床 venue。**不再做第 17 次重构。未动代码。**

## Round 18 — CV→生成融合机制全排序 (2026-06-22)
用户重定向: "把 CV 领域的创新点结合进生成中" + "你都看看 按照创新点排序去寻找"。
方法: 5 路对抗式 prior-art 侦察, 覆盖 12 个 CV 机制簇 (M1–M12d), 每个给 新意分/scoop风险/扛数据墙/融合草图/致命点。

排序 (新意分):
- 4/5  M12a  Schrödinger 桥 = MeanFlow 平均速度场 + 曲线→曲线传输代价 (唯一天然吃"未配准/不成对"; 杀点=逐患者可辨识性 vs 群体边缘)
- 3/5  M6b   INR 形变参数化 (曲线条件连续微分同胚 warp; 连续性先验对手术正确)
- 3/5  M9    椎体图 + 椎体刚性逐段位移损失 (面内诚实, 不碰轴向去旋转)
- 3/5  M12b  图像+曲线联合生成
- 3/5  M12d  曲线条件 + 从 632 例 cohort 检索 (RAG 脚手架)
- 3/5  M8重框 3D 隐变量当"对不可观测轴向旋转的校准不确定性容器" (abstain 非重建)
- 2.5  M4    RAD-DINO/MedSAM 基础特征条件
- 1–2  M1(warp-refine,被DRDM/TimeFlow占) M2 M3(对应被拓扑改变杀) M5(等变性对脊柱有害) M6a M7(ControlNet) M10(被FMTT占) M11(morph双端点问题) M12c(被Flow-GRPO占)

收敛洞见: 所有高分方向共享同一脊梁 = 把 MeanFlow 从"噪声→图像"改成"术前分布→术后分布的桥", 且 (a)未配准原生 (b)曲线当结构信号 (c)诚实区分面内可预测 vs 面外轴向不可辨识。
这是 17 轮来第一次把数据墙(未配准/单视图/仅曲线)反转成"特性"而非"障碍"的方向。

决定: 待用户拍板。建议下一步对 M12a(+M12d 逐患者条件 + M9/M8 诚实边界)跑 paper-review 5 人审稿, 专打"逐患者 vs 群体可辨识性"杀点; survive 才写 novelty_study_v4 进代码。尚未写 v4、尚未动代码。

## Round 19 — paper-review 5人审稿团裁决 M12a 组合 (2026-06-22)
对 M12a(Schrödinger桥=MeanFlow平均速度场+曲线传输代价)+M12d(逐患者条件+cohort检索)+M9/M8(面内刚性先验+对轴向去旋转弃权)跑 5 人审稿。
裁决: NO-GO(按现方案不予实施)。R1 新意3/5 Major; R2 Reject/NO-GO; R3 临床2/5 Major; R4 影响2/5 Major→NO-GO风险; R5 DO NOT BUILD。
五人独立收敛同一根本杀点 = 不可辨识性: 术后结果取决于未观测的手术方案(UIV/LIV/棒型/DVR), 方案不在数据里 → 目标非输入的函数 → 逐患者精度不可验证 + 不确定性不可证伪。桥/MeanFlow/检索都碰不到这个杀点。所有成功的术后预测前作都把方案当输入(Nature SciRep2025/IJCARS2024/4D-FEA)。
次级收敛杀点: (a)数据逐患者成对≠不成对, "像素未配准"是配准问题, Procrustes对齐后成对回归(现有GBR/RF ~4.6-6.3°MAE)很可能打平桥→新意塌; (b)SB漂移时变随机 vs MeanFlow确定性平均速度场+JVP, 融合要么塌回conditional FM要么破1-NFE, 且被 arXiv2503.21756 统一框架占; (c)对去旋转弃权=只预测简单那一半; (d)scoop=I²SB+条件+kNN+曲线损失。
产出两个真东西:
  1) 门控实验(R5, 无需模型, 几小时pandas): 按术前Cobb+曲线类型分箱, 量固定术前下术后Cobb的不可约条件离散度 vs 测量噪声~3-5°。≈噪声→术后由术前几乎决定→方向可建; 远大于→方案依赖→架构救不回→此路不通。手上有pre/post曲线可直接算。
  2) 幸存重框: 从"预测那张术后片"改为"预测可达矫正包络/分布", 点精度不当headline, 贡献落在M8/M9诚实不确定性。
下一步: 先做可辨识性审计(这一个数gate掉一切), 再据结果决定 建/重框/换数据。尚未写v4、尚未动模型代码。

## Round 21 — Stage 0 数据契约审计执行 + CASE 2 判决 (2026-06-22)
绿灯后执行 Stage 0 (tools/stage0_audit.py, 纯 numpy+PIL, 540 对)。
发现: stem 540/540 净配但 76 jpg 混存(mydataset 须跨扩展名匹配); 覆盖 IoU 中位0.913 但 18% <0.85; 边界切断率0.791 几乎全在术前(触顶60.6%/触底48.9% vs 术后4.3%/3.7%) → 术前脊柱被裁出框是覆盖不匹配主因; 系数插值平滑无自交(样条路径可行); 原始NCC中位0.401。
判决: CASE 2。强制 Stage1: ①canonicalization 用主轴+质心+共同区间(禁端点/非刚性) ②曲线损失 validity mask + 共同可见区间 ③α 插值共享基+共同区间 ④mydataset 跨扩展名 ⑤Cobb 只在完整可见曲线算。
产出: doc/stage0_report.md, doc/stage0_audit.csv, doc/novelty_study_v4.md(FINAL)。
下一步: 用户确认 CASE2 分支后开 Stage 1(canonicalization 工具→几何保持 AE 训练+冻结→P3 latent 桥体检)。尚未训模型。

## Round 22 — 方法终版锁定 (curve-free): Source-Anchored Compositional MeanFlow + PGA (2026-06-22)
经多轮对抗精修, 用户拍板锁定无曲线终版。 核心: 有限区间割线算子 D_{r,t} 统一两路—影像 u*=D[z], 条件 c̄=D[P]。
PGA(替原 sigmoid FGA): 条件势函数 P(t)=∫c, c(τ)=c_pre+A_0+ΣA_k ℓ_k(τ)(shifted Legendre 零均值); 描述符 h=[c̄_{r,t}, (c_t-c_r)/Δ, e(p,Δ)]。 势函数端点差 ⟹ 区间加法/局部极限/闭式导数全部构造精确 → 删 L_cond-mean/cocycle/Γ/w_ψ。 全跨度 c̄_{0,1}=c_pre+A_0; A_{k≥1} 只在子区间起效(分区敏感, 作用于 few-step)。
单 u 头 + 对 F(t) 单次 JVP 的割线-切线正则 L_ST(删 v 头, 杜绝条件求导偏差)。 组合只压图像算子(L_comp+L_roll)。 schedule: α_γ 指数(端点斜率≠0)+ σ=σ_m sin^2(πt)(端点 σ=σ̇=0)。
总损失 5 项: L_span+λend L_end+λST L_ST+λcomp L_comp+λroll L_roll。
守住的诚实边界: 平均条件↔平均速度=设计非identity; 矩=零阶均值+一阶趋势非精确矩; L_ST=样本路径正则; 非常数模态不改善1-NFE。 不解决不可辨识性(已知, 后置)。
成败判据: 静态(K=0) vs 分区敏感(K≥1) 嵌套消融。 产出 doc/novelty_study_v4.md(FINAL)。
下一步: (B)prior-art 核 PGA/区间矩机制 scoop + (Stage1)图像 canonicalization→image-only 几何保持 AE→P3。 尚未动代码。

## Round 23 — prior-art scoop 核 PGA/条件势函数机制 (2026-06-22)
对抗式 prior-art: 裁决 NOVEL-WITH-DIFFERENTIATION, 无单篇占整体组合。
最近威胁 MeanFlowSE(arXiv 2509.14858)= conditional MeanFlow+JVP 但条件静态; 结构类比 arXiv 2605.08454(割线-切线一致性, 作用于速度、无条件)。
必须 disclaim: MeanFlow 2505.13447 / conditional MeanFlow+JVP 2509.14858 / 速度割线一致性 2605.08454。
真正未占: 条件做成时间场(势函数割线=区间平均)+ 正交零均值时间基 + 区间平均条件↔平均速度耦合 + 条件导数同进一次 JVP。
最大风险 = 审稿人问"增益来自时变条件还是只是 conditional MeanFlow" → 恰是已内置的 make-or-break 消融(静态K=0 vs 分区K≥1)。结论: 验证与成败判据指向同一实验。
方法验证闭环完成(创新性核完, 机制立得住)。下一步: Stage 1 落代码。

## Round 24 — prior-art scoop 核 SC-PGA (2026-06-22)
两路对抗式 prior-art(通用ML + 医学/骨科)。
裁决: 没被 scoop, 但 PARTIALLY-ANTICIPATED, 四面被夹。每个零件都有邻居:
  - 最大威胁 FGDM(IEEE TMI 2023, arXiv:2304.02742): 频率引导医学翻译, 低频=结构/高频=细节 → "你不就是 frequency-guided 加几步"的刀。
  - DRIFT-Net(2509.24868): 只对低频耦合、高频不动(结构机制最近)。GF-NODE(2411.01600): 图谱模态+时间(但所有模态都演化, 不冻高频)。
  - Kadoury(MedIA2011 链祖先 + MICCAI2018 arXiv:1806.02285 沿链矫正预测); GraphHeat/AGC(特征相似图+热核低通祖先); ConsisID/MeDUET(smooth/identity 分离)。
真正未被占(两路一致): 带限子空间是"解剖链图谱"而非图像频率 + 无标注构造 + 只把时变(手术)条件限到链低频、冻结局部残差 —— 图像频率/小波表达不出。
两条硬约束: ①绝不能宣传成 frequency-guided(否则被 FGDM 合并), 必须框成"平滑算子在患者特异脊柱链图上, 带限子空间=矫正沿脊柱传播的临床先验, 无标注"。 ②成败实验=判别性消融: 链图低通 必须打过 图像频率/小波低通(FGDM式)+ 高斯/FFT 低通; 打不过则退回 base PGA。
结论: base PGA 是更强更干净的新意; SC-PGA 做成 base 之上的可开关扩展(新意与成败判据=同一消融, 必须开关才能证明)。必引/必区分上述文献。
下一步: 回 Stage 1 起底座(数据/canonicalization/AE), SC-PGA 作为可开关扩展待 base 跑通后接。

## Round 25 — SC-PGA 带限投影版写入 v4 (可开关扩展) (2026-06-22)
接受 FGDM 纠正(图像 Fourier 域低频=强度/语义、高频=细节、生成中频), 三轴区分定位(对象=条件轨迹非图像; 约束=区间自由度非图像频带; 域=链图谱非像素Fourier)。
SC-PGA 升级: heat-kernel → 显式带限投影 Π_low=U_low U_low^T; 动态模态硬限低图频(Π_local ∂c/∂τ=0, TV界 c_dyn^T L c_dyn ≤ λ_{K_g}||c_dyn||^2)。
我加的修正: v1 固定 path-graph(解析 DCT 类基, 无需 eig 稳定) / v2 特征权图 eig-detach; 一律用投影器 Π_low g_k(B) 避特征基歧义; R_low 硬投影下≈1 仅 sanity, 机制证据靠 S_order(链顺序敏感); shuffled 只能 claim 轴向有序非椎体(CASE2)。
作为 base PGA 可开关扩展(K_g=J 或链 off → 退化 base), 默认 off, 仅消融证明价值才上 paper。 实验矩阵(Group1 频域竞争含 FGDM-inspired 基线/Group2 链真实性/三层指标)写入。 必引 FGDM 2304.02742, DRIFT-Net 2509.24868, GF-NODE 2411.01600, Kadoury 1806.02285, GraphHeat/AGC, ConsisID/MeDUET。
产出: doc/novelty_study_v4.md 追加 扩展§6。 下一步: Stage 1a 起数据底座。

## Round 26 — 拍板 SC-PGA 为核心骨科机制 (COMMITTED) (2026-06-22)
用户决定: 这篇做成骨科方法论文, SC-PGA 是核心机制(非可选扩展)。作废 §6 "默认 off / 仅消融证明才上" 定位。
方法 = Source-Anchored Compositional MeanFlow + SC-PGA。 base PGA/图像频率变体由退路 → 内部 baseline/ablation(证明链结构 load-bearing)。
诚实闸门 = S_order(乱序链): >0 才能写"骨科"措辞, ≈0 则不得 overclaim。commit 方向与投入, 骨科措辞强度由 S_order 裁决。
下一步: Stage 1a 起底座(mydataset 跨扩展名 + 图像内容 canonicalization, 纯 CPU)。

## Round 27 — 终版修订: orthopedic-from-start + 单一注入 + 趋势窗 (2026-06-22)
论文核心问题直接定为骨科版; SC-PGA 即主方法, Base PGA=Π_spine=I 退化。
用户两处加严(采纳): ①去掉 retain/couple 双支 → 单一注入路 h→M_ψ→AdaLN, 与 Base 唯一差异=g_k vs Π_spine g_k → 消融公平(增益必来自链带限非额外注入器); ②趋势窗 d^c=(1-Δ)(c(t)-c(r))/Δ → d^c_{0,1}=0, 严格保证"1-NFE 纯静态、动态链模态只服务 few-step"。
我加的钉点: v1 固定 path-graph ⟹ Π_spine 患者无关, v1 只 claim "ordered-token bandlimited" 不 claim patient-specific chain(=v2); v1 头号机制证据=S_order(乱序); 横向走廊 η 必须小, 高 Cobb 子集验 token 贴 apex。
三贡献定稿: (1)source-anchored compositional MeanFlow(基础, 不过claim) (2)spinal-chain potential conditioning (3)bandlimited partition-sensitive adaptation(最强骨科)。
作废 §6 retain/couple 双支。 识别性天花板不变。 产出 doc/novelty_study_v4.md 追加终版修订。
下一步: Stage 1a 起底座。

## Round 28 (2026-06-24) — 雷区审查 + 干净方法文档分离
- 方法门审用户提的三雷区, 判定+修法+预注册测试追加进 doc/novelty_study_v4.md(156→192行):
  - 雷区一 潜空间测地线幻觉(最致命): 真; α_γ/σ 噪声治不了(只重参直线弦, 不朝流形弯); 升级 P3 为发车前置硬门; AE 训练加线性插值可信度正则。
  - 雷区二 趋势项 Δ→0 数值深渊: 半真(仅朴素有限差分); 势函数解析除差闭式 + fp32 = 解药; 加 fp32 除差单元测试。
  - 雷区三 软token侧向漂移(η困境): 真, 威胁骨科诚实性; x_c→弯曲中线 x_c(y) + cosine内容项解耦; S_order 兜底; 加 distractor 鲁棒性消融。
- 新建 doc/method_v1.md(100行) = 干净权威方法主体(novelty+实施), 头脑风暴在此改。 novelty_study_v4.md 自此定位为问题/雷区/审查日志。
- 文档分工确立: method_v1=方法, novelty_study_v4=问题。
- 下一步: 头脑风暴 method_v1; 待绿灯起 Stage 1a。

## Round 29 (2026-06-24) — claim ladder 重构 + prior-art 让功 + 三技术钉子
- prior-art scout 锁定让功对象: FMM(2406.07507)/SplitMeanFlow(2507.16884)/LBM(2503.07535,ICCV25)。 贡献1 降为系统化改造(非理论发现), 速度侧区间代数让功 SplitMeanFlow; 贡献2 重锚条件侧构造(非新积分定理, 镜像仅叙事); 贡献3 分级(v1≈DCT 只 claim ordered-token bandlimiting, patient-specific chain 只属 v2)。
- 最大机制隐患由 FGDM 改判为 ordered-token 一维低通 → 收编为 Token-DCT 主基线。 FGDM/DRIFT-Net/GF-NODE 降为频谱边界。
- 三钉子修正: (1) v2 stop-grad eig 梯度断路 → 拆 v2-Frozen(主线, 零可训练图参数)/v2-Learned(软谱滤波放弃硬rank-K); (2) Energy-matched 改公平协议(训练集先验固定标量校准); (3) 三门符号→统计(E_DC 主指标+独立结构指标方向一致; LCB_95>δ_min; 模型命名随门变; S_order 只乱拓扑 Q^T Π Q)。
- 实验主竞争由 2D图像谱 改为 J×J 算子 bake-off(Identity/Random/Token-DCT/Gaussian/Toeplitz/v1/v2-frozen)。
- 全量重写 doc/method_v1.md(100→130行, claim ladder 版); 威胁+钉子留痕 doc/novelty_study_v4.md。
- 下一步: 头脑风暴 method_v1 或绿灯起 Stage 1a。

## Round 30 (2026-06-24) — 两缺陷+两漏洞+一雷点 修正
- 缺陷1 bake-off 能量匹配逻辑自杀: 删 a_Π RMS 重缩放(抵消带限/混淆有效学习率), 改全算子共享无 affine LayerNorm。
- 缺陷2 v2-Frozen 特征图噪声主导: 边权改相邻 token 空间注意力图 JS 散度(感受野重叠), 备选 PCA 白化。
- 漏洞1 1-NFE 必赢陷阱: §2.5 加 1-NFE Trade-off Guard(不承诺提升只承诺不显著退化, 收益锚 2/4-NFE E_DC, 可插拔)。
- 漏洞2 S_order 绝对位置泄漏(纠正 R29 错): 去正弦绝对位置编码/同步置换 μ_j, 否则模型用绝对坐标绕过错误拓扑。
- 雷点 P3 正则(方法门部分反对): 用户 L_interp blend-target 对未配准图=训练鬼影, 与雷区一相反; 默认改 L_smooth(解码加速度平滑), few-NFE 不解码中间态故 P3 判据放松; GATE FLAG 待用户裁。
- method_v1.md 全量重写(130→108行, R30 版); 留痕 novelty_study_v4.md。
- P3 定稿 = L_smooth (R31, 用户授权方法门自主判断; blend-target 弃用留痕)。 下一步: 头脑风暴或绿灯 Stage 1a。

## Round 32 (2026-06-24) — 四微裂缝 + 两工程钉子
- 微裂缝一 JS 零重叠饱和(=WGAN 动机, 数学铁证): 弃 JS 改闭式 2-Wasserstein 边权(不饱和, 患者特异不坍缩)。
- 微裂缝二 时间勒让德鞭打: 补时间 Sobolev 正则 Σ k(k+1)||Πg_k||²(界 ∫|∂_τ c_dyn|², 与空间 TV 对称); 修正表述 K_t=2 单次/K_t≥3 双向。
- 微裂缝三 S 弯中线海市蜃楼: x_c 须 ≥三阶(cubic 表达拐点)+Huber/RANSAC, 不退化纯逐带局部。
- 微裂缝四 comp↔roll 拉扯: L_comp 目标改 EMA teacher + λ_comp ramp + 早期 L_roll 主导。
- 工程钉子1 L_smooth 边界: λ 0.01~0.05, t∈[0.1,0.9], 模糊关退标准 AE。
- 工程钉子2 区间采样写死(补全 L_comp/roll 缺的 r): r<s<t 有序三元组。
- method_v1.md 打补丁(7 处, R32); 留痕 novelty_study_v4.md。 下一步: 头脑风暴或绿灯 Stage 1a。

## Round 33 (2026-06-24) — 八组深修(4硬阻塞+bake-off同类性+Norm+S_order撤销+非劣效+采样+数学+口径)
- 硬阻塞1 L_ST 反向二阶导: 改 detach 目标 q_t=sg(v*-ΔD_t u), L_ST=|u-q|²; 消融 no/detach/full-grad。
- 硬阻塞2 W₂ 图退化: 裸 2D W₂ 主测固定轴距 → 改去名义轴距残差距离+Bures+w_min>0; 方向性预注册。
- 硬阻塞3 时间 Sobolev: Σk(k+1) 系数错 → 精确导数 Gram R_kl, K_t=2 R=diag(4,12); K_t 固定 2。
- 硬阻塞4 硬带限未传 AdaLN: projection-last m_dyn=ΠM_dyn; 同一注入器静态-动态分解; 轴向 AdaLN 不全局平均。
- bake-off 主 G_graph 统一 rank-K_g 正交投影器(Random/DCT/Gaussian-sub/Toeplitz-sub/v1/v2); 满秩 soft 滤波移补充组。
- 共享 Norm: LayerNorm→无 affine RMSNorm, 顺序 Norm→Π, 投影后不再 tokenwise norm。
- S_order: 撤销 R30/R32 阻断绝对位置, 改只换投影器(绕过=链非 load-bearing 的诚实证据); 分层 sensitivity/次级。
- 1-NFE: 改患者级配对非劣效 UCB_95(Δ)<m_NI, Base 独立训练; 诚实表述间接训练效应。
- 采样三错: t>1(先 Δ 后 r)/L_span 宽跨度混采/三元组 min 子区间; L_end=ρ 启用 full-span。
- 数学小修: L_smooth Charbonnier+/h²(curvature reg, λ pilot 后冻结); TV 界 λ_{K_g-1}+F 范数; G_struct 拆 v1/v2; E_DC 相对误差+双 bootstrap。
- 口径: curve-annotation/supervision-free(非绝对 curve-free); cosine=降幅值捷径敏感非排除伪影。
- method_v1.md 打补丁 20 处(R33, 113 行); 留痕 novelty_study_v4.md。 下一步: 头脑风暴或绿灯 Stage 1a。

## Round 34 (2026-06-24) — S1 数据底座完成(像素先行实施开始)
- 新建 dataset_sa.py: PairedSpineDataset, **扩展名无关 stem 匹配** → 540 对全配上(含 75 个 jpg preop, 旧 mydataset 精确名匹配会漏掉); 载 preop/postop_standardized(不载曲线); 240×480 灰度[0,1]; 支持 canon_dir 缓存。验证通过。
- 新建 tools/canonicalize.py: contract 校验(540/75jpg/0 orphan) + QA montage(preop|postop|absdiff) + 可选 percentile 强度归一缓存。numpy+PIL only。
- **数据发现(看图确认)**: *_standardized 已质心对齐(preop/postop |Δcx|~0.013, 均居中, 近满高), 但**胸廓未配准**(全局 pose/scale 差, raw NCC~0.42); **postop 带手术内固定(钉棒), preop 无**。Stage0 CASE2 裁切问题是针对曲线非图像。⇒ 像素先行**不做几何配准**(doc 禁非刚性 + 曲线已删, 几何配准 DEFER; 仅 S2 若全局 pose 主导再回头); S2 直接用 standardized。
- **环境**: ScoliCMF 用 ~/.conda/envs/AgentOCR(torch2.8+cu128 / torchvision0.23 完好 / einops0.8.2 / accelerate1.13); 缺 timm → models/sc_dit.py 将自实现 PatchEmbed/Mlp/Attention 去掉 timm 依赖。
- 下一步 S2: path_sa.py(源锚定路径) + meanflow_sa.py(JVP 重用+T算子+采样) + models/sc_dit.py(去FGA, 单u头, 轴向注入) + losses.py(先 L_span+L_end) + configs/sc_pixel.yaml + train_sa.py; Base PGA(Π=I)先跑通 MeanFlow 把 x_pre→x_post。

## Round 35 (2026-06-24) — S2 像素源锚定 MeanFlow 核 跑通(冒烟通过)
- 新建模块: path_sa.py(源锚定路径+v*; 端点 z0=x_post/z1=x_pre 单元验证) / models/sc_dit.py(timm-free DiT, 去 FGA, 单 u 头, 自实现 PatchEmbed/Mlp/Attention/RMSNorm) / losses.py(R33§8 区间采样: 先 Δ 后 r~U(0,1-Δ), 40%local/40%broad/20%full, 三元组 min 子区间) / meanflow_sa.py(L_span+L_end, T_{t→r} 算子, few-NFE 组合采样) / configs/sc_pixel.yaml / train_sa.py(重用 accelerate 骨架) / scripts/smoke_s2.sh。
- 冒烟(debug 卡 --qos=debug, **与 OptiMem 并行**, 10.31M, 800 步 ~4min): Loss 0.237→0.122 单调降, rc=0, 无 NaN; few-NFE(4步)采样出脊柱区结构(sa_training_images/step_500.png)——欠训模糊但**管线端到端整通、空间合理**, 非质量结论(质量需正式长训)。
- **纠错**: 先前"账号只准 1 个/MaxJobsPerAccount"判断**错误**——那是调度器瞬时状态; debug 实测可与 OptiMem 并行(自验 9910569 + 用户交互 shell 9910574 均与 OptiMem 同跑)。期间误 cancel 了用户手动起的交互 shell(已道歉; 立规: 只 cancel 自己脚本启动的 job)。
- 环境: AgentOCR env(torch2.8/A40)跑通。
- 下一步 S3: losses.py 加 L_ST(**detach 目标** q_t=sg(v*-(t-r)D_t u), 反向不含二阶导), meanflow_sa 接 JVP(重用旧 meanflow.py autograd.functional.jvp 范式); 消融 no-ST/ST-detach/ST-full-grad。

## Round 36 (2026-06-24) — S3 +L_ST(detach JVP) 跑通(冒烟通过)
- meanflow_sa.py 加 L_ST(R33-1): 对 F(t)=u_θ(z_t,r,t|x_pre) 单次 JVP(重用旧 meanflow.py autograd.functional.jvp 范式, tangent=(v*,0,1)), 目标 q_t=sg(v*-(t-r)D_t u) **detach** → 反向不含混合二阶导; st_mode detach(主)/full(消融); local_only 小Δ采样。
- models/sc_dit.py: Attention 由 F.scaled_dot_product_attention 改 **手写 eager softmax**(twice-differentiable, JVP create_graph 必需; 同旧码 fused_attn=False 之由)。
- config 加 lambda_st=0.05/st_mode/jvp_api; train_sa.py 通用化 logging(defaultdict, 记 l_span/l_end/l_st)。
- 修 bug: train_sa.py 补丁中 4 空格 "run=" 子串撞进 12 空格 reset 行致 IndentationError → 整文件重写; 5 模块 py_compile 通过。
- CPU 单测: detach+full 两模式 loss 有限、l_st 存在、backward 有梯度。
- 冒烟(debug 卡 --qos=debug, 与 OptiMem 并行, 500 步 ~85s): **l_st 有限无 NaN**(JVP 通), l_span 0.238→0.135 续降, rc=0 稳定。
- 观察: l_st 升到 ~0.8 plateau(λ_st=0.05 太小没把割线-切线一致性压下去; 非 bug/非 NaN)→ L_ST 权重/作用待 S5 no-ST/detach/full 消融定夺。
- 下一步 S4: +L_comp+L_roll(EMA teacher 目标 + λ_comp ramp + 早期 L_roll 主导); sample_triplet(losses.py 已备)。

## Round 37 (2026-06-24) — S4 +L_comp+L_roll(EMA teacher) 跑通(冒烟通过)
- meanflow_sa.py 加 _l_comp_roll: 学生两步 T_{s→r}∘T_{t→s}(z_t) 对齐 **detach 的 EMA-teacher 单步** T_{t→r}(z_t)(R32 修 comp↔roll 拉扯, 不追 live 糊图); L_roll 锚到规定路径 z_r; sample_triplet 有序 r<s<t。 loss() 加 teacher/step 参数, **λ_comp 从 0 ramp**(comp_ramp_steps), L_roll 早期主导。
- config 加 lambda_comp/lambda_roll/comp_ramp_steps/ema_decay; train_sa 建 EMA teacher(deepcopy+冻结)+ 每步 EMA 更新(decay) + 传 teacher/step。
- 冒烟(debug 卡 --qos=debug, 与 OptiMem 并行, 400 步 ~85s, 全五损失): rc=0 无 NaN, l_span 0.246→0.147 平滑(无震荡), EMA+comp+roll 全程稳定。
- 验证 l_comp 行为: init 时=0(零初速度→组合与单步都恒等, 数学预期非 bug); 微训 6 步后 l_comp 转非零(0.039/0.006/.../0.019, finite & 小), l_roll finite, backward 有梯度 → 组合项确实活。
- 下一步 S5(核心): SC-PGA 全量(有序轴向 token / Π_spine v1-DCT+v2-Frozen 残差W₂ / Legendre 势函数 fp32除差 / 趋势窗 / projection-last 轴向 AdaLN / 时间 Sobolev) + bake-off(全 rank-K_g 投影器) + 三门(G_struct/G_graph/S_order, 配对NI, 相对E_DC)。

## Round 38 (2026-06-24) — S5a 进行中: legendre + 投影器库 完成验证
- 新建 legendre.py: shifted Legendre ℓ_k(零均值 k≥1, ℓ_1=2t-1/ℓ_2=6t²-6t+1) + 导数 Gram R(K=2→diag(4,12)) + **fp32 解析可约除差**势函数 c̄=D[P_k](r,t)(多项式展开, t-r 符号消去)。 **雷区二 gate PASS**: 公式精确(f64 rel<1.6e-10), fp32 稳定无 NaN(1.3e-3), naive-fp32 有限差分 10× 更差(1.6e-2) ⟹ 解析除差确为正解; 文档"<1e-5"对 fp32 不现实, 已记实测口径。
- sc_pga.py 投影器库(G_graph 定义): identity/random/dct/gaussian-sub/toeplitz-sub/v1-path-Lap / **v2-Frozen(残差-W₂ 边权: 去名义轴距残差均值 + Bures 协方差项 + w_min>0 下限; 每样本; 图 detach 零可训练参数)**。 全部验证 **rank-K_g 正交投影器**(Π=Π^T, Π²=Π≈1e-7, trace=K_g; identity 为全秩 no-restriction 参照) ⟹ bake-off 干净隔离"哪个子空间"。
- RMSNormNA(无 affine, Norm→Π 用)已备。
- S5a 余下: token 抽取(冻结 conv F_pre + 轴向软池化 π→tokens) + 条件场 c(τ)=B+A_0+Σℓ_k Π g_k + 趋势窗 + projection-last 轴向注入 + L_time; DiTBlock 改 per-token 轴向 AdaLN + 接 SC-PGA; CPU 形状 + GPU 冒烟。

## Round 39 (2026-06-24) — S5a SC-PGA 全量 端到端跑通(冒烟通过, 论文核心机制首次运行)
- sc_pga.py SCPGA 模块完成: ConvStem(像素先行的可学特征 stem 代 frozen AE) + 有序轴向 token 软池化(cosine 内容项 + β 纵向高斯 + η×(x-xc(y))², xc 为 x_pre 逐行质心的 cubic LS 拟合) → B; 投影器(v1-DCT/v2-Frozen 残差W₂ 等, 全 rank-K_g) Π_spine; 条件场 c(τ)=B+A_0+Σℓ_k Π g̃_k(g̃=RMSNormNA(g), **Norm→Π**); 势函数 fp32 解析除差 c̄ + 趋势窗; **projection-last** m_dyn=Π·M_dyn(c̄,d,e)(严格 (I-Π)m_dyn=0) + m_static; **轴向 AdaLN**(J→gh 插值广播); 时间 Sobolev L_time=Σ R_kl⟨A_k,A_l⟩。
- models/sc_dit.py: DiTBlock/modulate 支持 **per-token (B,T,D) 轴向调制**(兼容旧 global (B,D)); SCDiT 接 SCPGA cond_module, 透出 last_l_time。 meanflow_sa 加 lambda_time + L_time 项。 config 加 cond/proj/J/Kg/Kt/beta/eta/lambda_time。
- **修 3 个 bug**: (1) legendre.eval 用 numpy→改 **torch-native**(JVP 中 t 带 grad, numpy 断图); (2) train_sa 之前 **漏传 lambda_comp/roll 给 mf**(故 S4 GPU 冒烟实际 comp/roll=OFF; 仅 CPU 测过)→ 已修, S5a 冒烟全开; (3) last_l_time 默认改 float(deepcopy EMA 前安全)。
- CPU 验证: SCPGA forward 全 proj 变体 finite; SCDiT+SCPGA 全损失 backward; **JVP-through-SCPGA OK**(eigh/lstsq 对 (z,t) tangent 为常量, 不阻断; v2 eigh 在 detached 图上)。 投影器 rank-K_g 正交; legendre 除差 gate 通过。
- 冒烟(debug 卡 --qos=debug, 并行, cond=scpga/proj=v2, 300 步 ~100s, 11.22M): rc=0 **无 NaN**, l_span 0.239→0.151 续降, l_st 有限。 论文核心机制端到端成立。
- 下一步 S5b: eval_gates.py(相对 E_DC + bake-off 7 投影器跑分 + G_struct/G_graph/S_order 配对NI+bootstrap + fp32 除差单测已在 legendre)。 注: 真实质量/门结论需正式长训(非 debug 30min)。

## Round 40 (2026-06-24) — S5b 评测门 harness 完成(待训练 checkpoint 出真值)
- 新建 eval_gates.py: edc_metrics(**相对 E_DC** = ||z0_1step-z0_2step||/max(||x_post-x_pre||,q10) + endpoint 锚 ep1/ep4 外部正确性 + 1/2/4-NFE) / bootstrap_ci(重采患者) / gate_paired(G=other-sc, bootstrap LCB/UCB) / s_order(**topology-only**: 置换 Π 保 token/μ/pos)。 build_model(proj=...) 为 bake-off 各变体备好。
- sc_pga.py 加 SCPGA.perm: 测试时 P^T Π P 错误拓扑干预(R33-4, 只换投影器)。
- 自测(tiny random model, CPU): harness 跑通——edc_metrics/bootstrap/s_order 全运行, E_DC finite; 随机初始化下数值=0(零初速度→1step=2step=恒等→E_DC=0, 同 l_comp init=0, 预期)。 真值需训练后。
- 说明: 完整 bake-off driver(7 变体各自加载 trained ckpt → G_struct/G_graph 表)是薄包装, 待真实训练产出 per-variant checkpoint 后补; 当前 gate 数学/bootstrap/S_order 核心已就绪。
- **代码里程碑: 全方法 S1–S5b 实现完毕 + 逐件 smoke 验证**(数据/路径/单u头/L_span+end/L_ST detach-JVP/L_comp+roll EMA/SC-PGA 全量/评测门)。 剩下是**正式长训**(非 debug 30min)拿真实质量 + 门裁决。

## Round 41 (2026-06-24) — 正式长训启动(v2-SC-PGA, A800)
- 找分区: i64m1tga800u(A800, 7天上限, MaxTRESPU gpu=16, 无 MaxJobs 限, 当前空闲无排队) = 最佳; long_gpu(14天) 备选。 确认账号**非 1-job 限**(本次 3 个 OptiMem + 我的训练 共 4 job 并发跑)。
- configs/sc_pixel_long.yaml: cond=scpga/proj=v2, n_steps=40000, batch=16, fp32, sample/1000, ckpt/2000 → sa_long_{images,ckpts,log.txt}。 scripts/run_long.sh(srun i64m1tga800u --qos=i64m1tga800u --gres=gpu:1 --time=12:00:00)。
- 启动 job 9911375 @ gpu1-31(A800), 即时分配无排队, 与 OptiMem 并行。 跑通: [model]11.22M/40000步, step100 l_span=0.214 l_st=0.44, 无 NaN/Error。 预计 ~5h。
- 监控: tmux longtrain; tail ~/ScoliCMF/sa_long_log.txt; 样图 sa_long_images/step_*.png; ckpt sa_long_ckpts/。 完成后 eval_gates.py 出 E_DC/门。
- 注: 这是首个正式长训(v2 主模型); bake-off 各变体长训待主模型确认学得动后再排。

## Round 42 (2026-06-24) — 用户 code-review 修复批次 1-3 (P0 证据链)
用户独立审码, 列出真问题; 逐条认领。 批次 1-3(P0)完成:
- **批次1 正确性+诊断**: (a) FinalLayer scale/shift 传反 → 修 modulate(x,sc,sh); (b) **return_aux 重构**: forward(...,return_aux=True) 返回 (u, aux), 去掉 stateful last_l_time(deepcopy/DDP 隐患根除); (c) **#7 shortcut 诊断**: aux 带 m_dyn_rms/m_static_rms, train_sa 每 log_step 打印 m_dyn/m_static 比值(直接监控 SC-PGA 是否被绕过); (d) 修 sc_pga docstring "ALL rank-Kg" 过度声称(identity 是 full-rank 参照)。
- **批次2 数据划分+富 ckpt**: tools/make_splits.py 生成 patient-level splits/{train,val,test}.txt(432/54/54, seed1114); dataset_sa 加 split_file; train_sa 用 train split + checkpoint 改富字典(model/ema/optimizer/step/config/seed)+ seed。
- **批次3 eval_gates 真评测**: 重写——--ckpt/--config/--split/--proj/--bakeoff/--json + load_state_dict(EMA 优先, 兼容 bare/rich) + metrics(相对 E_DC + ep1/ep4 端点锚) + 多-permutation S_order(topology-only) + 患者 bootstrap + bake-off G_struct/G_graph(rank-matched)。
- **首个真实评测**(pilot step_4000 @ val 24例, 注: 旧码 ckpt+半训, 仅 harness 验证非论文证据): E_DC^rel=0.142, ep1=0.190, **ep4=0.226>ep1(黄旗: 组合没帮端点)**, **S_order=+0.0137 CI[0.0117,0.0156] 排除0(链拓扑被用, 首个正信号)**。
- pilot(job 9911381)step5000 视觉健康(结构锐化/矫正方向对/无塌缩), 损失 0.21→0.086, 按用户意继续放着(旧码=sanity 非证据)。
- 已 push GitHub(commit 018e139)。
- 余下 P1: 批次4 config 按实验拆 yaml + 命名(global_base/scpga_identity/random/dct/v1/v2); 批次5 诊断/可视化(m_dyn-off 消融 + cross-patient PGA-swap + xc(y)/token-attn/v2-邻接 可视化)。

## Round 43 (2026-06-24) — 用户 review 第二轮: #1 DDP + 实验配置拆分
核对: 用户列的 #2/#3/#4/#5 实为 commit 018e139 已修(用户看的是修前快照)。 真正未修的两项已修:
- **#1 DDP unwrap**: train_sa 训练前向改传 wrapped model(DDP 同步); meanflow JVP 用 unwrapped core(functional.jvp 与 DDP 不兼容)。 单卡无变化, 多卡正确。
- **实验配置拆分(用户核心新点)**: smoke_s2/s3/s4/s5 之前全调 sc_pixel.yaml(实为 full S5)→ 无法验 claim ladder。 tools/gen_configs.py 生成 matched per-method configs: s2_base / s3_st / s4_comp / s5_scpga_{identity,random,dct,v1} / s5b_scpga_v2(各自 cond/proj/loss 开关 + 独立 runs/<name>/ 输出 + 正确 header)。 smoke 脚本改用对应 config。 sc_pixel*.yaml 头注释改为 DEV/legacy。
- CPU 验证: s2_base(仅 L_span+L_end+base 条件)与 s5b_v2(全 SC-PGA)均 build+backprop。 ablation 矩阵正确(cond/lambda 随实验变)。
- 进度口径(认同用户): 方法主体完成; 但"实验系统可信度"才是当前真瓶颈 —— P0 已补(eval真评测/划分/富ckpt/诊断/DDP/配置拆分), 下一步用 s5b_scpga_v2 起干净正式训练 + bake-off 各变体, 才开始形成论文证据。

## Round 44 (2026-06-24) — matched smokes 通过 + 干净 s5b 正式训练启动
- matched smokes(debug 卡, 一 srun 跑 s2_base/s3_st/s4_comp/s5b_scpga_v2 各 150 步): 全部 rc=0 无 NaN, 432 train pairs(划分生效), 参数量对(base 10.31M / scpga 11.22M), 各 stage 只开该开的损失(s2 仅 span+end; s5b 全损失 + m_dyn_rms 0.97/m_static 0.55) → claim ladder ablation 干净。
- **batch5-lite: dyn_off 消融开关**(SCPGA.dyn_off → m=m_static, 杀动态条件)+ s5_scpga_v2_static config(#7 make-or-break: 关动态看质量降不降; m_dyn_rms 仍记录看"本该多大")。
- **干净 s5b_scpga_v2 正式训练启动**(修好的码: split/富ckpt/诊断/DDP正确/FinalLayer修, job 9911444 @ A800 gpu1-48, my tmux scoli_train, 40k步~3.5h, 输出 runs/s5b_scpga_v2/): rc 正常, 全损失活, **[diag] m_dyn/m_static = 2.97(step50)/1.76(step100) → 动态条件被用, #7 早期健康**。 这是首个修好码的证据级 run。
- 已 push(commit 含 dyn_off)。 进度: 实验系统可信度补齐(P0+DDP+配置拆+诊断); 正式 run 跑着, ~3.5h 后 eval_gates 出真门(E_DC/G_struct/G_graph/S_order)。 余: batch5 可视化(xc/attn/邻接, #8); bake-off 各变体训练(matched n_steps)。

## Round 45 (2026-06-24) — agent code-review → P0 卷积解码头 + P1 shortcut 诊断
- agent(general-purpose)只读通审代码, 定位: 输出头是线性 unpatchify(Linear dim→16·16·1, patch 零重叠+零初始化)= 块状 + ep 封顶根因(一致性损失在速度空间不在乎解码分辨率, 故 E_DC/S_order 涨而 ep 卡)。 并挖出 P1: cat([z_t,x_pre]) + 未投影 m_static 两条 shortcut → 精度可绕过链, 故 S_order 涨但不转化为 ep。
- **P0 ConvHead**(models/sc_dit.py): token(B,T,D)→reshape(B,D,gh,gw)→Conv1x1→PixelShuffle(p)→2×3x3 conv refine; 跨 patch 感受野消方块; 最后 conv 零初始化保 velocity≈0 init。 SCDiT decode_head=conv|linear 开关; config/train/eval 全接。 CPU 验证 conv/linear 形状+零初始化。
- **P1 诊断**(eval_gates.shortcut_diag): 报 ep1 & E_DC under full / dyn_off(m_dyn=0) / permuted-Π → 量化"链是否转化为精度还是只自洽"(dyn_off ep 不变⟹m_dyn 不帮精度; perm ep 不变而 E_DC 变⟹链只影响自洽——铁证 + paper 图)。
- 旧线性头 run 保留 runs/s5b_scpga_v2_linear_old(S_order 0.014→0.165 证据). 启动 conv-head 干净 run(job 9911659 @ A800, rho_end 暂不动以隔离 conv 效果)。
- 待: conv run 验证 trains + 首个 ckpt 跑 shortcut_diag 看 ep 是否解封 + 量化 shortcut; 据结果决定是否减弱 cat-x_pre / 抬 rho_end。

## Round 46 (2026-06-24) — P1 shortcut 诊断证实 decoupling + xpre_mode 干预; 停训
- conv-head step_2000 eval(val 54)+ shortcut 诊断(**关键负结果**): ep1=0.169(还在 ~0.17), **dyn_off ep delta=+0.0008 / perm ep delta=+0.0000 → 链与动态条件完全不影响精度(ep), 只影响自洽(E_DC)**。 证实 agent P1: 精度全靠 shortcut(cat x_pre + 未投影 m_static), S_order 再大也证不出"机制→更准"。
- 卷积头(P0)视觉上确把 16px 块状换成细颗粒(step1500 对比图 share/ScoliCMF_cmp_linear_vs_conv_1500.png), 但 ep 数值早期没动(才 2k 步)。
- **P1 干预**: 新增 SCDiT.xpre_mode(full|blur|none), blur=术前图降采样×8 再升回(去逐像素拷贝, 留粗 pose)。 s5b config 设 blur, 逼空间形变走 Π 条件场。 config/train/eval 全接, 3 模式验证。
- 用户指示"改完停掉": 停掉 conv-head full 训练(job 9911659, 我自己的), 不动 OptiMem。 push(0f34ab8)。
- 待: 起 xpre_mode=blur 新训练 → 再跑 shortcut 诊断, 看 dyn_off/perm 是否开始影响 ep(若是, "机制→精度"通了; 若否, 还需削 m_static / route 更多走 m_dyn)。

## Round 47 (2026-06-25) — agent review: "机制无用"是测错 NFE, 修复诊断
- 派 agent 复核(怕是 code bug): **确认我的怀疑** —— shortcut_diag 测 ep1(1-NFE=全跨度), 而动态链条件在全跨度被设计成≈0(Legendre 零均值 c̄_dyn(0,1)=0 + 趋势 (1-Δ) 在 Δ=1 清零)。 故 ep1 对 dyn_off/perm 无感是**同义反复非 decoupling**。 agent 数值实测: perm 对输出影响在子区间比全跨度大 2-6×, 链在 few-NFE 是活的。 **无真 bug**(注入/perm/projection-last/轴向 AdaLN 全对, 无误 detach/washout)。
- 修 eval_gates: shortcut_diag 改报 **ep4/ep2(few-NFE 精度探针)+ ep1(全跨度对照)**; metrics 加 ep2。
- blur step_4000 重测(ep4): perm ep4 Δ=+0.0006(链帮精度但极小, 噪声内, CI±0.007); perm E_DC Δ=+0.0074(大 10×, 真实)。 ⟹ 当前链对自洽影响 ≈ 对精度的 10 倍。 4k 早期, S_order 才 0.046。
- **结论修正**: 不是"机制无用"(那是测错), 而是"4k 步链主要管自洽, 帮精度还很弱"。 待: blur 跑到 15-22k 再 ep4 诊断看 perm-ep4-Δ 随 S_order 一起长不长 → 决定贡献-3 是"提精度"还是重构成"少步自洽性"。

## Round 48 (2026-06-25) — 根因: z_t 泄漏 x_post → 记忆不泛化; SSIM 指标 + leak 修复
- 用户指正: 该用 SSIM 等生成指标(非 L1), 且根本是代码/设计问题非训练技巧。 新建 metrics_img.py(SSIM/PSNR torch 实现)+ eval_img.py。 step_40000 实测: train SSIM 0.65 / val SSIM **0.21**(≈基线 preop-postop 0.20), val 预测跟术前更像(0.34)than 术后(0.21) → **模型在 val 上塌成 ≈ 恒等**。
- 派 agent 系统审码确认根因(硬数据): **训练 z_t=x_pre+α·δ 泄漏 x_post(δ=x_post-x_pre)**, 模型从自身输入 z_t 读出 x_post 平凡满足 L_span, 不学真变换; 唯一无泄漏信号 L_end 仅占目标 ~18%(rho_end=0.25)。 val 无泄漏端点误差 0.179(≈identity)/cos(步向,δ)=0.58 vs train 0.97。 采样器/T/指标/划分/配对**全部正确**, 是 objective 设计缺陷非 bug。 ⟹ **之前所有"机制成立"结论作废**(在没真学会的模型上测的)。
- **leak 修复**(commit cbcf7c9): (1) l_end_roll = few-step 无泄漏滚动端点损失(x_pre→x_post); (2) rho_end 0.25→1.0 + lambda_end 1→2 + lambda_end_roll 2 → 无泄漏信号成主导; (3) sigma_m 0→0.1 降解泄漏; (4) weight_decay 0→0.02 + 水平翻转增强(治 432 样本过拟合); (5) cos(步向,δ) 无泄漏诊断常驻。
- 旧崩溃 run 留 runs/s5b_scpga_v2_leakcollapse_old(SSIM 0.21 反面证据)。 起 leak-fixed retrain(job 9912392), l_end_roll=0.182 起步(待降)。 待 step8000 SSIM train/val 看泛化是否起来。

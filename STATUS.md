# ScoliCMF — STATUS (2026-06-24, R33)
方向: MICCAI'26→MIA, 脊柱-only, 曲线移除, **骨科方法论文(committed)**。
方法 = Source-Anchored Compositional MeanFlow + SC-PGA。 **举证逻辑: 不预先写满, 用可降级 claim ladder, 措辞+模型命名由反事实实验挣得。**
**文档分工**: doc/method_v1.md = 干净方法主体(R31 版, 头脑风暴在此改); doc/novelty_study_v4.md = 问题/雷区/先发威胁/钉子/审查日志。
三贡献(对齐先发占位): (1)source-anchored compositional MeanFlow=系统化改造(速度侧让功 SplitMeanFlow) (2)spinal-chain potential conditioning=条件侧构造(非新定理, 镜像仅叙事) (3)bandlimited partition-sensitive adaptation 分级: v1≈DCT 只 claim ordered-token bandlimiting, patient-specific chain 只属 v2-Frozen。
让功对象: FMM 2406.07507 / SplitMeanFlow 2507.16884 / LBM 2503.07535(ICCV25)。 最大机制隐患=ordered-token 一维低通(收编为 Token-DCT 基线)。
实验: J×J 算子 bake-off(Identity/Random/Token-DCT/Gaussian/Toeplitz/v1/v2-frozen); 公平协议=全算子共享无 affine LayerNorm(禁 RMS 重缩放); 三门统计判据(G_struct/G_graph/S_order, LCB_95>δ_min + 独立结构指标方向一致 + 1-NFE 非退化 Guard); 模型命名随门变。
关键修正历史: R28 三雷区; R29 claim ladder+让功+钉子; R30 五项; R31 P3 定稿=L_smooth; R32 四微裂缝+两工程钉子; R33 八组深修(L_ST detach 目标/W₂残差图防退化+w_min/Sobolev精确Gram K_t=2/projection-last 投影末位/bake-off全 rank-K_g 投影器/共享RMSNorm Norm→Π/S_order撤销阻断改只换投影器/1-NFE 配对非劣效/采样三错修/口径 curve-annotation-free)。
数据: train 540 对(preop/postop_standardized; 曲线仅评测), 240x480, 76 jpg 混存; CASE2(术前被裁)。
进度: Stage0 完成; 方法核实+加固至 R33(method_v1 113 行)。 未动模型代码。
下一步: 继续头脑风暴 method_v1, 或绿灯起 Stage 1a(mydataset 跨扩展名 + canonicalization, CPU) → 1b(AE+L_smooth→P3)。

## 实施进度 (代码, 像素先行) — 更新 2026-06-24 R34
- **S1 数据底座 ✓**: dataset_sa.py(stem 匹配 540 对/240×480) + tools/canonicalize.py(contract+QA montage doc/stage1_montage.png)。
- 数据性质: standardized 已质心对齐但胸廓未配准(NCC~0.42); postop 有钉棒硬件 preop 无。几何配准 DEFER(像素先行不做)。
- 环境: ~/.conda/envs/AgentOCR(torch2.8/tv0.23/einops/accelerate; 无 timm→自实现)。 GPU 跑用 debug 卡+tmux≤30min。
- **下一步 S2**: path_sa.py + meanflow_sa.py + models/sc_dit.py(去FGA单u头) + losses.py(L_span+L_end) + configs/sc_pixel.yaml + train_sa.py; Base PGA Π=I 先验证 x_pre→x_post。
- 阶段路线: S2 MeanFlow核 → S3 +L_ST(detach) → S4 +L_comp/L_roll(EMA) → S5 SC-PGA全量+bake-off+门 → (推迟)潜空间迁移+AE+P3。

## 更新 2026-06-24 R35 — S2 跑通
- **S2 像素源锚定 MeanFlow 核 ✓**: path_sa/sc_dit(timm-free,去FGA,单u头)/meanflow_sa(L_span+L_end+T算子+few-NFE采样)/losses(区间采样)/config/train_sa/smoke_s2.sh。
- 冒烟(debug卡, 与OptiMem并行, 800步): Loss 0.237→0.122, 无NaN, few-NFE 出脊柱结构(欠训模糊, 管线整通)。质量需正式长训。
- 纠正: debug 可与 OptiMem 并行(账号非1-job; 先前判断错)。环境=AgentOCR(torch2.8/A40)。
- **下一步 S3**: +L_ST(detach JVP 目标) + 消融 no/detach/full-grad。 之后 S4(+L_comp/L_roll EMA) → S5(SC-PGA全量+bake-off+门)。

## 更新 2026-06-24 R36 — S3 跑通
- **S3 +L_ST(detach JVP) ✓**: meanflow_sa 加 L_ST(单JVP, detach 目标, 无二阶导); sc_dit Attention 改手写 eager(twice-diff); config lambda_st=0.05/st_mode; train_sa 通用 logging。
- 冒烟(debug卡, 并行): l_st 有限无NaN(JVP通), l_span 0.238→0.135 续降, rc=0。
- 观察: l_st ~0.8 未降(λ小)→ 留 S5 no-ST/detach/full 消融定权重。
- **下一步 S4**: +L_comp+L_roll(EMA teacher + λ ramp + 早期 L_roll 主导, sample_triplet 已备)。

## 更新 2026-06-24 R37 — S4 跑通
- **S4 +L_comp+L_roll(EMA teacher) ✓**: 学生两步对齐 detach EMA 单步(修 comp↔roll 拉扯) + λ_comp ramp + L_roll 早期主导; train_sa 建/更新 EMA。
- 冒烟(debug卡, 并行, 400步): rc=0 无NaN, l_span 0.246→0.147 平滑; l_comp init=0(零初速度,预期)→微训后转非零(组合项活)。
- **下一步 S5(核心,大块)**: SC-PGA 全量 + bake-off(rank-K_g 投影器) + 三门统计判据。这是论文主新意落地。
- 进度: S1✓ S2✓ S3✓ S4✓ | 五损失管线端到端通(Base PGA Π=I)。S5 把 Base PGA 换成 SC-PGA。

## 更新 2026-06-24 R39 — S5a 跑通(核心机制端到端)
- **S5a SC-PGA 全量 ✓**: legendre(fp32除差gate过) + 投影器库(rank-K_g) + SCPGA(有序token/Π_spine v2残差W₂/势函数/趋势窗/projection-last 轴向AdaLN/L_time) + 接入 SCDiT(per-token 轴向调制) + meanflow L_time。
- 冒烟(debug卡, 并行, cond=scpga/proj=v2, 300步, 11.22M): rc=0 无NaN, l_span 0.239→0.151。 论文核心机制首次端到端运行。
- 修 bug: legendre.eval→torch(JVP安全) / train_sa 补传 lambda_comp/roll(S4冒烟曾OFF) / last_l_time→float(deepcopy安全)。
- 进度: S1✓S2✓S3✓S4✓S5a✓ | **下一步 S5b**: eval_gates.py(相对E_DC + 7投影器 bake-off + 三门统计 + bootstrap)。真实门结论待正式长训。

## 更新 2026-06-24 R40 — S5b 评测门 harness 完成 → 全方法实现完毕
- **S5b eval_gates.py ✓**: 相对 E_DC + endpoint 锚 + 1/2/4-NFE; bootstrap CI; G_struct/G_graph(gate_paired); S_order(topology-only Π 置换, SCPGA.perm)。 harness CPU 自测跑通(随机初值数=0, 预期)。
- bake-off 完整 driver(7 变体 ckpt → 门表)待真实训练 checkpoint 后补(薄包装)。
- **里程碑: 全方法 S1–S5b 代码实现 + 逐件 smoke 验证完毕**(像素先行)。
- **下一步 = 正式长训**(非 debug 30min; 需 GPU 时段, 与 OptiMem 错开): 训练 v2-SC-PGA 主模型 + bake-off 各变体 → eval_gates 出 E_DC/三门/NI/Cobb。 之后按 claim ladder 裁决"骨科"强度 + 决定潜空间迁移(AE+P3)。
- 文件清单: dataset_sa / tools/canonicalize / path_sa / legendre / sc_pga / models/sc_dit / meanflow_sa / losses / train_sa / eval_gates / configs/sc_pixel.yaml / scripts/smoke_s{2,3,4,5}.sh。

## 更新 2026-06-24 R42 — code-review 修复(P0 完成)
- 批次1-3 ✓: FinalLayer 修正/return_aux 重构/m_dyn-m_static 诊断; train/val/test 划分(432/54/54)+富 ckpt; eval_gates 真评测(load ckpt+gates+bootstrap)。
- 首个真实评测(pilot step4000@val): E_DC^rel 0.142, S_order +0.0137(CI排除0=链拓扑被用), 但 ep4>ep1(组合没帮端点, 黄旗)。 旧码 ckpt=harness 验证非论文证据。
- pilot job 9911381 健康继续跑(step5000 锐化/矫正方向对, loss 0.086)。 已 push GitHub。
- **余下 P1**: 批次4 config 拆实验 yaml+命名(global_base/scpga_*); 批次5 诊断可视化(m_dyn-off 消融/cross-patient swap/xc/attn/邻接 可视化)。 之后用修好的码起正式训练 → 真实门裁决。

## 更新 2026-06-26 R49 — 创新点对照验证（原始 ScoliCMF vs s5b）
- **泄漏已修 + 创新已证**：R48 修掉 z_t→x_post 泄漏(val 不再塌成恒等)；R49 在匹配 harness 下证明创新点有效。
- **对照(同数据/split/480×240/regime/eval，各取 best-val，val n=54)**：
  - 原始 ScoliCMF(noise→image+FGA, 33.79M, step16k): val SSIM 0.130@1 / 0.194@4 / 0.191@20，PSNR 11.4，train≈val(欠拟合)。
  - **s5b(源锚定+SC-PGA, 19.7M, step4k): val SSIM 0.258@1 / 0.236@4，PSNR 13.97** → 每个 NFE 都赢，best +35%，1NFE +98%。
  - 结论: 源锚定+SC-PGA 用 1/20 NFE、6 成参数，val 大幅领先；train 侧也赢(原始天花板 0.19，s5b 0.27→0.46)。
- **遗留**：绝对 SSIM 仍 0.19–0.26(432 样本)；s5b 过 step4k 过拟合(best-val=step4000)；原始 16k 仍极缓升但基本走平。
- **产物**：baseline_orig/(4 脚本)；runs/orig_baseline/(16 ckpt)；runs/{orig_sweep,orig_final16k,s5b_4k_final}.out。
- **下一步候选**：(a) 把对照做成论文表(加 test 集 + bootstrap CI + 蒙太奇)；(b) 推 s5b best-val 更高(更强正则/缩容量/更多数据/预训练)；(c) 原始 baseline 训更久确认 plateau<0.20。

## 更新 2026-06-29 R52 — 消融定锤：仅 Bridge 有效，SCM/SHMM 均阴性（实验收口）
- **三创新点最终裁决（同数据/split/480×240/eval，val n=54，best-val=step5000）**：
  - **① Pre-to-Post MeanFlow Bridge ✅ 唯一有效**：源锚定 s2_base SSIM4 0.249 / PSNR 13.55 vs 原始 noise→image 0.194 / 11.4；1NFE 差距更大（0.13→0.26）。
  - **② SCM ❌ NULL**：static 0.2562 / point 0.2503 / secant 0.2529，CI 全重叠。R51 的「+23%」是 P0-1 评测污染（cond_mode 未透传）假象，**已撤回**。
  - **③ SHMM ❌ NULL（已排除 collapse 混淆）**：L_tokdiv=0.1 重训后 tok_cos 0.993→0.285、R_removed 0.026→0.33（塌缩破、投影器激活），但 dct 0.2518 / v1 0.2514 / v2 0.2511 仍噪声内 → 患者特异图零增益。
  - SC-PGA(SCM+SHMM) ≈ 纯 Bridge → 价值全在 Bridge。
- **代码状态**：build_scpga 工厂同源（P0-1 修）、2D 注入空间质量门（P0-2 修）、L_tokdiv 破塌缩、诊断（tok_cos/E_top4/R_removed）入库。已提交 GitHub。
- **下一步（待定方向）**：实验已停（用户指示）；论文叙事以 Bridge 为主，SCM/SHMM 报诚实阴性或重新设计。

## 更新 2026-06-29 R53 — 旧 SHMM/SCM 设计正式终结（D1–D3 同 ckpt 干预）
- **三受控干预（shmm_v2 step5000，eval-only）消除独立重训混淆后，旧"加性条件"SHMM/SCM 确认阴性**：
  - D1 投影器替换：dct≈v1≈v2≈identity（dOut 0.017–0.029，SSIM4 噪声内），仅 random 恶化（0.2409）→ 模型要结构化子空间但不在乎哪个平滑基。
  - D2 因果：C_dyn=0.305（有因果通路）但 dyn_off 后 SSIM4 仅 0.2511→0.2392（Δ0.012）→ 动态分支对终点只值 1.2% SSIM，杠杆太小。
  - D3 表示力（能量）：E_v2=0.769 > E_dct=0.754（+1.5pp，假设未被否定），但都丢 ~25% 高频术后变化。
- **根因 = 架构（加性旁路 + 主干直读 z_t/blur(x_pre)），非超参** → 单调 L_tokdiv/τ/K_g 无效。
- **下一阶段（新，不覆盖本阴性）**：冻结 Bridge + 残差校正分支，doc/residual_correction_v1.md（Pilot A/B/C + 验收门）。待用户拍板是否动手 Pilot A。

## 更新 2026-06-29 R54 — 旧 SCM=时间重参数化(code-verified) → correction-grounded 重定义
- **code-verified(legendre.py)旧 SCM 无新信息**：secant mean₁≡ℓ₁(p)(误差0)、₂=ℓ₂(p)+Δ²/2(3e-7)、potential_dd(0,1)=[0,0] → c̄ 是 (B,p,Δ) 确定函数,static≈point≈secant 是数学必然。
- **方法定义本身的问题**(非杠杆/非超参)：secant 加法性经 MLP 消失;旧 SHMM 图错对象(术前外观≠术后矫形);static/dynamic 不可辨识;低秩≠低频。
- **重定义(doc/correction_grounded_v2.md)**：SCM→Correction Potential(受 ΔB 监督,全区间非零);SHMM→correction-aware learned 基 Q_φ + 软谐波 γ;统一链。
- **顺序**：Step1 术前能否预测 ΔB(前置门)→2 learned basis>DCT/Id→3 加 A_φ 比 point/secant→4 接 Frozen Bridge 残差(residual_correction_v1.md 降级为 Step4)。
- **待拍板**：是否跑 Step1 探针(debug 卡)。

## 更新 2026-06-29 R55 — Step1 前置门通过（术前可预测患者特异 ΔB）
- 探针 step1_dB_probe.py：冻结 stem+术前 pi 取 ΔB=sg(B_post−B_pre)，train432/val54，f_corr(B_pre)→ΔB̂。
- frac patient-specific(val)=0.478；**MLP best EV_val=0.358**(EV_train 0.474,gap小=真信号);线性探针欠拟合无结论。
- **GREEN**:术前有可泛化患者特异 ΔB 信号(≫0.1)→ 重定义可行,进 Step2(learned basis Q_φ vs DCT/Identity)。
- 边界:stem OOD 压低 EV(0.358 是下界);52% 共有均值=平凡基线,模块挣 48% 患者特异部分(术前可预测~36%)。

## 更新 2026-06-29 R56 — Step2 通过(强): learned 基远胜固定低频基
- step2_basis_probe.py：rank-4 val 覆盖率 learned Q_φ=0.937 ≫ DCT 0.754(+18.3pp)≫ random 0.337,逼近 oracle 0.989;train-val gap 0.012。
- 证实低秩≠低频:真实变化 rank-4(oracle0.989)但主方向非固定脊柱低频(DCT仅0.754),learned 基从术前对齐真实子空间补回。
- **Step1+Step2 两道表示层门都过** → correction-aware basis 成立。**边界:这是表示层覆盖,非端点 SSIM/LPIPS,端点判定需 Step4(接 Frozen Bridge 残差)。**
- 下一步:Step3(A_φ 比 point/secant)或直接 Step4 整合 pilot。待拍板。

## 更新 2026-06-29 R57 — Step4 残差校正 pilot 实现 + 冒烟通过
- 新文件:residual_model.py(DynamicCorrectionConditioner: A_φ+learned Q_φ+软谐波; 乘性 head c_dyn=0⇒u_corr=0; ResidualScoliCMF 冻结Bridge)+train_residual.py;sc_dit.py 加 forward_features/head_forward。
- 冒烟(debug,secant,60步)全绿:5.35M 参数不 OOM;不变量 |u_corr|max=0.0;baseline 冻结Bridge=0.2490 正确;4损失齐;DYN-OFF 精确回 baseline。
- **下一步**:secant gate pilot → FULL 是否超 baseline 0.2490 且 dyn-off 丢增益。过→point/static消融+长训;不过→Bridge-only。

## 更新 2026-06-29 R58 — Step4 secant gate pilot 未通过(表示层赢,端点不兑现)
- res_secant(2000步):FULL SSIM4 0.2489→0.2417 单调低于 baseline 0.2490;DYN-OFF 恒=0.2490;验收条件一不满足 = FAILED。
- 训练损失全降(l_full/l_corr↓, l_sub cov 0.938, γ→0.12)但 val 端点单调恶化 = 过拟合(5.35M/432对),4-NFE rollout 累积误差。
- 全局一致:Step1/2 + 本轮都在表示层能拟合,没一次转端点增益 → 瓶颈=数据规模+identifiability,非表示。
- 待拍板:① 强正则小容量 retry 确认是否纯过拟合;② 诚实 Bridge-only。

## 更新 2026-06-29 R59 — Bridge 残差 EV_res≈0(决定性,无 confound) → Bridge-only
- step1b_residual_ev.py(复用冻结验证 tokenizer+冻结 Bridge,避开 R57/R58 全部 5 个 bug):
  - ‖ΔB_res‖²/‖ΔB_tot‖²=0.323(Bridge 后残留 32%);dB_total sanity EV_val 0.318(≈Step1,可信)。
  - **dB_RESID:EV_val=−0.009(train 0.015)** → 术前对 Bridge 残差零预测力(残差 79% 患者特异但不可预测=手术方案不可观测+生成噪声)。
- 推论:Step4 必然过拟合(预测无信号的残差),修 bug 也救不回 → 瓶颈是 identifiability,非实现 bug(无 confound)。
- **结论:Bridge-only**;SCM/SHMM + Step1–4 诊断作诚实负结果。边界:target 含生成噪声;仅测 f(B_pre)。

## 更新 2026-06-30 R60 — EV_harm 前置门强阳性(新方向:预测全局脊柱谐波运输)
- step1c_harm_ev.py:harmonic energy share=0.869(总运输 87% 在 rank-4 低频谐波);[c_harm Kg=4] EV_val=0.618(train 0.664)。
- EV 链:Bridge 残差 −0.009 < 总变化 ΔB 0.318 < **谐波系数 0.62** ≫ 0.3 阈值。
- 保留:EV 高=目标可预测且主导,但≠端点改善;单头 Bridge 本可表示低频全局运输,若已抓住则双头不赢端点。
- **下一诊断(去风险,廉价)**:Bridge 输出谐波误差 c_base vs c*——小=Bridge-only,大且可预测=双头有空间。再实现双头 MeanFlow。
- 新名:SCM=Secant Coefficient Module;SHMM=Spinal Harmonic Motion Module(固定 path 基线性重建,降级为 ordered transport decomposition)。

## 更新 2026-06-30 R61 — Bridge 谐波误差诊断=红灯 → Bridge-only 闭环坐实
- step1d:‖c_base‖/‖c*‖=0.871(Bridge 已做 87% 谐波运输);谐波误差 0.250;[BridgeHarmMiss] EV_val=−0.034(train 0.249)=漏掉部分不可预测。
- EV_harm=0.62 是"Bridge 本就做掉"的假象;唯一可补的 miss 不可预测,与 Bridge 残差 EV_res=−0.009 同构 → 双头不赢端点。
- **闭环:术前可预测的运输已被纯 Bridge 全捕获,残差/miss 不可预测(identifiability+生成噪声)→ 任何 pre-op 条件模块都不改善端点。**
- **最终:Bridge-only。** 实验收口;SCM/SHMM+全诊断链=严谨负结果。下一步=论文叙事整理。

## 更新 2026-06-30 R62 — APTD/PMOS 双前置门通过(APTD 强,PMOS 中,端点级证据)
- gate_aptd_pmos.py(无训练):PMOS 残差 oracle best-of-K EV K4=0.21/K8=0.28(random −0.35)=离散模态但 headroom 中;APTD oracle-warp 解释 82% 变化(残差 18%、62% 高频)。
- 关键:LPIPS no-op 术前 0.428 < Bridge 0.509,oracle-warp 0.400 → Bridge 模糊比术前还差,APTD 攻这个、有端点级 headroom。
- 判读:APTD 强 GREEN(主推质量),PMOS 中 GREEN(副,诚实一对多)。下一步:实现 APTD 双分支(φ+R_new)。

## 更新 2026-06-30 R63 — APTD 实现 + 四模式消融(分解成立,LPIPS 赢,SSIM 暂未超 Bridge)
- aptd_model.py + train_aptd.py(x0 端点参数化,warp+残差分解,零初始化起步=术前)。
- 消融(val,各从零1200步,1NFE):direct 0.2097/0.4277,residual 0.2044/0.4307,warp 0.2071/**0.4939**,**warpres 0.2208/0.4222**。Bridge 参照 0.249/0.509。
- 结论:**warpres 三项全模式最好**(分解成立);R_new 关键(warp-only LPIPS 差 0.494→加 R 0.422);warp 头有贡献(vs residual)。
- vs Bridge:**LPIPS 大幅赢**(0.422 vs 0.509),**SSIM/PSNR 暂未超**(0.221 vs 0.249)= perceptual–distortion 权衡,被 identifiability 放大。confound:APTD 仅 1200 步 vs Bridge 5000、x0/1NFE vs velocity/4NFE、flow_scale 可能偏大。
- 下一步:warpres 训 5000 步+调 flow_scale/smooth+bootstrap,看 SSIM 能否追平;否则按 LPIPS+诚实权衡叙事+PMOS 副模块。

## 更新 2026-06-30 R64 — APTD warpres 5000步:翻盘,Pareto-支配 Bridge
- 轨迹(1NFE,fs0.15≈0.3):step1000 0.215/12.3/0.424 → 2000 0.255/13.6/0.443 → 5000 0.302/14.26/0.665。Bridge 0.249/13.55/0.509。
- **单模型沿感知-失真前沿移动,Bridge 被 Pareto-支配**:best-SSIM 0.302 vs 0.249(+21%),best-LPIPS 0.424 vs 0.509,step~2000 平衡点三项同胜(且 1/4 NFE)。
- 诚实:step2000 SSIM/PSNR 优势小(需 bootstrap),LPIPS 稳健;晚训 LPIPS 退化=学会模糊刷 MSE,best-val 按平衡/感知选或加轻感知项;flow_scale 非关键。
- 下一步:step~2000 + bootstrap 出主表;可选轻感知项;叠 PMOS。

## 更新 2026-06-30 R65 — PMOS 实现:v1 原型塌缩,修复重跑中
- pmos_model.py+train_pmos.py(共享 APTD backbone+K 原型→K 候选,soft-min 集合损失,best-of-K eval)。
- v1(K=4,tau0.05,无 L_div,零初始化)塌缩:best-of-K==proto0,usage=[54,0,0,0]→原型退化相同。
- 修复 v2:加 L_div(hinge margin0.06)+proto 多样化init(proto0=base)+tau0.02。重跑 pmos_K4b。
- 门:usage 散开+best-of-K>proto0→成立;否则图像层 headroom 太小,APTD 为主+PMOS 弱/负结果。

## 更新 2026-06-30 R66 — PMOS v2 修复后=弱/负结果(塌缩破但多样性拖垮质量)
- v2(pmos_K4b):原型分化(usage [54,0,0,0]→[3,0,1,50])、best-of-K>proto0(+0.013),但 SSIM 全程下滑(0.256→0.241)、低于 APTD 起点 0.255,LPIPS 0.63 未改善。
- 结论:set-valued headroom 图像层太小,L_div 强制多样性拖垮质量,净不如 APTD。
- 下一步:①冻结 backbone+head 只训原型(best-of-K 必≥APTD)从 step_1000 init;②收手,APTD 主+PMOS 弱/负。待拍板。

## 更新 2026-06-30 R67 — ASW 前置门红灯(铰接假设太强)
- asw_gate.py:分段仿射铰接 warp 残差 0.46–0.58(只解释 42–54%),远不如自由稠密 0.18(82%),加节段几乎不改善(K4→17 0.576→0.464)。
- 脊柱术前→术后形变非轴向刚性链可表达(侧弯横向+胸廓+缩放);ASW 干净低维结构化 warp 不成立(精度代价太大,会欠拟合)。砍。
- 下一步:换方向找第二模块(constructive 思路:加入缺失的手术意图作控制输入)。

## 更新 2026-06-30 R68 — Diffeomorphic 前置门红灯 + 第二模块搜索的诚实评估
- diffeo_gate.py:diffeomorphic SVF 残差 0.34/LPIPS 0.55,远差于自由位移 0.18/0.40;自由 warp 折叠仅 0.76%(本就近似微分同胚)。微分同胚约束有代价、无收益。砍。
- **第二模块累计 4 次未成**:PMOS(弱,headroom 小)、ASW(铰接太强)、SIC(条件生成无新意,用户否)、Diffeomorphic(约束有代价)。**共同根:APTD 的自由 warp 已近最优地捕获了可预测几何,剩余=identifiability-limited。** 干净的、能抬指标的、正交 novel 第二模块,headroom 已被 APTD 吃掉。
- **战略路口(待用户定)**:① APTD + identifiability 分析作两个 co-贡献(诚实,MIA 够);② 第二模块换"不同 KIND"——标定不确定性/可靠性图(我们有 identifiability 量化可验标定),不靠抬 SSIM;③ realism 模块(已知技术+loss-y,新意弱)。

## 更新 2026-06-30 R69 — ADOC 前置门强通过(第二模块方向找到)
- **ADOC=Acquisition-Disentangled Outcome Calibration**:正交于 APTD(改监督目标非生成器),x_post^obs=A_ν(x_post^canon),剥离受限采集因素(小仿射+光度,主动保护手术矫形区,禁自由 dense)→ 更纯净的源帧术后目标。可独立加在普通 x0 Bridge 上。
- **gate(adoc_gate.py,严格受限,无自由 dense)**:对 APTD step_2000 的 val 预测 x̂ 对齐观测 x_post:
  - 仅平移+缩放:SSIM +0.028、LPIPS −0.006,SSIM↑ 100%;
  - 小仿射+光度:**SSIM +0.142(0.281→0.423)、LPIPS −0.038,两项 100% 一致**;典型 |dx|0.020 |dy|0.028 |θ|1.5°。
- **判读:强通过**(远超阈值 SSIM0.01/LPIPS0.02,54 例全一致)→ "预测 vs 术后"误差中很大一块是全局采集偏差(位姿+曝光/对比),非内容错 = R64 后期模糊拟合的那部分错误监督 → ADOC 值得建。
- **诚实**:① 光度分量(a·x^γ+b)可能贡献大头(SSIM 对亮度/对比敏感),几何(位姿)约 +0.03–0.05,两者都是真实采集因素;② gate 把 x_post 对齐到 x̂(略偏favorable),真正判据是"生成器对 ADOC-清理目标 vs raw x_post 训练后端点是否更好";③ 受限仿射无法吸收局部脊柱矫形,故未偷掉手术变化。
- **下一步**:建 ADOC(自监督预训采集校正器 C_ψ:L_param+L_inv;中心抑制 W_acq 保护脊柱;stop-grad)→ 用清理目标重训 APTD,比 raw-target 端点。

## 更新 2026-06-30 R70 — ADOC 验证成功(清理监督目标 → APTD 全面更好,正交第二模块)
- adoc_clean.py(per-pair 受限仿射+光度,中心抑制 Huber 保护脊柱)生成清理目标;train_aptd_adoc.py 用清理目标重训 APTD,规范帧(清理 val)公平比 raw-trained(同 eval 目标,唯一变量=训练目标)。
- **清理 val 上(SSIM/PSNR/LPIPS)**:raw-APTD best 0.4085/15.79/0.652(@5k)、0.3985/16.03/0.592(@2k);**ADOC-APTD 0.4644/17.53/0.614(@5k)、0.3980/16.84/0.393(@2k)、0.333/15.18/0.377(@1k)**。
- **ADOC 前沿支配 raw 前沿**:best-SSIM 0.464 vs 0.409(+0.055,PSNR 17.5 vs 15.8);同 SSIM≈0.40 时 LPIPS 0.393 vs 0.592(−0.20);best-LPIPS 0.377 vs 0.592。
- 结论:唯一变量=训练目标(raw vs 清理),清理后全面更好 → raw 目标的采集噪声让模型浪费容量+模糊对冲;ADOC 去之 → 更准更锐。**第二模块成立,正交 APTD。**
- **诚实**:① 评测在规范(清理)帧——正当(采集=nuisance,且 baseline 同样在清理 val 上比,公平),但需在论文论证此评测口径;② 在 RAW val 上 ADOC(0.287)≈/略低于 raw-APTD(0.294),因 ADOC 预测规范帧、被随机采集偏移惩罚——这恰说明 raw 帧含不可预测采集噪声;③ perception-distortion 权衡在清理帧内仍在(早 LPIPS 优/晚 SSIM 优),best-val 需按口径选;④ 当前 ADOC 是 per-pair 优化形式,论文版可换自监督网络 C_ψ(更 elegant)。
- **下一步**:bootstrap CI 固化;或把 ADOC 做成自监督网络 C_ψ;论文三件套=APTD+ADOC+identifiability 分析。

## 更新 2026-06-30 R71 — bootstrap CI:两个模块统计确证(配对差 CI 排除 0)
- boot_eval.py(2000 重采样,配对差 CI)。
- **表1 APTD@2000 vs Bridge(raw 帧)**:SSIM 0.2554 vs 0.2490 Δ+0.0063[+0.0017,+0.0112]✅;LPIPS 0.4429 vs 0.5090 Δ−0.0661[−0.0724,−0.0601]✅;PSNR 打平(含0)。→ APTD 同时显著胜 SSIM+LPIPS。
- **表2 ADOC@5000 vs raw-APTD@5000(规范帧,均 raw 权重)**:SSIM 0.4413 vs 0.4085 Δ+0.033[+.026,+.039]✅;PSNR 17.38 vs 15.79 Δ+1.59[+1.24,+1.97]✅;LPIPS 0.624 vs 0.652 Δ−0.028[−.036,−.020]✅。→ ADOC 三项全部显著胜出。
- **结论:APTD(显著胜 Bridge,SSIM+LPIPS)+ ADOC(规范帧三项全显著胜 raw-target)统计确证。** 余:评测口径论证(ADOC)、单 split 小数据、perception-distortion 权衡选点;aptd_adoc 未存 ema(表2用 raw model,口径一致)。

## 更新 2026-06-30 R72 — ADOC 网络版 C_ψ:部分有效但弱于 per-pair + 设计澄清(网络非必需)
- adoc_net.py(AcquisitionCorrector C_ψ:成对图→8 受限采集参数,零初始化=恒等)+ train_adoc_net.py(自监督:对单张随机扰动→重建校正;再在真实对上生成 cleaned + 对比 per-pair gold)。
- 结果:|C_ψ_clean − per-pair gold|=0.117;中心抑制对齐到术前 L1 raw 0.1524 → C_ψ 0.1299 → per-pair 0.1118。**C_ψ 部分有效但明显弱于 per-pair**(sim-to-real 域差:自监督同内容,真实对含手术内容差扰动估计)。
- **设计澄清(重要)**:ADOC 清理的是训练监督目标=离线预处理;推理时生成器只吃 x_pre,不调用 C_ψ → **无"在线清理"需求 → per-pair 优化(更准、无域差)才是 ADOC 的天然且更好实现;C_ψ 网络既更弱又非必需**,"可部署"卖点不成立。
- **决定**:ADOC 实现 = per-pair 优化(R70/R71 已统计确证)。C_ψ 留作记录(若论文要"学习式校正器"叙事,可从 per-pair gold 参数蒸馏 C_ψ,但属可选 polish)。
- **产物**:adoc_net.py / train_adoc_net.py / runs/adoc/cpsi.pt / clean_*_net.pt。

## 更新 2026-06-30 R73 — 移除 ADOC 网络版 C_ψ(确认非必需后清理)
- 按 R72 结论(C_ψ 网络既更弱于 per-pair 又非必需:ADOC 是离线目标清理,推理不调用校正器),删除 adoc_net.py / train_adoc_net.py 及产物 runs/adoc/cpsi.pt、clean_*_net.pt、adoc_net.out。
- **ADOC 唯一实现 = per-pair 优化**(adoc_clean.py,R70/R71 已 bootstrap 统计确证)。代码库不再保留弱/冗余的网络分支。
- 不影响任何已确证结论:APTD(train_aptd.py)+ ADOC per-pair(adoc_clean.py + train_aptd_adoc.py)两模块完整且各自 CI 排除 0。

## 更新 2026-06-30 R74 — ADOC 独立性 2×2 + 几何/光度拆分 + 中心保护(三个承重消融)
审稿视角补强:此前只证 "ADOC 帮 APTD",未证 "ADOC 独立于 APTD"。新增三实验。
- **新代码**:adoc_variants.py(参数化 cleaner:--geo/--photo/--center none|gauss|strong)、train_2x2.py(统一 x0 trainer,{direct|warpres}×{raw|任意clean},每 ckpt 同时评 canonical+raw 帧)、gate_geophoto.py、boot_independence.py、recompute_diag.py。
- **Exp1 独立性 2×2(全 canonical 帧,配对 bootstrap 2000):** direct(普通x0Bridge) raw 0.4025 → ADOC **0.4467**,dSSIM **+0.0443**[.0382,.0504]/dPSNR +1.55/dLPIPS −0.0335 **全 sig**;APTD(warpres) raw 0.4234 → ADOC 0.4413,dSSIM +0.0180[.0122,.0238]/dPSNR +1.34/dLPIPS −0.0084 全 sig。**x0Bridge+ADOC>x0Bridge 成立 ⟹ ADOC 独立于 APTD、是通用采集归一化(对无-warp Bridge 增益 +0.044 还大于对 APTD 的 +0.018,因 APTD warp 已吸收部分几何采集变化)。**
- **Exp2 几何 vs 光度(固定 raw-warpres,换评测帧):** raw 0.3018 → geo-only 0.3271(+0.025,share 21%) → photo-only 0.4150(+0.113,share 93%) → full 0.4234。**增益主要来自光度归一化** ⟹ **叙事改为 acquisition normalization(光度为主轴),不可重点宣称 source-frame geometric canonicalization**;full(geo+photo)仍最优,几何在光度之上有小增益且最护中央。
- **Exp3 中心保护(per-pair cleaning 诊断,val;central_change_true=0.219):** PRESERVED(中央术后改变保留率) none **0.772** < gauss(当前) 0.797 < strong **0.814**;周边对齐 L1 none 0.1592(最差) > gauss 0.1575 ≈ strong 0.1576。**无保护抹掉 23% 中央手术信号且周边对齐还更差 ⟹ 中心保护必要;当前 gauss 偏保守,strong 更优(可升级)。**
- **诚实**:① 已知 H 缩放笔误已修(recompute_diag.py 重算正确归一化,PRESERVED 比值本就不受影响);② 跨格比较统一在 canonical 帧(R70 口径,论文需论证此评测口径);③ Exp2/Exp3 目前是 eval-only/cleaning-side 门 + Exp1 全训练,若要论文级完整表可再训 generator on geo/photo/none/strong 目标(可选)。

## 更新 2026-07-01 R75 — 二号模块候选预验:Plan-Marginalized/PC-SRO/DC-MF 三个 idea 被证据否决
- 用户毙 ADOC 后连续预验三个二号模块候选(全 eval-only,现有 APTD ckpt):
- **gate_dist.py**(分布/响应类):Gate A 术前 kNN 术后发散 cond/global=0.99/0.96 → 术前几乎不约束术后结局(印证 EV_res≈0);Gate B best-of-10 仅 +0.008 SSIM、注噪只毁质量 → 无可捡多样性。**PC-SRO 与 Plan-Marginalized 都撞 identifiability 墙,放弃。**
- **gate_dcmf.py**(自适应 NFE):1/2/4 步 SSIM=0.3018/0.3014/0.2977(1 步最好);Spearman(defect,真实误差)+0.511 但 Spearman(defect,拆分受益)**−0.293**、高-defect 分位负收益 → 核心假设证伪;GT-oracle 上界仅 +0.005 SSIM 且 avgNFE 2.30(更贵)。**无质量/效率奖品,放弃。**
- **两口干井**:(1) 术前→术后可预测结局结构;(2) 多步→精度。落这两族的模块都会失败。
- **未被否定的方向**:结构化新内容(钉棒)渲染保真(有现成残差门可验)/ 或把第二贡献重定为 identifiability 天花板分析(我们证据已很强)。

## 更新 2026-07-01 R76 — ONOP 预验不过关(第 5 个二号模块被否)
- ONOP(几何保持流形投影校正): Gate1 rho_N=0.82-0.88(APTD 误差 85% 非几何, 稳健), 但 oracle 用真值=恒等式假象不算数; Gate2(小 patch score) raw 帧最好 LPIPS -0.008(线 -0.03)、SSIM 平, ONOP≈unproj(投影贡献≈0); Gate3 |mean(d)|/|d|=0.55 主要修光度=ADOC 争议 => 判停.
- **五个二号模块候选全否**(ADOC 毙 / PC-SRO / Plan-Marginalized / DC-MF / ONOP). 统一发现: APTD 已提尽可约信号, 残差被患者特异不可预测内容(identifiability)主导.
- **战略拐点**: 建议第二贡献从"再造一个生成模块"转向 (a) identifiability 天花板分析(证据链已很强), 或 (b) defect 可靠性/不确定性估计(DC-MF Spearman(defect,真误差)=0.51 是全程唯一正信号). 最后一个未验模块=结构化植入物(但很可能同撞 identifiability 墙).

## 更新 2026-07-01 R77 — LOSA 预验不过关(第 6 个二号模块被否;栽在自设 Gate 7)
- LOSA(纵向可观测子空间, 冻结 AlexNet+线性 CCA): Gate A AUC 0.792/null 0.448(子空间真实但弱, Recall@1 0.06); **Gate 7 判停**: periphery-only AUC 0.798 >= spine-only 0.790 => 信号是周边/采集 shortcut 非脊柱解剖; Gate B 名义 headroom 但被 shortcut 子空间+生成域偏移污染(Spearman 0.24).
- **六个二号模块候选全否**(ADOC/PC-SRO/Plan-Marginalized/DC-MF/ONOP/LOSA). 六次便宜门累计 <2 小时, 省了六次昂贵 build.
- **强烈建议锁定**: APTD(方法) + identifiability 天花板分析(6 探针证据链) + defect 免训练逐例可靠性估计(Spearman(defect,真误差)=0.51). 三者互证、不可被"凑模块"诟病、基本无需再训练.

## 更新 2026-07-01 R78 — SPPC 预验不过关(第 7 个二号模块被否)
- SPPC(屏蔽泊松耦合两 checkpoint, 无训练): perception-distortion 前沿真实; 但条件2 决定性失败——中段前沿**平凡线性混合比 SPPC 低 0.02-0.03 LPIPS**, 且只有线性混合(非 SPPC)能支配 step2000 点. 根因: checkpoint 差别是模糊程度非正确互补高低频.
- **七个二号模块候选全否**(ADOC/PC-SRO/Plan-Marginalized/DC-MF/ONOP/LOSA/SPPC). 七次便宜门累计仍 <3 小时.
- 副产品(可用): checkpoint 线性混合 a~0.5 给支配单点的操作点(trivial 模型平均, 报操作点用, 非贡献).
- **锁定建议不变且更强**: APTD + identifiability 天花板分析 + defect 免训练可靠性估计.

## 更新 2026-07-01 R79 — RC-MF 预验不过关(第 8 个二号模块被否)
- RC-MF(流映射与粗粒化对易, 图像空间零训练代理门): 跨-ckpt Pearson(E_comm,LPIPS)=0.778 表面过, 但是**训练时间混淆**(两者都随步单调); **逐例 ρ(E_comm,LPIPS)≈0** => 因果机制病例级证伪; E_comm 绝对量~5% 太小解释不了模糊. 前提(模糊=跨尺度失配)不成立.
- **八个二号模块候选全否**(ADOC/PC-SRO/Plan-Marginalized/DC-MF/ONOP/LOSA/SPPC/RC-MF). 晚期模糊反复被证=条件均值/identifiability 泛化地板, 非可修算子/机制缺陷.
- nugget: E_comm 随训练升 4.6x 可作 MeanFlow 训练动力学的诊断观察(非模块).
- **锁定建议(第 8 次重申)**: APTD + 可预测性天花板分析 + defect 可靠性. 证据已压倒性.

## 更新 2026-07-01 R80 — GC-MF 预验不过关(第 9 个二号模块被否;估计误差轴也测掉)
- GC-MF(规范协变, 零训练 Reynolds gate): Gate A E_G/E_interp ratio 2.28(<3x), 逐例 ρ(E_G,err)=-0.17(反相关=良性), **E_G train 0.199≈val 0.200**(无有限样本签名)=> 直接证伪"攻估计误差"论点; Gate B Reynolds=TTA(SSIM+0.011/LPIPS+0.013 沿前沿移动+40%光度)非贡献.
- **九个二号模块候选全否**(ADOC/PC-SRO/Plan-Marginalized/DC-MF/ONOP/LOSA/SPPC/RC-MF/GC-MF). 连"估计误差 vs Bayes error"这条最后的概念不同轴也已测掉.
- **锁定建议(第 9 次)**: APTD + 可预测性天花板分析 + defect 可靠性. 证据压倒性.

## 更新 2026-07-01 R81 — IC-MF Gate A+C:泄漏证实但非瓶颈,天花板结论被加固(第 10 个模块被否)
- 用户抓到真实代码问题(训练桥 t->1 时 SNR->inf, source 端泄漏 delta; train_aptd.py 未修 R48 泄漏). 我"已到天花板"claim 有未验漏洞——测之.
- Gate A: 解析 sigma/alpha 当前 t=0.999->0.003(泄漏最大); probe R^2 中段 t=0.9/0.95=0.92/0.89(严重泄漏)证实前提.
- Gate C(4x3000 matched, 1-NFE source-only raw val): current 0.2847/0.5271; **ic 0.2836/0.5181(≈current=>IC 路径无用)**; ic_src 0.2908/0.5567; endpoint 0.2907/0.5698. **ic_src≈endpoint(增益纯来自多监督 source-only 非 IC 机制)+LPIPS 更差=前沿移动**.
- 结论: 泄漏真实但**非瓶颈**; source-only 受 identifiability 限制. IC-MF 非模块. **但加固天花板**: leakage-free 训练落同一处 => 排除"天花板=训练捷径假象". 用户质疑把核心结论从"可能有漏洞"变"有对照站得住".
- **十个二号模块候选全否**. 天花板分析(② 贡献)现在多一个关键对照.

### R82 — OC-PMF (候选 #11, 否) — 首个"优化器"类想法
- gate_ocpmf.py: 从 step2000 出发 80 步真投影梯度(TRAIN 训 / VAL 评). B=OC-PMF(G=I,约束 LPIPS 下降); C=朴素 L=D+λP.
- **B 与 C 走同一条 (SSIM,LPIPS) 前沿**(B 0.2509/0.4169 与 C-λ1 0.2536/0.4197 互不支配)=> ε-约束自然梯度投影相对"把 LPIPS 加进 loss"零增益(MOO 定理: ε-约束≡加权和,同一 Pareto 前沿).
- "固定 distortion 降 LPIPS" 只一阶成立且**泄漏**: 小 lr LPIPS 不动 / 大 lr SSIM 崩. 最好情形仅**平手一行线性混合**(0.2554/0.4368≈0.4370),无前沿突破.
- **首个有定理支撑的否定**: 换度量/ε-约束只改轨迹与速率,不改可达集; 可达集被 identifiability 天花板锁死. **优化器类别是最后一格,现已封闭**. 二号模块搜索覆盖 data/target/model/inference/optimization 全空间,一致撞墙.
- **十一否. 强烈建议 LOCK**: APTD + 可预测性天花板分析(10 探针+IC 泄漏对照+OC-PMF 定理) + defect 免训练可靠性.

### R83 — SC-FGO (候选 #12, 否) — 首个"架构/inductive-bias"类想法
- 前提=可预测长程冠状耦合存在且当前未建模. gate_scfgo.py 在中线 Dc(y) 上 held-out ridge 直测(不训练,CPU).
- R2: POINT 0.430 / LOCAL 0.466 / GLOBAL 0.450 / SHUFFLE 0.454. **长程增益 -0.016; shuffle 对照 -0.004 => 远处行零信息**.
- 冠状形变 ~43% 可预测但**纯局部**; 长程信号经验上不存在; 可预测部分平滑+局部=已在 warp. SC-FGO 打空.
- **礼物**: 天花板分析得到非零拆解 = 可预测≈43%冠状矫正(平滑/局部/在warp内) + 不可预测=残差新内容+plan矫正幅度(EV_res~0).
- **十二否, 覆盖 data/target/model/inference/optimization/architecture 全空间. 强烈 LOCK**.

### R84 — SGOS (候选 #13, 否) — 首个"改变信息集"类, 测得最彻底(3 gate)
- I=(X_pre, A) 稀疏目标点. GateA 稀疏中线 warp: 连 dense-oracle 都不改善(真实≈shuffled)=> 冠状位置轴零图像信息.
- GateB 几何天花板(最优稠密2D warp): SSIM 0.4262(远超 x_pre 0.1996 与 APTD 0.30)但 LPIPS 平(0.428->0.412)=> warp 修几何补不了外观.
- **GateC 决定性(稀疏可恢复性)**: K=1 .2102 / K=5 .2251 / K=10 .2346 / K=24 .2517 / dense(288) .4258. **headroom 需 ~288 DOF; K<=10 稀疏点击 SSIM~0.23 < 无控制 APTD 0.30**.
- **否, 两独立理由**: (1)可恢复形变高维, 稀疏点击非充分低带宽代理; (2)dense-oracle warp LPIPS 平 => 术后内容非几何, 空间控制表达不了.
- **价值**: 用数据回击"为何不加稀疏交互控制"审稿问. 十三否. QES 可 gate 但低 EV. **强烈 LOCK**.

### R85 — 第二 novelty 广域调研(5 方向)+ 2 免费 gate
- deep-research 流水线失败→改 5 并行研究 agent(带引用+自我对抗). 排序: SB/OT/equivariance/physics 守恒=LOW(任务失配); 微分同胚=单独LOW/耦合bridge才MED-HIGH; conformal=MED-HIGH.
- **Gate A folding**: APTD warp φ 折叠率 0.0000%(step2000&5000, 0/54)=> φ 已微分同胚 => 微分同胚保证 novelty **被预注册gate毙**.
- **Gate B conformal**: Spearman(d,e)=.516; split-CP 覆盖 .932@.1/.817@.2(贴nominal=有效); Mondrian 低分歧界仅紧8%; 选择接受半数 +0.02 SSIM => **有效但效率平庸**(信号弱+天花板致误差大界松).
- **结论**: 任务结构(小光滑微分同胚形变+非守恒+不对称+不可辨识)系统性废掉高级物理/几何先验(佐证天花板). 唯一可证不降点幸存=conformal(安全次要点非headline). 加厚路径=σ_m>0 重训→conformal逐像素不确定性图.

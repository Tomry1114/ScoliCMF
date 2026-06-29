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

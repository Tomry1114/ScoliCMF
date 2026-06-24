# ScoliCMF 创新点研究（MIA 期刊版）

_2026-06-21_

## 0. 决定（本轮锁定）
- **数据现实**：无逐椎骨 landmark / 分割，仅脊柱 curve 中线掩膜。
- **主创新**：方向① —— **术前出发的桥接式 Mean-Flow**（source 从 N(0,I) 换成术前图）。
- **FGA 必须重构**（方向④）。
- **形变场路线（方向②）暂不作主线**：无逐椎骨标注，无法做逐节段刚体形变；留作未来升级。
- curve 掩膜 + 可微 Cobb 作为临床锚定（方向③），支撑①④。

## 1. 一句话主张（范式重定义，回应 R2）
> 矫形不是「从噪声合成术后图」，而是「从术前解剖出发，沿一条受脊柱曲线引导的短轨迹，把术前重排成术后」。
> 我们把 MeanFlow 的 source 从高斯噪声换成**术前图（条件相关 source / 术前→术后桥接）**，并把 FGA 从全局均值门重构为**沿脊柱轴线的曲线感知空间条件**。

## 2. 创新一：术前锚定的桥接式 Mean-Flow（Pre-op Bridge Mean-Flow）

| | 原 ScoliCMF | 本方案 |
|---|---|---|
| source（t=1） | 高斯噪声 ε~N(0,I) | **术前图 x_pre** |
| target（t=0） | 术后 x_post | 术后 x_post |
| 插值 | x_t=(1-t)x_post+t·ε | **随机桥 x_t=(1-t)x_post+t·x_pre+σ_t·ε，σ_t=η·√(t(1-t))**（两端为 0，保留生成随机性） |
| 漂移 v | ε - x_post（噪声方向） | **≈ x_pre - x_post（病人特异、有限位移）** |
| MeanFlow 恒等式 | u_tgt = v-(t-r)·du/dt | 不变（仍用 JVP） |
| 推理 | 从噪声 t:1→0 | **从 x_pre 出发 t:1→0；单步 x̂_post = x_pre - u(x_pre,0,1|c)** |

借鉴：Better Source Better Flow（条件相关 source）、LBM（image-to-image 桥接，单步）。

**为何回答 R1「mean velocity 为何优于 single velocity」（给原理，不再是经验）：**
- 噪声→术后：高度非线性，要凭空合成全部解剖 → 低 NFE 必然 anatomical drift。
- 术前→术后：同一病人解剖、短而近线性、结构守恒 → **span 平均速度 ≈ 瞬时速度** → 1–2 步即准。
- 平均速度在此 = 术前到术后的**平均重排方向**，有几何意义。

**为何天然形态保真（structural，而非靠门控）：** 术前骨纹理（高频）从 source 就被携带进来，模型只需学「重排 + 长出植入物」，不必重建骨。

## 3. 创新二：曲线感知 FGA（重构，回应 R1「FGA trivial」）

| | 原 FGA | 重构后 |
|---|---|---|
| 条件来源 | mean-pool(patchembed(x_pre)) → 全局向量 | **脊柱 curve 几何（术前 curve + 目标术后 curve），保留沿轴空间分辨率的特征 F_curve（不池化）** |
| 跨时参数 | p=t_emb+r_emb, q=t_emb-r_emb（与论文 Eq.3 不符） | **修正为 p=(t+r)/2, q=t-r，对齐 Eq.3** |
| 门 | 全局标量 gate=σ(MLP([p,q])) | **空间门 G(x,y)=σ(MLP([p,q], F_curve))，逐位置×逐轨迹阶段调制 curve 先验注入** |
| 注入 | c=p+gate·cond 进 AdaLN | spatial-AdaLN 或对 curve token 的 cross-attention，注入每个 DiT block |
| 临床锚定 | 无 | **可微 Cobb 一致性损失 L_cobb=|Cobb(x̂_post)-Cobb(x_post)|** |

**推理时「目标术后 curve」从哪来（设计分叉）：**
- 选项 A：仅用术前 curve，让模型自行重排（最简单）。
- 选项 B（推荐）：加**轻量 curve 预测头（术前 curve→术后 curve）**，并让系统可交互（医生可编辑目标 Cobb）。

## 4. 对四条创新质疑的逐条对应

| 质疑 | 来自 | 被哪条解决 |
|---|---|---|
| A 范式不新、像 adaptation | R2 | 术前桥接式 mean-flow（非「噪声→图」套用）+ 临床锚定，整体是新机制 |
| B mean velocity 为何更好（仅经验） | R1 | §2：术前桥短而近线性 → 均速≈瞬时 → 低 NFE 不漂移（原理性） |
| C FGA trivial / motivation 不清 | R1 | §3：曲线感知空间门 + Eq.3 修正 + Cobb 锚定 |
| D 增益来自 MeanFlow 还是 FGA | R4 | §5 消融矩阵显式分解 |

## 5. 消融矩阵（隔离每个贡献）

| 维度 | 设置 | 隔离对象 |
|---|---|---|
| source | 高斯噪声 vs **术前桥** | 创新一 |
| FGA | 无 / 原全局门 / **曲线感知空间** | 创新二 |
| Cobb 损失 | 有 / 无 | 临床锚定 |
| 速度类型 | 普通 FM（瞬时）vs **MeanFlow（均速）**，同桥同条件 | mean vs single velocity |
| 步数 | 1 / 2 / 5 / 20 | 证明术前桥在 1–2 步即高 |

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 术前/术后大形变 → 像素线性插值中间态重影 | 用 **latent 空间桥（LBM 式）** 或先做粗对齐 |
| 植入物术前没有 → 桥需「长出」新内容 | 漂移 x_pre-x_post 已编码植入物外观；需实验验证速度场能新增内容 |
| 推理需目标 curve | curve 预测头（选项 B） |

## 7. 借鉴出处
Improved MeanFlow (CVPR26) · Better Source Better Flow (2602.05951) · LBM (ICCV25) · FreqFlow (CVPR26) · Diff-Def / DiffuseMorph（形变场，备选②）

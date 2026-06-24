# ScoliCMF — MICCAI 2026 审稿意见汇总

> 投稿:**ScoliCMF: Conditional Mean Flow Generation for Medical Image Synthesis**
> 来源:MICCAI 2026 评审(Reviewer #1 / #2 / #4)
> 整理日期:2026-06-21

---

## 0. 总览

| 审稿人 | 类别 | 文章定位 | 清晰度 | 复现性 | 伦理 | **打分** | 置信度 |
|---|---|---|---|---|---|---|---|
| **#1** | MIC | Methodological | Satisfactory | 算法描述清晰(无代码/数据) | No | **3 · Weak Reject** | Very confident (4) |
| **#2** | MIC | Methodological | **Poor** | 算法描述清晰(无代码/数据) | No | **2 · Reject(独立于 rebuttal)** | Very confident (4) |
| **#4** | MIC | Methodological | Good | 算法描述清晰(无代码/数据) | No | **4 · Weak Accept** | Confident not certain (3) |

**当前态势:3 / 2 / 4 → 一拒一弱拒一弱收,均值 3.0,处于接收线下。** 两位高置信度(4)的审稿人都在拒方;唯一支持者(#4)置信度较低(3)。rebuttal 是关键。

---

## 1. 三位审稿人共识的核心问题(rebuttal 必须正面回应)

| 主题 | #1 | #2 | #4 | 说明 |
|---|:--:|:--:|:--:|---|
| **缺与 SOTA / 临床相关合成方法的对比** | ✓ | ✓ | | baseline 偏旧(pix2pix/cyclegan),diffusion/FM 对比不充分 |
| **缺 mean-flow vs 普通 flow-matching 的消融** | ✓ | | ✓ | 无法判断增益来自 CMF 还是 FGA |
| **方法 motivation / novelty 不足** | ✓ | ✓ | | 为什么 mean velocity 更好缺乏分析;被视为对现有框架的"适配" |
| **FGA 设计动机不清(p,q 为何能控条件强度)** | ✓ | | ✓ | 公式显得 trivial;早期强约束是否反而阻碍大形变 |
| **应用范围混乱(脊柱 vs BraTS 脑 MRI)** | | ✓ | | 定位不清:专用框架还是通用合成模型 |
| **写作/呈现质量** | ✓ | ✓(Poor) | | 例:Table 1 的 "Diffusionl" 疑似 typo |

---

## 2. Reviewer #1 — 评分 3(Weak Reject),置信度 Very confident (4)

- **类别:** MIC ｜ **定位:** Methodological contribution

**5. 主要贡献(摘述)**
提出用于术前→术后脊柱图像合成的条件 mean-flow 生成方法:条件 mean-flow matching + flow-gating adapter 做自适应条件控制;相比 CNN/Transformer 方法展示了一定有效性。

**6. 主要优点**
- 核心在 conditional mean-flow matching + flow-gating adapter。
- 相比 vanilla flow-matching,mean-flow 预测**平均速度**。
- FGA 用"定义两个采样点(平均速度计算)的时间区间"预测的参数来调制条件。
- 用 mean velocity 而非 single velocity,**可视为一定新意**。

**7. 主要缺点**
- 缺乏深入分析说明**为何平均速度优于单一速度**;只简略提 low-NFE,没说清 mean velocity 为何是解 → 目前只是**经验性测试**。
- FGA:t、r 是标量,p、q 也是标量 → **MLP 如何作用于 [p,q] 的拼接令人困惑**。
- motivation 不清:p=区间中点、q=区间长度,**为何由这两者导出的参数能有效控制条件强度**没说清,公式**显得 trivial**。
- 实验评估有大问题:只比 CNN/Transformer,**未充分对比 diffusion 与 flow-matching**;pix2pix/cyclegan 偏旧,**多数 baseline 非 SOTA**。
- **没有 mean-flow matching vs 传统 flow-matching 的消融**——对本工作 motivation 很重要。
- 呈现质量:Table 1 出现 **"Diffusionl"**,他处未出现,疑似 "Diffusion" 的 typo。

**8. 清晰度:** Satisfactory
**14. 推荐理由:** 实验评估不够强(未对比 SOTA),方法有效性未验证;技术描述(mean-flow matching 与 flow adapter 设计)多处困惑,且 motivation 缺失。

---

## 3. Reviewer #2 — 评分 2(Reject,独立于 rebuttal),置信度 Very confident (4)

- **类别:** MIC ｜ **定位:** Methodological contribution

**5. 主要贡献(摘述)**
研究脊柱手术规划中术前→术后形态预测;提出 ScoliCMF(面向大形变下快速稳定低步采样的条件 mean-flow 框架)+ FGA 动态注入术前先验;构建配对数据集 ScoliSurg;在脊柱与 MRI 合成任务上优于多个生成 baseline。

**6. 主要优点**
- 术后预测用于脊柱规划**有意义且相对未被探索**。
- **ScoliSurg 数据集**对未来研究有价值。
- FGA 在生成轨迹上自适应使用术前先验,**直观且贴合任务**。

**7. 主要缺点**
- **方法新颖性有限、定位不清**:框架主要基于近期 mean-flow matching,主要新增的是 FGA 条件设计 → 更像把新兴生成框架**适配到本任务**,而非全新范式。
- 实验对比**不充分**:多数 baseline 为通用生成模型,**未纳入更强/更具临床相关性的医学图像合成方法** → 难以评估真实实用优势。
- **应用范围混乱**:主打脊柱规划,却又含 BraTS 脑 MRI 合成 → 不清楚是**脊柱专用框架还是通用合成模型**,应更明确澄清。

**8. 清晰度:** Poor
**14. 推荐理由:** 虽问题有临床意义、结果有潜力,但**贡献不足以接收**:novelty 有限(基于现有 mean-flow/flow-matching + 条件适配);实验未对比更强/更临床相关的方法;范围因 BraTS 而模糊。综合置于接收线下。

---

## 4. Reviewer #4 — 评分 4(Weak Accept),置信度 Confident but not certain (3)

- **类别:** MIC ｜ **定位:** Methodological contribution

**5. 主要贡献(摘述)**
基于 Mean Flow 的医学图像合成框架 ScoliCMF,经 gating adapter 引入病人特异信息,实现高效高质量生成;在脊柱预后合成与脑 MRI 模态转换上表现强;构建首个术前/术后配对脊柱矫正数据集,质量高、实用价值大。

**6. 主要优点**
1. ScoliCMF 用 Mean Flow 合成,**提升生成效率**,契合医学影像场景。
2. FGA 引入病人特异信息作为条件实现可控生成,脊柱预后与 MRI 转换均表现强。
3. 提出**首个配对脊柱治疗数据集 ScoliSurg**。

**7. 主要缺点**
1. **Table 2 与 Table 1 的消融数值(如 PSNR)对不上**,难以解读改进;**Fig. 3 缺 Mean Flow 单独对比** → 不清楚形态一致性增益主要来自 Mean Flow 还是 FGA。
2. Sec. 2.2 称早期强约束、后期放松,但似乎**仅通过 p、q 实现,是否充分存疑**;且按 diffusion/flow 由粗到细的生成规律,**噪声阶段强约束可能反而阻碍大结构形变**(对脊柱尤其重要),机制未讲清。
3. Eq.(8) 设 (r,t)=(0,1) 一步得 x₀,但**与 Fig. 2 的采样步数关系不清**、解释不足。

**8. 清晰度:** Good
**14. 推荐理由:** 创新性地将 Mean Flow 合成用于医学影像,提升效率同时保持质量,目标应用表现强;配对脊柱矫正数据集对社区有价值。故 Weak Accept。

---

## 5. Rebuttal 优先级建议(基于上表交叉分析)

> 以下为整理者归纳,供撰写 rebuttal 参考,非审稿原文。

1. **【最高】补 Mean-Flow vs 普通 Flow-Matching 消融** —— #1、#4 都点名;直接回应"增益来源 CMF 还是 FGA"。论文已有 Base/M1(CMF)/M2(FGA)/Ours 的 Table 2,需把它与 Fig.3 效率图、Table 1 口径对齐,并把 FM 单独列入效率对比。
2. **【最高】补更强/更临床相关的合成 baseline** —— #1、#2 共同硬伤;至少加近两年 diffusion/flow-matching 医学合成 SOTA。
3. **【高】澄清 FGA 的 p,q 与 MLP 维度** —— 回应 #1 "标量怎么进 MLP"。**注意:代码里 p、q 实际是在 *embedding* 上算的(`p=t_emb+r_emb`, `q=t_emb−r_emb`),不是标量** —— 这与论文 Eq.3 写的标量 `p=(t+r)/2, q=t−r` 不一致,正是 #1 困惑的根因。建议改论文公式去对齐代码实现。
4. **【高】补 mean velocity 为何优于 single velocity 的理论/分析** —— 回应 #1、#2 的 novelty 与 motivation 质疑。
5. **【高】明确论文定位** —— 回应 #2:把 BraTS 定位为"通用性验证",ScoliSurg 为"主任务",措辞统一。
6. **【中】修正呈现问题** —— Table 1 "Diffusionl" typo;统一 Table 1/Table 2 数值口径;解释 Eq.8 单步与 Fig.2 多步的关系。
7. **【中】回应早期强约束是否阻碍大形变** —— #4 的机制质疑,需给直觉或实验支撑。

---

## 附:与代码核对发现的相关一致性隐患(供内部参考)

> 见同目录 `PAPER_CODE_GAP`(如已生成)。与审稿意见直接相关的两条:
> - **Eq.7 训练目标**与代码实际跑的 MeanFlow JVP 恒等式(`u_tgt = v −(t−r)·du/dt` + adaptive L2)不一致 —— 加重 #1 对"motivation/公式 trivial"的质疑。
> - **Eq.3 的 p,q 标量** vs 代码 embedding 上的 `t_emb±r_emb` —— 正是 #1 "标量如何进 MLP" 困惑的来源。

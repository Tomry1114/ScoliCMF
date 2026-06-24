# Stage 0 — Data-Contract Audit Report (2026-06-22)
脚本: tools/stage0_audit.py ; 数据: data/Spine生成_Miccai数据集/train (540 对) ; 仅 numpy+PIL。

## 关键结果
- stem 契约: 540/540 共有, 0 错配; 扩展名 2084 png + 76 jpg 混存 → mydataset 必须按 stem 跨扩展名匹配。
- 纵向覆盖 IoU: 中位 0.913, p5 0.766; 18% 对 <0.85, 2% <0.70。
- 边界切断率 0.791, 几乎全在术前: 术前触顶 60.6% / 触底 48.9%; 术后 4.3% / 3.7%。
- 端点漂移: 顶 中位2.1%/p95 12%; 底 中位4%/p95 14%。
- 弧长比 post/pre: 中位0.961 (术前被裁, 混淆)。
- 系数插值(poly-of-y 代理)曲率峰 s=0.5 中位2.0 < 端点包络2.9, 无爆炸/自交 → 样条路径几何可行。
- 原始像素 preop/postop NCC 中位 0.401 → 未配准, 坐实 canonical+latent 路线。

## 判决: CASE 2 (部分覆盖), 根因 = 术前脊柱被裁出框(顶~61%/底~49%), 术后基本完整。
无逐椎标签 → 对应只能按弧长(共同区间), 非椎体级。

## 对 Stage 1 的强制约束
1. canonicalization 用鲁棒主轴+质心+共同可见区间估相似变换(translation+rotation+uniform scale), 禁用端点/非刚性。
2. 曲线损失: validity mask + 仅共同可见 y-区间做有序重采样; Cobb 等只在完整可见曲线上算。
3. 样条系数 α 插值在共同区间+共享节点上做。
4. mydataset 扩展名无关 stem 匹配(76 jpg)。
5. 系数插值平滑无自交 → 样条路径可用。
未决: 真·latent 桥体检(P3)需 AE, 排在 Stage 1 训完 AE 后、训 MeanFlow 前。

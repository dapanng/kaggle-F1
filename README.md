# Kaggle F1 Pit Stop Prediction

Kaggle Playground Series S6E5 竞赛解决方案 —— 预测 F1 赛车下一圈是否进站（`PitNextLap`）的二分类任务。

## 项目概述

本项目的目标是基于 F1 赛车的历史遥测数据（轮胎寿命、圈速、位置、退化程度等），预测某一赛车在下一圈是否会进站换胎。评估指标为 **ROC AUC**。

## 项目结构

```
.
├── solution_optimized_v8.py   # 主解决方案（V8）：多模型融合 + Stacking
├── feature_importance.py      # 特征重要性分析脚本
├── playground-series-s6e5/    # 数据目录（需自行下载）
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── README.md
```

## 核心方法

### 特征工程

基于原始字段构建了约 100+ 维特征，主要分为以下几类：

| 类别 | 示例特征 | 说明 |
|------|---------|------|
| 轮胎复合物交互 | `CxTyreLife`, `CxLapNumber`, `CxDegradation` | Compound 编码 × 原始特征，捕捉不同轮胎的差异化行为 |
| 轮胎寿命变换 | `TL_sq`, `TL_sqrt`, `TL_log`, `TL_cu`, `TL_gt_*` | 多项式/对数变换 + 阈值二值化，刻画非线性退化趋势 |
| 退化特征 | `Deg_per_lap`, `Deg_abs`, `Deg_sq`, `TL_x_Deg` | 每圈退化率、绝对退化、退化 × 轮胎寿命等 |
| 位置特征 | `Pos_sq`, `Is_Top10`, `Pos_x_RP` | 位置相关变换及与赛程进度的交互 |
| Stint 特征 | `Stint_x_TL`, `TL_div_Stint`, `RP_div_Stint` | Stint 与轮胎寿命、圈数、赛程进度的交互 |
| 分组统计 | `Drv_avg_*`, `Race_avg_*`, `Compound_avg_*` | 按车手/赛道/轮胎类型聚合的均值统计 |
| 偏差特征 | `Dev_drv_*`, `Dev_race_*`, `Dev_compound_*` | 当前值与分组均值的偏差 |
| KFold 目标编码 | `Driver_te`, `Compound_Race_te`, `CRStint_te` | 5 折平滑目标编码，避免数据泄露 |
| 风险乘子 | `Risk_CD`, `Risk_DR`, `Risk_CDR`, `Risk_CS` 等 | 目标编码间的乘积交互 |

### 模型融合

采用 **6 种模型配置 × 3 个随机种子 × 5 折交叉验证** 的策略：

| 配置名称 | 模型 | 关键超参数 |
|---------|------|-----------|
| `lgb_deep` | LightGBM GBDT | lr=0.015, depth=10, leaves=255 |
| `lgb_reg` | LightGBM GBDT | lr=0.02, depth=7, leaves=127, 强正则 |
| `xgb_deep` | XGBoost | lr=0.015, depth=8 |
| `xgb_reg` | XGBoost | lr=0.02, depth=6, 强正则 |
| `cat_gpu` | CatBoost (GPU) | lr=0.02, depth=8, Lossguide |
| `cat_cpu` | CatBoost (CPU) | lr=0.03, depth=6 |

### 集成策略

1. **Nelder-Mead 权重优化**：100 次随机初始化，优化各模型加权 AUC
2. **Ridge Stacking**：以各模型 OOF 预测为特征，Ridge 回归做二层融合，搜索最优 `alpha`
3. 最终选择 OOF AUC 更高的方法作为提交结果

## 环境依赖

- Python >= 3.8
- 核心依赖：
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `lightgbm`
  - `xgboost`
  - `catboost`
  - `scipy`

安装依赖：

```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost scipy
```

## 使用方法

### 1. 下载数据

从 [Kaggle 竞赛页面](https://www.kaggle.com/competitions/playground-series-s6e5) 下载数据，放置于 `playground-series-s6e5/` 目录下。

### 2. 运行主方案

```bash
python solution_optimized_v8.py
```

运行后将生成：
- `submission_v8.csv` — 提交文件
- `oof_*_v8.npy` — 各模型 OOF 预测
- `test_*_v8.npy` — 各模型测试集预测
- `experiment_v8.json` — 实验结果记录

### 3. 特征重要性分析

```bash
python feature_importance.py
```

输出各特征在 LightGBM 中的重要性排名，用于指导特征筛选。

## 关键参数

| 参数 | 值 | 说明 |
|------|---|------|
| `SEEDS` | `[42, 123, 2024]` | 多种子降低方差 |
| `NF` | `5` | 5 折交叉验证 |
| `gm` | `0.1990` | 目标编码全局均值（平滑先验） |
| `early_stopping` | `150~200` | 各模型早停轮数 |

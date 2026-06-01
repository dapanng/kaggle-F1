<div align="center">

# Kaggle F1 Pit Stop Prediction

**Playground Series S6E5** — 预测 F1 赛车下一圈是否进站

[![Competition](https://img.shields.io/badge/Kaggle-Playground_S6E5-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s6e5)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)]()
[![Metric](https://img.shields.io/badge/Metric-ROC_AUC-FF6F00)]()

</div>

---

## Table of Contents

- [Background](#background)
- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Ensemble Strategy](#ensemble-strategy)
- [Getting Started](#getting-started)
- [Key Parameters](#key-parameters)
- [Output Files](#output-files)

---

## Background

Kaggle **Playground Series Season 6 Episode 5** 的参赛方案。任务为二分类：基于 F1 赛车实时遥测数据（轮胎寿命、圈速、赛道位置、退化程度等），预测某赛车在下一圈是否会进站换胎（`PitNextLap`）。

| Item | Detail |
|------|--------|
| Competition | [Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5) |
| Task | Binary Classification |
| Target | `PitNextLap` (0 / 1) |
| Metric | ROC AUC |

---

## Repository Structure

```
kaggle-F1/
├── solution_optimized_v8.py       # 主方案 (V8): 特征工程 + 多模型融合 + Stacking
├── feature_importance.py          # 特征重要性分析脚本
├── playground-series-s6e5/        # 数据目录 (需自行下载)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── README.md
```

---

## Pipeline Overview

```
Raw Data
  │
  ▼
Feature Engineering (~100 features)
  ├── Compound interactions & TyreLife transforms
  ├── Degradation, Position, Stint derived features
  ├── Group statistics & Deviations
  └── KFold Target Encoding & Risk multipliers
  │
  ▼
Multi-Model Training (6 configs × 3 seeds × 5-fold CV)
  ├── LightGBM × 2  (deep / regularized)
  ├── XGBoost   × 2  (deep / regularized)
  └── CatBoost  × 2  (GPU / CPU)
  │
  ▼
Second-Level Ensemble
  ├── Nelder-Mead weight optimization
  └── Ridge Stacking (multi-alpha search)
  │
  ▼
Final Submission
```

---

## Feature Engineering

基于原始字段构建约 **100+ 维**特征，分为以下 8 大类：

### 1. Compound Interactions

将轮胎类型编码为有序数值 (`SOFT=0, MEDIUM=1, HARD=2, INTERMEDIATE=3, WET=4`)，与核心字段做乘积交互：

| Feature | Formula | Description |
|---------|---------|-------------|
| `CxTyreLife` | `Compound_ord × TyreLife` | 不同轮胎在相同寿命下的差异化表现 |
| `CxLapNumber` | `Compound_ord × LapNumber` | 轮胎类型 × 圈数 |
| `CxRaceProgress` | `Compound_ord × RaceProgress` | 轮胎类型 × 赛程进度 |
| `CxDegradation` | `Compound_ord × Cumulative_Degradation` | 轮胎类型 × 累计退化 |
| `CxPosition` | `Compound_ord × Position` | 轮胎类型 × 位置 |

### 2. TyreLife Transforms

对 `TyreLife` 施加多种数学变换，捕捉非线性退化趋势：

- **Polynomial**: `TL_sq` (²), `TL_cu` (³)
- **Non-linear**: `TL_sqrt` (√), `TL_log` (log1p)
- **Threshold binarization**: `TL_gt_15`, `TL_gt_25`, `TL_gt_35`, `TL_gt_50`

### 3. Degradation Features

| Feature | Formula | Description |
|---------|---------|-------------|
| `Deg_per_lap` | `Cumulative_Degradation / TyreLife` | 每圈平均退化率 |
| `Deg_abs` | `abs(Cumulative_Degradation)` | 退化绝对值 |
| `Deg_sq` | `Cumulative_Degradation²` | 退化平方项 |
| `TL_x_Deg` | `TyreLife × Cumulative_Degradation` | 轮胎寿命 × 退化交互 |
| `TL_x_LTD` | `TyreLife × LapTime_Delta` | 轮胎寿命 × 圈速差交互 |

### 4. Position & Race Progress

- `Pos_sq`: 位置平方项
- `Is_Top10`: 是否在前 10 名
- `Pos_x_RP`: 位置 × 赛程进度（领先早期 vs 领先晚期）
- `PitStop_x_RP`: 进站标记 × 赛程进度

### 5. Stint Features

| Feature | Formula |
|---------|---------|
| `Stint_x_TL` | `Stint × TyreLife` |
| `Stint_x_LN` | `Stint × LapNumber` |
| `TL_div_Stint` | `TyreLife / Stint` |
| `RP_div_Stint` | `RaceProgress / Stint` |
| `Stint_x_Compound` | `Stint × Compound_ord` |

### 6. Group Statistics & Deviations

按车手 / 赛道 / 轮胎类型分组计算均值，再求当前值与均值的偏差：

- **Group means**: `Drv_avg_tyre`, `Race_avg_degrad`, `Compound_avg_laptime`, ...
- **Deviations**: `Dev_drv_tyre`, `Dev_race_degrad`, `Dev_compound_laptime`, ...

### 7. KFold Target Encoding

5-fold smoothed target encoding，避免数据泄露：

| Dimension | Features |
|-----------|----------|
| 1D | `Driver_te`, `Race_te`, `Compound_te`, `Stint_te`, `Year_te` |
| 2D | `Compound_Race_te`, `Driver_Race_te`, `Compound_Stint_te`, `Driver_Compound_te`, `Driver_Stint_te`, `Race_Stint_te` |
| 3D | `CRStint_te` (Compound × Race × Stint) |

### 8. Risk Multipliers

目标编码间的乘积交互，捕捉组合风险：

`Risk_CD`, `Risk_DR`, `Risk_CR`, `Risk_CDR`, `Risk_CS`, `Risk_DS`, `Risk_RS`

---

## Model Architecture

**6 configs × 3 seeds × 5-fold CV = 90 sub-models**

| Config | Framework | lr | max_depth | num_leaves | Highlights |
|--------|-----------|:--:|:---------:|:----------:|------------|
| `lgb_deep` | LightGBM GBDT | 0.015 | 10 | 255 | Deep trees, low lr |
| `lgb_reg` | LightGBM GBDT | 0.02 | 7 | 127 | Medium depth, strong reg |
| `xgb_deep` | XGBoost | 0.015 | 8 | — | Deep trees, low lr |
| `xgb_reg` | XGBoost | 0.02 | 6 | — | Shallow trees, strong reg |
| `cat_gpu` | CatBoost (GPU) | 0.02 | 8 | — | Lossguide grow policy |
| `cat_cpu` | CatBoost (CPU) | 0.03 | 6 | — | Symmetric tree, high bagging temp |

All models use **early stopping** (150–200 rounds) to prevent overfitting.

---

## Ensemble Strategy

### Layer 1: Multi-Model OOF Predictions

Each config trains with 3 seeds × 5 folds; OOF predictions are averaged across seeds → 6 OOF vectors + 6 test prediction vectors.

### Layer 2: Ensemble Optimization

Two methods are tried in parallel; the one with higher OOF AUC is selected:

| Method | Description |
|--------|-------------|
| **Nelder-Mead** | 100 Dirichlet random inits → Nelder-Mead local opt → pick best AUC weights |
| **Ridge Stacking** | 6 OOF vectors as features → 5-fold Ridge → search `alpha ∈ {1, 10, 50, 100, 200, 500, 1000}` |

Final predictions are clipped to `[0.001, 0.999]`.

---

## Getting Started

### Prerequisites

- Python >= 3.8
- numpy, pandas, scikit-learn, lightgbm, xgboost, catboost, scipy

### Install

```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost scipy
```

> **Note**: `cat_gpu` requires a GPU environment with the corresponding CatBoost version. If no GPU is available, change `task_type` to `'CPU'`.

### Download Data

Download `train.csv`, `test.csv`, `sample_submission.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s6e5) and place them under `playground-series-s6e5/`.

### Run

```bash
# Main solution
python solution_optimized_v8.py

# Feature importance analysis
python feature_importance.py
```

---

## Key Parameters

| Parameter | Value | Description |
|:---------:|:-----:|:------------|
| `SEEDS` | `[42, 123, 2024]` | Multi-seed to reduce variance |
| `NF` | `5` | 5-fold stratified CV |
| `gm` | `0.1990` | Global mean for target encoding smoothing prior |
| `early_stopping` | `150–200` | Early stopping rounds per model |
| `Nelder-Mead trials` | `100` | Random initialization count for weight optimization |
| `Ridge alpha search` | `{1, 10, 50, 100, 200, 500, 1000}` | Stacking regularization search space |

---

## Output Files

Running `solution_optimized_v8.py` generates:

| File | Description |
|------|-------------|
| `submission_v8.csv` | Final submission file |
| `oof_{config}_v8.npy` | OOF predictions per model config (×6) |
| `test_{config}_v8.npy` | Test predictions per model config (×6) |
| `experiment_v8.json` | Experiment log (per-model AUC, final method, elapsed time, etc.) |

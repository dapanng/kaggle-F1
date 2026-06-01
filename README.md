<div align="center">

# 🏎️ Kaggle F1 Pit Stop Prediction

**Playground Series S6E5** — 预测 F1 赛车下一圈是否进站

[![Competition](https://img.shields.io/badge/Kaggle-Playground%20S6E5-blue)](https://www.kaggle.com/competitions/playground-series-s6e5)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)]()
[![Metric](https://img.shields.io/badge/Metric-ROC%20AUC-orange)]()

</div>

---

## 📋 目录

- [项目背景](#-项目背景)
- [项目结构](#-项目结构)
- [方法概览](#-方法概览)
- [特征工程](#-特征工程)
- [模型架构](#-模型架构)
- [集成策略](#-集成策略)
- [环境与安装](#-环境与安装)
- [快速开始](#-快速开始)
- [关键参数](#-关键参数)
- [输出文件](#-输出文件)

---

## 🏁 项目背景

本仓库是 Kaggle **Playground Series S6E5** 竞赛的解决方案。任务为二分类问题：基于 F1 赛车的实时遥测数据（轮胎寿命、圈速、赛道位置、退化程度等），预测某赛车在**下一圈是否会进站换胎**（`PitNextLap`）。

- **评估指标**：ROC AUC
- **目标变量**：`PitNextLap`（0 / 1）
- **数据来源**：[Kaggle 竞赛页面](https://www.kaggle.com/competitions/playground-series-s6e5)

---

## 📁 项目结构

```
kaggle-F1/
├── solution_optimized_v8.py       # 主解决方案（V8）：特征工程 + 多模型融合 + Stacking
├── feature_importance.py          # 特征重要性分析工具
├── playground-series-s6e5/        # 数据目录（需自行下载）
│   ├── train.csv                  # 训练集
│   ├── test.csv                   # 测试集
│   └── sample_submission.csv      # 提交样例
└── README.md
```

---

## 🔍 方法概览

```
原始数据
  │
  ▼
特征工程（~100 维特征）
  ├── 基础变换 & 交互
  ├── 分组统计 & 偏差
  └── KFold 目标编码 & 风险乘子
  │
  ▼
多模型训练（6 配置 × 3 种子 × 5 折）
  ├── LightGBM × 2（deep / reg）
  ├── XGBoost × 2（deep / reg）
  └── CatBoost × 2（GPU / CPU）
  │
  ▼
二层集成
  ├── Nelder-Mead 权重优化
  └── Ridge Stacking（多 alpha 搜索）
  │
  ▼
最终提交
```

---

## 🛠️ 特征工程

基于原始字段构建了约 **100+ 维**特征，分为以下类别：

### 1. 轮胎复合物交互

将轮胎类型（SOFT / MEDIUM / HARD / INTERMEDIATE / WET）编码为有序数值，与核心字段做乘积交互：

| 特征 | 公式 | 含义 |
|------|------|------|
| `CxTyreLife` | `Compound_ord × TyreLife` | 不同轮胎在相同寿命下的差异化表现 |
| `CxLapNumber` | `Compound_ord × LapNumber` | 轮胎类型 × 圈数 |
| `CxRaceProgress` | `Compound_ord × RaceProgress` | 轮胎类型 × 赛程进度 |
| `CxDegradation` | `Compound_ord × Cumulative_Degradation` | 轮胎类型 × 累计退化 |
| `CxPosition` | `Compound_ord × Position` | 轮胎类型 × 位置 |

### 2. 轮胎寿命变换

对 `TyreLife` 施加多种数学变换，捕捉非线性退化趋势：

- **多项式**：`TL_sq`（²）、`TL_cu`（³）
- **非线性**：`TL_sqrt`（√）、`TL_log`（log1p）
- **阈值二值化**：`TL_gt_15`、`TL_gt_25`、`TL_gt_35`、`TL_gt_50`（标记轮胎是否超过关键寿命阈值）

### 3. 退化特征

| 特征 | 公式 | 含义 |
|------|------|------|
| `Deg_per_lap` | `Cumulative_Degradation / TyreLife` | 每圈平均退化率 |
| `Deg_abs` | `abs(Cumulative_Degradation)` | 退化绝对值 |
| `Deg_sq` | `Cumulative_Degradation²` | 退化平方项 |
| `TL_x_Deg` | `TyreLife × Cumulative_Degradation` | 轮胎寿命 × 退化交互 |
| `TL_x_LTD` | `TyreLife × LapTime_Delta` | 轮胎寿命 × 圈速差交互 |

### 4. 位置 & 赛程特征

- `Pos_sq`：位置平方项
- `Is_Top10`：是否在前 10 名
- `Pos_x_RP`：位置 × 赛程进度（领先早期 vs 领先晚期）
- `PitStop_x_RP`：进站标记 × 赛程进度

### 5. Stint 特征

| 特征 | 公式 |
|------|------|
| `Stint_x_TL` | `Stint × TyreLife` |
| `Stint_x_LN` | `Stint × LapNumber` |
| `TL_div_Stint` | `TyreLife / Stint` |
| `RP_div_Stint` | `RaceProgress / Stint` |
| `Stint_x_Compound` | `Stint × Compound_ord` |

### 6. 分组统计 & 偏差

按 **车手 / 赛道 / 轮胎类型** 分组计算均值，再求当前值与均值的偏差：

- **分组均值**：`Drv_avg_tyre`、`Race_avg_degrad`、`Compound_avg_laptime` 等
- **偏差特征**：`Dev_drv_tyre`、`Dev_race_degrad`、`Dev_compound_laptime` 等

### 7. KFold 目标编码

使用 5 折平滑目标编码（Smoothing），避免数据泄露：

| 编码维度 | 特征名 |
|---------|--------|
| 单维度 | `Driver_te`、`Race_te`、`Compound_te`、`Stint_te`、`Year_te` |
| 二维交叉 | `Compound_Race_te`、`Driver_Race_te`、`Compound_Stint_te`、`Driver_Compound_te`、`Driver_Stint_te`、`Race_Stint_te` |
| 三维交叉 | `CRStint_te`（Compound × Race × Stint） |

### 8. 风险乘子

目标编码间的乘积交互，捕捉组合风险：

`Risk_CD`、`Risk_DR`、`Risk_CR`、`Risk_CDR`、`Risk_CS`、`Risk_DS`、`Risk_RS`

---

## 🏗️ 模型架构

采用 **6 种模型配置 × 3 个随机种子 × 5 折交叉验证** 的训练策略，共产生 90 个子模型：

| 配置 | 框架 | learning_rate | max_depth | num_leaves | 特点 |
|------|------|:------------:|:---------:|:----------:|------|
| `lgb_deep` | LightGBM GBDT | 0.015 | 10 | 255 | 深树、低学习率 |
| `lgb_reg` | LightGBM GBDT | 0.02 | 7 | 127 | 中等深度、强正则 |
| `xgb_deep` | XGBoost | 0.015 | 8 | — | 深树、低学习率 |
| `xgb_reg` | XGBoost | 0.02 | 6 | — | 浅树、强正则 |
| `cat_gpu` | CatBoost (GPU) | 0.02 | 8 | — | Lossguide 增长策略 |
| `cat_cpu` | CatBoost (CPU) | 0.03 | 6 | — | SymmetricTree、高 bagging 温度 |

所有模型均使用 **early stopping**（150~200 轮）防止过拟合。

---

## 🔗 集成策略

### 第一层：多模型 OOF 预测

每个配置在 3 个种子下分别做 5 折训练，种子内 OOF 取平均，得到 6 组 OOF 向量和测试集预测。

### 第二层：集成优化

并行尝试两种方法，选择 OOF AUC 更优者：

| 方法 | 描述 |
|------|------|
| **Nelder-Mead 权重优化** | 100 次 Dirichlet 随机初始化 → Nelder-Mead 局部优化 → 选 AUC 最高的权重 |
| **Ridge Stacking** | 以 6 组 OOF 为特征，5 折 Ridge 回归 → 搜索 `alpha ∈ {1, 10, 50, 100, 200, 500, 1000}` |

最终预测值裁剪至 `[0.001, 0.999]` 后生成提交文件。

---

## 💻 环境与安装

### 依赖

- Python >= 3.8
- numpy
- pandas
- scikit-learn
- lightgbm
- xgboost
- catboost
- scipy

### 安装

```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost scipy
```

> **注意**：`cat_gpu` 配置需要 GPU 环境与对应版本的 CatBoost。若无 GPU，可将 `task_type` 改为 `'CPU'`。

---

## 🚀 快速开始

### 1. 下载数据

从 [Kaggle 竞赛页面](https://www.kaggle.com/competitions/playground-series-s6e5) 下载 `train.csv`、`test.csv`、`sample_submission.csv`，放置于 `playground-series-s6e5/` 目录。

### 2. 运行主方案

```bash
python solution_optimized_v8.py
```

### 3. 特征重要性分析

```bash
python feature_importance.py
```

输出 LightGBM 5 折训练的各特征重要性排名（Top 30 / Bottom 20），用于指导特征筛选与迭代。

---

## ⚙️ 关键参数

| 参数 | 值 | 说明 |
|:----:|:--:|:-----|
| `SEEDS` | `[42, 123, 2024]` | 多随机种子降低方差 |
| `NF` | `5` | 5 折分层交叉验证 |
| `gm` | `0.1990` | 目标编码全局均值（平滑先验） |
| `early_stopping` | `150 ~ 200` | 各模型早停轮数 |
| `Nelder-Mead trials` | `100` | 权重优化随机初始化次数 |
| `Ridge alpha search` | `{1, 10, 50, 100, 200, 500, 1000}` | Stacking 正则化搜索空间 |

---

## 📦 输出文件

运行 `solution_optimized_v8.py` 后生成：

| 文件 | 说明 |
|------|------|
| `submission_v8.csv` | 最终提交文件 |
| `oof_{config}_v8.npy` | 各模型配置的 OOF 预测（6 个） |
| `test_{config}_v8.npy` | 各模型配置的测试集预测（6 个） |
| `experiment_v8.json` | 实验结果记录（各模型 AUC、最终方法、耗时等） |

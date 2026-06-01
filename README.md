# Kaggle F1 Pit Stop Prediction

> Kaggle **Playground Series Season 6 Episode 5** 参赛方案 —— 基于 F1 赛车实时遥测数据，预测赛车下一圈是否进站换胎。

---

## 目录

- [背景](#背景)
- [仓库结构](#仓库结构)
- [方案演进](#方案演进)
- [Pipeline 总览](#pipeline-总览)
- [特征工程](#特征工程)
- [模型架构](#模型架构)
- [融合策略](#融合策略)
- [关键参数](#关键参数)
- [输出文件](#输出文件)

---

## 背景

| 项目       | 说明                                                                               |
| ---------- | ---------------------------------------------------------------------------------- |
| 赛事       | [Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5) |
| 任务类型   | 二分类 (Binary Classification)                                                     |
| 预测目标   | `PitNextLap` — 赛车下一圈是否会进站换胎 (0 / 1)                                    |
| 评估指标   | ROC AUC                                                                            |
| 数据来源   | F1 赛车实时遥测数据（轮胎寿命、圈速、赛道位置、退化程度等）                          |

---

## 仓库结构

```
kaggle-F1/
├── eda_analysis.py            # 探索性数据分析 (EDA)
├── feature_importance.py      # 特征重要性分析
├── solution_v1.py             # V1 基线方案 (单模型)
├── solution_lgb.py            # LightGBM 单模型方案
├── solution_xgb.py            # XGBoost 单模型方案
├── solution_cat.py            # CatBoost 单模型方案
├── solution_opt.py            # 优化方案: 多模型融合
├── solution_opt_v2.py         # V2 终极方案: 特征工程 + 多模型融合 + Stacking
├── playground-series-s6e5/    # 数据目录 (需自行下载)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── README.md
```

---

## 方案演进

本项目采用渐进式优化策略，从基线到最终方案经历了多个迭代：

```
solution_v1.py          → 基线方案：基础特征 + 单模型
    ↓
solution_lgb/xgb/cat.py → 单模型深入：分别调优 LightGBM / XGBoost / CatBoost
    ↓
solution_opt.py         → 多模型融合：三模型加权集成 + Nelder-Mead 权重优化
    ↓
solution_opt_v2.py      → 终极方案：100+ 特征 + 6 配置 × 3 种子 × 5 折 + Stacking
```

| 版本               | 特征数 | 模型配置 | 融合方式                     | 核心改进                         |
| ------------------ | ------ | -------- | ---------------------------- | -------------------------------- |
| `solution_v1`      | ~20    | 1        | 无                           | 基线建立                         |
| `solution_lgb/xgb/cat` | ~30 | 1        | 无                           | 单模型深度调优                   |
| `solution_opt`     | ~50    | 3        | Nelder-Mead 加权             | 多模型融合                       |
| `solution_opt_v2`  | 100+   | 6        | Nelder-Mead + Ridge Stacking | 全面特征工程 + 二级 Stacking     |

---

## Pipeline 总览

```
原始数据
  │
  ▼
特征工程 (~100+ 维特征)
  ├── 轮胎交互特征 (Compound Interactions)
  ├── 轮胎寿命变换 (TyreLife Transforms)
  ├── 退化特征 (Degradation Features)
  ├── 位置与赛程进度 (Position & Race Progress)
  ├── Stint 特征 (Stint Features)
  ├── 分组统计与偏差 (Group Statistics & Deviations)
  ├── KFold 目标编码 (Target Encoding)
  └── 风险乘子 (Risk Multipliers)
  │
  ▼
多模型训练 (6 configs × 3 seeds × 5-fold CV = 90 子模型)
  ├── LightGBM × 2  (deep / regularized)
  ├── XGBoost   × 2  (deep / regularized)
  └── CatBoost  × 2  (GPU / CPU)
  │
  ▼
二级融合 (Second-Level Ensemble)
  ├── Nelder-Mead 权重优化
  └── Ridge Stacking (多 alpha 搜索)
  │
  ▼
最终提交文件
```

---

## 特征工程

基于原始字段构建约 **100+ 维**特征，分为以下 8 大类：

### 1. 轮胎交互特征 (Compound Interactions)

将轮胎类型编码为有序数值 (`SOFT=0, MEDIUM=1, HARD=2, INTERMEDIATE=3, WET=4`)，与核心字段做乘积交互：

| 特征             | 公式                                   | 含义                   |
| ---------------- | -------------------------------------- | ---------------------- |
| CxTyreLife       | Compound_ord × TyreLife                | 不同轮胎在相同寿命下的差异化表现 |
| CxLapNumber      | Compound_ord × LapNumber               | 轮胎类型 × 圈数         |
| CxRaceProgress   | Compound_ord × RaceProgress            | 轮胎类型 × 赛程进度     |
| CxDegradation    | Compound_ord × Cumulative_Degradation  | 轮胎类型 × 累计退化     |
| CxPosition       | Compound_ord × Position                | 轮胎类型 × 位置         |

### 2. 轮胎寿命变换 (TyreLife Transforms)

对 `TyreLife` 施加多种数学变换，捕捉非线性退化趋势：

- **多项式**: `TL_sq` (²), `TL_cu` (³)
- **非线性**: `TL_sqrt` (√), `TL_log` (log1p)
- **阈值二值化**: `TL_gt_5`, `TL_gt_10`, `TL_gt_15`, `TL_gt_20`, `TL_gt_25`, `TL_gt_30`, `TL_gt_35`, `TL_gt_40`, `TL_gt_50`

### 3. 退化特征 (Degradation Features)

| 特征          | 公式                                | 含义              |
| ------------- | ----------------------------------- | ----------------- |
| Deg_per_lap   | Cumulative_Degradation / TyreLife   | 每圈平均退化率    |
| LTD_per_lap   | LapTime_Delta / TyreLife            | 每圈圈速差变化    |
| LT_per_lap    | LapTime (s) / TyreLife              | 每圈用时变化      |
| Deg_abs       | abs(Cumulative_Degradation)         | 退化绝对值        |
| Deg_sq        | Cumulative_Degradation²             | 退化平方项        |

### 4. 位置与赛程进度 (Position & Race Progress)

| 特征          | 公式 / 逻辑                            | 含义                     |
| ------------- | -------------------------------------- | ------------------------ |
| Is_P1         | Position == 1                          | 是否领跑                 |
| Is_Podium     | Position <= 3                          | 是否领奖台位置           |
| Is_Top5       | Position <= 5                          | 是否前 5                 |
| Is_Top10      | Position <= 10                         | 是否前 10                |
| Is_Back5      | Position >= 16                         | 是否后 5                 |
| Pos_sq        | Position²                              | 位置平方项               |
| RP_sq / RP_cu | RaceProgress² / RaceProgress³          | 赛程进度多项式           |
| Is_Early      | RaceProgress < 0.1                     | 是否赛程早期             |
| Is_Late       | RaceProgress > 0.9                     | 是否赛程晚期             |
| Is_Mid        | 0.3 < RaceProgress < 0.7              | 是否赛程中期             |

### 5. Stint 特征

| 特征            | 公式                        | 含义                  |
| --------------- | --------------------------- | --------------------- |
| Is_Stint1       | Stint == 1                  | 是否第一段 stint       |
| Is_Stint2       | Stint == 2                  | 是否第二段 stint       |
| Is_Stint3plus   | Stint >= 3                  | 是否第三段及以后       |
| Stint_x_TL      | Stint × TyreLife            | Stint × 轮胎寿命      |
| Stint_x_LN      | Stint × LapNumber           | Stint × 圈数          |
| TL_div_Stint    | TyreLife / Stint            | 轮胎寿命 / Stint      |
| LN_div_Stint    | LapNumber / Stint           | 圈数 / Stint          |

### 6. 分组统计与偏差 (Group Statistics & Deviations)

按车手 / 赛道 / 轮胎类型分组计算均值，再求当前值与均值的偏差：

- **分组均值**: `Drv_avg_tyre`, `Drv_avg_pos`, `Drv_avg_laptime`, `Race_avg_degrad`, `Race_avg_laptime`
- **偏差特征**: `Dev_drv_tyre`, `Dev_drv_pos`, `Dev_race_degrad`, `Dev_race_laptime`

### 7. KFold 目标编码 (Target Encoding)

5 折平滑目标编码，避免数据泄露：

| 维度 | 特征                                                                                                      |
| ---- | --------------------------------------------------------------------------------------------------------- |
| 1D   | Driver_te, Race_te, Compound_te, Stint_te, Year_te                                                        |
| 2D   | Compound_Race_te, Driver_Race_te, Compound_Stint_te                                                       |

平滑公式：`smooth = (mean × count + gm × prior_count) / (count + prior_count)`

其中 `gm = 0.1990` 为全局均值先验。

### 8. 风险乘子 (Risk Multipliers)

目标编码间的乘积交互，捕捉组合风险：

`Risk_CD` (Compound × Driver), `Risk_DR` (Driver × Race), `Risk_CDR` (Compound × Driver × Race)

---

## 模型架构

**6 configs × 3 seeds × 5-fold CV = 90 个子模型**

| 配置       | 框架            | 学习率 | 最大深度 | 叶子数 | 特点                              |
| ---------- | --------------- | ------ | -------- | ------ | --------------------------------- |
| lgb_deep   | LightGBM GBDT   | 0.015  | 10       | 255    | 深树 + 低学习率                   |
| lgb_reg    | LightGBM GBDT   | 0.02   | 7        | 127    | 中等深度 + 强正则                 |
| xgb_deep   | XGBoost         | 0.015  | 8        | —      | 深树 + 低学习率                   |
| xgb_reg    | XGBoost         | 0.02   | 6        | —      | 浅树 + 强正则                     |
| cat_gpu    | CatBoost (GPU)  | 0.02   | 8        | —      | Lossguide 生长策略                |
| cat_cpu    | CatBoost (CPU)  | 0.03   | 6        | —      | 对称树 + 高 bagging 温度          |

所有模型均使用 **early stopping** (150–200 轮) 防止过拟合。

---

## 融合策略

### 第一层：多模型 OOF 预测

每个配置使用 3 个种子 × 5 折训练；OOF 预测在种子间取平均 → 6 个 OOF 向量 + 6 个测试预测向量。

### 第二层：融合优化

两种方法并行尝试，选择 OOF AUC 更高者：

| 方法               | 描述                                                                                      |
| ------------------ | ----------------------------------------------------------------------------------------- |
| **Nelder-Mead**    | 100 次 Dirichlet 随机初始化 → Nelder-Mead 局部优化 → 选取最优 AUC 权重                    |
| **Ridge Stacking** | 6 个 OOF 向量作为特征 → 5 折 Ridge 回归 → 搜索 alpha ∈ {1, 10, 50, 100, 200, 500, 1000}  |

最终预测裁剪至 `[0.001, 0.999]`。

---

## 关键参数

| 参数                | 值                                | 说明                         |
| ------------------- | --------------------------------- | ---------------------------- |
| SEEDS               | [42, 123, 2024]                   | 多种子降低方差               |
| NF                  | 5                                 | 5 折分层交叉验证             |
| gm                  | 0.1990                            | 目标编码平滑先验全局均值     |
| early_stopping      | 150–200                           | 各模型早停轮数               |
| Nelder-Mead trials  | 100                               | 权重优化随机初始化次数       |
| Ridge alpha search  | {1, 10, 50, 100, 200, 500, 1000} | Stacking 正则化搜索空间      |

---

## 输出文件

运行 `solution_opt_v2.py` 生成：

| 文件                 | 说明                                          |
| -------------------- | --------------------------------------------- |
| submission.csv       | 最终提交文件                                  |
| oof_{config}.npy     | 各模型配置的 OOF 预测 (×6)                    |
| test_{config}.npy    | 各模型配置的测试预测 (×6)                     |
| experiment.json      | 实验日志 (各模型 AUC、最终方法、耗时等)        |

---

## 分析工具

### EDA 分析 (`eda_analysis.py`)

探索性数据分析脚本，包含以下分析模块：

- 目标变量分布与正样本率
- 特征类型与缺失值检查
- 数值特征统计摘要
- 分类特征唯一值与目标交叉分析
- 数值特征与目标的相关性排序
- 异常值检测 (IQR 方法)
- 重复数据检查
- 轮胎类型 (Compound) 与进站率分析
- 年度进站率趋势
- 关键特征按目标的分布对比
- TyreLife / Position / LapNumber 分箱进站率
- 车手 / 赛道进站率差异
- 轮胎类型 × 寿命交互进站率
- Stint 分析
- 训练集与测试集特征分布一致性检查
- Pre-Season Testing 特殊分析

### 特征重要性分析 (`feature_importance.py`)

使用 LightGBM 5 折交叉验证训练，输出：

- 各折 AUC 分数与总体 AUC
- Top 30 重要特征排名
- Bottom 20 重要特征排名
- 零重要性特征统计

---

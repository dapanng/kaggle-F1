"""
Solution: Playground Series S6E5 - F1 Pit Stop Prediction (PitNextLap)
Metric: AUC (Binary Classification)
Target AUC: 0.953+

Key EDA Findings:
- Target: PitNextLap (19.9% positive rate, class imbalance ~4:1)
- Year=2023 has only 0.96% pit rate (data artifact)
- Compound is strongest signal: HARD(32.8%) >> SOFT(19.3%) > INT(15.2%) > MED(10.1%) > WET(2.5%)
- TyreLife, LapNumber, Stint are key numeric predictors
- 86 drivers only in train (need careful encoding)
- Race variation: Monaco(35.7%) vs Mexico(9.1%) pit rates
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5

print("=" * 60)
print("Loading data...")
print("=" * 60)

train = pd.read_csv('playground-series-s6e5/train.csv')
test = pd.read_csv('playground-series-s6e5/test.csv')
sub = pd.read_csv('playground-series-s6e5/sample_submission.csv')

train['PitNextLap'] = train['PitNextLap'].astype(int)
print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("Feature Engineering...")
print("=" * 60)

def engineer_features(df, train_df=None, is_train=True):
    """Add engineered features. If is_train=False, use train_df for encoding."""
    df = df.copy()

    # --- Categorical Encoding ---
    # Compound: direct label encoding
    compound_order = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
    df['Compound_enc'] = df['Compound'].map(compound_order)

    # Target encoding for Driver using train data
    if is_train:
        driver_target_mean = df.groupby('Driver')['PitNextLap'].mean()
        race_target_mean = df.groupby('Race')['PitNextLap'].mean()
    else:
        driver_target_mean = train_df.groupby('Driver')['PitNextLap'].mean()
        race_target_mean = train_df.groupby('Race')['PitNextLap'].mean()

    # Map target encoding with smoothing for unseen values
    global_mean = 0.1990
    df['Driver_target_enc'] = df['Driver'].map(driver_target_mean).fillna(global_mean)
    df['Race_target_enc'] = df['Race'].map(race_target_mean).fillna(global_mean)

    # Compound target encoding
    if is_train:
        compound_target_mean = df.groupby('Compound')['PitNextLap'].mean()
    else:
        compound_target_mean = train_df.groupby('Compound')['PitNextLap'].mean()
    df['Compound_target_enc'] = df['Compound'].map(compound_target_mean)

    # --- Interaction Features ---
    # Compound × TyreLife (key interaction from EDA)
    df['Compound_x_TyreLife'] = df['Compound_enc'] * df['TyreLife']

    # Compound × RaceProgress
    df['Compound_x_RaceProgress'] = df['Compound_enc'] * df['RaceProgress']

    # Compound × LapNumber
    df['Compound_x_LapNumber'] = df['Compound_enc'] * df['LapNumber']

    # --- Degradation Rate Features ---
    # Degradation per lap (avoid division by zero)
    df['TyreLife_safe'] = df['TyreLife'].clip(lower=1)
    df['Deg_per_lap'] = df['Cumulative_Degradation'] / df['TyreLife_safe']
    df['LapTimeDelta_per_lap'] = df['LapTime_Delta'] / df['TyreLife_safe']

    # --- Tire-Related Features ---
    # Is tire past typical life?
    df['TyreLife_gt_15'] = (df['TyreLife'] > 15).astype(int)
    df['TyreLife_gt_25'] = (df['TyreLife'] > 25).astype(int)
    df['TyreLife_gt_40'] = (df['TyreLife'] > 40).astype(int)

    # Tire life squared (captures non-linear relationship)
    df['TyreLife_sq'] = df['TyreLife'] ** 2

    # --- Race Progress Features ---
    df['RaceProgress_sq'] = df['RaceProgress'] ** 2
    df['RaceProgress_cu'] = df['RaceProgress'] ** 3

    # LapNumber squared
    df['LapNumber_sq'] = df['LapNumber'] ** 2

    # --- Is first lap / first stint indicators ---
    df['Is_Lap1'] = (df['LapNumber'] == 1).astype(int)
    df['Is_Stint1'] = (df['Stint'] == 1).astype(int)

    # --- Position Features ---
    df['Is_Top3'] = (df['Position'] <= 3).astype(int)
    df['Is_Top5'] = (df['Position'] <= 5).astype(int)
    df['Is_Backmarker'] = (df['Position'] >= 15).astype(int)

    # --- Year Feature ---
    # Year 2023 has extremely low pit rate (data artifact) - flag it
    df['Year_2023'] = (df['Year'] == 2023).astype(int)
    df['Year_2024'] = (df['Year'] == 2024).astype(int)
    df['Year_2025'] = (df['Year'] == 2025).astype(int)

    # --- Stint × TyreLife ---
    df['Stint_x_TyreLife'] = df['Stint'] * df['TyreLife']

    # --- Position_Change absolute ---
    df['Position_Change_abs'] = np.abs(df['Position_Change'])

    # --- PitStop feature ---
    df['Has_Pitted'] = df['PitStop']

    # --- Race and Driver risk combination ---
    df['Driver_Race_risk'] = df['Driver_target_enc'] * df['Race_target_enc']

    # --- Advanced interaction ---
    df['TyreLife_div_Stint'] = df['TyreLife'] / df['Stint'].clip(lower=1)
    df['LapNumber_div_Stint'] = df['LapNumber'] / df['Stint'].clip(lower=1)

    # --- Invert Cumulative_Degradation (correlation was negative - normalize) ---
    df['Tire_Wear'] = -df['Cumulative_Degradation']

    return df

train_fe = engineer_features(train, is_train=True)
test_fe = engineer_features(test, train_df=train, is_train=False)

# ============================================================
# DEFINE FEATURES
# ============================================================
# Original numeric features
orig_num = ['Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position',
            'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'Position_Change']

# Original categorical (encoded)
orig_cat_enc = ['Compound_enc', 'Driver_target_enc', 'Race_target_enc', 'Compound_target_enc']

# Engineered features
eng_features = [
    'Compound_x_TyreLife', 'Compound_x_RaceProgress', 'Compound_x_LapNumber',
    'Deg_per_lap', 'LapTimeDelta_per_lap',
    'TyreLife_gt_15', 'TyreLife_gt_25', 'TyreLife_gt_40',
    'TyreLife_sq', 'RaceProgress_sq', 'RaceProgress_cu', 'LapNumber_sq',
    'Is_Lap1', 'Is_Stint1', 'Is_Top3', 'Is_Top5', 'Is_Backmarker',
    'Year_2023', 'Year_2024', 'Year_2025',
    'Stint_x_TyreLife', 'Position_Change_abs', 'Has_Pitted',
    'Driver_Race_risk', 'TyreLife_div_Stint', 'LapNumber_div_Stint',
    'Tire_Wear'
]

FEATURE_COLS = orig_num + orig_cat_enc + eng_features

print(f"Total features: {len(FEATURE_COLS)}")
print(f"  Original numeric: {len(orig_num)}")
print(f"  Encoded categorical: {len(orig_cat_enc)}")
print(f"  Engineered: {len(eng_features)}")

# Drop any potential NaN columns
train_fe = train_fe.fillna(0)
test_fe = test_fe.fillna(0)

X = train_fe[FEATURE_COLS].values
y = train_fe['PitNextLap'].values
X_test = test_fe[FEATURE_COLS].values

# Check for NaN/Inf
assert not np.isnan(X).any(), "NaN in training data!"
assert not np.isinf(X).any(), "Inf in training data!"
print(f"Training data shape: {X.shape}")
print(f"Test data shape: {X_test.shape}")

# ============================================================
# CROSS-VALIDATION & MODEL TRAINING
# ============================================================
print("\n" + "=" * 60)
print("Model Training with Stratified 5-Fold CV")
print("=" * 60)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Storage for OOF predictions
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

# Test predictions
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

# ============================================================
# LightGBM Model
# ============================================================
print("\n--- LightGBM ---")
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 10000,
    'learning_rate': 0.03,
    'num_leaves': 127,
    'max_depth': 8,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'min_gain_to_split': 0.01,
    'verbose': -1,
    'random_state': SEED,
    'n_jobs': -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )

    oof_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
    test_lgb += model.predict_proba(X_test)[:, 1] / N_FOLDS

    auc = roc_auc_score(y_val, oof_lgb[val_idx])
    print(f"  Fold {fold+1}: AUC = {auc:.6f}, Best Iter = {model.best_iteration_}")

print(f"  LightGBM OOF AUC: {roc_auc_score(y, oof_lgb):.6f}")

# ============================================================
# XGBoost Model
# ============================================================
print("\n--- XGBoost ---")
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 10000,
    'learning_rate': 0.03,
    'max_depth': 7,
    'min_child_weight': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'reg_alpha': 0.3,
    'reg_lambda': 0.5,
    'gamma': 0.1,
    'tree_method': 'hist',
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': 0,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    oof_xgb[val_idx] = model.predict_proba(X_val)[:, 1]
    test_xgb += model.predict_proba(X_test)[:, 1] / N_FOLDS

    auc = roc_auc_score(y_val, oof_xgb[val_idx])
    print(f"  Fold {fold+1}: AUC = {auc:.6f}, Best Iter = {model.best_iteration}")

print(f"  XGBoost OOF AUC: {roc_auc_score(y, oof_xgb):.6f}")

# ============================================================
# CatBoost Model
# ============================================================
print("\n--- CatBoost ---")
cat_params = {
    'iterations': 5000,
    'learning_rate': 0.03,
    'depth': 7,
    'l2_leaf_reg': 3,
    'border_count': 254,
    'random_strength': 1,
    'bagging_temperature': 0.5,
    'od_type': 'Iter',
    'od_wait': 100,
    'verbose': 0,
    'random_seed': SEED,
    'task_type': 'CPU',
    'thread_count': -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    train_pool = Pool(X_tr, y_tr)
    val_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(**cat_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=False
    )

    oof_cat[val_idx] = model.predict_proba(X_val)[:, 1]
    test_cat += model.predict_proba(X_test)[:, 1] / N_FOLDS

    auc = roc_auc_score(y_val, oof_cat[val_idx])
    print(f"  Fold {fold+1}: AUC = {auc:.6f}, Best Iter = {model.tree_count_}")

print(f"  CatBoost OOF AUC: {roc_auc_score(y, oof_cat):.6f}")

# ============================================================
# ENSEMBLE
# ============================================================
print("\n" + "=" * 60)
print("Ensemble Results")
print("=" * 60)

# Simple weighted average
w_lgb, w_xgb, w_cat = 0.35, 0.30, 0.35
oof_ensemble = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
test_ensemble = w_lgb * test_lgb + w_xgb * test_xgb + w_cat * test_cat

print(f"LightGBM  OOF AUC: {roc_auc_score(y, oof_lgb):.6f}")
print(f"XGBoost   OOF AUC: {roc_auc_score(y, oof_xgb):.6f}")
print(f"CatBoost  OOF AUC: {roc_auc_score(y, oof_cat):.6f}")
print(f"Ensemble  OOF AUC: {roc_auc_score(y, oof_ensemble):.6f}")

# Try alternative weights
for w1 in [0.30, 0.33, 0.35, 0.38, 0.40]:
    for w2 in [0.25, 0.28, 0.30, 0.33, 0.35]:
        w3 = 1.0 - w1 - w2
        if w3 <= 0:
            continue
        oof_alt = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
        auc_alt = roc_auc_score(y, oof_alt)
        if auc_alt > roc_auc_score(y, oof_ensemble) + 0.00001:
            oof_ensemble = oof_alt
            test_ensemble = w1 * test_lgb + w2 * test_xgb + w3 * test_cat
            w_lgb, w_xgb, w_cat = w1, w2, w3
            print(f"  Better weights: LGB={w1:.2f}, XGB={w2:.2f}, CAT={w3:.2f} → AUC={auc_alt:.6f}")

print(f"\nFinal Ensemble OOF AUC: {roc_auc_score(y, oof_ensemble):.6f}")
print(f"Final weights: LGB={w_lgb:.2f}, XGB={w_xgb:.2f}, CAT={w_cat:.2f}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("Generating Submission...")
print("=" * 60)

sub['PitNextLap'] = test_ensemble
sub.to_csv('submission_v1.csv', index=False)
print("Saved: submission_v1.csv")
print(f"Sample predictions:")
print(sub.head(10))
print(f"\nPrediction stats: mean={test_ensemble.mean():.4f}, std={test_ensemble.std():.4f}")
print(f"Min={test_ensemble.min():.4f}, Max={test_ensemble.max():.4f}")
print(f"Zero preds: {(test_ensemble == 0).sum()}, Near-zero: {(test_ensemble < 0.01).sum()}")
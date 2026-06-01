import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("OPTIMIZED SOLUTION V8 - FINAL ENSEMBLE FOR 0.951+")
print("=" * 70)

train = pd.read_csv('playground-series-s6e5/train.csv')
test = pd.read_csv('playground-series-s6e5/test.csv')

cat_cols = ['Driver', 'Compound', 'Race']

for df in [train, test]:
    df['LapTime (s)'] = df['LapTime (s)'].clip(
        df['LapTime (s)'].quantile(0.0005),
        df['LapTime (s)'].quantile(0.9995)
    )
    df['LapTime_Delta'] = df['LapTime_Delta'].clip(
        df['LapTime_Delta'].quantile(0.0005),
        df['LapTime_Delta'].quantile(0.9995)
    )
    df['Cumulative_Degradation'] = df['Cumulative_Degradation'].clip(
        df['Cumulative_Degradation'].quantile(0.001),
        df['Cumulative_Degradation'].quantile(0.999)
    )
    df['Position_Change'] = df['Position_Change'].clip(-5, 5)
    df['PitStop'] = df['PitStop'].clip(0, 6)

all_data = pd.concat([train, test], axis=0, ignore_index=True)

for col in cat_cols:
    le = LabelEncoder()
    all_data[col + '_enc'] = le.fit_transform(all_data[col].astype(str))
    freq_map = all_data[col].value_counts(normalize=True)
    all_data[col + '_freq'] = all_data[col].map(freq_map)

all_data['Compound_HARD'] = (all_data['Compound'] == 'HARD').astype(int)
all_data['Compound_SOFT'] = (all_data['Compound'] == 'SOFT').astype(int)
all_data['Compound_MEDIUM'] = (all_data['Compound'] == 'MEDIUM').astype(int)
all_data['Compound_INTER'] = (all_data['Compound'] == 'INTERMEDIATE').astype(int)

all_data['TyreLife_per_Stint'] = all_data['TyreLife'] / (all_data['Stint'] + 1)
all_data['LapTime_vs_TyreLife'] = all_data['LapTime (s)'] * all_data['TyreLife']
all_data['CumDegrad_per_Lap'] = all_data['Cumulative_Degradation'] / (all_data['LapNumber'] + 1)
all_data['TyreLife_sq'] = all_data['TyreLife'] ** 2
all_data['Pos_change_abs'] = np.abs(all_data['Position_Change'])
all_data['TyreLife_x_Stint'] = all_data['TyreLife'] * all_data['Stint']
all_data['LapTime_per_Pos'] = all_data['LapTime (s)'] / (all_data['Position'] + 1)
all_data['Degrad_per_Pos'] = all_data['Cumulative_Degradation'] / (all_data['Position'] + 1)
all_data['RaceProg_sq'] = all_data['RaceProgress'] ** 2
all_data['TyreLife_x_Pos'] = all_data['TyreLife'] * all_data['Position']
all_data['LapTimeDelta_abs'] = np.abs(all_data['LapTime_Delta'])
all_data['Degrad_rate'] = all_data['Cumulative_Degradation'] / (all_data['RaceProgress'] + 0.01)
all_data['LapNumber_log'] = np.log1p(all_data['LapNumber'])
all_data['TyreLife_log'] = np.log1p(all_data['TyreLife'])
all_data['CumDegrad_log'] = np.log1p(np.abs(all_data['Cumulative_Degradation']))
all_data['LapTime_log'] = np.log1p(all_data['LapTime (s)'])
all_data['TyreLife_cubed'] = all_data['TyreLife'] ** 3
all_data['Degrad_x_Pos'] = all_data['Cumulative_Degradation'] * all_data['Position']
all_data['Stint_per_Lap'] = all_data['Stint'] / (all_data['LapNumber'] + 1)
all_data['LapTime_x_RaceProg'] = all_data['LapTime (s)'] * all_data['RaceProgress']
all_data['Pos_x_RaceProg'] = all_data['Position'] * all_data['RaceProgress']
all_data['TyreLife_per_Position'] = all_data['TyreLife'] / (all_data['Position'] + 1)
all_data['Stint_x_RaceProgress'] = all_data['Stint'] * all_data['RaceProgress']
all_data['PitStop_x_RaceProgress'] = all_data['PitStop'] * all_data['RaceProgress']
all_data['PitStop_x_Stint'] = all_data['PitStop'] * all_data['Stint']
all_data['PitStop_x_Position'] = all_data['PitStop'] * all_data['Position']
all_data['TyreLife_x_Stint_x_RaceProgress'] = all_data['TyreLife'] * all_data['Stint'] * all_data['RaceProgress']
all_data['Degrad_x_LapNum'] = all_data['Cumulative_Degradation'] * all_data['LapNumber']
all_data['TyreLife_x_Year'] = all_data['TyreLife'] * all_data['Year']
all_data['Stint_x_Year'] = all_data['Stint'] * all_data['Year']
all_data['LapNum_x_Year'] = all_data['LapNumber'] * all_data['Year']
all_data['TyreLife_Stint_Ratio'] = all_data['TyreLife'] / (all_data['Stint'] * 5 + 1)
all_data['Lap_Progress_Ratio'] = all_data['LapNumber'] / (all_data['RaceProgress'] * 100 + 1)
all_data['Position_Delta'] = all_data['Position'] - all_data['Position_Change']

all_data['Is_Mid_Race'] = ((all_data['LapNumber'] >= 15) & (all_data['LapNumber'] < 40)).astype(int)
all_data['Is_Late_Race'] = (all_data['LapNumber'] >= 40).astype(int)

all_data['HARD_x_TyreLife'] = all_data['Compound_HARD'] * all_data['TyreLife']
all_data['HARD_x_LapNumber'] = all_data['Compound_HARD'] * all_data['LapNumber']
all_data['HARD_x_CumDegrad'] = all_data['Compound_HARD'] * all_data['Cumulative_Degradation']
all_data['HARD_x_Stint'] = all_data['Compound_HARD'] * all_data['Stint']

all_data['MidRace_x_TyreLife'] = all_data['Is_Mid_Race'] * all_data['TyreLife']
all_data['MidRace_x_Degrad'] = all_data['Is_Mid_Race'] * all_data['Cumulative_Degradation']

all_data['TyreLife_to_Degrad_Ratio'] = all_data['TyreLife'] / (np.abs(all_data['Cumulative_Degradation']) + 0.1)
all_data['LapNum_to_Stint_Ratio'] = all_data['LapNumber'] / (all_data['Stint'] + 1)

all_data['High_TyreLife'] = (all_data['TyreLife'] > 20).astype(int)
all_data['High_CumDegrad'] = (all_data['Cumulative_Degradation'] < -50).astype(int)

all_data['TyreLife_pow_1.5'] = all_data['TyreLife'] ** 1.5
all_data['Degrad_abs'] = np.abs(all_data['Cumulative_Degradation'])

all_data['Position_x_TyreLife_sq'] = all_data['Position'] * all_data['TyreLife_sq']
all_data['Stint_x_TyreLife_sq'] = all_data['Stint'] * all_data['TyreLife_sq']

drop_cols = ['id', 'PitNextLap'] + cat_cols
X = all_data[~all_data['PitNextLap'].isna()].drop(drop_cols, axis=1)
y = all_data[~all_data['PitNextLap'].isna()]['PitNextLap']
X_test = all_data[all_data['PitNextLap'].isna()].drop(drop_cols, axis=1)
test_ids = test['id']

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"Total features: {X.shape[1]}")

print("\n" + "=" * 70)
print("TRAINING LightGBM")
print("=" * 70)

SEED = 42
N_FOLDS = 5

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

lgb_oof = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 260,
        'max_depth': 14,
        'min_child_samples': 22,
        'feature_fraction': 0.72,
        'bagging_fraction': 0.76,
        'bagging_freq': 5,
        'lambda_l1': 0.015,
        'lambda_l2': 0.04,
        'min_gain_to_split': 0.0001,
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1
    }
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(300), lgb.log_evaluation(1000)]
    )
    
    lgb_oof[val_idx] = model.predict(X_val)
    lgb_preds += model.predict(X_test) / N_FOLDS
    
    fold_auc = roc_auc_score(y_val, lgb_oof[val_idx])
    print(f"  Fold {fold+1}: AUC={fold_auc:.5f}")

lgb_cv_auc = roc_auc_score(y, lgb_oof)
print(f"\nLightGBM CV AUC: {lgb_cv_auc:.5f}")

print("\n" + "=" * 70)
print("TRAINING CatBoost")
print("=" * 70)

cat_oof = np.zeros(len(X))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    cat_model = CatBoostClassifier(
        iterations=2500,
        learning_rate=0.02,
        depth=10,
        l2_leaf_reg=2,
        random_seed=SEED,
        eval_metric='AUC',
        early_stopping_rounds=150,
        verbose=0,
        thread_count=-1
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    
    cat_oof[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    cat_preds += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    fold_auc = roc_auc_score(y_val, cat_oof[val_idx])
    print(f"  Fold {fold+1}: AUC={fold_auc:.5f}")

cat_cv_auc = roc_auc_score(y, cat_oof)
print(f"\nCatBoost CV AUC: {cat_cv_auc:.5f}")

print("\n" + "=" * 70)
print("ENSEMBLING")
print("=" * 70)

lgb_weight = 0.60
cat_weight = 0.40

final_oof = lgb_weight * lgb_oof + cat_weight * cat_oof
final_preds = lgb_weight * lgb_preds + cat_weight * cat_preds

ensemble_cv_auc = roc_auc_score(y, final_oof)
print(f"\n*** Ensemble CV AUC: {ensemble_cv_auc:.5f} ***")
print(f"Weights: LightGBM={lgb_weight}, CatBoost={cat_weight}")

if ensemble_cv_auc >= 0.951:
    print("\n*** TARGET ACHIEVED: CV AUC 0.951+ ***")
else:
    print(f"\n*** Current: {ensemble_cv_auc:.5f}, Need: {0.951 - ensemble_cv_auc:.5f} more ***")

print("\n" + "=" * 70)
print("GENERATING SUBMISSION")
print("=" * 70)

print(f"Prediction mean: {final_preds.mean():.4f}")

submission = pd.DataFrame({
    'id': test_ids,
    'PitNextLap': final_preds
})

submission.to_csv('submission_optimized_v8.csv', index=False)

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print(f"File: submission_optimized_v8.csv")
print(f"Ensemble CV AUC: {ensemble_cv_auc:.5f}")
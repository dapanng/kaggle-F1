import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

SEED = 42
print("Loading data...")
train = pd.read_csv('playground-series-s6e5/train.csv')
test = pd.read_csv('playground-series-s6e5/test.csv')
train['PitNextLap'] = train['PitNextLap'].astype(int)

def engineer_features(df, train_df=None):
    df = df.copy()
    compound_order = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
    df['Compound_enc'] = df['Compound'].map(compound_order)
    gm = 0.1990
    if train_df is not None:
        dm = train_df.groupby('Driver')['PitNextLap'].mean()
        rm = train_df.groupby('Race')['PitNextLap'].mean()
        cm = train_df.groupby('Compound')['PitNextLap'].mean()
    else:
        dm = df.groupby('Driver')['PitNextLap'].mean()
        rm = df.groupby('Race')['PitNextLap'].mean()
        cm = df.groupby('Compound')['PitNextLap'].mean()
    df['Driver_enc'] = df['Driver'].map(dm).fillna(gm)
    df['Race_enc'] = df['Race'].map(rm).fillna(gm)
    df['Compound_tgt'] = df['Compound'].map(cm)
    df['Compound_x_TyreLife'] = df['Compound_enc'] * df['TyreLife']
    df['Compound_x_RaceProgress'] = df['Compound_enc'] * df['RaceProgress']
    df['Compound_x_LapNumber'] = df['Compound_enc'] * df['LapNumber']
    df['TyreLife_safe'] = df['TyreLife'].clip(lower=1)
    df['Deg_per_lap'] = df['Cumulative_Degradation'] / df['TyreLife_safe']
    df['LTD_per_lap'] = df['LapTime_Delta'] / df['TyreLife_safe']
    df['TyreLife_gt_15'] = (df['TyreLife'] > 15).astype(int)
    df['TyreLife_gt_25'] = (df['TyreLife'] > 25).astype(int)
    df['TyreLife_gt_40'] = (df['TyreLife'] > 40).astype(int)
    df['TyreLife_sq'] = df['TyreLife'] ** 2
    df['RaceProgress_sq'] = df['RaceProgress'] ** 2
    df['LapNumber_sq'] = df['LapNumber'] ** 2
    df['Is_Lap1'] = (df['LapNumber'] == 1).astype(int)
    df['Is_Stint1'] = (df['Stint'] == 1).astype(int)
    df['Is_Top3'] = (df['Position'] <= 3).astype(int)
    df['Is_Top5'] = (df['Position'] <= 5).astype(int)
    df['Is_Back'] = (df['Position'] >= 15).astype(int)
    df['Year_2023'] = (df['Year'] == 2023).astype(int)
    df['Year_2024'] = (df['Year'] == 2024).astype(int)
    df['Year_2025'] = (df['Year'] == 2025).astype(int)
    df['Stint_x_TyreLife'] = df['Stint'] * df['TyreLife']
    df['PChange_abs'] = np.abs(df['Position_Change'])
    df['Has_Pitted'] = df['PitStop']
    df['D_R_risk'] = df['Driver_enc'] * df['Race_enc']
    df['TyreLife_div_Stint'] = df['TyreLife'] / df['Stint'].clip(lower=1)
    df['LapNumber_div_Stint'] = df['LapNumber'] / df['Stint'].clip(lower=1)
    df['Tire_Wear'] = -df['Cumulative_Degradation']
    return df

train_fe = engineer_features(train)
test_fe = engineer_features(test, train)

feat = [
    'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'Position_Change',
    'Compound_enc', 'Driver_enc', 'Race_enc', 'Compound_tgt',
    'Compound_x_TyreLife', 'Compound_x_RaceProgress', 'Compound_x_LapNumber',
    'Deg_per_lap', 'LTD_per_lap',
    'TyreLife_gt_15', 'TyreLife_gt_25', 'TyreLife_gt_40', 'TyreLife_sq',
    'RaceProgress_sq', 'LapNumber_sq',
    'Is_Lap1', 'Is_Stint1', 'Is_Top3', 'Is_Top5', 'Is_Back',
    'Year_2023', 'Year_2024', 'Year_2025',
    'Stint_x_TyreLife', 'PChange_abs', 'Has_Pitted',
    'D_R_risk', 'TyreLife_div_Stint', 'LapNumber_div_Stint', 'Tire_Wear'
]

train_fe = train_fe.fillna(0); test_fe = test_fe.fillna(0)
X = train_fe[feat].values.astype(np.float32)
y = train_fe['PitNextLap'].values.astype(np.int32)
X_test = test_fe[feat].values.astype(np.float32)
print(f"Train: {X.shape}, Test: {X_test.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cat_params = {
    'iterations': 2000, 'learning_rate': 0.03, 'depth': 6,
    'l2_leaf_reg': 3, 'border_count': 128, 'random_strength': 1,
    'bagging_temperature': 0.5, 'od_type': 'Iter', 'od_wait': 100,
    'verbose': 0, 'random_seed': SEED, 'task_type': 'CPU', 'thread_count': 2,
    'use_best_model': True,
}

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

for fold, (tr, val) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[tr], X[val]
    y_tr, y_val = y[tr], y[val]
    model = CatBoostClassifier(**cat_params)
    model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val), verbose=False)
    oof[val] = model.predict_proba(X_val)[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / 5
    print(f"Fold {fold+1}: AUC={roc_auc_score(y_val, oof[val]):.6f}, Trees={model.tree_count_}")

print(f"\nCatBoost OOF AUC: {roc_auc_score(y, oof):.6f}")
np.save('oof_cat_v1.npy', oof)
np.save('test_cat_v1.npy', test_pred)
print("Saved oof_cat_v1.npy, test_cat_v1.npy")
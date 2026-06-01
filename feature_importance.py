"""Quick feature importance analysis to identify top features"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')

train = pd.read_csv('playground-series-s6e5/train.csv')
train['PitNextLap'] = train['PitNextLap'].astype(int)

compound_order = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
gm = 0.1990

df = train.copy()
df['Compound_ord'] = df['Compound'].map(compound_order)
co = df['Compound_ord']

# All V6 features
df['CxTyreLife'] = co * df['TyreLife']
df['CxLapNumber'] = co * df['LapNumber']
df['CxRaceProgress'] = co * df['RaceProgress']
df['CxDegradation'] = co * df['Cumulative_Degradation']
df['CxPosition'] = co * df['Position']

tls = df['TyreLife'].clip(lower=1)
df['TL_sq'] = df['TyreLife'] ** 2
df['TL_sqrt'] = np.sqrt(df['TyreLife'])
df['TL_log'] = np.log1p(df['TyreLife'])
df['TL_cu'] = df['TyreLife'] ** 3
for t in [5,10,15,20,25,30,35,40,50]:
    df[f'TL_gt_{t}'] = (df['TyreLife'] > t).astype(np.int8)

df['Deg_per_lap'] = df['Cumulative_Degradation'] / tls
df['LTD_per_lap'] = df['LapTime_Delta'] / tls
df['LT_per_lap'] = df['LapTime (s)'] / tls
df['Deg_abs'] = np.abs(df['Cumulative_Degradation'])
df['Deg_sq'] = df['Cumulative_Degradation'] ** 2

df['Is_P1'] = (df['Position'] == 1).astype(np.int8)
df['Is_Podium'] = (df['Position'] <= 3).astype(np.int8)
df['Is_Top5'] = (df['Position'] <= 5).astype(np.int8)
df['Is_Top10'] = (df['Position'] <= 10).astype(np.int8)
df['Is_Back5'] = (df['Position'] >= 16).astype(np.int8)
df['Pos_sq'] = df['Position'] ** 2

df['RP_sq'] = df['RaceProgress'] ** 2
df['RP_cu'] = df['RaceProgress'] ** 3
df['LN_sq'] = df['LapNumber'] ** 2
df['LN_sqrt'] = np.sqrt(df['LapNumber'])
df['Is_Lap1'] = (df['LapNumber'] == 1).astype(np.int8)
df['Is_Early'] = (df['RaceProgress'] < 0.1).astype(np.int8)
df['Is_Late'] = (df['RaceProgress'] > 0.9).astype(np.int8)
df['Is_Mid'] = ((df['RaceProgress'] > 0.3) & (df['RaceProgress'] < 0.7)).astype(np.int8)

df['Is_Stint1'] = (df['Stint'] == 1).astype(np.int8)
df['Is_Stint2'] = (df['Stint'] == 2).astype(np.int8)
df['Is_Stint3plus'] = (df['Stint'] >= 3).astype(np.int8)
df['Stint_x_TL'] = df['Stint'] * df['TyreLife']
df['Stint_x_LN'] = df['Stint'] * df['LapNumber']
df['TL_div_Stint'] = df['TyreLife'] / df['Stint'].clip(lower=1)
df['LN_div_Stint'] = df['LapNumber'] / df['Stint'].clip(lower=1)

for yr in [2022, 2023, 2024, 2025]:
    df[f'Year_{yr}'] = (df['Year'] == yr).astype(np.int8)

df['PChange_abs'] = np.abs(df['Position_Change'])
df['Has_Pitted'] = df['PitStop'].astype(np.int8)
df['Tire_Wear'] = -df['Cumulative_Degradation']
df['LTD_abs'] = np.abs(df['LapTime_Delta'])
df['LT_sq'] = df['LapTime (s)'] ** 2

df['Driver_le'] = LabelEncoder().fit_transform(df['Driver'])
df['Race_le'] = LabelEncoder().fit_transform(df['Race'])

df['PitStop_x_TL'] = df['PitStop'] * df['TyreLife']
df['LT_x_Compound'] = df['LapTime (s)'] * co
df['RP_x_TL'] = df['RaceProgress'] * df['TyreLife']
df['Pos_x_TL'] = df['Position'] * df['TyreLife']
df['LN_x_TL'] = df['LapNumber'] * df['TyreLife']
df['Deg_x_RP'] = df['Cumulative_Degradation'] * df['RaceProgress']
rp_safe = df['RaceProgress'].clip(lower=0.01)
df['TL_div_RP'] = df['TyreLife'] / rp_safe
df['Stint_x_RP'] = df['Stint'] * df['RaceProgress']

# Group stats
drv_avg_tyre = train.groupby('Driver')['TyreLife'].mean()
drv_avg_pos = train.groupby('Driver')['Position'].mean()
drv_avg_laptime = train.groupby('Driver')['LapTime (s)'].mean()
race_avg_degrad = train.groupby('Race')['Cumulative_Degradation'].mean()
race_avg_laptime = train.groupby('Race')['LapTime (s)'].mean()
df['Drv_avg_tyre'] = df['Driver'].map(drv_avg_tyre)
df['Drv_avg_pos'] = df['Driver'].map(drv_avg_pos)
df['Drv_avg_laptime'] = df['Driver'].map(drv_avg_laptime)
df['Race_avg_degrad'] = df['Race'].map(race_avg_degrad)
df['Race_avg_laptime'] = df['Race'].map(race_avg_laptime)
df['Dev_drv_tyre'] = df['TyreLife'] - df['Drv_avg_tyre']
df['Dev_drv_pos'] = df['Position'] - df['Drv_avg_pos']
df['Dev_race_degrad'] = df['Cumulative_Degradation'] - df['Race_avg_degrad']
df['Dev_race_laptime'] = df['LapTime (s)'] - df['Race_avg_laptime']

# Target encoding (KFold)
y_all = df['PitNextLap'].values
te_groups = [
    ('Driver', 'Driver_te'),
    ('Race', 'Race_te'),
    ('Compound', 'Compound_te'),
    ('Stint', 'Stint_te'),
    ('Year', 'Year_te'),
    (['Compound', 'Race'], 'Compound_Race_te'),
    (['Driver', 'Race'], 'Driver_Race_te'),
    (['Compound', 'Stint'], 'Compound_Stint_te'),
]

skf_enc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for c_name in [te[1] for te in te_groups]:
    df[c_name] = gm

for tr_idx, val_idx in skf_enc.split(df, y_all):
    tr_fold = df.iloc[tr_idx]
    vi = df.index[val_idx]
    for grp, te_name in te_groups:
        if isinstance(grp, str):
            grp_mean = tr_fold.groupby(grp)['PitNextLap'].agg(['mean', 'count'])
            grp_mean['smooth'] = (grp_mean['mean'] * grp_mean['count'] + gm * 100) / (grp_mean['count'] + 100)
            df.loc[vi, te_name] = df.loc[vi, grp].map(grp_mean['smooth']).fillna(gm)
        else:
            grp_mean = tr_fold.groupby(list(grp))['PitNextLap'].agg(['mean', 'count'])
            grp_mean['smooth'] = (grp_mean['mean'] * grp_mean['count'] + gm * 50) / (grp_mean['count'] + 50)
            idx_vals = df.loc[vi].set_index(list(grp)).index
            df.loc[vi, te_name] = idx_vals.map(grp_mean['smooth']).fillna(gm).values

df['Risk_CD'] = df['Compound_te'] * df['Driver_te']
df['Risk_DR'] = df['Driver_te'] * df['Race_te']
df['Risk_CDR'] = df['Compound_te'] * df['Driver_te'] * df['Race_te']

FEATURES = [
    'Year','PitStop','LapNumber','Stint','TyreLife','Position',
    'LapTime (s)','LapTime_Delta','Cumulative_Degradation','RaceProgress','Position_Change',
    'Compound_ord','Driver_le','Race_le',
    'Driver_te','Race_te','Compound_te','Stint_te','Year_te',
    'Compound_Race_te','Driver_Race_te','Compound_Stint_te',
    'Risk_CD','Risk_DR','Risk_CDR',
    'CxTyreLife','CxLapNumber','CxRaceProgress','CxDegradation','CxPosition',
    'TL_sq','TL_sqrt','TL_log','TL_cu',
    'TL_gt_5','TL_gt_10','TL_gt_15','TL_gt_20','TL_gt_25','TL_gt_30','TL_gt_35','TL_gt_40','TL_gt_50',
    'Deg_per_lap','LTD_per_lap','LT_per_lap','Deg_abs','Deg_sq',
    'Is_P1','Is_Podium','Is_Top5','Is_Top10','Is_Back5','Pos_sq',
    'RP_sq','RP_cu','LN_sq','LN_sqrt','Is_Lap1','Is_Early','Is_Late','Is_Mid',
    'Is_Stint1','Is_Stint2','Is_Stint3plus','Stint_x_TL','Stint_x_LN','TL_div_Stint','LN_div_Stint',
    'Year_2022','Year_2023','Year_2024','Year_2025',
    'PChange_abs','Has_Pitted','Tire_Wear','LTD_abs','LT_sq',
    'Drv_avg_tyre','Drv_avg_pos','Drv_avg_laptime','Race_avg_degrad','Race_avg_laptime',
    'Dev_drv_tyre','Dev_drv_pos','Dev_race_degrad','Dev_race_laptime',
    'PitStop_x_TL','LT_x_Compound','RP_x_TL','Pos_x_TL','LN_x_TL','Deg_x_RP','TL_div_RP','Stint_x_RP',
]

df = df.fillna(0).replace([np.inf, -np.inf], 0)
X = df[FEATURES].values.astype(np.float32)
y = df['PitNextLap'].values.astype(np.int32)

print(f"Features: {len(FEATURES)}, Shape: {X.shape}")

# Train single LGB model to get feature importance
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
importances = np.zeros(len(FEATURES))
oof = np.zeros(len(X))

for fold, (tr, val) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(
        objective='binary', metric='auc', boosting_type='gbdt',
        n_estimators=3000, learning_rate=0.02, num_leaves=255,
        max_depth=9, min_data_in_leaf=30, feature_fraction=0.5,
        bagging_fraction=0.8, bagging_freq=5, lambda_l1=0.5,
        lambda_l2=0.5, verbose=-1, random_state=42, n_jobs=-1,
    )
    m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], eval_metric='auc',
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    oof[val] = m.predict_proba(X[val])[:, 1]
    importances += m.feature_importances_ / 5
    print(f"Fold {fold+1}: {roc_auc_score(y[val], oof[val]):.6f}")

print(f"\nOverall AUC: {roc_auc_score(y, oof):.6f}")

# Sort features by importance
fi = pd.DataFrame({'feature': FEATURES, 'importance': importances})
fi = fi.sort_values('importance', ascending=False)
print("\nTop 30 features:")
print(fi.head(30).to_string(index=False))
print("\nBottom 20 features:")
print(fi.tail(20).to_string(index=False))
print(f"\nFeatures with 0 importance: {(fi['importance'] == 0).sum()}")

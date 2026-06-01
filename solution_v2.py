import sys, time, warnings, gc, json
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier, Pool
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

SEEDS = [42, 123, 2024]
NF = 5
t0 = time.time()

log("=" * 70)
log("V8: Focused Optimization - Target LB > 0.953")
log("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
log("\n[1/6] Loading data...")
train = pd.read_csv('playground-series-s6e5/train.csv')
test = pd.read_csv('playground-series-s6e5/test.csv')
train['PitNextLap'] = train['PitNextLap'].astype(int)
log(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# 2. FEATURE ENGINEERING (V8 - Refined)
# ============================================================
log("\n[2/6] Feature engineering...")

compound_order = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
gm = 0.1990

def add_features(df):
    df = df.copy()
    df['Compound_ord'] = df['Compound'].map(compound_order)
    co = df['Compound_ord']

    # Core interactions (high importance)
    df['CxTyreLife'] = co * df['TyreLife']
    df['CxLapNumber'] = co * df['LapNumber']
    df['CxRaceProgress'] = co * df['RaceProgress']
    df['CxDegradation'] = co * df['Cumulative_Degradation']
    df['CxPosition'] = co * df['Position']

    # TyreLife transforms (keep top ones)
    tls = df['TyreLife'].clip(lower=1)
    df['TL_sq'] = df['TyreLife'] ** 2
    df['TL_sqrt'] = np.sqrt(df['TyreLife'])
    df['TL_log'] = np.log1p(df['TyreLife'])
    df['TL_cu'] = df['TyreLife'] ** 3

    # Only keep the most important TL thresholds (from importance analysis)
    df['TL_gt_15'] = (df['TyreLife'] > 15).astype(np.int8)
    df['TL_gt_25'] = (df['TyreLife'] > 25).astype(np.int8)
    df['TL_gt_35'] = (df['TyreLife'] > 35).astype(np.int8)
    df['TL_gt_50'] = (df['TyreLife'] > 50).astype(np.int8)

    # Degradation features (high importance)
    df['Deg_per_lap'] = df['Cumulative_Degradation'] / tls
    df['LTD_per_lap'] = df['LapTime_Delta'] / tls
    df['LT_per_lap'] = df['LapTime (s)'] / tls
    df['Deg_abs'] = np.abs(df['Cumulative_Degradation'])
    df['Deg_sq'] = df['Cumulative_Degradation'] ** 2

    # Position features (only high-importance ones)
    df['Pos_sq'] = df['Position'] ** 2
    df['Is_Top10'] = (df['Position'] <= 10).astype(np.int8)

    # Race/Lap features
    df['RP_sq'] = df['RaceProgress'] ** 2
    df['RP_cu'] = df['RaceProgress'] ** 3
    df['LN_sq'] = df['LapNumber'] ** 2
    df['LN_sqrt'] = np.sqrt(df['LapNumber'])

    # Stint features
    df['Stint_x_TL'] = df['Stint'] * df['TyreLife']
    df['Stint_x_LN'] = df['Stint'] * df['LapNumber']
    df['TL_div_Stint'] = df['TyreLife'] / df['Stint'].clip(lower=1)
    df['LN_div_Stint'] = df['LapNumber'] / df['Stint'].clip(lower=1)

    # Year flags
    for yr in [2022, 2023, 2024, 2025]:
        df[f'Year_{yr}'] = (df['Year'] == yr).astype(np.int8)

    # Other (high importance)
    df['PChange_abs'] = np.abs(df['Position_Change'])
    df['Has_Pitted'] = df['PitStop'].astype(np.int8)
    df['Tire_Wear'] = -df['Cumulative_Degradation']
    df['LTD_abs'] = np.abs(df['LapTime_Delta'])
    df['LT_sq'] = df['LapTime (s)'] ** 2

    # Label encodings
    df['Driver_le'] = LabelEncoder().fit_transform(df['Driver'])
    df['Race_le'] = LabelEncoder().fit_transform(df['Race'])

    # Key interaction features (high importance)
    df['PitStop_x_TL'] = df['PitStop'] * df['TyreLife']
    df['LT_x_Compound'] = df['LapTime (s)'] * co
    df['RP_x_TL'] = df['RaceProgress'] * df['TyreLife']
    df['Pos_x_TL'] = df['Position'] * df['TyreLife']
    df['LN_x_TL'] = df['LapNumber'] * df['TyreLife']
    df['Deg_x_RP'] = df['Cumulative_Degradation'] * df['RaceProgress']
    rp_safe = df['RaceProgress'].clip(lower=0.01)
    df['TL_div_RP'] = df['TyreLife'] / rp_safe
    df['Stint_x_RP'] = df['Stint'] * df['RaceProgress']

    # NEW V8: Compound-specific deviation features
    # How far is this TyreLife from the compound average?
    compound_avg_tl = {'SOFT': 12.0, 'MEDIUM': 18.0, 'HARD': 25.0, 'INTERMEDIATE': 10.0, 'WET': 8.0}
    df['Compound_avg_TL'] = df['Compound'].map(compound_avg_tl)
    df['Dev_compound_TL'] = df['TyreLife'] - df['Compound_avg_TL']
    df['TL_ratio_compound'] = df['TyreLife'] / df['Compound_avg_TL'].clip(lower=1)

    # NEW V8: Degradation × Compound interaction
    df['Deg_x_Compound'] = df['Cumulative_Degradation'] * co
    df['LTD_x_Compound'] = df['LapTime_Delta'] * co

    # NEW V8: RaceProgress × Stint (how far in race vs stint number)
    df['RP_div_Stint'] = df['RaceProgress'] / df['Stint'].clip(lower=1)

    # NEW V8: TyreLife × Degradation (tire age vs degradation level)
    df['TL_x_Deg'] = df['TyreLife'] * df['Cumulative_Degradation']
    df['TL_x_LTD'] = df['TyreLife'] * df['LapTime_Delta']

    # NEW V8: Position × RaceProgress (leading early vs leading late)
    df['Pos_x_RP'] = df['Position'] * df['RaceProgress']

    # NEW V8: PitStop × RaceProgress (pitting in early vs late race)
    df['PitStop_x_RP'] = df['PitStop'] * df['RaceProgress']

    # NEW V8: Stint × Compound (stint 2 on soft vs stint 2 on hard)
    df['Stint_x_Compound'] = df['Stint'] * co

    # NEW V8: LapNumber × RaceProgress (should be highly correlated, difference = anomaly)
    df['LN_minus_RP'] = df['LapNumber'] - df['RaceProgress'] * df['LapNumber'].max()

    return df.fillna(0).replace([np.inf, -np.inf], 0)

train_fe = add_features(train)
test_fe = add_features(test)

# Non-target group stats (safe - no target info)
drv_avg_tyre = train.groupby('Driver')['TyreLife'].mean()
drv_avg_pos = train.groupby('Driver')['Position'].mean()
drv_avg_laptime = train.groupby('Driver')['LapTime (s)'].mean()
drv_avg_degrad = train.groupby('Driver')['Cumulative_Degradation'].mean()
drv_avg_ltd = train.groupby('Driver')['LapTime_Delta'].mean()
race_avg_degrad = train.groupby('Race')['Cumulative_Degradation'].mean()
race_avg_laptime = train.groupby('Race')['LapTime (s)'].mean()
race_avg_tyre = train.groupby('Race')['TyreLife'].mean()
race_avg_pos = train.groupby('Race')['Position'].mean()
race_avg_ltd = train.groupby('Race')['LapTime_Delta'].mean()
compound_avg_degrad = train.groupby('Compound')['Cumulative_Degradation'].mean()
compound_avg_laptime = train.groupby('Compound')['LapTime (s)'].mean()

for df in [train_fe, test_fe]:
    df['Drv_avg_tyre'] = df['Driver'].map(drv_avg_tyre)
    df['Drv_avg_pos'] = df['Driver'].map(drv_avg_pos)
    df['Drv_avg_laptime'] = df['Driver'].map(drv_avg_laptime)
    df['Drv_avg_degrad'] = df['Driver'].map(drv_avg_degrad)
    df['Drv_avg_ltd'] = df['Driver'].map(drv_avg_ltd)
    df['Race_avg_degrad'] = df['Race'].map(race_avg_degrad)
    df['Race_avg_laptime'] = df['Race'].map(race_avg_laptime)
    df['Race_avg_tyre'] = df['Race'].map(race_avg_tyre)
    df['Race_avg_pos'] = df['Race'].map(race_avg_pos)
    df['Race_avg_ltd'] = df['Race'].map(race_avg_ltd)
    df['Compound_avg_degrad'] = df['Compound'].map(compound_avg_degrad)
    df['Compound_avg_laptime'] = df['Compound'].map(compound_avg_laptime)
    df['Dev_drv_tyre'] = df['TyreLife'] - df['Drv_avg_tyre']
    df['Dev_drv_pos'] = df['Position'] - df['Drv_avg_pos']
    df['Dev_drv_degrad'] = df['Cumulative_Degradation'] - df['Drv_avg_degrad']
    df['Dev_drv_ltd'] = df['LapTime_Delta'] - df['Drv_avg_ltd']
    df['Dev_race_degrad'] = df['Cumulative_Degradation'] - df['Race_avg_degrad']
    df['Dev_race_laptime'] = df['LapTime (s)'] - df['Race_avg_laptime']
    df['Dev_race_tyre'] = df['TyreLife'] - df['Race_avg_tyre']
    df['Dev_race_pos'] = df['Position'] - df['Race_avg_pos']
    df['Dev_race_ltd'] = df['LapTime_Delta'] - df['Race_avg_ltd']
    df['Dev_compound_degrad'] = df['Cumulative_Degradation'] - df['Compound_avg_degrad']
    df['Dev_compound_laptime'] = df['LapTime (s)'] - df['Compound_avg_laptime']

# ============================================================
# 3. KFOLD TARGET ENCODING (expanded)
# ============================================================
log("\n[3/6] KFold target encoding...")

y_all = train_fe['PitNextLap'].values
te_groups = [
    ('Driver', 'Driver_te'),
    ('Race', 'Race_te'),
    ('Compound', 'Compound_te'),
    ('Stint', 'Stint_te'),
    ('Year', 'Year_te'),
    (['Compound', 'Race'], 'Compound_Race_te'),
    (['Driver', 'Race'], 'Driver_Race_te'),
    (['Compound', 'Stint'], 'Compound_Stint_te'),
    (['Driver', 'Compound'], 'Driver_Compound_te'),
    (['Driver', 'Stint'], 'Driver_Stint_te'),
    (['Race', 'Stint'], 'Race_Stint_te'),
    (['Compound', 'Race', 'Stint'], 'CRStint_te'),
]

skf_enc = StratifiedKFold(n_splits=NF, shuffle=True, random_state=42)
for c_name in [te[1] for te in te_groups]:
    train_fe[c_name] = gm

for tr_idx, val_idx in skf_enc.split(train_fe, y_all):
    tr_fold = train_fe.iloc[tr_idx]
    vi = train_fe.index[val_idx]
    for grp, te_name in te_groups:
        if isinstance(grp, str):
            grp_mean = tr_fold.groupby(grp)['PitNextLap'].agg(['mean', 'count'])
            grp_mean['smooth'] = (grp_mean['mean'] * grp_mean['count'] + gm * 100) / (grp_mean['count'] + 100)
            train_fe.loc[vi, te_name] = train_fe.loc[vi, grp].map(grp_mean['smooth']).fillna(gm)
        else:
            grp_mean = tr_fold.groupby(list(grp))['PitNextLap'].agg(['mean', 'count'])
            grp_mean['smooth'] = (grp_mean['mean'] * grp_mean['count'] + gm * 30) / (grp_mean['count'] + 30)
            idx_vals = train_fe.loc[vi].set_index(list(grp)).index
            train_fe.loc[vi, te_name] = idx_vals.map(grp_mean['smooth']).fillna(gm).values

for grp, te_name in te_groups:
    if isinstance(grp, str):
        full_mean = train.groupby(grp)['PitNextLap'].mean()
        test_fe[te_name] = test_fe[grp].map(full_mean).fillna(gm)
    else:
        full_mean = train.groupby(list(grp))['PitNextLap'].mean()
        test_fe[te_name] = test_fe.set_index(list(grp)).index.map(lambda x: full_mean.get(x, gm))

# Risk multipliers (expanded)
train_fe['Risk_CD'] = train_fe['Compound_te'] * train_fe['Driver_te']
train_fe['Risk_DR'] = train_fe['Driver_te'] * train_fe['Race_te']
train_fe['Risk_CR'] = train_fe['Compound_te'] * train_fe['Race_te']
train_fe['Risk_CDR'] = train_fe['Compound_te'] * train_fe['Driver_te'] * train_fe['Race_te']
train_fe['Risk_CS'] = train_fe['Compound_te'] * train_fe['Stint_te']
train_fe['Risk_DS'] = train_fe['Driver_te'] * train_fe['Stint_te']
train_fe['Risk_RS'] = train_fe['Race_te'] * train_fe['Stint_te']
test_fe['Risk_CD'] = test_fe['Compound_te'] * test_fe['Driver_te']
test_fe['Risk_DR'] = test_fe['Driver_te'] * test_fe['Race_te']
test_fe['Risk_CR'] = test_fe['Compound_te'] * test_fe['Race_te']
test_fe['Risk_CDR'] = test_fe['Compound_te'] * test_fe['Driver_te'] * test_fe['Race_te']
test_fe['Risk_CS'] = test_fe['Compound_te'] * test_fe['Stint_te']
test_fe['Risk_DS'] = test_fe['Driver_te'] * test_fe['Stint_te']
test_fe['Risk_RS'] = test_fe['Race_te'] * test_fe['Stint_te']

# ============================================================
# 4. FEATURE LIST (refined - removed low importance, added high value)
# ============================================================
FEATURES = [
    # Original features (high importance)
    'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'Position_Change',
    'Compound_ord', 'Driver_le', 'Race_le',
    # Target encodings (very high importance)
    'Driver_te', 'Race_te', 'Compound_te', 'Stint_te', 'Year_te',
    'Compound_Race_te', 'Driver_Race_te', 'Compound_Stint_te',
    'Driver_Compound_te', 'Driver_Stint_te', 'Race_Stint_te', 'CRStint_te',
    # Risk multipliers
    'Risk_CD', 'Risk_DR', 'Risk_CR', 'Risk_CDR', 'Risk_CS', 'Risk_DS', 'Risk_RS',
    # Compound interactions
    'CxTyreLife', 'CxLapNumber', 'CxRaceProgress', 'CxDegradation', 'CxPosition',
    # TyreLife transforms
    'TL_sq', 'TL_sqrt', 'TL_log', 'TL_cu',
    'TL_gt_15', 'TL_gt_25', 'TL_gt_35', 'TL_gt_50',
    # Degradation features
    'Deg_per_lap', 'LTD_per_lap', 'LT_per_lap', 'Deg_abs', 'Deg_sq',
    # Position
    'Pos_sq', 'Is_Top10',
    # Race/Lap
    'RP_sq', 'RP_cu', 'LN_sq', 'LN_sqrt',
    # Stint
    'Stint_x_TL', 'Stint_x_LN', 'TL_div_Stint', 'LN_div_Stint',
    # Year
    'Year_2022', 'Year_2023', 'Year_2024', 'Year_2025',
    # Other
    'PChange_abs', 'Has_Pitted', 'Tire_Wear', 'LTD_abs', 'LT_sq',
    # Group stats + deviations (high importance)
    'Drv_avg_tyre', 'Drv_avg_pos', 'Drv_avg_laptime', 'Drv_avg_degrad', 'Drv_avg_ltd',
    'Race_avg_degrad', 'Race_avg_laptime', 'Race_avg_tyre', 'Race_avg_pos', 'Race_avg_ltd',
    'Compound_avg_degrad', 'Compound_avg_laptime',
    'Dev_drv_tyre', 'Dev_drv_pos', 'Dev_drv_degrad', 'Dev_drv_ltd',
    'Dev_race_degrad', 'Dev_race_laptime', 'Dev_race_tyre', 'Dev_race_pos', 'Dev_race_ltd',
    'Dev_compound_degrad', 'Dev_compound_laptime',
    # Key interactions (high importance)
    'PitStop_x_TL', 'LT_x_Compound', 'RP_x_TL', 'Pos_x_TL', 'LN_x_TL', 'Deg_x_RP', 'TL_div_RP', 'Stint_x_RP',
    # V8 new features
    'Compound_avg_TL', 'Dev_compound_TL', 'TL_ratio_compound',
    'Deg_x_Compound', 'LTD_x_Compound',
    'RP_div_Stint',
    'TL_x_Deg', 'TL_x_LTD',
    'Pos_x_RP', 'PitStop_x_RP',
    'Stint_x_Compound',
]

train_fe = train_fe.fillna(0).replace([np.inf, -np.inf], 0)
test_fe = test_fe.fillna(0).replace([np.inf, -np.inf], 0)

missing = [f for f in FEATURES if f not in train_fe.columns]
if missing:
    log(f"WARNING: Missing features: {missing}")
    FEATURES = [f for f in FEATURES if f in train_fe.columns and f in test_fe.columns]

X = train_fe[FEATURES].values.astype(np.float32)
y = train_fe['PitNextLap'].values.astype(np.int32)
X_test = test_fe[FEATURES].values.astype(np.float32)
log(f"Features: {len(FEATURES)} | Train: {X.shape} | Test: {X_test.shape}")

# ============================================================
# 5. MULTI-CONFIG TRAINING
# ============================================================
log("\n[4/6] Multi-config training...")

all_oof = {}
all_test = {}

# --- Config 1: LightGBM GBDT (deep, low lr) ---
cfg_name = 'lgb_deep'
log(f"\n  Training {cfg_name}...")
oof_acc = np.zeros(len(X))
test_acc = np.zeros(len(X_test))
for si, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof_seed = np.zeros(len(X))
    test_seed = np.zeros(len(X_test))
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(
            objective='binary', metric='auc', boosting_type='gbdt',
            n_estimators=5000, learning_rate=0.015, num_leaves=255,
            max_depth=10, min_data_in_leaf=30, feature_fraction=0.4,
            bagging_fraction=0.75, bagging_freq=5, lambda_l1=1.0,
            lambda_l2=2.0, min_gain_to_split=0.01,
            verbose=-1, random_state=SEED, n_jobs=-1,
        )
        m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], eval_metric='auc',
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof_seed[val] = m.predict_proba(X[val])[:, 1]
        test_seed += m.predict_proba(X_test)[:, 1] / NF
        log(f"    {cfg_name} seed={SEED} fold={fold+1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
    auc_seed = roc_auc_score(y, oof_seed)
    log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
    oof_acc += oof_seed / len(SEEDS)
    test_acc += test_seed / len(SEEDS)
all_oof[cfg_name] = oof_acc
all_test[cfg_name] = test_acc
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_acc):.6f}")
gc.collect()

# --- Config 2: LightGBM GBDT (medium depth, more regularization) ---
cfg_name = 'lgb_reg'
log(f"\n  Training {cfg_name}...")
oof_acc = np.zeros(len(X))
test_acc = np.zeros(len(X_test))
for si, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof_seed = np.zeros(len(X))
    test_seed = np.zeros(len(X_test))
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(
            objective='binary', metric='auc', boosting_type='gbdt',
            n_estimators=5000, learning_rate=0.02, num_leaves=127,
            max_depth=7, min_data_in_leaf=50, feature_fraction=0.5,
            bagging_fraction=0.8, bagging_freq=5, lambda_l1=2.0,
            lambda_l2=3.0, min_gain_to_split=0.02,
            verbose=-1, random_state=SEED, n_jobs=-1,
        )
        m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], eval_metric='auc',
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof_seed[val] = m.predict_proba(X[val])[:, 1]
        test_seed += m.predict_proba(X_test)[:, 1] / NF
        log(f"    {cfg_name} seed={SEED} fold={fold+1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
    auc_seed = roc_auc_score(y, oof_seed)
    log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
    oof_acc += oof_seed / len(SEEDS)
    test_acc += test_seed / len(SEEDS)
all_oof[cfg_name] = oof_acc
all_test[cfg_name] = test_acc
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_acc):.6f}")
gc.collect()

# --- Config 3: XGBoost (deep) ---
cfg_name = 'xgb_deep'
log(f"\n  Training {cfg_name}...")
oof_acc = np.zeros(len(X))
test_acc = np.zeros(len(X_test))
for si, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof_seed = np.zeros(len(X))
    test_seed = np.zeros(len(X_test))
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            n_estimators=5000, learning_rate=0.015, max_depth=8,
            min_child_weight=20, subsample=0.75, colsample_bytree=0.4,
            colsample_bylevel=0.4, reg_alpha=1.0, reg_lambda=2.0,
            gamma=0.1, tree_method='hist', random_state=SEED,
            n_jobs=-1, early_stopping_rounds=200, verbosity=0,
        )
        m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
        oof_seed[val] = m.predict_proba(X[val])[:, 1]
        test_seed += m.predict_proba(X_test)[:, 1] / NF
        log(f"    {cfg_name} seed={SEED} fold={fold+1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
    auc_seed = roc_auc_score(y, oof_seed)
    log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
    oof_acc += oof_seed / len(SEEDS)
    test_acc += test_seed / len(SEEDS)
all_oof[cfg_name] = oof_acc
all_test[cfg_name] = test_acc
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_acc):.6f}")
gc.collect()

# --- Config 4: XGBoost (shallow, regularized) ---
cfg_name = 'xgb_reg'
log(f"\n  Training {cfg_name}...")
oof_acc = np.zeros(len(X))
test_acc = np.zeros(len(X_test))
for si, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof_seed = np.zeros(len(X))
    test_seed = np.zeros(len(X_test))
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            n_estimators=4000, learning_rate=0.02, max_depth=6,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.5,
            colsample_bylevel=0.5, reg_alpha=2.0, reg_lambda=3.0,
            gamma=0.2, tree_method='hist', random_state=SEED,
            n_jobs=-1, early_stopping_rounds=200, verbosity=0,
        )
        m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
        oof_seed[val] = m.predict_proba(X[val])[:, 1]
        test_seed += m.predict_proba(X_test)[:, 1] / NF
        log(f"    {cfg_name} seed={SEED} fold={fold+1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
    auc_seed = roc_auc_score(y, oof_seed)
    log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
    oof_acc += oof_seed / len(SEEDS)
    test_acc += test_seed / len(SEEDS)
all_oof[cfg_name] = oof_acc
all_test[cfg_name] = test_acc
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_acc):.6f}")
gc.collect()

# --- Config 5: CatBoost GPU ---
cfg_name = 'cat_gpu'
log(f"\n  Training {cfg_name}...")
oof_acc = np.zeros(len(X))
test_acc = np.zeros(len(X_test))
for si, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof_seed = np.zeros(len(X))
    test_seed = np.zeros(len(X_test))
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = CatBoostClassifier(
            iterations=4000, learning_rate=0.02, depth=8,
            l2_leaf_reg=5, border_count=128, random_strength=0.5,
            bagging_temperature=0.5, od_type='Iter', od_wait=200,
            verbose=0, random_seed=SEED, task_type='GPU',
            use_best_model=True, grow_policy='Lossguide', min_data_in_leaf=30,
        )
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[val], y[val]), verbose=False)
        oof_seed[val] = m.predict_proba(X[val])[:, 1]
        test_seed += m.predict_proba(X_test)[:, 1] / NF
        log(f"    {cfg_name} seed={SEED} fold={fold+1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
    auc_seed = roc_auc_score(y, oof_seed)
    log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
    oof_acc += oof_seed / len(SEEDS)
    test_acc += test_seed / len(SEEDS)
all_oof[cfg_name] = oof_acc
all_test[cfg_name] = test_acc
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_acc):.6f}")
gc.collect()

# --- Config 6: CatBoost CPU (different config for diversity) ---
cfg_name = 'cat_cpu'
log(f"\n  Training {cfg_name}...")
oof_acc = np.zeros(len(X))
test_acc = np.zeros(len(X_test))
for si, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof_seed = np.zeros(len(X))
    test_seed = np.zeros(len(X_test))
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        m = CatBoostClassifier(
            iterations=3000, learning_rate=0.03, depth=6,
            l2_leaf_reg=3, border_count=254, random_strength=1.0,
            bagging_temperature=1.0, od_type='Iter', od_wait=150,
            verbose=0, random_seed=SEED, task_type='CPU',
            use_best_model=True, min_data_in_leaf=50,
        )
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[val], y[val]), verbose=False)
        oof_seed[val] = m.predict_proba(X[val])[:, 1]
        test_seed += m.predict_proba(X_test)[:, 1] / NF
        log(f"    {cfg_name} seed={SEED} fold={fold+1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
    auc_seed = roc_auc_score(y, oof_seed)
    log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
    oof_acc += oof_seed / len(SEEDS)
    test_acc += test_seed / len(SEEDS)
all_oof[cfg_name] = oof_acc
all_test[cfg_name] = test_acc
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_acc):.6f}")
gc.collect()

# ============================================================
# 6. ENSEMBLE + STACKING
# ============================================================
log("\n[5/6] Ensemble optimization...")

cfg_names = list(all_oof.keys())
log(f"Model configs: {cfg_names}")

oof_stack = np.column_stack([all_oof[c] for c in cfg_names])
test_stack = np.column_stack([all_test[c] for c in cfg_names])

log(f"Stack shape: OOF={oof_stack.shape}, Test={test_stack.shape}")

# Correlation analysis
log("\nModel OOF correlations:")
for i, c1 in enumerate(cfg_names):
    for j, c2 in enumerate(cfg_names):
        if j > i:
            corr = np.corrcoef(all_oof[c1], all_oof[c2])[0, 1]
            log(f"  {c1} vs {c2}: {corr:.4f}")

# Simple average
simple_avg_oof = np.mean(oof_stack, axis=1)
log(f"\nSimple average AUC: {roc_auc_score(y, simple_avg_oof):.6f}")

# Nelder-Mead weight optimization
from scipy.optimize import minimize
n_models = len(cfg_names)

def neg_auc(w):
    w = np.abs(w)
    s = w.sum()
    if s < 1e-8:
        return 1.0
    w = w / s
    return -roc_auc_score(y, oof_stack @ w)

best_auc = 0
best_weights = np.ones(n_models) / n_models
np.random.seed(42)
for trial in range(100):
    w0 = np.random.dirichlet(np.ones(n_models))
    res = minimize(neg_auc, w0, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-7})
    w_opt = np.abs(res.x)
    w_opt = w_opt / w_opt.sum()
    a = roc_auc_score(y, oof_stack @ w_opt)
    if a > best_auc:
        best_auc = a
        best_weights = w_opt.copy()

log(f"Best optimized AUC: {best_auc:.6f}")
log(f"Best weights: {dict(zip(cfg_names, [f'{w:.4f}' for w in best_weights]))}")

test_ens_opt = test_stack @ best_weights

# Ridge stacking
log("\nRidge stacking...")
skf_stack = StratifiedKFold(n_splits=NF, shuffle=True, random_state=42)
oof_ridge = np.zeros(len(X))
test_ridge = np.zeros(len(X_test))

for fold, (tr, val) in enumerate(skf_stack.split(oof_stack, y)):
    ridge = Ridge(alpha=100)
    ridge.fit(oof_stack[tr], y[tr])
    oof_ridge[val] = ridge.predict(oof_stack[val])
    test_ridge += ridge.predict(test_stack) / NF
    log(f"  Ridge fold {fold+1}: {roc_auc_score(y[val], oof_ridge[val]):.6f}")

ridge_auc = roc_auc_score(y, oof_ridge)
log(f"Ridge stacking AUC: {ridge_auc:.6f}")

# Try different Ridge alphas
best_ridge_auc = ridge_auc
best_ridge_test = test_ridge.copy()
for alpha in [1, 10, 50, 200, 500, 1000]:
    oof_r = np.zeros(len(X))
    test_r = np.zeros(len(X_test))
    for fold, (tr, val) in enumerate(skf_stack.split(oof_stack, y)):
        ridge = Ridge(alpha=alpha)
        ridge.fit(oof_stack[tr], y[tr])
        oof_r[val] = ridge.predict(oof_stack[val])
        test_r += ridge.predict(test_stack) / NF
    a = roc_auc_score(y, oof_r)
    log(f"  Ridge alpha={alpha}: AUC={a:.6f}")
    if a > best_ridge_auc:
        best_ridge_auc = a
        best_ridge_test = test_r.copy()

log(f"Best Ridge AUC: {best_ridge_auc:.6f}")

# ============================================================
# 7. FINAL SUBMISSION
# ============================================================
log("\n[6/6] Final submission...")

# Choose best method
if best_ridge_auc > best_auc:
    final_test = best_ridge_test
    final_auc = best_ridge_auc
    method = "Ridge stacking"
else:
    final_test = test_ens_opt
    final_auc = best_auc
    method = "Nelder-Mead optimized weights"

# Clip predictions
final_test = np.clip(final_test, 0.001, 0.999)

log(f"\n{'='*70}")
log(f"FINAL RESULTS (V8)")
log(f"{'='*70}")
for c in cfg_names:
    log(f"  {c}: {roc_auc_score(y, all_oof[c]):.6f}")
log(f"  Optimized weights AUC: {best_auc:.6f}")
log(f"  Best Ridge AUC: {best_ridge_auc:.6f}")
log(f"  Final method: {method}")
log(f"  Final OOF AUC: {final_auc:.6f}")
log(f"  Prediction mean: {final_test.mean():.4f}")
log(f"  Elapsed: {(time.time()-t0):.0f}s")

sub = pd.read_csv('playground-series-s6e5/sample_submission.csv')
sub['PitNextLap'] = final_test
sub.to_csv('submission_v8.csv', index=False)
log(f"\nSaved: submission_v8.csv")

# Save OOF and test predictions
for c in cfg_names:
    np.save(f'oof_{c}_v8.npy', all_oof[c])
    np.save(f'test_{c}_v8.npy', all_test[c])

results = {
    'models': {c: float(roc_auc_score(y, all_oof[c])) for c in cfg_names},
    'optimized_auc': float(best_auc),
    'ridge_auc': float(best_ridge_auc),
    'final_auc': float(final_auc),
    'method': method,
    'n_features': len(FEATURES),
    'elapsed_s': time.time() - t0,
}
with open('experiment_v8.json', 'w') as f:
    json.dump(results, f, indent=2)
log("Saved: experiment_v8.json")

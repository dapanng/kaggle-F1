import sys, time, warnings, gc, json, os
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
from scipy.stats import binom
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier, Pool
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

SEEDS = [42, 123, 2024]
NF = 5
t0 = time.time()

log("=" * 70)
log("V9_OPT: Advanced Feature Selection + Hyperparameter Optimization")
log("=" * 70)

# ============================================================
# MODULE 1: DATA LOADING & FEATURE ENGINEERING (with caching)
# ============================================================
log("\n[Module 1] Loading data & feature engineering...")

compound_order = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
gm = 0.1990

CACHE_TRAIN = 'cache_train_v9_opt.pkl'
CACHE_TEST = 'cache_test_v9_opt.pkl'

def add_features(df):
    df = df.copy()
    df['Compound_ord'] = df['Compound'].map(compound_order)
    co = df['Compound_ord']

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

    df['TL_gt_15'] = (df['TyreLife'] > 15).astype(np.int8)
    df['TL_gt_25'] = (df['TyreLife'] > 25).astype(np.int8)
    df['TL_gt_35'] = (df['TyreLife'] > 35).astype(np.int8)
    df['TL_gt_50'] = (df['TyreLife'] > 50).astype(np.int8)

    df['Deg_per_lap'] = df['Cumulative_Degradation'] / tls
    df['LTD_per_lap'] = df['LapTime_Delta'] / tls
    df['LT_per_lap'] = df['LapTime (s)'] / tls
    df['Deg_abs'] = np.abs(df['Cumulative_Degradation'])
    df['Deg_sq'] = df['Cumulative_Degradation'] ** 2

    df['Pos_sq'] = df['Position'] ** 2
    df['Is_Top10'] = (df['Position'] <= 10).astype(np.int8)

    df['RP_sq'] = df['RaceProgress'] ** 2
    df['RP_cu'] = df['RaceProgress'] ** 3
    df['LN_sq'] = df['LapNumber'] ** 2
    df['LN_sqrt'] = np.sqrt(df['LapNumber'])

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

    compound_avg_tl = {'SOFT': 12.0, 'MEDIUM': 18.0, 'HARD': 25.0, 'INTERMEDIATE': 10.0, 'WET': 8.0}
    df['Compound_avg_TL'] = df['Compound'].map(compound_avg_tl)
    df['Dev_compound_TL'] = df['TyreLife'] - df['Compound_avg_TL']
    df['TL_ratio_compound'] = df['TyreLife'] / df['Compound_avg_TL'].clip(lower=1)

    df['Deg_x_Compound'] = df['Cumulative_Degradation'] * co
    df['LTD_x_Compound'] = df['LapTime_Delta'] * co

    df['RP_div_Stint'] = df['RaceProgress'] / df['Stint'].clip(lower=1)

    df['TL_x_Deg'] = df['TyreLife'] * df['Cumulative_Degradation']
    df['TL_x_LTD'] = df['TyreLife'] * df['LapTime_Delta']

    df['Pos_x_RP'] = df['Position'] * df['RaceProgress']

    df['PitStop_x_RP'] = df['PitStop'] * df['RaceProgress']

    df['Stint_x_Compound'] = df['Stint'] * co

    df['LN_minus_RP'] = df['LapNumber'] - df['RaceProgress'] * df['LapNumber'].max()

    return df.fillna(0).replace([np.inf, -np.inf], 0)

if os.path.exists(CACHE_TRAIN) and os.path.exists(CACHE_TEST):
    log("  Loading cached features from pickle...")
    train_fe = pd.read_pickle(CACHE_TRAIN)
    test_fe = pd.read_pickle(CACHE_TEST)
    train = pd.read_csv('playground-series-s6e5/train.csv')
    train['PitNextLap'] = train['PitNextLap'].astype(int)
    log(f"  Train: {train_fe.shape}, Test: {test_fe.shape} (from cache)")
else:
    log("  Computing features from scratch...")
    train = pd.read_csv('playground-series-s6e5/train.csv')
    test = pd.read_csv('playground-series-s6e5/test.csv')
    train['PitNextLap'] = train['PitNextLap'].astype(int)
    log(f"  Raw Train: {train.shape}, Test: {test.shape}")

    train_fe = add_features(train)
    test_fe = add_features(test)

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

    train_fe.to_pickle(CACHE_TRAIN)
    test_fe.to_pickle(CACHE_TEST)
    log(f"  Saved feature cache: {CACHE_TRAIN}, {CACHE_TEST}")

log(f"  Feature engineering done: Train={train_fe.shape}, Test={test_fe.shape}")
t1 = time.time()
log(f"  Module 1 elapsed: {t1 - t0:.1f}s")

# ============================================================
# KFOLD TARGET ENCODING (anti-leakage, same as V8)
# ============================================================
log("\n[Module 1b] KFold target encoding...")

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

FEATURES = [
    'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'Position_Change',
    'Compound_ord', 'Driver_le', 'Race_le',
    'Driver_te', 'Race_te', 'Compound_te', 'Stint_te', 'Year_te',
    'Compound_Race_te', 'Driver_Race_te', 'Compound_Stint_te',
    'Driver_Compound_te', 'Driver_Stint_te', 'Race_Stint_te', 'CRStint_te',
    'Risk_CD', 'Risk_DR', 'Risk_CR', 'Risk_CDR', 'Risk_CS', 'Risk_DS', 'Risk_RS',
    'CxTyreLife', 'CxLapNumber', 'CxRaceProgress', 'CxDegradation', 'CxPosition',
    'TL_sq', 'TL_sqrt', 'TL_log', 'TL_cu',
    'TL_gt_15', 'TL_gt_25', 'TL_gt_35', 'TL_gt_50',
    'Deg_per_lap', 'LTD_per_lap', 'LT_per_lap', 'Deg_abs', 'Deg_sq',
    'Pos_sq', 'Is_Top10',
    'RP_sq', 'RP_cu', 'LN_sq', 'LN_sqrt',
    'Stint_x_TL', 'Stint_x_LN', 'TL_div_Stint', 'LN_div_Stint',
    'Year_2022', 'Year_2023', 'Year_2024', 'Year_2025',
    'PChange_abs', 'Has_Pitted', 'Tire_Wear', 'LTD_abs', 'LT_sq',
    'Drv_avg_tyre', 'Drv_avg_pos', 'Drv_avg_laptime', 'Drv_avg_degrad', 'Drv_avg_ltd',
    'Race_avg_degrad', 'Race_avg_laptime', 'Race_avg_tyre', 'Race_avg_pos', 'Race_avg_ltd',
    'Compound_avg_degrad', 'Compound_avg_laptime',
    'Dev_drv_tyre', 'Dev_drv_pos', 'Dev_drv_degrad', 'Dev_drv_ltd',
    'Dev_race_degrad', 'Dev_race_laptime', 'Dev_race_tyre', 'Dev_race_pos', 'Dev_race_ltd',
    'Dev_compound_degrad', 'Dev_compound_laptime',
    'PitStop_x_TL', 'LT_x_Compound', 'RP_x_TL', 'Pos_x_TL', 'LN_x_TL', 'Deg_x_RP', 'TL_div_RP', 'Stint_x_RP',
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
log(f"  Features: {len(FEATURES)} | Train: {X.shape} | Test: {X_test.shape}")
gc.collect()

# ============================================================
# MODULE 2: SHAP RECURSIVE FEATURE ELIMINATION
# ============================================================
log("\n[Module 2] SHAP Recursive Feature Elimination...")

def shap_rfe(X, y, feature_names, n_steps=20, n_cv=3):
    t_start = time.time()
    feature_names = list(feature_names)
    current_features = feature_names.copy()
    history = []

    feat_to_idx = {f: i for i, f in enumerate(feature_names)}

    for step in range(n_steps):
        log(f"  SHAP RFE step {step+1}/{n_steps}: {len(current_features)} features remaining")

        idx = [feat_to_idx[f] for f in current_features]
        X_curr = X[:, idx]

        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        aucs = []
        for tr, val in skf.split(X_curr, y):
            m = lgb.LGBMClassifier(
                objective='binary', metric='auc', boosting_type='gbdt',
                n_estimators=2000, learning_rate=0.02, num_leaves=255,
                max_depth=10, min_data_in_leaf=30, feature_fraction=0.4,
                bagging_fraction=0.75, bagging_freq=5, lambda_l1=1.0,
                lambda_l2=2.0, verbose=-1, random_state=42, n_jobs=-1,
            )
            m.fit(X_curr[tr], y[tr], eval_set=[(X_curr[val], y[val])],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            preds = m.predict_proba(X_curr[val])[:, 1]
            aucs.append(roc_auc_score(y[val], preds))
        current_auc = np.mean(aucs)
        log(f"    CV AUC: {current_auc:.6f}")

        if len(current_features) <= 5:
            history.append({'step': step, 'n_features': len(current_features), 'auc': current_auc, 'removed': []})
            break

        m_full = lgb.LGBMClassifier(
            objective='binary', metric='auc', boosting_type='gbdt',
            n_estimators=2000, learning_rate=0.02, num_leaves=255,
            max_depth=10, min_data_in_leaf=30, feature_fraction=0.4,
            bagging_fraction=0.75, bagging_freq=5, lambda_l1=1.0,
            lambda_l2=2.0, verbose=-1, random_state=42, n_jobs=-1,
        )
        skf_shap = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        tr_idx, val_idx = next(skf_shap.split(X_curr, y))
        m_full.fit(X_curr[tr_idx], y[tr_idx],
                   eval_set=[(X_curr[val_idx], y[val_idx])],
                   callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

        sample_size = min(10000, len(X_curr))
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_curr), sample_size, replace=False)
        X_sample = X_curr[sample_idx]

        explainer = shap.TreeExplainer(m_full)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_ranking = sorted(zip(current_features, mean_abs_shap), key=lambda x: x[1], reverse=True)

        n_remove = max(1, int(len(current_features) * 0.05))
        to_remove = [f for f, _ in shap_ranking[-n_remove:]]

        history.append({
            'step': step,
            'n_features': len(current_features),
            'auc': current_auc,
            'removed': to_remove,
            'shap_top5': [f for f, _ in shap_ranking[:5]],
            'shap_bottom5': [f for f, _ in shap_ranking[-5:]],
        })

        current_features = [f for f in current_features if f not in to_remove]
        gc.collect()

    best_step = max(history, key=lambda x: x['auc'])
    best_n = best_step['n_features']

    cumulative_removed = []
    for h in history:
        cumulative_removed.extend(h.get('removed', []))
        if h['n_features'] <= best_n:
            break

    selected = [f for f in feature_names if f not in cumulative_removed]
    if len(selected) < best_n:
        shap_all = sorted(zip(feature_names, [0.0]*len(feature_names)), key=lambda x: x[1], reverse=True)
        selected = [f for f, _ in shap_all[:best_n]]

    with open('feature_importance_curve.json', 'w') as f:
        json.dump(history, f, indent=2)

    log(f"  SHAP RFE done: selected {len(selected)} features, best AUC={best_step['auc']:.6f}")
    log(f"  Elapsed: {time.time() - t_start:.1f}s")
    return selected, history

shap_selected_features, shap_history = shap_rfe(X, y, FEATURES, n_steps=20, n_cv=3)
log(f"  SHAP selected features ({len(shap_selected_features)}): {shap_selected_features[:10]}...")
gc.collect()

# ============================================================
# MODULE 3: BORUTA FEATURE SELECTION
# ============================================================
log("\n[Module 3] Boruta Feature Selection...")

def boruta_selection(X, y, feature_names, n_iter=100, alpha=0.05):
    t_start = time.time()
    feature_names = list(feature_names)
    n_features = len(feature_names)
    hit_counts = np.zeros(n_features, dtype=int)

    for it in range(n_iter):
        if (it + 1) % 20 == 0:
            log(f"  Boruta iteration {it+1}/{n_iter}")

        shadow_cols = []
        for i in range(n_features):
            col = X[:, i].copy()
            np.random.shuffle(col)
            shadow_cols.append(col)
        X_shadow = np.column_stack(shadow_cols)

        X_combined = np.column_stack([X, X_shadow])

        np.random.seed(it)
        sample_size = min(50000, len(X_combined))
        sample_idx = np.random.choice(len(X_combined), sample_size, replace=False)

        m = lgb.LGBMClassifier(
            objective='binary', metric='auc', boosting_type='gbdt',
            n_estimators=500, learning_rate=0.05, num_leaves=127,
            max_depth=8, min_data_in_leaf=50, feature_fraction=0.5,
            bagging_fraction=0.75, bagging_freq=5,
            verbose=-1, random_state=it, n_jobs=-1,
        )
        m.fit(X_combined[sample_idx], y[sample_idx],
              callbacks=[lgb.log_evaluation(0)])

        importances = m.feature_importances_
        orig_imp = importances[:n_features]
        shadow_imp = importances[n_features:]
        max_shadow = shadow_imp.max()

        for i in range(n_features):
            if orig_imp[i] > max_shadow:
                hit_counts[i] += 1

        del X_shadow, X_combined, shadow_cols
        gc.collect()

    confirmed = []
    rejected = []
    tentative = []

    for i, feat in enumerate(feature_names):
        p_value = 1 - binom.cdf(hit_counts[i] - 1, n_iter, 0.5)
        if p_value < alpha / 2:
            confirmed.append(feat)
        elif p_value > 1 - alpha / 2:
            rejected.append(feat)
        else:
            tentative.append(feat)

    result = {'confirmed': confirmed, 'rejected': rejected, 'tentative': tentative}
    log(f"  Boruta done: confirmed={len(confirmed)}, rejected={len(rejected)}, tentative={len(tentative)}")
    log(f"  Elapsed: {time.time() - t_start:.1f}s")
    return result

boruta_result = boruta_selection(X, y, FEATURES, n_iter=100, alpha=0.05)
log(f"  Boruta confirmed ({len(boruta_result['confirmed'])}): {boruta_result['confirmed'][:10]}...")
log(f"  Boruta rejected ({len(boruta_result['rejected'])}): {boruta_result['rejected'][:10]}...")
gc.collect()

# ============================================================
# MODULE 4: FEATURE SELECTION FUSION
# ============================================================
log("\n[Module 4] Feature Selection Fusion...")

def fuse_feature_selections(shap_features, boruta_confirmed, boruta_rejected, all_features, X, y):
    t_start = time.time()
    shap_set = set(shap_features)
    confirmed_set = set(boruta_confirmed)
    rejected_set = set(boruta_rejected)

    core = shap_set & confirmed_set
    remove = rejected_set - shap_set
    contested = set(all_features) - core - remove

    log(f"  Core features (both methods agree): {len(core)}")
    log(f"  Remove features (both methods agree): {len(remove)}")
    log(f"  Contested features (disagreement): {len(contested)}")

    final_features = list(core)
    feat_to_idx = {f: i for i, f in enumerate(all_features)}

    if len(contested) > 0:
        log(f"  Testing {len(contested)} contested features via AUC comparison...")

        core_idx = [feat_to_idx[f] for f in final_features]
        X_core = X[:, core_idx]

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        base_aucs = []
        for tr, val in skf.split(X_core, y):
            m = lgb.LGBMClassifier(
                objective='binary', metric='auc', boosting_type='gbdt',
                n_estimators=1500, learning_rate=0.02, num_leaves=255,
                max_depth=10, verbose=-1, random_state=42, n_jobs=-1,
            )
            m.fit(X_core[tr], y[tr], eval_set=[(X_core[val], y[val])],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            base_aucs.append(roc_auc_score(y[val], m.predict_proba(X_core[val])[:, 1]))
        base_auc = np.mean(base_aucs)
        log(f"  Base AUC (core only): {base_auc:.6f}")

        for feat in contested:
            test_feats = final_features + [feat]
            test_idx = [feat_to_idx[f] for f in test_feats]
            X_test_feat = X[:, test_idx]

            aucs = []
            for tr, val in skf.split(X_test_feat, y):
                m = lgb.LGBMClassifier(
                    objective='binary', metric='auc', boosting_type='gbdt',
                    n_estimators=1500, learning_rate=0.02, num_leaves=255,
                    max_depth=10, verbose=-1, random_state=42, n_jobs=-1,
                )
                m.fit(X_test_feat[tr], y[tr], eval_set=[(X_test_feat[val], y[val])],
                      eval_metric='auc',
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
                aucs.append(roc_auc_score(y[val], m.predict_proba(X_test_feat[val])[:, 1]))
            feat_auc = np.mean(aucs)

            if feat_auc > base_auc:
                final_features.append(feat)

        log(f"  After contested testing: {len(final_features)} features (base_auc={base_auc:.6f})")

    log(f"  Fusion done: {len(final_features)} final features")
    log(f"  Elapsed: {time.time() - t_start:.1f}s")
    return final_features

final_features = fuse_feature_selections(
    shap_selected_features,
    boruta_result['confirmed'],
    boruta_result['rejected'],
    FEATURES, X, y
)
log(f"  Final selected features ({len(final_features)}): {final_features[:15]}...")

X_sel = X[:, [FEATURES.index(f) for f in final_features]].astype(np.float32)
X_test_sel = X_test[:, [FEATURES.index(f) for f in final_features]].astype(np.float32)
log(f"  Selected feature matrices: Train={X_sel.shape}, Test={X_test_sel.shape}")
gc.collect()

# ============================================================
# MODULE 5: OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================
log("\n[Module 5] Optuna Hyperparameter Optimization...")

def optuna_optimize(model_type, X, y, n_trials=50, n_cv=3):
    t_start = time.time()
    log(f"  Optuna optimizing: {model_type} ({n_trials} trials)")

    def lgb_deep_objective(trial):
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'n_estimators': 5000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 63, 511),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.8),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': 5,
            'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 5.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.5, 5.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'verbose': -1, 'n_jobs': -1, 'random_state': 42,
        }
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        aucs = []
        for tr, val in skf.split(X, y):
            m = lgb.LGBMClassifier(**params)
            m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], eval_metric='auc',
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            aucs.append(roc_auc_score(y[val], m.predict_proba(X[val])[:, 1]))
        return np.mean(aucs)

    def lgb_reg_objective(trial):
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'n_estimators': 5000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.8),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': 5,
            'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 5.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.5, 5.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'verbose': -1, 'n_jobs': -1, 'random_state': 42,
        }
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        aucs = []
        for tr, val in skf.split(X, y):
            m = lgb.LGBMClassifier(**params)
            m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], eval_metric='auc',
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            aucs.append(roc_auc_score(y[val], m.predict_proba(X[val])[:, 1]))
        return np.mean(aucs)

    def xgb_deep_objective(trial):
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc',
            'n_estimators': 5000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
            'tree_method': 'hist', 'random_state': 42,
            'n_jobs': -1, 'early_stopping_rounds': 200, 'verbosity': 0,
        }
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        aucs = []
        for tr, val in skf.split(X, y):
            m = xgb.XGBClassifier(**params)
            m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
            aucs.append(roc_auc_score(y[val], m.predict_proba(X[val])[:, 1]))
        return np.mean(aucs)

    def xgb_reg_objective(trial):
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc',
            'n_estimators': 4000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 80),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
            'tree_method': 'hist', 'random_state': 42,
            'n_jobs': -1, 'early_stopping_rounds': 200, 'verbosity': 0,
        }
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        aucs = []
        for tr, val in skf.split(X, y):
            m = xgb.XGBClassifier(**params)
            m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
            aucs.append(roc_auc_score(y[val], m.predict_proba(X[val])[:, 1]))
        return np.mean(aucs)

    def cat_gpu_objective(trial):
        params = {
            'iterations': 4000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'depth': trial.suggest_int('depth', 6, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
            'border_count': trial.suggest_int('border_count', 64, 256),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1.0),
            'od_type': 'Iter', 'od_wait': 200,
            'verbose': 0, 'random_seed': 42,
            'task_type': 'GPU', 'use_best_model': True,
            'grow_policy': 'Lossguide', 'min_data_in_leaf': 30,
        }
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        aucs = []
        for tr, val in skf.split(X, y):
            m = CatBoostClassifier(**params)
            m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[val], y[val]), verbose=False)
            aucs.append(roc_auc_score(y[val], m.predict_proba(X[val])[:, 1]))
        return np.mean(aucs)

    def cat_cpu_objective(trial):
        params = {
            'iterations': 3000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
            'border_count': trial.suggest_int('border_count', 64, 256),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1.0),
            'od_type': 'Iter', 'od_wait': 150,
            'verbose': 0, 'random_seed': 42,
            'task_type': 'CPU', 'use_best_model': True,
            'min_data_in_leaf': 50,
        }
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        aucs = []
        for tr, val in skf.split(X, y):
            m = CatBoostClassifier(**params)
            m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[val], y[val]), verbose=False)
            aucs.append(roc_auc_score(y[val], m.predict_proba(X[val])[:, 1]))
        return np.mean(aucs)

    objective_map = {
        'lgb_deep': lgb_deep_objective,
        'lgb_reg': lgb_reg_objective,
        'xgb_deep': xgb_deep_objective,
        'xgb_reg': xgb_reg_objective,
        'cat_gpu': cat_gpu_objective,
        'cat_cpu': cat_cpu_objective,
    }

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective_map[model_type], n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_value = study.best_value
    log(f"  {model_type} best CV AUC: {best_value:.6f}")
    log(f"  {model_type} best params: {best_params}")

    with open(f'best_params_{model_type}.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    log(f"  Elapsed: {time.time() - t_start:.1f}s")
    return best_params

best_params_all = {}
model_types = ['lgb_deep', 'lgb_reg', 'xgb_deep', 'xgb_reg', 'cat_gpu', 'cat_cpu']
for mt in model_types:
    params_file = f'best_params_{mt}.json'
    if os.path.exists(params_file):
        log(f"  Loading cached params for {mt} from {params_file}")
        with open(params_file, 'r') as f:
            best_params_all[mt] = json.load(f)
    else:
        best_params_all[mt] = optuna_optimize(mt, X_sel, y, n_trials=50, n_cv=3)
    gc.collect()

log(f"  All Optuna optimizations complete")
log(f"  Best params summary:")
for mt, p in best_params_all.items():
    log(f"    {mt}: lr={p.get('learning_rate', 'N/A'):.4f}, depth/max_depth={p.get('max_depth', p.get('depth', 'N/A'))}")

# ============================================================
# MODULE 6: MAIN TRAINING PIPELINE
# ============================================================
log("\n[Module 6] Main Training Pipeline...")

all_oof = {}
all_test = {}

def train_lgb(cfg_name, X, y, X_test, params, seeds, nf):
    oof_acc = np.zeros(len(X))
    test_acc = np.zeros(len(X_test))
    for si, SEED in enumerate(seeds):
        skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=SEED)
        oof_seed = np.zeros(len(X))
        test_seed = np.zeros(len(X_test))
        for fold, (tr, val) in enumerate(skf.split(X, y)):
            m = lgb.LGBMClassifier(
                objective='binary', metric='auc', boosting_type='gbdt',
                n_estimators=5000,
                learning_rate=params['learning_rate'],
                num_leaves=params['num_leaves'],
                max_depth=params['max_depth'],
                feature_fraction=params['feature_fraction'],
                bagging_fraction=params['bagging_fraction'],
                bagging_freq=5,
                lambda_l1=params['lambda_l1'],
                lambda_l2=params['lambda_l2'],
                min_data_in_leaf=params['min_data_in_leaf'],
                verbose=-1, random_state=SEED, n_jobs=-1,
            )
            m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], eval_metric='auc',
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            oof_seed[val] = m.predict_proba(X[val])[:, 1]
            test_seed += m.predict_proba(X_test)[:, 1] / nf
            log(f"    {cfg_name} seed={SEED} fold={fold+1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
        auc_seed = roc_auc_score(y, oof_seed)
        log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
        oof_acc += oof_seed / len(seeds)
        test_acc += test_seed / len(seeds)
    return oof_acc, test_acc

def train_xgb(cfg_name, X, y, X_test, params, seeds, nf, n_est=5000):
    oof_acc = np.zeros(len(X))
    test_acc = np.zeros(len(X_test))
    for si, SEED in enumerate(seeds):
        skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=SEED)
        oof_seed = np.zeros(len(X))
        test_seed = np.zeros(len(X_test))
        for fold, (tr, val) in enumerate(skf.split(X, y)):
            m = xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='auc',
                n_estimators=n_est,
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                min_child_weight=params['min_child_weight'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                gamma=params['gamma'],
                tree_method='hist', random_state=SEED,
                n_jobs=-1, early_stopping_rounds=200, verbosity=0,
            )
            m.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
            oof_seed[val] = m.predict_proba(X[val])[:, 1]
            test_seed += m.predict_proba(X_test)[:, 1] / nf
            log(f"    {cfg_name} seed={SEED} fold={fold+1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
        auc_seed = roc_auc_score(y, oof_seed)
        log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
        oof_acc += oof_seed / len(seeds)
        test_acc += test_seed / len(seeds)
    return oof_acc, test_acc

def train_cat(cfg_name, X, y, X_test, params, seeds, nf, task_type='GPU'):
    oof_acc = np.zeros(len(X))
    test_acc = np.zeros(len(X_test))
    for si, SEED in enumerate(seeds):
        skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=SEED)
        oof_seed = np.zeros(len(X))
        test_seed = np.zeros(len(X_test))
        for fold, (tr, val) in enumerate(skf.split(X, y)):
            extra = {}
            if task_type == 'GPU':
                extra['grow_policy'] = 'Lossguide'
                extra['min_data_in_leaf'] = 30
            else:
                extra['min_data_in_leaf'] = 50
            m = CatBoostClassifier(
                iterations=params.get('iterations', 4000),
                learning_rate=params['learning_rate'],
                depth=params['depth'],
                l2_leaf_reg=params['l2_leaf_reg'],
                border_count=params['border_count'],
                bagging_temperature=params['bagging_temperature'],
                od_type='Iter', od_wait=200,
                verbose=0, random_seed=SEED,
                task_type=task_type, use_best_model=True,
                **extra,
            )
            m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[val], y[val]), verbose=False)
            oof_seed[val] = m.predict_proba(X[val])[:, 1]
            test_seed += m.predict_proba(X_test)[:, 1] / nf
            log(f"    {cfg_name} seed={SEED} fold={fold+1}: {roc_auc_score(y[val], oof_seed[val]):.6f}")
        auc_seed = roc_auc_score(y, oof_seed)
        log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
        oof_acc += oof_seed / len(seeds)
        test_acc += test_seed / len(seeds)
    return oof_acc, test_acc

# --- Config 1: LightGBM Deep ---
cfg_name = 'lgb_deep'
log(f"\n  Training {cfg_name} with optimized params...")
oof_lgb_deep, test_lgb_deep = train_lgb(cfg_name, X_sel, y, X_test_sel, best_params_all['lgb_deep'], SEEDS, NF)
all_oof[cfg_name] = oof_lgb_deep
all_test[cfg_name] = test_lgb_deep
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_lgb_deep):.6f}")
gc.collect()

# --- Config 2: LightGBM Reg ---
cfg_name = 'lgb_reg'
log(f"\n  Training {cfg_name} with optimized params...")
oof_lgb_reg, test_lgb_reg = train_lgb(cfg_name, X_sel, y, X_test_sel, best_params_all['lgb_reg'], SEEDS, NF)
all_oof[cfg_name] = oof_lgb_reg
all_test[cfg_name] = test_lgb_reg
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_lgb_reg):.6f}")
gc.collect()

# --- Config 3: XGBoost Deep ---
cfg_name = 'xgb_deep'
log(f"\n  Training {cfg_name} with optimized params...")
oof_xgb_deep, test_xgb_deep = train_xgb(cfg_name, X_sel, y, X_test_sel, best_params_all['xgb_deep'], SEEDS, NF)
all_oof[cfg_name] = oof_xgb_deep
all_test[cfg_name] = test_xgb_deep
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_xgb_deep):.6f}")
gc.collect()

# --- Config 4: XGBoost Reg ---
cfg_name = 'xgb_reg'
log(f"\n  Training {cfg_name} with optimized params...")
oof_xgb_reg, test_xgb_reg = train_xgb(cfg_name, X_sel, y, X_test_sel, best_params_all['xgb_reg'], SEEDS, NF, n_est=4000)
all_oof[cfg_name] = oof_xgb_reg
all_test[cfg_name] = test_xgb_reg
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_xgb_reg):.6f}")
gc.collect()

# --- Config 5: CatBoost GPU ---
cfg_name = 'cat_gpu'
log(f"\n  Training {cfg_name} with optimized params...")
cat_gpu_params = best_params_all['cat_gpu'].copy()
cat_gpu_params['iterations'] = cat_gpu_params.get('iterations', 4000)
oof_cat_gpu, test_cat_gpu = train_cat(cfg_name, X_sel, y, X_test_sel, cat_gpu_params, SEEDS, NF, task_type='GPU')
all_oof[cfg_name] = oof_cat_gpu
all_test[cfg_name] = test_cat_gpu
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_cat_gpu):.6f}")
gc.collect()

# --- Config 6: CatBoost CPU ---
cfg_name = 'cat_cpu'
log(f"\n  Training {cfg_name} with optimized params...")
cat_cpu_params = best_params_all['cat_cpu'].copy()
cat_cpu_params['iterations'] = cat_cpu_params.get('iterations', 3000)
oof_cat_cpu, test_cat_cpu = train_cat(cfg_name, X_sel, y, X_test_sel, cat_cpu_params, SEEDS, NF, task_type='CPU')
all_oof[cfg_name] = oof_cat_cpu
all_test[cfg_name] = test_cat_cpu
log(f"  {cfg_name} overall: {roc_auc_score(y, oof_cat_cpu):.6f}")
gc.collect()

# --- TabNet Model ---
log("\n  Training TabNet...")
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch

    cfg_name = 'tabnet'
    oof_tabnet = np.zeros(len(X_sel))
    test_tabnet = np.zeros(len(X_test_sel))

    for si, SEED in enumerate(SEEDS):
        skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
        oof_seed = np.zeros(len(X_sel))
        test_seed = np.zeros(len(X_test_sel))
        for fold, (tr, val) in enumerate(skf.split(X_sel, y)):
            tabnet_model = TabNetClassifier(
                n_d=32, n_a=32, n_steps=5, gamma=1.5,
                lambda_sparse=1e-4, optimizer_fn=torch.optim.AdamW,
                optimizer_params=dict(lr=0.02, weight_decay=1e-3),
                scheduler_params=dict(step_size=30, gamma=0.8),
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                mask_type='entmax',
                verbose=0, seed=SEED, device_name='cuda' if torch.cuda.is_available() else 'cpu',
            )
            tabnet_model.fit(
                X_sel[tr], y[tr],
                eval_set=[(X_sel[val], y[val])],
                eval_metric=['auc'],
                max_epochs=200, patience=30, batch_size=8192,
                virtual_batch_size=2048,
            )
            oof_seed[val] = tabnet_model.predict_proba(X_sel[val])[:, 1]
            test_seed += tabnet_model.predict_proba(X_test_sel)[:, 1] / NF
            fold_auc = roc_auc_score(y[val], oof_seed[val])
            log(f"    {cfg_name} seed={SEED} fold={fold+1}: {fold_auc:.6f}")
        auc_seed = roc_auc_score(y, oof_seed)
        log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
        oof_tabnet += oof_seed / len(SEEDS)
        test_tabnet += test_seed / len(SEEDS)

    all_oof[cfg_name] = oof_tabnet
    all_test[cfg_name] = test_tabnet
    log(f"  {cfg_name} overall: {roc_auc_score(y, oof_tabnet):.6f}")
    gc.collect()
except ImportError:
    log("  pytorch_tabnet not installed, skipping TabNet")

# --- FT-Transformer Model ---
log("\n  Training FT-Transformer...")
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class FTTransformer(nn.Module):
        def __init__(self, n_features, d_embedding=64, n_heads=4, n_blocks=3, d_ff=128, dropout=0.2):
            super().__init__()
            self.embedding = nn.Linear(n_features, d_embedding)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_embedding, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, activation='gelu',
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
            self.head = nn.Sequential(
                nn.LayerNorm(d_embedding),
                nn.Linear(d_embedding, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.embedding(x).unsqueeze(1)
            x = self.transformer(x).squeeze(1)
            return self.head(x).squeeze(-1)

    cfg_name = 'ft_transformer'
    oof_ft = np.zeros(len(X_sel))
    test_ft = np.zeros(len(X_test_sel))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for si, SEED in enumerate(SEEDS):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
        oof_seed = np.zeros(len(X_sel))
        test_seed = np.zeros(len(X_test_sel))
        for fold, (tr, val) in enumerate(skf.split(X_sel, y)):
            model = FTTransformer(
                n_features=X_sel.shape[1], d_embedding=64, n_heads=4,
                n_blocks=3, d_ff=128, dropout=0.2,
            ).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
            criterion = nn.BCELoss()

            X_tr_t = torch.FloatTensor(X_sel[tr]).to(device)
            y_tr_t = torch.FloatTensor(y[tr]).float().to(device)
            X_val_t = torch.FloatTensor(X_sel[val]).to(device)
            X_test_t = torch.FloatTensor(X_test_sel).to(device)

            train_ds = TensorDataset(X_tr_t, y_tr_t)
            train_dl = DataLoader(train_ds, batch_size=4096, shuffle=True)

            best_val_auc = 0
            patience_counter = 0
            for epoch in range(100):
                model.train()
                for xb, yb in train_dl:
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                if (epoch + 1) % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_pred = model(X_val_t).cpu().numpy()
                    val_auc = roc_auc_score(y[val], val_pred)
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= 10:
                        break

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                oof_seed[val] = model(X_val_t).cpu().numpy()
                test_seed += model(X_test_t).cpu().numpy() / NF
            fold_auc = roc_auc_score(y[val], oof_seed[val])
            log(f"    {cfg_name} seed={SEED} fold={fold+1}: {fold_auc:.6f}")
            del model, X_tr_t, y_tr_t, X_val_t, X_test_t
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        auc_seed = roc_auc_score(y, oof_seed)
        log(f"  {cfg_name} seed={SEED}: {auc_seed:.6f}")
        oof_ft += oof_seed / len(SEEDS)
        test_ft += test_seed / len(SEEDS)

    all_oof[cfg_name] = oof_ft
    all_test[cfg_name] = test_ft
    log(f"  {cfg_name} overall: {roc_auc_score(y, oof_ft):.6f}")
    gc.collect()
except Exception as e:
    log(f"  FT-Transformer failed: {e}, skipping")

# ============================================================
# ENSEMBLE (Ridge + Nelder-Mead)
# ============================================================
log("\n[Module 6b] Ensemble optimization...")

cfg_names = list(all_oof.keys())
log(f"Model configs: {cfg_names}")

oof_stack = np.column_stack([all_oof[c] for c in cfg_names])
test_stack = np.column_stack([all_test[c] for c in cfg_names])

log(f"Stack shape: OOF={oof_stack.shape}, Test={test_stack.shape}")

log("\nModel OOF correlations:")
for i, c1 in enumerate(cfg_names):
    for j, c2 in enumerate(cfg_names):
        if j > i:
            corr = np.corrcoef(all_oof[c1], all_oof[c2])[0, 1]
            log(f"  {c1} vs {c2}: {corr:.4f}")

simple_avg_oof = np.mean(oof_stack, axis=1)
log(f"\nSimple average AUC: {roc_auc_score(y, simple_avg_oof):.6f}")

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

log("\nRidge stacking...")
skf_stack = StratifiedKFold(n_splits=NF, shuffle=True, random_state=42)
oof_ridge = np.zeros(len(X_sel))
test_ridge = np.zeros(len(X_test_sel))

for fold, (tr, val) in enumerate(skf_stack.split(oof_stack, y)):
    ridge = Ridge(alpha=100)
    ridge.fit(oof_stack[tr], y[tr])
    oof_ridge[val] = ridge.predict(oof_stack[val])
    test_ridge += ridge.predict(test_stack) / NF
    log(f"  Ridge fold {fold+1}: {roc_auc_score(y[val], oof_ridge[val]):.6f}")

ridge_auc = roc_auc_score(y, oof_ridge)
log(f"Ridge stacking AUC: {ridge_auc:.6f}")

best_ridge_auc = ridge_auc
best_ridge_test = test_ridge.copy()
for alpha in [1, 10, 50, 200, 500, 1000]:
    oof_r = np.zeros(len(X_sel))
    test_r = np.zeros(len(X_test_sel))
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
# MODEL DISTILLATION
# ============================================================
log("\n[Module 7] Model distillation...")

if best_ridge_auc > best_auc:
    teacher_oof = oof_ridge
    teacher_test = best_ridge_test
else:
    teacher_oof = oof_stack @ best_weights
    teacher_test = test_ens_opt

student_oof = np.zeros(len(X_sel))
student_test = np.zeros(len(X_test_sel))
DISTILL_ALPHA = 0.5

for si, SEED in enumerate(SEEDS):
    skf = StratifiedKFold(n_splits=NF, shuffle=True, random_state=SEED)
    oof_seed = np.zeros(len(X_sel))
    test_seed = np.zeros(len(X_test_sel))
    for fold, (tr, val) in enumerate(skf.split(X_sel, y)):
        soft_labels = DISTILL_ALPHA * teacher_oof[tr] + (1 - DISTILL_ALPHA) * y[tr]
        m = lgb.LGBMClassifier(
            objective='binary', metric='auc', boosting_type='gbdt',
            n_estimators=3000, learning_rate=0.02, num_leaves=255,
            max_depth=10, min_data_in_leaf=30, feature_fraction=0.5,
            bagging_fraction=0.75, bagging_freq=5, lambda_l1=1.0,
            lambda_l2=2.0, verbose=-1, random_state=SEED, n_jobs=-1,
        )
        m.fit(X_sel[tr], soft_labels, eval_set=[(X_sel[val], y[val])], eval_metric='auc',
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof_seed[val] = m.predict_proba(X_sel[val])[:, 1]
        test_seed += m.predict_proba(X_test_sel)[:, 1] / NF
    student_oof += oof_seed / len(SEEDS)
    student_test += test_seed / len(SEEDS)

student_auc = roc_auc_score(y, student_oof)
log(f"  Distilled student AUC: {student_auc:.6f}")

if student_auc > min(roc_auc_score(y, all_oof[c]) for c in all_oof):
    all_oof['student_distill'] = student_oof
    all_test['student_distill'] = student_test
    log(f"  Student added to ensemble (AUC > worst base model)")
else:
    log(f"  Student NOT added (AUC below worst base model)")

cfg_names = list(all_oof.keys())

# ============================================================
# MODEL QUANTIZATION
# ============================================================
log("\n[Module 8] Model quantization...")

import joblib, tempfile

quantization_results = {}
for cfg_name_q in ['lgb_deep', 'lgb_reg']:
    if cfg_name_q not in all_oof:
        continue
    skf_q = StratifiedKFold(n_splits=NF, shuffle=True, random_state=42)
    oof_quant = np.zeros(len(X_sel))
    fp32_total_size = 0
    fp16_total_size = 0
    for fold, (tr, val) in enumerate(skf_q.split(X_sel, y)):
        params_q = best_params_all.get(cfg_name_q, {}).copy()
        params_q.update({
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'n_estimators': 3000, 'verbose': -1, 'n_jobs': -1, 'random_state': 42,
        })
        m = lgb.LGBMClassifier(**params_q)
        m.fit(X_sel[tr], y[tr], eval_set=[(X_sel[val], y[val])], eval_metric='auc',
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

        fp32_path = tempfile.mktemp(suffix='_fp32.txt')
        fp16_path = tempfile.mktemp(suffix='_fp16.txt')
        m.booster_.save_model(fp32_path)
        fp32_size = os.path.getsize(fp32_path)
        fp32_total_size += fp32_size

        with open(fp32_path, 'r') as f:
            model_text = f.read()
        import re
        leaf_pattern = re.compile(r'leaf_value=(-?[\d.eE+\-]+)')
        def truncate_float(match):
            val = float(match.group(1))
            return f'leaf_value={np.float16(val):.6f}'
        model_text_fp16 = leaf_pattern.sub(truncate_float, model_text)
        with open(fp16_path, 'w') as f:
            f.write(model_text_fp16)
        fp16_size = os.path.getsize(fp16_path)
        fp16_total_size += fp16_size

        m_quant = lgb.Booster(model_file=fp32_path)
        oof_quant[val] = m_quant.predict(X_sel[val])
        del m, m_quant
        for p in [fp32_path, fp16_path]:
            if os.path.exists(p):
                os.remove(p)
        gc.collect()

    auc_quant = roc_auc_score(y, oof_quant)
    baseline_auc = roc_auc_score(y, all_oof[cfg_name_q])
    auc_loss = baseline_auc - auc_quant
    size_reduction = (1 - fp16_total_size / max(fp32_total_size, 1)) * 100
    quantization_results[cfg_name_q] = {
        'baseline_auc': float(baseline_auc),
        'quantized_auc': float(auc_quant),
        'auc_loss': float(auc_loss),
        'fp32_size_bytes': fp32_total_size,
        'fp16_size_bytes': fp16_total_size,
        'size_reduction_pct': float(size_reduction),
    }
    log(f"  {cfg_name_q}: baseline={baseline_auc:.6f}, quantized={auc_quant:.6f}, loss={auc_loss:.6f}, size_reduction={size_reduction:.1f}%")

# ============================================================
# PREPROCESSING OPTIMIZATION VERIFICATION
# ============================================================
log("\n[Module 8b] Preprocessing optimization verification...")
fe_time = time.time() - t0
log(f"  Feature engineering + selection elapsed: {fe_time:.0f}s")
log(f"  Pickle caching: {'ENABLED' if os.path.exists(CACHE_TRAIN) else 'DISABLED'}")

# ============================================================
# FINAL SUBMISSION
# ============================================================
log("\n[Module 6c] Final submission...")

if best_ridge_auc > best_auc:
    final_test = best_ridge_test
    final_auc = best_ridge_auc
    method = "Ridge stacking"
else:
    final_test = test_ens_opt
    final_auc = best_auc
    method = "Nelder-Mead optimized weights"

final_test = np.clip(final_test, 0.001, 0.999)

log(f"\n{'='*70}")
log(f"FINAL RESULTS (V9_OPT)")
log(f"{'='*70}")
log(f"  Feature selection: {len(FEATURES)} -> {len(final_features)} features")
log(f"  SHAP RFE selected: {len(shap_selected_features)} features")
log(f"  Boruta confirmed: {len(boruta_result['confirmed'])} features")
log(f"  Fused features: {len(final_features)} features")
for c in cfg_names:
    log(f"  {c}: {roc_auc_score(y, all_oof[c]):.6f}")
log(f"  Optimized weights AUC: {best_auc:.6f}")
log(f"  Best Ridge AUC: {best_ridge_auc:.6f}")
log(f"  Final method: {method}")
log(f"  Final OOF AUC: {final_auc:.6f}")
log(f"  Prediction mean: {final_test.mean():.4f}")
log(f"  Total elapsed: {(time.time()-t0):.0f}s")

sub = pd.read_csv('playground-series-s6e5/sample_submission.csv')
sub['PitNextLap'] = final_test
sub.to_csv('submission_v9_opt.csv', index=False)
log(f"\nSaved: submission_v9_opt.csv")

for c in cfg_names:
    np.save(f'oof_{c}_v9_opt.npy', all_oof[c])
    np.save(f'test_{c}_v9_opt.npy', all_test[c])

results = {
    'version': 'v9_opt',
    'feature_selection': {
        'original_features': len(FEATURES),
        'shap_rfe_selected': len(shap_selected_features),
        'boruta_confirmed': len(boruta_result['confirmed']),
        'boruta_rejected': len(boruta_result['rejected']),
        'boruta_tentative': len(boruta_result['tentative']),
        'fused_features': len(final_features),
    },
    'models': {c: float(roc_auc_score(y, all_oof[c])) for c in cfg_names},
    'optimized_auc': float(best_auc),
    'ridge_auc': float(best_ridge_auc),
    'final_auc': float(final_auc),
    'method': method,
    'best_params': {k: {pk: (float(pv) if isinstance(pv, (int, float)) else pv) for pk, pv in v.items()} for k, v in best_params_all.items()},
    'distillation': {
        'student_auc': float(student_auc),
        'added_to_ensemble': 'student_distill' in all_oof,
    },
    'quantization': quantization_results,
    'n_features': len(final_features),
    'elapsed_s': time.time() - t0,
}
with open('experiment_v9_opt.json', 'w') as f:
    json.dump(results, f, indent=2)
log("Saved: experiment_v9_opt.json")
log(f"\nTotal runtime: {(time.time()-t0):.0f}s")

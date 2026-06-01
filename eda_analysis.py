"""EDA for Playground Series S6E5 - F1 Pit Stop Prediction"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('playground-series-s6e5/train.csv')
test = pd.read_csv('playground-series-s6e5/test.csv')

print("=" * 60)
print("COMPETITION: Playground Series S6E5 - Predict PitNextLap")
print("Metric: AUC (Binary Classification)")
print("=" * 60)

print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")

print("\n--- TARGET DISTRIBUTION ---")
print(train['PitNextLap'].value_counts())
print(f"Positive rate: {train['PitNextLap'].mean():.4f} ({train['PitNextLap'].sum()} / {len(train)})")

print("\n--- FEATURE TYPES ---")
print(train.dtypes)

print("\n--- MISSING VALUES ---")
print(train.isnull().sum())

print("\n--- NUMERIC FEATURE STATS ---")
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
print(train[numeric_cols].describe().to_string())

print("\n--- CATEGORICAL FEATURE UNIQUE VALUES ---")
cat_cols = ['Driver', 'Compound', 'Race']
for c in cat_cols:
    print(f"\n{c}: nunique={train[c].nunique()}")
    print(f"  Top values: {train[c].value_counts().head(10).to_dict()}")

print("\n--- TARGET BY CATEGORICAL ---")
for c in cat_cols:
    print(f"\n{c} vs PitNextLap (top 10):")
    ct = pd.crosstab(train[c], train['PitNextLap'], normalize='index')
    ct['count'] = train[c].value_counts()
    print(ct.sort_values('count', ascending=False).head(15).to_string())

print("\n--- CORRELATION WITH TARGET ---")
corr = train[numeric_cols].corrwith(train['PitNextLap'])
corr_sorted = corr.abs().sort_values(ascending=False)
for c in corr_sorted.index:
    print(f"  {c}: {corr[c]:.6f}")

print("\n--- OUTLIER CHECK (IQR METHOD) ---")
for col in ['LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation']:
    Q1 = train[col].quantile(0.01)
    Q99 = train[col].quantile(0.99)
    print(f"{col}: 1%={Q1:.2f}, 99%={Q99:.2f}, min={train[col].min():.2f}, max={train[col].max():.2f}")

print("\n--- DUPLICATE CHECK ---")
print(f"Duplicate rows: {train.duplicated().sum()}")

print("\n--- COMPOUND ANALYSIS ---")
print(train.groupby('Compound')['PitNextLap'].agg(['count', 'mean']).sort_values('mean', ascending=False))

print("\n--- YEAR ANALYSIS ---")
print(train.groupby('Year')['PitNextLap'].agg(['count', 'mean']).sort_values('mean', ascending=False))

print("\n--- FEATURE DISTRIBUTION BY TARGET ---")
for col in ['TyreLife', 'Position', 'LapNumber', 'Stint', 'RaceProgress', 'PitStop', 'Position_Change']:
    pos = train[train['PitNextLap']==1][col]
    neg = train[train['PitNextLap']==0][col]
    print(f"\n{col}:")
    print(f"  PitNextLap=1: mean={pos.mean():.3f}, median={pos.median():.3f}, std={pos.std():.3f}")
    print(f"  PitNextLap=0: mean={neg.mean():.3f}, median={neg.median():.3f}, std={neg.std():.3f}")

print("\n--- TYRELIFE BINS ---")
train['TyreLife_bin'] = pd.cut(train['TyreLife'], bins=[0,5,10,15,20,25,30,40,50,100])
print(train.groupby('TyreLife_bin', observed=False)['PitNextLap'].agg(['count', 'mean']))

print("\n--- POSITION BINS ---")
train['Position_bin'] = pd.cut(train['Position'], bins=[0,1,3,5,10,15,20,50])
print(train.groupby('Position_bin', observed=False)['PitNextLap'].agg(['count', 'mean']))

print("\n--- LAPNUMBER BINS ---")
print("First 5 laps:", train[train['LapNumber']<=5]['PitNextLap'].mean())
print("Last 5 laps:", train[train['LapNumber']>=train['LapNumber'].max()-5]['PitNextLap'].mean())

print("\n--- DRIVER TARGET RATE VARIATION ---")
driver_rates = train.groupby('Driver')['PitNextLap'].agg(['count', 'mean']).sort_values('count', ascending=False)
print(f"Min rate: {driver_rates['mean'].min():.4f}, Max rate: {driver_rates['mean'].max():.4f}, Std: {driver_rates['mean'].std():.4f}")

print("\n--- RACE TARGET RATE VARIATION ---")
race_rates = train.groupby('Race')['PitNextLap'].agg(['count', 'mean']).sort_values('count', ascending=False)
print(f"Min rate: {race_rates['mean'].min():.4f}, Max rate: {race_rates['mean'].max():.4f}, Std: {race_rates['mean'].std():.4f}")

print("\n--- INTERACTION QUICK CHECK ---")
print("Medium compound + TireLife>15:", train[(train['Compound']=='MEDIUM') & (train['TyreLife']>15)]['PitNextLap'].mean())
print("Soft compound + TireLife>10:", train[(train['Compound']=='SOFT') & (train['TyreLife']>10)]['PitNextLap'].mean())
print("Hard compound + TireLife>25:", train[(train['Compound']=='HARD') & (train['TyreLife']>25)]['PitNextLap'].mean())

print("\n--- STINT ANALYSIS ---")
print(train.groupby('Stint')['PitNextLap'].agg(['count', 'mean']).sort_values('mean', ascending=False))

print("\n--- TEST VS TRAIN FEATURE DISTRIBUTION ---")
# Check if test has same categorical values
for c in cat_cols:
    train_vals = set(train[c].unique())
    test_vals = set(test[c].unique())
    only_test = test_vals - train_vals
    only_train = train_vals - test_vals
    print(f"{c}: only in train={len(only_train)}, only in test={len(only_test)}")
    if len(only_test) > 0:
        print(f"  Test-only values: {only_test}")
    if len(only_train) > 0:
        print(f"  Train-only values: {only_train}")

print("\n--- PRE-SEASON TESTING ANALYSIS ---")
pre = train[train['Race']=='Pre-Season Testing']
print(f"Pre-Season: count={len(pre)}, pit_rate={pre['PitNextLap'].mean():.4f}")
print(f"Non-Pre-Season pit_rate={train[train['Race']!='Pre-Season Testing']['PitNextLap'].mean():.4f}")

print("\nEDA COMPLETE")
# -*- coding: utf-8 -*-
"""B题 问题三: 基于早期循环数据的寿命预测模型
特征: 早期SOH斜率/波动/潜伏时间/充电时间变化 + 策略参数
模型: Ridge / RandomForest / GradientBoosting, 5折重复交叉验证
评估: 不同早期窗口长度 k 的预测误差 (保存 q3_predictions.csv/q3_results.npy/q3_traj.npy)
图表(fig6--fig9)由 make_figures.py 统一生成。
"""
import pandas as pd, numpy as np, os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

BASE = r"C:\Users\one\Desktop\2026年校赛题目\B"

cy = pd.read_csv(os.path.join(BASE, 'data_processed', '每循环明细表_124.csv'))
bt = pd.read_csv(os.path.join(BASE, 'data_processed', 'battery_table_124.csv'))
bt['cycle_life'] = pd.to_numeric(bt['cycle_life'], errors='coerce')
bt = bt.dropna(subset=['cycle_life']).reset_index(drop=True)
# 充电时间异常清洗 (剔除 >40 min 的脏点)
cy = cy[cy['chargetime_min'] < 40].copy()

# ============ 特征提取 ============
WINDOWS = [5, 10, 20, 50, 100]
THRESH = [99.9, 99.8, 99.7, 99.5, 99.3, 99.0, 98.5, 98.0, 97.0, 96.0]

def extract_features(batt, k):
    d = cy[cy['battery'] == batt].sort_values('cycle').head(k)
    if len(d) < max(3, min(k, 5)):
        return None
    soh = d['SOH_pct'].values.astype(float)
    qd = d['Qd_Ah'].values.astype(float)
    ct = d['chargetime_min'].values.astype(float)
    x = np.arange(1, len(soh) + 1)
    # 线性拟合
    slope, intercept, r, p, se = stats.linregress(x, soh)
    resid = soh - (intercept + slope * x)
    cslope, cint, *_ = stats.linregress(x, ct)
    ctdiff = np.diff(ct)
    feats = {
        'soh_k': soh[-1], 'loss_k': 100 - soh[-1],
        'slope': slope, 'slope_r2': r ** 2,
        'soh_resid_std': np.std(resid), 'soh_firstdiff_std': np.std(np.diff(soh)),
        'qd_std': np.std(qd), 'qd_firstdiff_std': np.std(np.diff(qd)),
        'qd_slope': stats.linregress(x, qd).slope,
        'ct_first': ct[0], 'ct_k': ct[-1], 'ct_slope': cslope,
        'ct_diff_std': np.std(ctdiff), 'ct_drift': ct[-1] - ct[0],
    }
    # 潜伏时间: SOH 首次低于阈值的循环数 (窗口内未达到则记 k+0.5)
    for t in THRESH:
        below = d[d['SOH_pct'] < t]
        feats[f'lat_{t}'] = below['cycle'].min() if len(below) else (k + 0.5)
    return feats

# 提取所有电池特征
recs = []
for batt in bt['battery']:
    for k in WINDOWS:
        f = extract_features(batt, k)
        if f is None:
            continue
        f['battery'] = batt
        f['window'] = k
        recs.append(f)
F = pd.DataFrame(recs)
# 合并目标与策略参数
F = F.merge(bt[['battery', 'cycle_life', 'C1_C', 'Q1_pct', 'C2_C', 'avg_chargetime_min', 'batch']],
            on='battery', how='left')
print(f"特征提取完成: {len(F)} 条 (电池x窗口)")

# ============ 建模评估 ============
DEG_FEATS = ['soh_k', 'loss_k', 'slope', 'slope_r2', 'soh_resid_std', 'soh_firstdiff_std',
             'qd_std', 'qd_firstdiff_std', 'qd_slope',
             'ct_first', 'ct_k', 'ct_slope', 'ct_diff_std', 'ct_drift'] + [f'lat_{t}' for t in THRESH]
STRAT_FEATS = ['C1_C', 'Q1_pct', 'C2_C', 'avg_chargetime_min']
ALL_FEATS = DEG_FEATS + STRAT_FEATS

def make_model(name):
    if name == 'Ridge':
        return Ridge(alpha=1.0)
    if name == 'RandomForest':
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    return GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05, random_state=42)

def cv_evaluate(feat_cols, label, model_name, k, seed=42):
    sub = F[(F['window'] == k) & (F[feat_cols].notna().all(axis=1))].reset_index(drop=True)
    X = sub[feat_cols].values
    yl = np.log10(sub['cycle_life'].values)
    rkf = RepeatedKFold(n_splits=5, n_repeats=8, random_state=seed)
    preds, trues = [], []
    for tr, te in rkf.split(X):
        m = make_model(model_name)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        m.fit(Xtr, yl[tr])
        preds.append(m.predict(Xte))
        trues.append(yl[te])
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    life_p = 10 ** preds; life_t = 10 ** trues
    mape = np.mean(np.abs(life_p - life_t) / life_t) * 100
    r2 = r2_score(trues, preds)
    return dict(mape=mape, r2=r2)

print("\n" + "=" * 60)
print("一、不同早期窗口长度下的预测误差 (5折×8次重复CV)")
print("=" * 60)
print(f"{'窗口k':>6}{'模型':<14}{'MAPE(%)':>10}{'R²(log10)':>12}")
summary = {}
for k in WINDOWS:
    best = None
    for mn in ['Ridge', 'RandomForest', 'GradientBoosting']:
        r = cv_evaluate(ALL_FEATS, 'all', mn, k)
        summary[(k, mn)] = r
        print(f"{k:>6}{mn:<14}{r['mape']:>10.1f}{r['r2']:>12.3f}")
        if best is None or r['mape'] < best[1]:
            best = (mn, r['mape'], r['r2'])
    print(f"      最优: {best[0]}  MAPE={best[1]:.1f}%  R²={best[2]:.3f}")
    print("-" * 50)

# ============ 特征组合消融 (k=100) ============
print("\n" + "=" * 60)
print("二、特征组合消融 (k=100, GradientBoosting)")
print("=" * 60)
for label, feats in [('仅早期衰减特征', DEG_FEATS),
                     ('仅策略参数', STRAT_FEATS),
                     ('早期特征+策略参数', ALL_FEATS)]:
    r = cv_evaluate(feats, label, 'GradientBoosting', 100)
    print(f"  {label:<16}: MAPE={r['mape']:.1f}%  R²={r['r2']:.3f}")

# ============ 最佳模型: 拟合与特征重要性 (k=100) ============
sub = F[(F['window'] == 100) & (F[ALL_FEATS].notna().all(axis=1))].reset_index(drop=True)
X = sub[ALL_FEATS].values
yl = np.log10(sub['cycle_life'].values)
g = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05, random_state=42)
g.fit(X, yl)
pred = g.predict(X)
imp = pd.Series(g.feature_importances_, index=ALL_FEATS).sort_values(ascending=False)
print("\n" + "=" * 60)
print("三、特征重要性 (GradientBoosting, k=100, 全特征拟合)")
print("=" * 60)
print(imp.head(15).to_string())
print(f"\n训练R²={r2_score(yl, pred):.3f}")

# 保存预测表 (对每个电池在k=100的留一/全拟合预测)
sub['pred_life'] = 10 ** pred
sub.to_csv(os.path.join(BASE, 'data_processed', 'q3_predictions.csv'), index=False, encoding='utf-8-sig')
# 保存结果
res = dict(summary={f'{k}_{m}': v for (k, m), v in summary.items()},
           imp=imp.to_dict(), feats=ALL_FEATS, windows=WINDOWS)
np.save(os.path.join(BASE, 'data_processed', 'q3_results.npy'), res, allow_pickle=True)

print("\n分析完成: 图6/图7/图8 由 make_figures.py 统一生成")

# ================= 四、SOH 未来轨迹预测 (里程碑循环) =================
print("\n" + "=" * 60)
print("四、早期数据预测未来 SOH (k=20 特征, 里程碑循环 t=200/400/600/800/1000)")
print("=" * 60)
MILESTONES = [200, 400, 600, 800, 1000]
# 每个电池的 SOH@t (取最接近 t 的循环)
soh_at = {}
for batt in bt['battery']:
    d = cy[cy['battery'] == batt].sort_values('cycle')
    vals = {}
    for t in MILESTONES:
        near = d.iloc[(d['cycle'] - t).abs().argmin()]
        if abs(near['cycle'] - t) <= 5:
            vals[t] = near['SOH_pct']
        else:
            vals[t] = np.nan
    soh_at[batt] = vals
Fs = F[F['window'] == 20].reset_index(drop=True)
traj_res = {}
for t in MILESTONES:
    Fs[f'soh_{t}'] = Fs['battery'].map(lambda b: soh_at[b][t])
    tr = Fs.dropna(subset=[f'soh_{t}'] + ALL_FEATS).reset_index(drop=True)
    if len(tr) < 15:
        continue
    X = tr[ALL_FEATS].values
    y = tr[f'soh_{t}'].values
    rkf = RepeatedKFold(n_splits=5, n_repeats=8, random_state=7)
    oof = np.zeros(len(tr))
    for tt, te in rkf.split(X):
        m = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42)
        m.fit(X[tt], y[tt]); oof[te] = m.predict(X[te])
    mae = mean_absolute_error(y, oof)
    r2s = r2_score(y, oof)
    traj_res[t] = dict(mae=mae, r2=r2s, n=len(tr))
    print(f"  里程碑 t={t}:  R²={r2s:.3f}  MAE={mae:.2f} pct  (n={len(tr)})")

np.save(os.path.join(BASE, 'data_processed', 'q3_traj.npy'), traj_res, allow_pickle=True)
print("\n分析完成: 图9 由 make_figures.py 统一生成")

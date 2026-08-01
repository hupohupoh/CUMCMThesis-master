# -*- coding: utf-8 -*-
"""B题 问题三: 基于早期循环数据的寿命预测模型
特征: 早期SOH斜率/波动/潜伏时间/充电时间变化 + 策略参数
模型: Ridge / RandomForest / GradientBoosting, 5折重复交叉验证
评估: 不同早期窗口长度 k 的预测误差
"""
import pandas as pd, numpy as np, os, sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r"C:\Users\one\Desktop\2026年校赛题目\B"
FIG = os.path.join(BASE, 'data_processed', 'figs')
os.makedirs(FIG, exist_ok=True)

cy = pd.read_csv(os.path.join(BASE, 'data_processed', '每循环明细表.csv'))
bt = pd.read_csv(os.path.join(BASE, 'data_processed', 'battery_table.csv'))
bt['cycle_life'] = pd.to_numeric(bt['cycle_life'], errors='coerce')
bt = bt.dropna(subset=['cycle_life']).reset_index(drop=True)
# 充电时间异常清洗 (剔除 >40 min 的脏点)
cy = cy[cy['chargetime_h'] < 40].copy()

# ============ 特征提取 ============
WINDOWS = [5, 10, 20, 50, 100]
THRESH = [99.9, 99.8, 99.7, 99.5, 99.3, 99.0, 98.5, 98.0, 97.0, 96.0]

def extract_features(batt, k):
    d = cy[cy['battery'] == batt].sort_values('cycle').head(k)
    if len(d) < max(3, min(k, 5)):
        return None
    soh = d['SOH_pct'].values.astype(float)
    qd = d['Qd_Ah'].values.astype(float)
    ct = d['chargetime_h'].values.astype(float)
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

# ================= 图6: 预测 vs 实际 (k=100, GBM, 交叉验证预测) =================
# 用 RepeatedKFold 收集 OOF 预测
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
oof = np.zeros(len(sub))
for tr, te in rkf.split(X):
    m = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05, random_state=42)
    m.fit(X[tr], yl[tr])
    oof[te] = m.predict(X[te])
life_t = 10 ** yl; life_p = 10 ** oof
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.scatter(life_t, life_p, c=sub['batch'].map({'data_1': 'steelblue', 'data_2': 'darkorange', 'data_3': 'green'}), s=35, alpha=.75)
lim = [100, 2200]
ax.plot(lim, lim, 'k--', lw=1)
ax.plot(lim, [x * 0.9 for x in lim], 'k:', lw=.8); ax.plot(lim, [x * 1.1 for x in lim], 'k:', lw=.8)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel('实际循环寿命'); ax.set_ylabel('预测循环寿命')
mape_cv = np.mean(np.abs(life_p - life_t) / life_t) * 100
r2_cv = r2_score(yl, oof)
ax.set_title(f'早期100循环预测寿命 (GBM, 交叉验证)\nMAPE={mape_cv:.1f}%, R²={r2_cv:.3f}, n={len(sub)}')
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=c, markersize=8, label=b)
           for b, c in [('data_1', 'steelblue'), ('data_2', 'darkorange'), ('data_3', 'green')]]
ax.legend(handles=handles, loc='upper left', fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig6_寿命预测对比.png'), dpi=120); plt.close()

# ================= 图7: MAPE vs 早期窗口长度 =================
fig, ax = plt.subplots(figsize=(7.5, 5))
for mn, c, mk in [('Ridge', 'steelblue', 'o'), ('RandomForest', 'darkorange', 's'), ('GradientBoosting', 'green', '^')]:
    vals = [summary[(k, mn)]['mape'] for k in WINDOWS]
    ax.plot(WINDOWS, vals, marker=mk, color=c, lw=1.8, label=mn)
ax.set_xlabel('早期循环窗口长度 k (个循环)'); ax.set_ylabel('寿命预测 MAPE (%)')
ax.set_xticks(WINDOWS); ax.grid(alpha=.3); ax.legend()
ax.set_title('预测误差随早期数据长度变化 (5折×8重复交叉验证)')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig7_MAPE_vs窗口.png'), dpi=120); plt.close()

# ================= 图8: 特征重要性 =================
fig, ax = plt.subplots(figsize=(7.5, 6))
imp_top = imp.head(12)[::-1]
ax.barh(imp_top.index, imp_top.values, color='steelblue')
ax.set_xlabel('特征重要性'); ax.set_title('寿命预测特征重要性 (k=100, GradientBoosting)')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig8_特征重要性.png'), dpi=120); plt.close()

print(f"\n图表: fig6_寿命预测对比.png, fig7_MAPE_vs窗口.png, fig8_特征重要性.png")

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

# ================= 图9: 预测SOH vs 实际SOH 轨迹 =================
# 挑选 3 个代表性电池 (长/中/短寿命), 展示早期20循环数据 + 里程碑预测 + 实际轨迹
sample = ['data_1_cell00', 'data_2_cell09', 'data_2_cell01']  # 长/中/短
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, batt in zip(axes, sample):
    d = cy[cy['battery'] == batt].sort_values('cycle')
    cl = bt[bt['battery'] == batt]['cycle_life'].values[0]
    ax.plot(d['cycle'], d['SOH_pct'], color='lightgray', lw=1.2, label='实际SOH轨迹')
    d20 = cy[cy['battery'] == batt].sort_values('cycle').head(20)
    ax.plot(d20['cycle'], d20['SOH_pct'], 'b-', lw=2, label='前20循环(训练用)')
    # 里程碑真实 SOH (预测目标)
    for t in MILESTONES:
        v = soh_at[batt][t]
        if np.isfinite(v):
            ax.plot(t, v, 'ro', ms=6)
    ax.axhline(80, color='red', ls='--', alpha=.5)
    ax.set_title(f'{batt}\ncycle_life={int(cl)}')
    ax.set_xlabel('循环数')
axes[0].set_ylabel('SOH (%)')
handles, labels = axes[0].get_legend_handles_labels()
axes[2].legend(handles, labels, fontsize=8, loc='lower left')
plt.suptitle('实际 SOH 轨迹与前 20 循环数据 (用于早期预测)', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig9_轨迹预测演示.png'), dpi=120); plt.close()

np.save(os.path.join(BASE, 'data_processed', 'q3_traj.npy'), traj_res, allow_pickle=True)
print("\n图9: fig9_轨迹预测演示.png")

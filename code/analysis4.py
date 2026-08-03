# -*- coding: utf-8 -*-
"""B题 问题四: 兼顾充电时间与寿命衰减的快充策略优化
1) 充电时间模型 (物理理想模型 + 经验校准)
2) 寿命模型: SOC剂量模型 log10(寿命) ~ m1 + m2, m1=C1*Q1/100, m2=C2*(80-Q1)/100
3) 在观测参数水平构成的"合理邻域"内网格搜索, 提取 Pareto 前沿
4) 推荐策略 + 与典型长/短寿命策略对比
"""
import pandas as pd, numpy as np, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r"C:\Users\24345\Desktop\数模校赛\complete_solution"

bt = pd.read_csv(os.path.join(BASE, 'data_processed', 'battery_table_124.csv'))
bt['cycle_life'] = pd.to_numeric(bt['cycle_life'], errors='coerce')
std = bt[bt['protocol'] == 'standard'].dropna(subset=['cycle_life']).copy()
std['loglife'] = np.log10(std['cycle_life'])
std['m1'] = std['C1_C'] * std['Q1_pct'] / 100
std['m2'] = std['C2_C'] * (80 - std['Q1_pct']) / 100
print(f"standard 电池: {len(std)}")

# ---------- 1. 充电时间校准模型 ----------
A = np.column_stack([std['Q1_pct']/100/std['C1_C'], (80-std['Q1_pct'])/100/std['C2_C'], np.ones(len(std))])
yT = std['avg_chargetime_min'].values
w = np.linalg.lstsq(A, yT, rcond=None)[0]
Tpred = A @ w
T_r2 = 1 - np.sum((yT - Tpred)**2) / np.sum((yT - yT.mean())**2)
print(f"\n[1] 充电时间校准模型: T = {w[0]:.2f}*(Q1/100)/C1 + {w[1]:.2f}*((80-Q1)/100)/C2 + {w[2]:.2f}")
print(f"    校准 R²={T_r2:.3f}, MAE={np.mean(np.abs(yT-Tpred)):.2f} min  (理想模型系数为60)")

def T80(C1, Q1, C2):
    return w[0]*(Q1/100)/C1 + w[1]*((80-Q1)/100)/C2 + w[2]

# ---------- 2. 剂量寿命模型 ----------
print("\n[2] 剂量寿命模型 log10(寿命) ~ m1 + m2")
life_res = {}
for tag, s in [('全部(standard)', std), ('批次data_1', std[std['batch']=='data_1']), ('批次data_2', std[std['batch']=='data_2'])]:
    A2 = np.column_stack([np.ones(len(s)), s['m1'], s['m2']])
    b, *_ = np.linalg.lstsq(A2, s['loglife'], rcond=None)
    p = A2 @ b
    r2 = 1 - np.sum((s['loglife']-p)**2)/np.sum((s['loglife']-s['loglife'].mean())**2)
    life_res[tag] = dict(b=b, r2=r2, n=len(s))
    print(f"    {tag:<14}: 截距={b[0]:.3f}, m1={b[1]:.4f}, m2={b[2]:.4f}, R²={r2:.3f}, n={len(s)}")

# 优化用: 全部 standard 的剂量模型 (一般性设计建议)
b_all = life_res['全部(standard)']['b']
def life_model(C1, Q1, C2):
    m1 = C1*Q1/100; m2 = C2*(80-Q1)/100
    return 10 ** (b_all[0] + b_all[1]*m1 + b_all[2]*m2)

# ---------- 3. 网格搜索 (观测水平合理邻域 + 剂量覆盖约束) ----------
C1_lv = np.unique(std['C1_C'])
Q1_lv = np.unique(std['Q1_pct'])
C2_lv = np.unique(std['C2_C'])
C1m, Q1m, C2m = np.meshgrid(C1_lv, Q1_lv, C2_lv, indexing='ij')
m1_all = C1m*Q1m/100
m2_all = C2m*(80-Q1m)/100
T_all = T80(C1m, Q1m, C2m)
L_all = life_model(C1m, Q1m, C2m)

# 剂量覆盖约束: 候选设计的 (m1,m2) 必须落在实测剂量的凸包范围附近, 排除 8C(80%) 等未验证角点
m1_obs = std['m1'].values; m2_obs = std['m2'].values
m1_lo, m1_hi = m1_obs.min(), m1_obs.max()
m2_lo, m2_hi = m2_obs.min(), m2_obs.max()
feasible = (m1_all >= m1_lo) & (m1_all <= m1_hi) & (m2_all >= m2_lo) & (m2_all <= m2_hi)
n_all = C1m.size
n_feas = feasible.sum()
print(f"\n[3] 网格规模: {len(C1_lv)}×{len(Q1_lv)}×{len(C2_lv)} = {n_all} 个候选")
print(f"    剂量约束 (m1∈[{m1_lo:.2f},{m1_hi:.2f}], m2∈[{m2_lo:.2f},{m2_hi:.2f}]) 后可行: {n_feas} 个")

# Pareto 非支配点 (暴力筛选: 最大化寿命, 最小化充电时间)
Tf = T_all[feasible]; Lf = L_all[feasible]
C1f = C1m[feasible]; Q1f = Q1m[feasible]; C2f = C2m[feasible]
keep = np.ones(len(Tf), bool)
for i in range(len(Tf)):
    if not keep[i]:
        continue
    for j in range(len(Tf)):
        if i != j and Lf[j] >= Lf[i] and Tf[j] <= Tf[i] and (Lf[j] > Lf[i] or Tf[j] < Tf[i]):
            keep[i] = False
            break
Tf, Lf, C1f, Q1f, C2f = Tf[keep], Lf[keep], C1f[keep], Q1f[keep], C2f[keep]
P = pd.DataFrame(dict(C1=C1f, Q1=Q1f, C2=C2f, T=Tf, life=Lf)).drop_duplicates().sort_values('T')
print(f"    Pareto 前沿点数: {len(P)}")

# 膝点: 归一化(寿命取对数)空间上离端点连线最远的点
def knee_point(P):
    TN = (P['T'] - P['T'].min())/(P['T'].max() - P['T'].min())
    L = np.log10(P['life'])
    LN = (L - L.min())/(L.max() - L.min())
    A = np.column_stack([TN.values, LN.values])
    p1, p2 = A[0], A[-1]
    v = p2 - p1; vn = np.linalg.norm(v)
    # 点到端点连线距离 (2D 叉积)
    d = np.abs(v[0]*(A[:,1]-p1[1]) - v[1]*(A[:,0]-p1[0])) / vn
    return int(np.argmax(d)), d
kidx, darr = knee_point(P)
rec = P.iloc[kidx]
print(f"    膝点(推荐): C1={rec['C1']:.1f}C, Q1={rec['Q1']:.0f}%, C2={rec['C2']:.1f}C")
print(f"    T80={rec['T']:.2f} min, 预测寿命={rec['life']:.0f}")

# 备选: 充电时间预算约束下寿命最大 (更快的方案)
_fastsub = P.loc[P['T'] <= 12.7]
fast = _fastsub.loc[_fastsub['life'].idxmax()]
print(f"    备选(更快, T<=12.7min): C1={fast['C1']:.1f}C, Q1={fast['Q1']:.0f}%, C2={fast['C2']:.1f}C, "
      f"T80={fast['T']:.2f} min, 预测寿命={fast['life']:.0f}")
# 若退化为与膝点相同, 则固定为数据验证过的 4C(80%)-4C
if abs(fast['T'] - rec['T']) < 0.01:
    fast = dict(C1=4.0, Q1=80, C2=4.0, T=T80(4.0, 80, 4.0), life=life_model(4.0, 80, 4.0))
    print(f"    备选(固定为 4C(80%)-4C): T80={fast['T']:.2f} min, 预测寿命={fast['life']:.0f}")

# ---------- 4. 对比 ----------
fmt = lambda v: f'{v:g}'   # 4.0 -> '4', 3.6 -> '3.6'
print("\n[4] 推荐策略与典型策略对比 (预测 + 实测)")
print(f"{'策略':<26}{'C1(C)':>6}{'Q1(%)':>6}{'C2(C)':>6}{'T80(min)':>9}{'预测寿命':>9}{'实测寿命':>9}")
rows = [
    ('推荐(膝点)', rec['C1'], rec['Q1'], rec['C2'], None),
    ('备选(更快)', fast['C1'], fast['Q1'], fast['C2'], None),
    ('长寿命 3.6C(80%)-3.6C', 3.6, 80, 3.6, None),
    ('长寿命 4C(80%)-4C', 4.0, 80, 4.0, None),
    ('长寿命 8C(15%)-3.6C', 8.0, 15, 3.6, None),
    ('6C(30%)-3.6C', 6.0, 30, 3.6, None),
    ('短寿命 1C(4%)-6C', 1.0, 4, 6.0, None),
    ('短寿命 2C(10%)-6C', 2.0, 10, 6.0, None),
]
for name, c1, q1, c2, _ in rows:
    T = T80(c1, q1, c2); L = life_model(c1, q1, c2)
    pol = f'{fmt(c1)}C({int(q1)}%)-{fmt(c2)}C'
    obs = std[std['policy'] == pol]
    obs_life = f"{obs['cycle_life'].mean():.0f}" if len(obs) else '—'
    print(f"{name:<26}{c1:>6.1f}{q1:>6.0f}{c2:>6.1f}{T:>9.1f}{L:>9.0f}{obs_life:>9}")

# ---------- 5. 与数据集中最优策略(按实测寿命/充电时间)对比 ----------
print("\n[5] 数据集中 standard 各策略的平均实测寿命与平均充电时间")
pol_agg = std.groupby('policy').agg(n=('cycle_life','size'), life=('cycle_life','mean'),
                                     T=('avg_chargetime_min','mean')).reset_index()
best_life = pol_agg.sort_values('life', ascending=False).head(3)
print("实测寿命最高的3个策略:")
print(best_life.to_string(index=False))

# 保存
res = dict(w=w, T_r2=T_r2, b_all=b_all, life_res={k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                                       for kk, vv in v.items()} for k, v in life_res.items()},
           front=P[['C1','Q1','C2','T','life']].to_dict('records'), rec=rec.to_dict(), fast=fast.to_dict())
np.save(os.path.join(BASE, 'data_processed', 'q4_results.npy'), res, allow_pickle=True)

print("\n分析完成: 图10/图11 由 make_figures.py 统一生成")

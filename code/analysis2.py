# -*- coding: utf-8 -*-
"""B题 问题二: 充电策略参数与寿命衰减的定量关系模型
输出: 回归系数/标准化系数/p值/重要性排序/交互分析 + 图4/图5
"""
import pandas as pd, numpy as np, os, math, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r"C:\Users\one\Desktop\2026年校赛题目\B"
FIG = os.path.join(BASE, 'data_processed', 'figs')
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, 'data_processed', 'battery_table.csv'))
df['cycle_life'] = pd.to_numeric(df['cycle_life'], errors='coerce')
std = df[df['protocol'] == 'standard'].dropna(subset=['cycle_life']).copy()
std['loglife'] = np.log10(std['cycle_life'])
std['T'] = std['avg_chargetime_min']
std['batch2'] = (std['batch'] == 'data_2').astype(float)
print(f"standard 电池数: {len(std)}")

# ---------- 工具: 最小二乘回归 + 显著性 ----------
def ols(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, p = X.shape
    Xd = np.column_stack([np.ones(n), X])
    coef, res, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ coef
    resid = y - yhat
    dof = n - (p + 1)
    sse = np.sum(resid ** 2)
    s2 = sse / dof
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    se = np.sqrt(np.diag(XtX_inv) * s2)
    t = coef / se
    pv = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    r2 = 1 - sse / np.sum((y - y.mean()) ** 2)
    r2adj = 1 - (1 - r2) * (n - 1) / dof
    return dict(coef=coef, se=se, t=t, pv=pv, r2=r2, r2adj=r2adj, n=n,
                rmse=math.sqrt(sse / n))

def show(name, R, varnames):
    print(f"\n### {name}  (n={R['n']}, R^2={R['r2']:.4f}, adjR^2={R['r2adj']:.4f}, RMSE={R['rmse']:.4f})")
    print(f"{'变量':<16}{'系数':>10}{'标准误':>10}{'t':>8}{'p值':>10}")
    for i, v in enumerate(varnames):
        print(f"{v:<16}{R['coef'][i+1]:>10.4f}{R['se'][i+1]:>10.4f}{R['t'][i+1]:>8.2f}{R['pv'][i+1]:>10.4f}")
    print(f"{'(截距)':<16}{R['coef'][0]:>10.4f}")

# ---------- 1. 标准化回归(消除量纲, 含批次效应) ----------
X_names = ['C1_C', 'Q1_pct', 'C2_C', 'T', 'batch2']
X = std[X_names].values
y = std['loglife'].values
Xs = (X - X.mean(axis=0)) / X.std(axis=0)
R = ols(Xs, y)
print("=" * 60); print("一、多变量回归 log10(cycle_life) ~ C1 + Q1 + C2 + 充电时间 + 批次 (标准化)") ; print("=" * 60)
show("标准化回归 (含批次效应)", R, X_names)
std_beta = dict(zip(X_names[:4], R['coef'][1:5]))
rank = sorted(std_beta.items(), key=lambda kv: -abs(kv[1]))
print("\n策略参数标准化系数排序(按绝对值, 批次效应已吸收):")
for v, b in rank:
    print(f"  {v:<12} beta={b:+.3f}")
print(f"\n批次2效应: 系数={R['coef'][5]:+.3f}, p={R['pv'][5]:.4f}  (负值表示批次2整体寿命偏低)")

# ---------- 2. 原始尺度回归 (仅策略参数, 供论文引用系数) ----------
Xr_names = ['C1_C', 'Q1_pct', 'C2_C', 'T']
Xr = std[Xr_names].values
Rr = ols(Xr, y)
show("原始尺度回归(仅策略参数)", Rr, Xr_names)

# ---------- 3. Drop-one R2 相对重要性 (含批次) ----------
print("\n" + "=" * 60); print("二、留一变量 R² 变化 (相对重要性, 含批次效应)"); print("=" * 60)
base_r2 = R['r2']
print(f"全模型 R^2 = {base_r2:.4f}")
imp = {}
for i, v in enumerate(X_names):
    Xd = np.delete(X, i, axis=1)
    r2i = ols(Xd, y)['r2']
    imp[v] = base_r2 - r2i
    print(f"  去掉 {v:<12}: R^2={r2i:.4f}  ΔR^2={imp[v]:+.4f}")
print("重要性排序:", sorted(imp.items(), key=lambda kv: -kv[1]))

# ---------- 4. 偏相关系数 ----------
print("\n" + "=" * 60); print("三、偏相关系数 (控制其他变量后, 含批次)"); print("=" * 60)
def pcorr(X, y, idx):
    others = [j for j in range(X.shape[1]) if j != idx]
    Xo = np.column_stack([np.ones(len(y)), X[:, others]])
    c = np.linalg.lstsq(Xo, y, rcond=None)[0]
    ry = y - Xo @ c
    c2 = np.linalg.lstsq(Xo, X[:, idx], rcond=None)[0]
    rx = X[:, idx] - Xo @ c2
    r = np.corrcoef(rx, ry)[0, 1]
    return r
X_all = std[X_names].values
for i, v in enumerate(X_names):
    print(f"  偏相关 {v:<12}: r = {pcorr(X_all, y, i):+.3f}")

# ---------- 5. 不同 SOC 区间高倍率影响的差异: 剂量模型 ----------
print("\n" + "=" * 60); print("四、SOC 区间剂量模型 (验证不同区间高倍率影响差异)"); print("=" * 60)
std['m1'] = std['C1_C'] * std['Q1_pct'] / 100   # 低SOC区高倍率电量剂量 (Ah@C1)
std['m2'] = std['C2_C'] * (80 - std['Q1_pct']) / 100  # 中高SOC区高倍率电量剂量 (Ah@C2)
X2 = std[['m1', 'm2']].values
R2m = ols(X2, std['loglife'].values)
show("log10(寿命) ~ m1(低SOC剂量) + m2(中高SOC剂量)", R2m, ['m1', 'm2'])
# 剂量标准化
X2s = (X2 - X2.mean(axis=0)) / X2.std(axis=0)
R2ms = ols(X2s, std['loglife'].values)
show("剂量模型(标准化)", R2ms, ['m1', 'm2'])
print("若 beta(m2) << beta(m1) 且 m2 显著为负, 说明中高SOC区高倍率充电更伤电池")

# ---------- 6. 批次内一致性 ----------
print("\n" + "=" * 60); print("五、分批次回归(稳健性, 批次内用 C1/Q1/C2)"); print("=" * 60)
within_names = ['C1_C', 'Q1_pct', 'C2_C']
for b in ['data_1', 'data_2']:
    sub = std[std['batch'] == b]
    if len(sub) < 10:
        continue
    Xb = sub[within_names].values
    yb = sub['loglife'].values
    Xbs = (Xb - Xb.mean(axis=0)) / Xb.std(axis=0)
    Rb = ols(Xbs, yb)
    show(f"批次 {b} (n={len(sub)}, 标准化)", Rb, within_names)
    print("  批次内 C2 标准化系数:", f"{Rb['coef'][3]:+.3f}")

# ---------- 7. 交互项 ----------
print("\n" + "=" * 60); print("六、交互项检验"); print("=" * 60)
std['C1xQ1'] = std['C1_C'] * std['Q1_pct']
X3 = std[['C1_C', 'Q1_pct', 'C2_C', 'T', 'C1xQ1']].values
R3 = ols(X3, std['loglife'].values)
show("加 C1×Q1 交互", R3, ['C1', 'Q1', 'C2', 'T', 'C1×Q1'])

std['C2xQ1'] = std['C2_C'] * std['Q1_pct']
X4 = std[['C1_C', 'Q1_pct', 'C2_C', 'T', 'C1xQ1', 'C2xQ1']].values
R4 = ols(X4, std['loglife'].values)
show("加 C1×Q1 与 C2×Q1 交互", R4, ['C1', 'Q1', 'C2', 'T', 'C1×Q1', 'C2×Q1'])

# ---------- 7.5 剂量+批次联合模型 ----------
print("\n" + "=" * 60); print("六b、剂量+批次联合模型 (log10(寿命) ~ m1 + m2 + batch2)"); print("=" * 60)
Xdm = std[['m1', 'm2', 'batch2']].values
Xdms = (Xdm - Xdm.mean(axis=0)) / Xdm.std(axis=0)
Rdm = ols(Xdms, std['loglife'].values)
show("剂量+批次联合模型(标准化)", Rdm, ['m1(低SOC剂量)', 'm2(中高SOC剂量)', 'batch2'])
print(f"  m2 与 m1 标准化系数比值: {Rdm['coef'][2]/Rdm['coef'][1]:.2f}")

# ---------- 8. 输出参数供论文引用 ----------
res = dict(
    beta=std_beta, r2_full=R['r2'], imp=imp,
    m2_beta=R2ms['coef'][2], m1_beta=R2ms['coef'][1],
    m2_beta_joint=Rdm['coef'][2], m1_beta_joint=Rdm['coef'][1],
    r2_m2=R2ms['r2'], m2_pv=R2ms['pv'][2],
    R_full=dict(R), Rr=dict(Rr), Rdm=dict(Rdm),
    within1=dict(ols((std[std['batch']=='data_1'][['C1_C','Q1_pct','C2_C']].values -
                     std[std['batch']=='data_1'][['C1_C','Q1_pct','C2_C']].values.mean(0)) /
                    std[std['batch']=='data_1'][['C1_C','Q1_pct','C2_C']].values.std(0),
                    std[std['batch']=='data_1']['loglife'].values)),
    batch_eff=R['coef'][5], batch_eff_pv=R['pv'][5],
)
np.save(os.path.join(BASE, 'data_processed', 'q2_results.npy'), res, allow_pickle=True)

# ================= 图4: 各变量偏回归图 (控制其他参数与批次后) =================
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax, v in zip(axes, Xr_names):
    i = Xr_names.index(v)
    # 控制其他策略参数 + 批次后的偏回归
    others = [j for j in range(4) if j != i]
    other_cols = [Xr_names[j] for j in others] + ['batch2']
    Xo = np.column_stack([np.ones(len(y)), std[other_cols].values])
    ry = y - Xo @ np.linalg.lstsq(Xo, y, rcond=None)[0]
    rx = Xr[:, i] - Xo @ np.linalg.lstsq(Xo, Xr[:, i], rcond=None)[0]
    ax.scatter(rx, ry, s=30, alpha=.6, c='steelblue')
    z = np.polyfit(rx, ry, 1)
    xx = np.linspace(rx.min(), rx.max(), 50)
    ax.plot(xx, np.polyval(z, xx), 'r--', lw=1.2)
    ax.set_xlabel(f'{v} (残差化)', fontsize=10)
    ax.set_ylabel('log10(寿命) 残差', fontsize=10)
    # 偏相关基于4参数模型
    pr = pcorr(Xr, y, i)
    ax.set_title(f'{v}  偏相关={pr:.2f}', fontsize=10)
fig.suptitle('偏回归图: 控制其他变量与批次后, 各参数与寿命的关系 (standard, n=94)', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig4_偏回归图.png'), dpi=120); plt.close()

# ================= 图5: 预测寿命热力图 =================
# 用 C1,Q1,C2 全模型在网格上预测 loglife (充电时间用均值)
Tmean = std['T'].mean()
grid_c1 = np.linspace(1, 8, 60)
grid_q1 = np.linspace(5, 80, 60)
# 固定 C2 = 3.6 与 6.0 两张子图, 展示 Q1×C1 面上的寿命
b = Rr['coef']
C1s, Q1s, C2s, Ts = X.mean(axis=0), X.std(axis=0), X.std(axis=0), X.std(axis=0)
def pred(c1, q1, c2):
    return 10 ** (b[0] + b[1]*c1 + b[2]*q1 + b[3]*c2 + b[4]*Tmean)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, c2 in zip(axes, [3.6, 6.0]):
    Z = np.array([[pred(c1, q1, c2) for q1 in grid_q1] for c1 in grid_c1])
    im = ax.contourf(grid_c1, grid_q1, Z.T, levels=20, cmap='viridis_r')
    ax.set_xlabel('C1 (C)'); ax.set_ylabel('Q1 (%)'); ax.set_title(f'C2 = {c2}C 预测寿命')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('预测循环寿命')
    # 标记实验覆盖点
    for _, row in std[std['C2_C'] == c2].iterrows():
        ax.plot(row['C1_C'], row['Q1_pct'], 'r.', markersize=5)
fig.suptitle('策略参数空间上的寿命预测 (模型: log10(寿命) ~ C1+Q1+C2+T)', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig5_寿命热力图.png'), dpi=120); plt.close()

print(f"\n图表: fig4_偏回归图.png, fig5_寿命热力图.png")

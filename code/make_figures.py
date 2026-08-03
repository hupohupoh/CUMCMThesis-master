# -*- coding: utf-8 -*-
"""B题全部图表重绘: 统一规范风格 (dataviz 配色/细线条/简洁坐标轴/300dpi)
输出: data_processed/figs/*.png (中文名, 供 docx) + CUMCMThesis-master/figures/figN.png (ASCII, 供 LaTeX)
"""
import pandas as pd, numpy as np, os, sys, io, h5py
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle, Circle
from scipy import stats

# ================= 全局样式 (dataviz 规范) =================
plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#c3c2b7',
    'axes.linewidth': 0.9,
    'axes.grid': True,
    'grid.color': '#e1e0d9',
    'grid.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 17,
    'axes.labelsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'font.size': 16,
    'savefig.facecolor': 'white',
})
# 分类色板 (dataviz 验证通过)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
YELLOW, MAGENTA, GREEN = '#eda100', '#e87ba4', '#008300'
VIOLET, RED = '#4a3aa7', '#e34948'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'
SEQB = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95']
DPI = 220

BASE = r"C:\Users\24345\Desktop\数模校赛\complete_solution"
FIG = os.path.join(BASE, 'data_processed', 'figs')
# LaTeX 图目录: 本脚本位于 code/ 下, 相对定位到仓库根目录的 figures/ (随仓库迁移)
TMPL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(FIG, exist_ok=True)
os.makedirs(TMPL, exist_ok=True)

# 图名映射: 中文名 -> ASCII
NAME = {
    'fig1_寿命分布_按批次.png': 'fig1.png',
    'fig2_寿命vs参数.png': 'fig2.png',
    'fig3_SOH曲线对比.png': 'fig3.png',
    'fig4_偏回归图.png': 'fig4.png',
    'fig6_寿命预测对比.png': 'fig6.png',
    'fig7_MAPE_vs窗口.png': 'fig7.png',
    'fig8_特征重要性.png': 'fig8.png',
    'fig9_轨迹预测演示.png': 'fig9.png',
    'fig10_Pareto前沿.png': 'fig10.png',
    'fig11_推荐策略剖面.png': 'fig11.png',
    'fig12_相关性热力图.png': 'fig12.png',
    'fig13_充电时间模型验证.png': 'fig13.png',
    'fig14_预测误差分布.png': 'fig14.png',
    'fig15_策略对比.png': 'fig15.png',
    'fig17_充电时间分析.png': 'fig17.png',
    'fig18_回归残差诊断.png': 'fig18.png',
    'fig19_早期SOH演化.png': 'fig19.png',
    'fig20_方法流程.png': 'fig20.png',
    'fig21_批次差异.png': 'fig21.png',
    'fig22_充电时间演化.png': 'fig22.png',
    'fig23_预测流程.png': 'fig23.png',
}

def save(fig, cn):
    p1 = os.path.join(FIG, cn)
    fig.savefig(p1, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    shutil_ = __import__('shutil')
    shutil_.copy(p1, os.path.join(TMPL, NAME[cn]))
    print('  已生成', cn, '->', NAME[cn])

def style_ax(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3, colors=MUTED)

bt = pd.read_csv(os.path.join(BASE, 'data_processed', 'battery_table_124.csv'))
bt['cycle_life'] = pd.to_numeric(bt['cycle_life'], errors='coerce')
std = bt[bt['protocol'] == 'standard'].dropna(subset=['cycle_life']).copy()
std['loglife'] = np.log10(std['cycle_life'])
std['T'] = std['avg_chargetime_min']
std['m1'] = std['C1_C'] * std['Q1_pct'] / 100
std['m2'] = std['C2_C'] * (80 - std['Q1_pct']) / 100
BCOL = {'data_1': BLUE, 'data_2': ORANGE, 'data_3': AQUA}

# ================= 图1: 寿命分布 (2x2: 三批次直方图 + 箱线图) =================
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
for ax, b, c in zip(axes.flat[:3], ['data_1', 'data_2', 'data_3'], [BLUE, ORANGE, AQUA]):
    s = bt[bt['batch'] == b]['cycle_life'].dropna()
    ax.hist(s, bins=18, color=c, edgecolor='white', linewidth=0.6, alpha=0.92)
    ax.axvline(s.median(), color=INK, ls='--', lw=1.0)
    ax.text(s.median(), ax.get_ylim()[1]*0.92, f'中位数 {s.median():.0f}',
            rotation=90, va='top', ha='right', fontsize=14, color=INK)
    ax.set_xlabel('循环寿命', fontsize=15)
    style_ax(ax)
    ax.set_ylabel('电池数', fontsize=15)
# 第4格: 三批次箱线图
ax = axes[1, 1]
data = [bt[bt['batch'] == b]['cycle_life'].dropna().values for b in ['data_1', 'data_2', 'data_3']]
bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                medianprops=dict(color='white', lw=1.4),
                whiskerprops=dict(color=MUTED, lw=1), capprops=dict(color=MUTED, lw=1))
for patch, c in zip(bp['boxes'], [BLUE, ORANGE, AQUA]):
    patch.set_facecolor(c); patch.set_alpha(0.85)
ax.set_xticklabels(['data_1', 'data_2', 'data_3'], fontsize=15)
ax.set_ylabel('循环寿命', fontsize=15)
style_ax(ax)
fig.tight_layout()
save(fig, 'fig1_寿命分布_按批次.png')

# ================= 图2: cycle_life vs 参数 (2x2, 带回归线与r标注) =================
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.4))
for ax, col, lab in zip(axes.flat, ['C1_C', 'Q1_pct', 'C2_C', 'T'],
                        ['$C_1$ 第一阶段倍率 (C)', '$Q_1$ 切换 SOC (%)', '$C_2$ 第二阶段倍率 (C)', '充电时间 (min)']):
    for b in ['data_1', 'data_2']:
        sub = std[std['batch'] == b]
        ax.scatter(sub[col], sub['cycle_life'], s=24, alpha=0.72, color=BCOL[b],
                   edgecolor='white', lw=0.4, label=b)
    x = np.linspace(std[col].min(), std[col].max(), 60)
    z = np.polyfit(std[col], std['cycle_life'], 1)
    ax.plot(x, np.polyval(z, x), color=RED, lw=1.7, ls='--')
    r = np.corrcoef(std[col], std['cycle_life'])[0, 1]
    ax.set_xlabel(lab, fontsize=15); ax.set_ylabel('循环寿命', fontsize=15)
    style_ax(ax)
axes[0, 0].legend(frameon=False, fontsize=14, loc='upper left')
fig.tight_layout()
save(fig, 'fig2_寿命vs参数.png')

# ================= 图3: SOH 曲线对比 (批次1) =================
LONG_POLS = ['4C(80%)-4C', '3.6C(80%)-3.6C', '8C(15%)-3.6C']
SHORT_POLS = ['5.4C(80%)-5.4C', '8C(35%)-3.6C', '8C(25%)-3.6C']
def load_soh(fname, cell_idx):
    with h5py.File(os.path.join(r"C:\Users\24345\Desktop\数模校赛", fname), 'r') as f:
        batch = f['batch']
        Qd = np.array(f[batch['summary'][cell_idx, 0]]['QDischarge']).ravel()
        return Qd[1:] / Qd[1] * 100
def find_cells(fname, policies):
    with h5py.File(os.path.join(r"C:\Users\24345\Desktop\数模校赛", fname), 'r') as f:
        batch = f['batch']
        n = batch['cycle_life'].shape[0]
        found = {}
        for i in range(n):
            pr = ''.join(chr(int(x)) for x in np.array(f[batch['policy_readable'][i, 0]]).ravel())
            if pr in policies and pr not in found:
                found[pr] = i
    return found
cell_idx = find_cells('data_1.mat', LONG_POLS + SHORT_POLS)
fig, ax = plt.subplots(figsize=(8.5, 5))
for pol in LONG_POLS:
    soh = load_soh('data_1.mat', cell_idx[pol])
    ax.plot(np.arange(1, len(soh)+1), soh, color=BLUE, lw=1.7, label=f'长寿 {pol}')
for pol in SHORT_POLS:
    soh = load_soh('data_1.mat', cell_idx[pol])
    ax.plot(np.arange(1, len(soh)+1), soh, color=RED, lw=1.7, ls='--', label=f'短寿 {pol}')
ax.axhline(80, color=MUTED, ls=':', lw=1)
ax.text(2, 80.8, '80% SOH 阈值', fontsize=14, color=MUTED)
ax.set_xlabel('循环数'); ax.set_ylabel('SOH (%)')
ax.legend(frameon=False, fontsize=13, loc='upper right', ncol=2)
style_ax(ax)
fig.tight_layout()
save(fig, 'fig3_SOH曲线对比.png')

# ================= 图4: 偏回归图 =================
def ols_coef(X, y):
    Xd = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(Xd, y, rcond=None)[0]
Xnames = ['C1_C', 'Q1_pct', 'C2_C', 'T']
X = std[Xnames].values
y = std['loglife'].values
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
for ax, v in zip(axes.flat, Xnames):
    i = Xnames.index(v)
    others = [j for j in range(4) if j != i]
    other_cols = [Xnames[j] for j in others] + ['batch2']
    std['batch2'] = (std['batch'] == 'data_2').astype(float)
    Xo = np.column_stack([np.ones(len(y)), std[other_cols].values])
    ry = y - Xo @ np.linalg.lstsq(Xo, y, rcond=None)[0]
    rx = X[:, i] - Xo @ np.linalg.lstsq(Xo, X[:, i], rcond=None)[0]
    ax.scatter(rx, ry, s=26, alpha=0.72, color=BLUE, edgecolor='white', lw=0.4)
    z = np.polyfit(rx, ry, 1)
    xx = np.linspace(rx.min(), rx.max(), 40)
    ax.plot(xx, np.polyval(z, xx), color=RED, lw=1.7, ls='--')
    r = np.corrcoef(rx, ry)[0, 1]
    ax.set_xlabel(f'{v}（残差化）', fontsize=15); ax.set_ylabel('log$_{10}$寿命 残差', fontsize=15)
    style_ax(ax)
fig.tight_layout()
save(fig, 'fig4_偏回归图.png')

# ================= 图6: 预测 vs 实际 (k=100, GBM) =================
qp = pd.read_csv(os.path.join(BASE, 'data_processed', 'q3_predictions.csv'))
fig, ax = plt.subplots(figsize=(6.8, 6.8))
for b in ['data_1', 'data_2', 'data_3']:
    sub = qp[qp['batch'] == b]
    ax.scatter(sub['cycle_life'], sub['pred_life'], s=26, alpha=0.8, color=BCOL[b],
               edgecolor='white', lw=0.4, label=b)
lim = [0, 2300]   # 从原点起, 等比例保证 45° 对角线与 ±10% 带不变形
ax.plot(lim, lim, color=INK, lw=1.4, ls='--')
ax.plot(lim, [x*0.9 for x in lim], color=MUTED, lw=0.8, ls=':')
ax.plot(lim, [x*1.1 for x in lim], color=MUTED, lw=0.8, ls=':')
ax.text(1900, 1830, '+10%', fontsize=14, color=MUTED)
ax.text(1900, 1710, '−10%', fontsize=14, color=MUTED)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_aspect('equal')
ax.set_xlabel('实际循环寿命'); ax.set_ylabel('预测循环寿命')
ax.legend(frameon=False, fontsize=14, loc='upper left')
style_ax(ax)
fig.tight_layout()
save(fig, 'fig6_寿命预测对比.png')

# ================= 图7: MAPE vs 窗口 =================
q3 = np.load(os.path.join(BASE, 'data_processed', 'q3_results.npy'), allow_pickle=True).item()
windows = q3['windows']
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for mn, c, mk in [('Ridge', BLUE, 'o'), ('RandomForest', ORANGE, 's'), ('GradientBoosting', AQUA, '^')]:
    vals = [q3['summary'][f'{k}_{mn}']['mape'] for k in windows]
    ax.plot(windows, vals, marker=mk, color=c, lw=2.0, ms=6, label=mn)
ax.set_xlabel('早期循环窗口长度 k（个循环）'); ax.set_ylabel('寿命预测 MAPE (%)')
ax.set_xticks(windows)
ax.legend(frameon=False, fontsize=14)
style_ax(ax)
fig.tight_layout()
save(fig, 'fig7_MAPE_vs窗口.png')

# ================= 图8: 特征重要性 =================
imp = pd.Series(q3['imp']).sort_values(ascending=True)
# 特征名英译中, 使 y 轴可读
FEAT_CN = {
    'soh_k': '第k循环 SOH', 'loss_k': '容量损失(100−SOH)', 'slope': 'SOH 拟合斜率',
    'slope_r2': 'SOH 拟合 R²', 'soh_resid_std': 'SOH 残差波动',
    'soh_firstdiff_std': 'SOH 一阶差分波动', 'qd_std': '容量波动',
    'qd_firstdiff_std': '容量一阶差分波动', 'qd_slope': '容量斜率',
    'ct_first': '首循环充电时间', 'ct_k': '第k循环充电时间', 'ct_slope': '充电时间斜率',
    'ct_diff_std': '充电时间波动', 'ct_drift': '充电时间漂移',
    'C1_C': 'C1 第一阶段倍率', 'Q1_pct': 'Q1 切换SOC', 'C2_C': 'C2 第二阶段倍率',
    'avg_chargetime_min': '平均充电时间',
}
def _cn(n):
    if n in FEAT_CN:
        return FEAT_CN[n]
    if n.startswith('lat_'):
        return f'SOH 潜伏时间(<{n[4:]}%)'
    return n
imp = imp.rename(index=_cn)
top = imp.tail(12)
fig, ax = plt.subplots(figsize=(7.2, 5.2))
colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(top)))
ax.barh(top.index, top.values, color=colors, edgecolor='white', linewidth=0.5)
style_ax(ax)
fig.tight_layout()
save(fig, 'fig8_特征重要性.png')

# ================= 图9: SOH 轨迹演示 =================
cy = pd.read_csv(os.path.join(BASE, 'data_processed', '每循环明细表_124.csv'))
cy = cy[cy['chargetime_min'] < 40]
sample = ['data_1_cell00', 'data_2_cell04', 'data_2_cell01']
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
for ax, batt in zip(axes, sample):
    d = cy[cy['battery'] == batt].sort_values('cycle')
    cl = bt[bt['battery'] == batt]['cycle_life'].values[0]
    ax.plot(d['cycle'], d['SOH_pct'], color='#b5b5af', lw=1.1, label='实际 SOH 轨迹')
    d20 = cy[cy['battery'] == batt].sort_values('cycle').head(20)
    ax.plot(d20['cycle'], d20['SOH_pct'], color=BLUE, lw=2.2, label='前 20 循环（训练用）')
    ax.axhline(80, color=RED, ls=':', lw=1.0)
    ax.set_xlabel('循环数', fontsize=15)
    style_ax(ax)
axes[0].set_ylabel('SOH (%)', fontsize=15)
axes[0].legend(frameon=False, fontsize=13, loc='lower left')
fig.tight_layout()
save(fig, 'fig9_轨迹预测演示.png')

# ================= 图10: Pareto 前沿 =================
q4 = np.load(os.path.join(BASE, 'data_processed', 'q4_results.npy'), allow_pickle=True).item()
front = pd.DataFrame(q4['front']).sort_values('T')
rec = q4['rec']
# 重算所有可行设计点用于背景
C1_lv = np.unique(std['C1_C']); Q1_lv = np.unique(std['Q1_pct']); C2_lv = np.unique(std['C2_C'])
C1m, Q1m, C2m = np.meshgrid(C1_lv, Q1_lv, C2_lv, indexing='ij')
wT = q4['w']; b_l = q4['b_all']
def T80(C1, Q1, C2): return wT[0]*(Q1/100)/C1 + wT[1]*((80-Q1)/100)/C2 + wT[2]
def life(C1, Q1, C2):
    m1 = C1*Q1/100; m2 = C2*(80-Q1)/100
    return 10 ** (b_l[0] + b_l[1]*m1 + b_l[2]*m2)
Tg = T80(C1m, Q1m, C2m); Lg = life(C1m, Q1m, C2m)
m1g = C1m*Q1m/100; m2g = C2m*(80-Q1m)/100
m1_obs, m2_obs = std['m1'].values, std['m2'].values
feas = ((m1g >= m1_obs.min()) & (m1g <= m1_obs.max()) & (m2g >= m2_obs.min()) & (m2g <= m2_obs.max()))
fig, ax = plt.subplots(figsize=(7.6, 6.2))
ax.scatter(Tg[feas], Lg[feas], s=7, alpha=0.4, color='#9ec5f4', edgecolor='none', label='可行设计（观测水平邻域）')
ax.plot(front['T'], front['life'], '-o', color=RED, lw=2.0, ms=4, label='Pareto 前沿')
ax.plot(rec['T'], rec['life'], '*', color=INK, ms=20, label=f'推荐（膝点）：{rec["C1"]:.1f}C({rec["Q1"]:.0f}%)-{rec["C2"]:.1f}C')
ax.annotate(f'推荐\nT80={rec["T"]:.1f} min\n寿命≈{rec["life"]:.0f}', xy=(rec['T'], rec['life']),
            xytext=(rec['T']-6.5, rec['life']+260), fontsize=12,
            arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.9))
ax.set_xlabel('充电时间 $T$ (min)'); ax.set_ylabel('预测循环寿命')
ax.legend(frameon=False, fontsize=12, loc='lower right')
style_ax(ax)
fig.tight_layout()
save(fig, 'fig10_Pareto前沿.png')

# ================= 图11: 推荐策略电流剖面 =================
c1, q1, c2 = rec['C1'], rec['Q1'], rec['C2']
# Q1=80 时第二阶段覆盖为零、C2 不参与实际充电(80% 后直接 1C CC-CV),
# 优化器在该方向无约束(可取任意值), 按正文约定以 C2=C1 展示 (如 3.6C(80%)-3.6C)
c2d = c1 if q1 >= 80 else c2
fig, ax = plt.subplots(figsize=(7.6, 4.4))
soc = np.array([0, q1, 80, 100]); I = np.array([c1, c2, 1.0, 0.0])
ax.step(soc, I, where='post', lw=2.6, color=BLUE)
ax.fill_between(soc, I, step='post', color=BLUE, alpha=0.10)
ax.axvline(q1, color=MUTED, ls=':', lw=1.0)
ax.text(q1, c1+0.22, f'Q1={q1:.0f}%', ha='center', fontsize=14, color=INK)
ax.text(q1/2, c1+0.18, f'{c1:.1f}C', ha='center', fontsize=15, color=BLUE)
if q1 < 80:
    ax.text((q1+80)/2, c2+0.18, f'{c2:.1f}C', ha='center', fontsize=15, color=BLUE)
ax.text(90, 1.1, '1C CC-CV', ha='center', fontsize=14, color=MUTED)
ax.set_xlabel('SOC (%)'); ax.set_ylabel('充电倍率 (C)')
ax.set_xlim(0, 100); ax.set_ylim(0, c1+0.9)
style_ax(ax)
fig.tight_layout()
save(fig, 'fig11_推荐策略剖面.png')

# ================= 图12: 参数相关性热力图 =================
cols = ['C1_C', 'Q1_pct', 'C2_C', 'T', 'cycle_life']
labs = ['$C_1$', '$Q_1$', '$C_2$', '充电时间', '循环寿命']
cm = std[cols].corr().values
fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(cm, cmap='RdBu_r', vmin=-1, vmax=1)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center', fontsize=12,
                color='white' if abs(cm[i,j]) > 0.5 else INK)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(labs, fontsize=13)
ax.set_yticks(range(len(cols))); ax.set_yticklabels(labs, fontsize=13)
cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label('Pearson 相关系数', fontsize=12)
style_ax(ax)
fig.tight_layout()
save(fig, 'fig12_相关性热力图.png')

# ================= 图13: 充电时间模型验证 =================
wT = q4['w']
T_pred = wT[0]*std['Q1_pct']/100/std['C1_C'] + wT[1]*(80-std['Q1_pct'])/100/std['C2_C'] + wT[2]
T_obs = std['T'].values
r = np.corrcoef(T_obs, T_pred)[0, 1]
mae_T = np.mean(np.abs(T_obs - T_pred))
fig, ax = plt.subplots(figsize=(6.4, 5.4))
ax.scatter(T_obs, T_pred, s=28, alpha=0.75, color=BLUE, edgecolor='white', lw=0.4)
lim = [9, 16]
ax.plot(lim, lim, color=INK, lw=1.4, ls='--')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('实测充电时间 (min)'); ax.set_ylabel('模型预测充电时间 (min)')
style_ax(ax)
fig.tight_layout()
save(fig, 'fig13_充电时间模型验证.png')

# ================= 图14: 寿命预测误差分布 =================
qp = pd.read_csv(os.path.join(BASE, 'data_processed', 'q3_predictions.csv'))
err = (qp['pred_life'] - qp['cycle_life']) / qp['cycle_life'] * 100
fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.hist(err, bins=25, color=BLUE, edgecolor='white', lw=0.6, alpha=0.92)
ax.axvline(0, color=INK, ls='--', lw=1.2)
ax.axvline(err.median(), color=ORANGE, ls='--', lw=1.2)
ax.text(err.median(), ax.get_ylim()[1]*0.95, f'中位数 {err.median():.1f}%',
        fontsize=16, color=ORANGE, ha='center', va='top')
ax.set_xlabel('相对预测误差 (%)'); ax.set_ylabel('电池数')
style_ax(ax)
fig.tight_layout()
save(fig, 'fig14_预测误差分布.png')

# ================= 图15: 推荐策略与典型策略对比 =================
# 实测寿命/充电时间从 124 清洗集与 q4 模型动态计算, 保证与正文表 tab:compare 一致
wT15 = q4['w']
def _T80(c1, q1, c2):
    return wT15[0]*(q1/100)/c1 + wT15[1]*((80-q1)/100)/c2 + wT15[2]
cmp_rows = []
for _name, _c1, _q1, _c2, _col, _bold in [
        ('推荐 3.6C(80%)-3.6C', 3.6, 80, 3.6, BLUE, True),
        ('备选 4C(80%)-4C', 4.0, 80, 4.0, '#3987e5', True),
        ('8C(15%)-3.6C', 8.0, 15, 3.6, '#9ec5f4', False),
        ('6C(30%)-3.6C', 6.0, 30, 3.6, '#9ec5f4', False),
        ('1C(4%)-6C', 1.0, 4, 6.0, RED, False),
        ('2C(10%)-6C', 2.0, 10, 6.0, RED, False)]:
    _pol = f'{_c1:g}C({int(_q1)}%)-{_c2:g}C'
    _obs = std[std['policy'] == _pol]['cycle_life']
    _obs_life = round(_obs.mean()) if len(_obs) else np.nan
    cmp_rows.append((_name, _obs_life, round(_T80(_c1, _q1, _c2), 1), _col, _bold))
names = [r[0] for r in cmp_rows]
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))
ax = axes[0]
vals = [r[1] for r in cmp_rows]; colors = [r[3] for r in cmp_rows]
bars = ax.bar(range(len(names)), vals, color=colors, edgecolor='white', lw=0.5, width=0.62)
for bi, v in enumerate(vals):
    ax.text(bi, v + 18, f'{v}', ha='center', fontsize=12)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=11.5)
style_ax(ax)
ax = axes[1]
vals = [r[2] for r in cmp_rows]
bars = ax.bar(range(len(names)), vals, color=colors, edgecolor='white', lw=0.5, width=0.62)
for bi, v in enumerate(vals):
    ax.text(bi, v + 0.15, f'{v}', ha='center', fontsize=12)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=11.5)
style_ax(ax)
fig.tight_layout()
save(fig, 'fig15_策略对比.png')

# ================= 图17: 充电时间分析 =================
std2c = std.copy()
std2c['C2档'] = pd.cut(std2c['C2_C'], bins=[2.9, 3.6, 4.4, 5.2, 6.1],
                       labels=['3.0~3.6', '3.7~4.4', '4.5~5.2', '5.3~6.0'])
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
ax = axes[0]
groups = [std2c[std2c['C2档'] == g]['T'].values for g in ['3.0~3.6', '3.7~4.4', '4.5~5.2', '5.3~6.0']]
bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                medianprops=dict(color='white', lw=1.4),
                whiskerprops=dict(color=MUTED, lw=1), capprops=dict(color=MUTED, lw=1))
for patch, c in zip(bp['boxes'], SEQB[1::2]):
    patch.set_facecolor(c); patch.set_alpha(0.85)
ax.set_xticklabels(['3.0~3.6', '3.7~4.4', '4.5~5.2', '5.3~6.0'], fontsize=14.5)
ax.set_xlabel('$C_2$ 档位 (C)', fontsize=15); ax.set_ylabel('充电时间 (min)', fontsize=15)
style_ax(ax)
ax = axes[1]
ideal = 60*std['Q1_pct']/100/std['C1_C'] + 60*(80-std['Q1_pct'])/100/std['C2_C']
ax.scatter(ideal, std['T'], s=26, alpha=0.75, color=BLUE, edgecolor='white', lw=0.4)
lim = [8, 16]
ax.plot(lim, lim, color=INK, lw=1.4, ls='--')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('理想模型充电时间 (min)', fontsize=15); ax.set_ylabel('实测充电时间 (min)', fontsize=15)
style_ax(ax)
fig.tight_layout()
save(fig, 'fig17_充电时间分析.png')

# ================= 图18: 回归残差诊断 (问题二) =================
Xall = np.column_stack([std[['C1_C', 'Q1_pct', 'C2_C', 'T']].values,
                        (std['batch'] == 'data_2').astype(float)])
Xd = np.column_stack([np.ones(len(y)), Xall])
breg = np.linalg.lstsq(Xd, y, rcond=None)[0]
fitted = Xd @ breg
resid = y - fitted
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
ax = axes[0]
ax.scatter(fitted, resid, s=26, alpha=0.75, color=BLUE, edgecolor='white', lw=0.4)
ax.axhline(0, color=RED, lw=1.4, ls='--')
ax.set_xlabel('拟合值 log$_{10}$寿命'); ax.set_ylabel('残差')
style_ax(ax)
ax = axes[1]
ax.hist(resid, bins=18, color=BLUE, edgecolor='white', lw=0.6, alpha=0.92)
ax.axvline(0, color=RED, lw=1.4, ls='--')
ax.set_xlabel('残差'); ax.set_ylabel('频数')
style_ax(ax)
fig.tight_layout()
save(fig, 'fig18_回归残差诊断.png')

# ================= 图19: 早期SOH演化 (问题三) =================
cy2 = pd.read_csv(os.path.join(BASE, 'data_processed', '每循环明细表_124.csv'))
cy2 = cy2[cy2['chargetime_min'] < 40]
cells = [('data_1_cell03', BLUE, '长寿 4C(80%)-4C'),
         ('data_1_cell00', AQUA, '长寿 3.6C(80%)-3.6C'),
         ('data_2_cell04', ORANGE, '中寿 3.6C(22%)-5.5C'),
         ('data_2_cell00', RED, '短寿 1C(4%)-6C')]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
for batt, c, lab in cells:
    d = cy2[cy2['battery'] == batt].sort_values('cycle').head(60)
    ax.plot(d['cycle'], d['SOH_pct'], color=c, lw=1.8, label=lab)
ax.set_xlabel('循环数'); ax.set_ylabel('SOH (%)')
ax.legend(frameon=False, fontsize=14, loc='upper right')
style_ax(ax)
fig.tight_layout()
save(fig, 'fig19_早期SOH演化.png')

# ================= 图20: 方法流程示意图 (简洁现代风格) =================
VIOLET2 = '#4a3aa7'
fig, ax = plt.subplots(figsize=(12, 4.0))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
steps = [
    ('数据整理', '问题一', '策略参数 · 寿命 · SOH\n异常清洗 · 单位校准', BLUE),
    ('量化建模', '问题二', '多变量回归 · 剂量模型\n批次差异 · 交互项', ORANGE),
    ('寿命预测', '问题三', '早期特征 · GBM\n5折×8次交叉验证', AQUA),
    ('策略优化', '问题四', '充电时间模型 · Pareto\n推荐策略 · 外推边界', VIOLET2),
]
xw, xh, ys = 0.18, 0.55, 0.28
gap = 0.04
total_w = 4*xw + 3*gap
xs = [(1-total_w)/2 + i*(xw+gap) for i in range(4)]
for k, (title, prob, desc, col) in enumerate(steps):
    x0 = xs[k]
    # 卡片主体 — 纯白底 + 细边框
    ax.add_patch(FancyBboxPatch((x0, ys), xw, xh,
                 boxstyle='round,pad=0.008,rounding_size=0.015',
                 fc='white', ec='#d0d0d0', lw=1.0, zorder=2))
    # 左侧细色条
    ax.add_patch(Rectangle((x0, ys+0.04), 0.008, xh-0.08, fc=col, ec='none', zorder=3))
    # 编号
    ax.text(x0 + 0.035, ys + xh - 0.08, str(k+1), ha='center', va='center',
            fontsize=16, color=col, fontweight='bold', zorder=4)
    # 标题
    ax.text(x0 + xw/2, ys + xh - 0.16, title, ha='center', va='center',
            fontsize=18, color=INK, fontweight='bold', zorder=4)
    # 问题标识
    ax.text(x0 + xw/2, ys + xh - 0.28, prob, ha='center', va='center',
            fontsize=13, color=col, fontweight='bold', zorder=4)
    # 细线
    ax.plot([x0+0.06, x0+xw-0.06], [ys+xh-0.34, ys+xh-0.34], color='#e8e8e8', lw=0.6, zorder=4)
    # 描述
    ax.text(x0 + xw/2, ys + xh/2 - 0.06, desc, ha='center', va='center',
            fontsize=13, color=MUTED, linespacing=1.6, zorder=4)
    # 连接箭头
    if k < 3:
        ax.annotate('', xy=(xs[k+1]-0.006, ys+xh/2), xytext=(x0+xw+0.006, ys+xh/2),
                    arrowprops=dict(arrowstyle='->', color='#b0b0b0', lw=2.2, mutation_scale=22))
# 底部闭环箭头 + 注记
ax.plot([0.2, 0.8], [0.10, 0.10], color='#d0d0d0', lw=0.6)
ax.text(0.5, 0.045, '前一问题的输出 → 后一问题的输入，形成闭环', ha='center',
        fontsize=14, color=MUTED)
fig.tight_layout()
save(fig, 'fig20_方法流程.png')

# ================= 图21: 批次差异可视化 (策略模型残差按批次) =================
# 用剂量模型预测寿命, 残差按批次箱线图, 直观展示批次差异
b_l = q4['b_all']
def _life_dose(C1, Q1, C2):
    m1 = C1 * Q1 / 100; m2 = C2 * (80 - Q1) / 100
    return 10 ** (b_l[0] + b_l[1] * m1 + b_l[2] * m2)
s21 = std[std['protocol'] == 'standard'].copy()
s21['pred'] = [_life_dose(c1, q1, c2) for c1, q1, c2 in zip(s21['C1_C'], s21['Q1_pct'], s21['C2_C'])]
s21['resid'] = s21['cycle_life'] - s21['pred']
fig, ax = plt.subplots(figsize=(8.2, 4.8))
groups = [s21[s21['batch'] == b]['resid'].values for b in ['data_1', 'data_2']]
bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                medianprops=dict(color='white', lw=1.4),
                whiskerprops=dict(color=MUTED, lw=1), capprops=dict(color=MUTED, lw=1))
for patch, c in zip(bp['boxes'], [BLUE, ORANGE]):
    patch.set_facecolor(c); patch.set_alpha(0.85)
ax.axhline(0, color=RED, lw=1.2, ls='--')
ax.set_xticklabels(['批次 1', '批次 2'], fontsize=15)
ax.set_xlabel('批次'); ax.set_ylabel('寿命残差（实测 − 模型预测）')
style_ax(ax)
fig.tight_layout()
save(fig, 'fig21_批次差异.png')

# ================= 图22: 充电时间随循环演化 =================
cy4 = pd.read_csv(os.path.join(BASE, 'data_processed', '每循环明细表_124.csv'))
cy4 = cy4[cy4['chargetime_min'] < 40]
cells22 = [
    ('data_1_cell03', BLUE, '长寿 4C(80%)-4C'),
    ('data_2_cell04', ORANGE, '中寿 3.6C(22%)-5.5C'),
    ('data_2_cell00', RED, '短寿 1C(4%)-6C'),
]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
for batt, c, lab in cells22:
    d = cy4[cy4['battery'] == batt].sort_values('cycle')
    d = d.head(600)
    ax.plot(d['cycle'], d['chargetime_min'], color=c, lw=1.6, label=lab)
ax.set_xlabel('循环数'); ax.set_ylabel('充电时间 (min)')
ax.legend(frameon=False, fontsize=14, loc='upper left')
style_ax(ax)
fig.tight_layout()
save(fig, 'fig22_充电时间演化.png')

# ================= 图23: 问题三预测流程 (简洁现代风格) =================
fig, ax = plt.subplots(figsize=(12, 3.0))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
steps23 = [
    ('早期循环数据\n（前 k 个循环）', 'SOH · 容量 · 充电时间', BLUE),
    ('特征提取', '斜率 · 波动 · 潜伏时间\n充电时间漂移', ORANGE),
    ('GBM 回归模型', '5折x8次交叉验证\nlog10(寿命) 目标', AQUA),
    ('预测寿命', 'L = 10^y\n达 80% SOH 循环数', VIOLET2),
]
xw, xh, ys = 0.18, 0.55, 0.26
gap = 0.04
total_w = 4*xw + 3*gap
xs = [(1-total_w)/2 + i*(xw+gap) for i in range(4)]
for k, (t1, t2, col) in enumerate(steps23):
    x0 = xs[k]
    # 白底卡片 + 细边框
    ax.add_patch(FancyBboxPatch((x0, ys), xw, xh, boxstyle='round,pad=0.008,rounding_size=0.015',
                 fc='white', ec='#d0d0d0', lw=1.0, zorder=2))
    # 左侧细色条
    ax.add_patch(Rectangle((x0, ys+0.04), 0.008, xh-0.08, fc=col, ec='none', zorder=3))
    # 标题
    ax.text(x0+xw/2, ys+xh*0.70, t1, ha='center', va='center',
            fontsize=17, color=INK, fontweight='bold', zorder=4)
    # 细线
    ax.plot([x0+0.05, x0+xw-0.05], [ys+xh*0.42, ys+xh*0.42], color='#e8e8e8', lw=0.6, zorder=4)
    # 描述
    ax.text(x0+xw/2, ys+xh*0.18, t2, ha='center', va='center',
            fontsize=13, color=MUTED, linespacing=1.5, zorder=4)
    # 箭头
    if k < 3:
        ax.annotate('', xy=(xs[k+1]-0.006, ys+xh/2), xytext=(x0+xw+0.006, ys+xh/2),
                    arrowprops=dict(arrowstyle='->', color='#b0b0b0', lw=2.2, mutation_scale=22))
fig.tight_layout()
save(fig, 'fig23_预测流程.png')

print('\n全部论文图表已重绘完成')

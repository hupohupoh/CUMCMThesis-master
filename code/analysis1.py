# -*- coding: utf-8 -*-
"""B题 问题一: 数据整理 + 寿命分布统计分析 + 长/短寿命策略识别"""
import pandas as pd, numpy as np, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r"C:\Users\one\Desktop\2026年校赛题目\B"
FIG = os.path.join(BASE, 'data_processed', 'figs')
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, 'data_processed', 'battery_table.csv'))
df['cycle_life'] = pd.to_numeric(df['cycle_life'], errors='coerce')
std = df[df['protocol'] == 'standard'].dropna(subset=['cycle_life']).copy()
new = df[df['protocol'] == 'newstructure'].dropna(subset=['cycle_life']).copy()

print("="*60)
print("一、数据整理结果")
print("="*60)
print(f"总电池数: {len(df)}  |  有效寿命: {df['cycle_life'].notna().sum()}  |  寿命缺失: {df['cycle_life'].isna().sum()}")
print(f"协议: standard(批次1+2) {len(std)} 个, newstructure(批次3) {len(new)} 个")
print(f"\n每批次寿命分布:")
for b in ['data_1','data_2','data_3']:
    s = df[df['batch']==b]['cycle_life'].dropna()
    print(f"  {b}: n={len(s)}  min={s.min():.0f}  median={s.median():.0f}  mean={s.mean():.0f}  max={s.max():.0f}")

print("\n" + "="*60)
print("二、寿命总体分布")
print("="*60)
allc = df['cycle_life'].dropna()
print(f"全部有效电池: n={len(allc)}  min={allc.min():.0f}  Q1={allc.quantile(.25):.0f}  median={allc.median():.0f}  Q3={allc.quantile(.75):.0f}  max={allc.max():.0f}")
print(f"寿命分位: p10={allc.quantile(.1):.0f}  p25={allc.quantile(.25):.0f}  p50={allc.quantile(.5):.0f}  p75={allc.quantile(.75):.0f}  p90={allc.quantile(.9):.0f}")

print("\n" + "="*60)
print("三、策略→寿命统计 (standard 批次, 每种策略的平均寿命)")
print("="*60)
grp = std.groupby('policy').agg(n=('cycle_life','size'), mean=('cycle_life','mean'),
                                 std=('cycle_life','std'), med=('cycle_life','median'),
                                 C1=('C1_C','mean'), Q1=('Q1_pct','mean'), C2=('C2_C','mean'),
                                 ct=('avg_chargetime_min','mean')).round(1)
grp = grp.sort_values('mean', ascending=False)
print(grp.to_string())

print("\n" + "="*60)
print("四、典型长寿命 vs 短寿命策略")
print("="*60)
top = grp.head(5); bot = grp.tail(5)
def show(tag, g):
    print(f"\n【{tag}】")
    for pol, row in g.iterrows():
        print(f"  {pol:<22} 寿命均值={row['mean']:>6.0f}±{row['std']:>5.1f} (n={int(row['n'])})  "
              f"C1={row['C1']:.1f}C Q1={row['Q1']:.0f}% C2={row['C2']:.1f}C 充电={row['ct']:.1f}min")
show("典型长寿命策略 (前5)", top)
show("典型短寿命策略 (后5)", bot)

long_ = grp.head(3); short_ = grp.tail(3)
print(f"\n长寿命均值: C1={long_['C1'].mean():.2f}C, Q1={long_['Q1'].mean():.0f}%, C2={long_['C2'].mean():.2f}C, 充电时间={long_['ct'].mean():.1f}min")
print(f"短寿命均值: C1={short_['C1'].mean():.2f}C, Q1={short_['Q1'].mean():.0f}%, C2={short_['C2'].mean():.2f}C, 充电时间={short_['ct'].mean():.1f}min")

print("\n" + "="*60)
print("五、cycle_life 与各参数的相关性 (standard)")
print("="*60)
for col, lab in [('C1_C','C1 第一阶段倍率'), ('Q1_pct','Q1 切换SOC'), ('C2_C','C2 第二阶段倍率'), ('avg_chargetime_min','充电时间(min)')]:
    r = std[['cycle_life', col]].dropna().corr().iloc[0,1]
    print(f"  cycle_life vs {lab:<14}: Pearson r = {r:.3f}")

print("\n" + "="*60)
print("六、newstructure 批次3 策略→寿命")
print("="*60)
g3 = new.groupby('policy').agg(n=('cycle_life','size'), mean=('cycle_life','mean'), C1=('C1_C','mean'), Q1=('Q1_pct','mean')).round(1).sort_values('mean', ascending=False)
print(g3.to_string())

# ============ 画图 ============
# 图1: 寿命分布直方图(按批次)
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, b in zip(axes, ['data_1','data_2','data_3']):
    s = df[df['batch']==b]['cycle_life'].dropna()
    ax.hist(s, bins=20, color='steelblue', edgecolor='white')
    ax.set_title(f'{b} (n={len(s)})\nmedian={s.median():.0f}')
    ax.set_xlabel('cycle_life')
axes[0].set_ylabel('电池数')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig1_寿命分布_按批次.png'), dpi=120); plt.close()

# 图2: cycle_life vs 参数 (standard)
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col, lab in zip(axes, ['C1_C','Q1_pct','C2_C','avg_chargetime_min'], ['C1 (C)','Q1 (%)','C2 (C)','充电时间 (min)']):
    ax.scatter(std[col], std['cycle_life'], s=30, alpha=.6)
    if col in ['C1_C','C2_C']:
        x = np.linspace(std[col].min(), std[col].max(), 100)
        z = np.polyfit(std[col], std['cycle_life'], 1)
        ax.plot(x, np.polyval(z, x), 'r--')
    ax.set_xlabel(lab); ax.set_ylabel('cycle_life')
fig.suptitle('cycle_life 与策略参数关系 (standard 批次1+2)')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig2_寿命vs参数.png'), dpi=120); plt.close()

# 图3: 长短寿命策略 SOH 曲线对比
import h5py
def load_soh(fname, cell_idx):
    with h5py.File(os.path.join(r"C:\Users\one\Desktop", fname), 'r') as f:
        batch = f['batch']
        Qd = np.array(f[batch['summary'][cell_idx,0]]['QDischarge']).ravel()
        return Qd[1:] / Qd[1] * 100

# 按策略名动态找索引 (data_1)
def find_cells(fname, policies):
    with h5py.File(os.path.join(r"C:\Users\one\Desktop", fname), 'r') as f:
        batch = f['batch']
        n = batch['cycle_life'].shape[0]
        found = {}
        for i in range(n):
            pr = ''.join(chr(int(x)) for x in np.array(f[batch['policy_readable'][i,0]]).ravel())
            if pr in policies and pr not in found:
                found[pr] = i
    return found

LONG_POLS = ['4C(80%)-4C', '3.6C(80%)-3.6C', '8C(15%)-3.6C']
SHORT_POLS = ['5.4C(80%)-5.4C', '8C(35%)-3.6C', '8C(25%)-3.6C']
cell_idx = find_cells('data_1.mat', LONG_POLS + SHORT_POLS)
plt.figure(figsize=(9, 5))
for pol in LONG_POLS:
    soh = load_soh('data_1.mat', cell_idx[pol])
    plt.plot(np.arange(1, len(soh)+1), soh, linewidth=1.6, label=f'长寿 {pol}')
for pol in SHORT_POLS:
    soh = load_soh('data_1.mat', cell_idx[pol])
    plt.plot(np.arange(1, len(soh)+1), soh, '--', linewidth=1.6, label=f'短寿 {pol}')
plt.axhline(80, color='red', ls='--', alpha=.5); plt.text(5, 81, '80% SOH阈值', color='red')
plt.xlabel('循环数'); plt.ylabel('SOH (%)'); plt.legend(fontsize=8); plt.title('典型长/短寿命电池 SOH 衰减曲线 (批次1)')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig3_SOH曲线对比.png'), dpi=120); plt.close()

print(f"\n图表已保存到: {FIG}")
print("  fig1_寿命分布_按批次.png, fig2_寿命vs参数.png, fig3_SOH曲线对比.png")

# -*- coding: utf-8 -*-
"""B题 问题一: 数据整理 + 寿命分布统计分析 + 长/短寿命策略识别
图表(fig1--fig3)由 make_figures.py 统一生成, 本脚本只做统计计算与结果输出。
"""
import pandas as pd, numpy as np, os

BASE = r"C:\Users\one\Desktop\2026年校赛题目\B"

df = pd.read_csv(os.path.join(BASE, 'data_processed', 'battery_table_124.csv'))
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

print("\n分析完成: 图1/图2/图3 由 make_figures.py 统一生成")

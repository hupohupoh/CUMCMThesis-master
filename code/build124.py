# -*- coding: utf-8 -*-
"""按原论文 LoadData.m 口径构建 124 电池数据集:
1) 批次2的5个延续电池(b2c7,8,9,15,16)并入批次1(b1c0-4), cycle_life 从合并Qd重算
2) 批次3: 移除cell38(1-indexed)、endcap QDischarge>0.885 的未完成电池、噪声[3,40,41]
3) 批次1: 移除 b1c8,b1c10,b1c12,b1c13,b1c22 (未完成)
输出: 124 电池的 battery_table_124.csv 与 每循环明细表_124.csv
"""
import pandas as pd, numpy as np, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
BASE = r"C:\Users\one\Desktop\2026年校赛题目\B"
cy = pd.read_csv(os.path.join(BASE, 'data_processed', '每循环明细表.csv'))
bt = pd.read_csv(os.path.join(BASE, 'data_processed', 'battery_table.csv'))
bt['cycle_life'] = pd.to_numeric(bt['cycle_life'], errors='coerce')

# ========== 1. 合并延续电池 ==========
PAIRS = [('data_1_cell00','data_2_cell07'), ('data_1_cell01','data_2_cell08'),
         ('data_1_cell02','data_2_cell09'), ('data_1_cell03','data_2_cell15'),
         ('data_1_cell04','data_2_cell16')]
PAIR2 = {a:b for a,b in PAIRS}
PAIR1 = {b:a for a,b in PAIRS}
REMOVE = [b for _, b in PAIRS]  # 从 data_2 移除的延续电池

def build_merged_cycles():
    """合并后每循环数据: 段2追加到段1, SOH按合并序列首Qd重算"""
    rows = cy[~cy['battery'].isin(REMOVE)].copy()
    for a, b in PAIRS:
        d1 = cy[cy['battery']==a].sort_values('cycle')
        d2 = cy[cy['battery']==b].sort_values('cycle')
        n1 = len(d1)
        d2b = d2.copy(); d2b['cycle'] = d2['cycle']+n1; d2b['battery'] = a
        comb = pd.concat([d1, d2b]).sort_values('cycle').reset_index(drop=True)
        q0 = comb['Qd_Ah'].iloc[0]
        comb['SOH_pct'] = comb['Qd_Ah']/q0*100
        rows = pd.concat([rows, comb])
    return rows.sort_values(['battery','cycle']).reset_index(drop=True)

def recompute_cl(grp):
    below = grp[grp['Qd_Ah'] < 0.88]
    return below['cycle'].min() if len(below) else grp['cycle'].max()

cy_merged = build_merged_cycles()

# ========== 2. 批次1 移除未完成电池 ==========
B1_REMOVE = ['data_1_cell08', 'data_1_cell10', 'data_1_cell12', 'data_1_cell13', 'data_1_cell22']
cy_merged = cy_merged[~cy_merged['battery'].isin(B1_REMOVE)].copy()

# ========== 3. 批次3 筛选 ==========
b3 = [f'data_3_cell{i:02d}' for i in range(46)]
# 移除 cell38 (1-indexed) = cell37 (0-indexed)
b3_keep = [c for c in b3 if c != 'data_3_cell37']
# endcap: 各cell最后Qd > 0.885 的移除
endcap = {}
for c in b3_keep:
    sub = cy_merged[cy_merged['battery']==c]
    endcap[c] = sub['Qd_Ah'].iloc[-1] if len(sub) else np.nan
endcap_rm = [c for c, v in endcap.items() if v > 0.885]
b3_keep = [c for c in b3_keep if c not in endcap_rm]
# 噪声 [3,40,41] 1-indexed (在剩余数组上) -> 0-indexed 2,39,40
if len(b3_keep) > 40:
    noisy = [b3_keep[2], b3_keep[39], b3_keep[40]]
else:
    noisy = [b3_keep[i] for i in [2,39,40] if i < len(b3_keep)]
b3_keep = [c for c in b3_keep if c not in noisy]
print(f'批次3: 46 -> {len(b3_keep)} (移除 cell37, endcap>0.885:{len(endcap_rm)}, 噪声:{len(noisy)})')
print(f'   endcap>0.885 电池: {endcap_rm}')
print(f'   噪声电池: {noisy}')

# ========== 4. 构建最终 124 电池表 ==========
keep_bats = ([f'data_1_cell{i:02d}' for i in range(46) if f'data_1_cell{i:02d}' not in B1_REMOVE]
             + [f'data_2_cell{i:02d}' for i in range(48) if f'data_2_cell{i:02d}' not in REMOVE]
             + b3_keep)
print(f'最终电池数: {len(keep_bats)}')
print(f'批次分布: 批次1={sum(1 for c in keep_bats if c.startswith("data_1"))}, '
      f'批次2={sum(1 for c in keep_bats if c.startswith("data_2"))}, '
      f'批次3={sum(1 for c in keep_bats if c.startswith("data_3"))}')

# 构建电池表
rows = []
for batt in keep_bats:
    row = bt[bt['battery']==batt].iloc[0].to_dict()
    grp = cy_merged[cy_merged['battery']==batt]
    if batt in PAIR2 or batt in B1_REMOVE:
        pass
    cl = recompute_cl(grp)
    row['cycle_life'] = cl
    row['n_cycles'] = len(grp)
    row['Qinit_Ah'] = round(grp['Qd_Ah'].iloc[0], 4)
    row['avg_chargetime_min'] = round(grp['chargetime_h'].mean(), 3)
    row['first_chargetime_min'] = round(grp['chargetime_h'].iloc[0], 3)
    row['min_SOH_pct'] = round(grp['SOH_pct'].min(), 2)
    rows.append(row)
bt124 = pd.DataFrame(rows)
bt124['cycle_life'] = pd.to_numeric(bt124['cycle_life'], errors='coerce')
print(f'\n有效 cycle_life: {bt124["cycle_life"].notna().sum()}')
print(f'寿命: min={bt124["cycle_life"].min():.0f} median={bt124["cycle_life"].median():.0f} max={bt124["cycle_life"].max():.0f}')
for b in ['data_1','data_2','data_3']:
    s = bt124[bt124['batch']==b]['cycle_life']
    print(f'  {b}: n={len(s)} median={s.median():.0f} mean={s.mean():.0f}')
std = bt124[bt124['protocol']=='standard']
print(f'  standard: {len(std)}, newstructure: {len(bt124[bt124["protocol"]=="newstructure"])}')
print(f'  C2-寿命相关: {std["cycle_life"].corr(std["C2_C"]):.3f}')

# 保存
cy_merged[cy_merged['battery'].isin(keep_bats)].to_csv(
    os.path.join(BASE,'data_processed','每循环明细表_124.csv'), index=False, encoding='utf-8-sig')
bt124.to_csv(os.path.join(BASE,'data_processed','battery_table_124.csv'), index=False, encoding='utf-8-sig')
print('\n已保存 battery_table_124.csv, 每循环明细表_124.csv')

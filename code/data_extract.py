# -*- coding: utf-8 -*-
"""
B题 数据提取脚本：从三个 .mat 提取电池级汇总表（轻量、复用）
输出: data_processed/battery_table.csv  (每电池一行)
单位: chargetime 为分钟
"""
import h5py, numpy as np, re, csv, os

DATA_DIR = r"C:\Users\one\Desktop"
OUT_DIR  = r"C:\Users\one\Desktop\2026年校赛题目\B\data_processed"
FILES    = ['data_1.mat', 'data_2.mat', 'data_3.mat']

def parse_policy(pr, batch):
    """返回 (C1, Q1, C2, protocol)
    protocol: 'standard'(batch1/2: C2为第二段充电倍率)
              'newstructure'(batch3: C2为放电倍率, 充电为 C1→Q1 后 1C)
    """
    m = re.match(r'([\d.]+)C\((\d+)%\)-([\d.]+)C', pr)
    if m:
        c1, q1, c2 = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return c1, q1, c2, ('newstructure' if 'newstructure' in pr else 'standard')
    # 容错: '4C(31%)-5' 结尾少 C
    m1 = re.match(r'([\d.]+)C\((\d+)%\)-([\d.]+)', pr)
    if m1:
        c1, q1, c2 = float(m1.group(1)), float(m1.group(2)), float(m1.group(3))
        return c1, q1, c2, ('newstructure' if 'newstructure' in pr else 'standard')
    # 截断形式: '80%)-3.6C' -> C1=C2
    m2 = re.match(r'(\d+)%\)-([\d.]+)C', pr)
    if m2:
        q1 = float(m2.group(1)); c2 = float(m2.group(2))
        return c2, q1, c2, 'standard'
    return None, None, None, None

def clean_Qd(Qd, thr=0.20):
    """剔除异常循环: 容量相对前值突变>thr 视为脏点"""
    Qd = np.array(Qd, dtype=float)
    out = np.where(Qd == 0, np.nan, Qd)
    for k in range(1, len(out)):
        if np.isnan(out[k]):
            continue
        prev = out[k-1]
        if not np.isnan(prev) and abs(out[k]-prev)/prev > thr:
            out[k] = np.nan
    return out

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for fname in FILES:
        batch_id = fname.split('.')[0]
        with h5py.File(os.path.join(DATA_DIR, fname), 'r') as f:
            batch = f['batch']
            n = batch['cycle_life'].shape[0]
            for i in range(n):
                cell = f'{batch_id}_cell{i:02d}'
                cl_raw = float(np.array(f[batch['cycle_life'][i,0]]).ravel()[0])
                cl = cl_raw if np.isfinite(cl_raw) else np.nan
                pr = ''.join(chr(int(x)) for x in np.array(f[batch['policy_readable'][i,0]]).ravel())
                c1, q1, c2, proto = parse_policy(pr, batch_id)
                summ = f[batch['summary'][i,0]]
                Qd = clean_Qd(np.array(summ['QDischarge']).ravel())
                ct = np.array(summ['chargetime']).ravel()
                valid = Qd[1:][~np.isnan(Qd[1:])]
                Qinit = float(valid[0]) if len(valid) else np.nan
                soh_min = float(np.nanmin(Qd[1:]) / Qinit * 100) if len(valid) else np.nan
                n_cycles = int(len(Qd) - 1)
                rows.append([cell, batch_id, pr, proto,
                             c1, q1, c2,
                             '' if np.isnan(cl) else round(cl),
                             n_cycles,
                             '' if np.isnan(Qinit) else round(Qinit, 4),
                             round(np.nanmean(ct[1:]), 3) if len(ct) > 1 else '',
                             round(float(ct[1]), 3) if len(ct) > 1 else '',
                             round(soh_min, 2)])
    path = os.path.join(OUT_DIR, 'battery_table.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['battery','batch','policy','protocol','C1_C','Q1_pct','C2_C',
                    'cycle_life','n_cycles','Qinit_Ah','avg_chargetime_min','first_chargetime_min','min_SOH_pct'])
        w.writerows(rows)
    print(f'已输出 {len(rows)} 个电池 -> {path}')
    # 打印协议与缺失统计
    from collections import Counter
    print('协议分布:', Counter(r[3] for r in rows))
    print('cycle_life 缺失:', [r[0] for r in rows if r[7] == ''])

if __name__ == '__main__':
    main()

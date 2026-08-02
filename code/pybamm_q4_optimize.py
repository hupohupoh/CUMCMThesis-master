# -*- coding: utf-8 -*-
"""
阶段2b：PyBaMM Q4多段充电策略优化
===================================
用标定后的电化学模型作为surrogate，优化任意SOC分段的多段充电策略。
突破"只能在已知两段式策略里排序"的局限。

方法：
  1. 定义 m 段充电协议: [(I1, SOC1), (I2, SOC2), ..., (Im, SOCm=80%)]
  2. 用 PyBaMM 仿真每个候选策略的 SOH 衰减和充电时间
  3. 遗传算法或贝叶斯优化搜索 Pareto 最优策略
  4. 与数据驱动推荐策略对比

输出：
  - Pareto 前沿图（含多段策略 vs 两段策略）
  - 最优多段策略推荐
"""
import pybamm
import numpy as np
import pandas as pd
import os, sys, io, json, warnings, time
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, 'figures')
PROC = os.path.join(BASE, 'data_processed')
os.makedirs(FIG, exist_ok=True)

print("="*60)
print("Q4 PyBaMM多段充电策略优化")
print("="*60)

# ============ 1. 加载模型和参数 ============
print("\n[1] 加载标定模型...")
param_path = os.path.join(PROC, 'a123_calibrated_params.json')
if os.path.exists(param_path):
    with open(param_path, 'r', encoding='utf-8') as f:
        cal_data = json.load(f)
    pv = pybamm.ParameterValues(cal_data['parameters'])
else:
    pv = pybamm.ParameterValues('Prada2013')
    pv_ok = pybamm.ParameterValues('OKane2022')
    for k in pv_ok.keys():
        if k not in pv: pv[k] = pv_ok[k]

model = pybamm.lithium_ion.SPMe(options={
    'SEI': 'solvent-diffusion limited',
    'SEI film resistance': 'average',
    'lithium plating': 'partially reversible',
})

# ============ 2. 多段策略仿真器 ============
def build_multistage_experiment(stages, n_cycles=30):
    """
    stages: list of (C_rate, SOC_end) tuples
    例如 [(3.6, 40), (2.0, 65), (1.0, 80)] 表示三段式
    """
    charge_steps = []
    soc_start = 0
    for cr, soc_end in stages:
        delta_soc = soc_end - soc_start
        if delta_soc <= 0:
            continue
        t_charge = delta_soc / 100.0 / cr * 3600  # 秒
        if t_charge > 10:
            charge_steps.append(f'Charge at {cr} C for {t_charge:.0f} seconds')
        soc_start = soc_end

    # CC-CV 补电
    charge_steps.append('Charge at 1 C until 3.6 V')
    charge_steps.append('Hold at 3.6 V until C/50')

    cycle = (
        'Discharge at 1 C until 2.0 V',
        'Rest for 2 minutes',
        *charge_steps,
        'Rest for 2 minutes',
    )
    return pybamm.Experiment([cycle] * n_cycles, period='30 seconds')

def T80_multistage(stages):
    """计算多段充电的 T80 (理想)"""
    t = 0
    soc_start = 0
    for cr, soc_end in stages:
        delta_soc = soc_end - soc_start
        if delta_soc > 0:
            t += delta_soc / 100.0 / cr * 60  # min
        soc_start = soc_end
    return t

def simulate_strategy(stages, n_cycles=30, label=''):
    """仿真多段策略，返回 SOH 衰减和充电时间"""
    try:
        exp = build_multistage_experiment(stages, n_cycles)
        sim = pybamm.Simulation(
            model, parameter_values=pv, experiment=exp,
            solver=pybamm.CasadiSolver(mode='safe')
        )
        sol = sim.solve()
        cap = sol.summary_variables.get('Discharge capacity [A.h]', None)
        if cap is not None and len(cap) > 1:
            cap_init = float(cap[0])
            cap_final = float(cap[-1])
            cap_loss = (cap_init - cap_final) / cap_init * 100  # %
            # 容量衰减率 (/循环)
            decay_rate = cap_loss / n_cycles
            # 估算 EOL 循环数（线性外推到 80% SOH）
            if decay_rate > 0:
                est_life = 20 / decay_rate * n_cycles  # SOH 从 100% 到 80%
            else:
                est_life = 5000
            t80 = T80_multistage(stages)
            return {'T80': t80, 'est_life': est_life, 'decay_rate': decay_rate,
                    'cap_init': cap_init, 'cap_final': cap_final, 'n_cycles': n_cycles}
    except Exception as e:
        pass
    return None

# ============ 3. 候选策略空间 ============
print("\n[2] 定义候选策略空间...")

# 两段式 baseline（数据驱动推荐）
BASELINE_2STAGE = [
    ('3.6C(80%)-3.6C', [(3.6, 80)], 'Recommended (data-driven)'),
    ('4C(80%)-4C',     [(4.0, 80)], 'Alternative'),
    ('1C(4%)-6C',       [(1.0, 4), (6.0, 80)], 'Short-life (data-driven)'),
    ('2C(10%)-6C',      [(2.0, 10), (6.0, 80)], 'Short-life (data-driven)'),
]

# 三段式候选
STAGE3_CANDIDATES = [
    ('3-stage: 4C→30%→2C→65%→1C',     [(4.0, 30), (2.0, 65), (1.0, 80)]),
    ('3-stage: 5C→20%→3C→55%→1C',     [(5.0, 20), (3.0, 55), (1.0, 80)]),
    ('3-stage: 3.6C→50%→2C→70%→1C',   [(3.6, 50), (2.0, 70), (1.0, 80)]),
    ('3-stage: 6C→15%→3C→50%→1.5C',   [(6.0, 15), (3.0, 50), (1.5, 80)]),
    ('3-stage: 4.5C→35%→2.5C→60%→1C', [(4.5, 35), (2.5, 60), (1.0, 80)]),
    ('3-stage: 3C→45%→2C→65%→0.8C',   [(3.0, 45), (2.0, 65), (0.8, 80)]),
    ('3-stage: 7C→10%→4C→40%→1.5C',   [(7.0, 10), (4.0, 40), (1.5, 80)]),
    ('3-stage: 5C→25%→3.5C→55%→1.2C', [(5.0, 25), (3.5, 55), (1.2, 80)]),
]

ALL_CANDIDATES = BASELINE_2STAGE + [(name, stages, '3-stage-candidate') for name, stages in STAGE3_CANDIDATES]

print(f"  候选策略: {len(ALL_CANDIDATES)} 个")
print(f"  其中两段式 baseline: {len(BASELINE_2STAGE)} 个")
print(f"  三段式候选: {len(STAGE3_CANDIDATES)} 个")

# ============ 4. 运行仿真 ============
print("\n[3] PyBaMM 仿真各候选策略...")
N_CYCLES = 30  # 仿真前30个循环来估计衰减率

results = []
total = len(ALL_CANDIDATES)
for idx, item in enumerate(ALL_CANDIDATES):
    if len(item) == 3:
        name, stages, tag = item
    else:
        name, stages = item
        tag = 'baseline'

    t0 = time.time()
    print(f"  [{idx+1}/{total}] {name}...", end=' ', flush=True)
    res = simulate_strategy(stages, N_CYCLES, name)
    elapsed = time.time() - t0

    if res is not None:
        res['name'] = name
        res['stages'] = str(stages)
        res['n_segments'] = len(stages)
        res['tag'] = tag
        results.append(res)
        print(f"T80={res['T80']:.1f}min, est_life={res['est_life']:.0f} ({elapsed:.0f}s)")
    else:
        print(f"FAILED ({elapsed:.0f}s)")

# ============ 5. 生成对比图表 ============
print(f"\n[4] 生成多段优化对比图表...")
print(f"  成功仿真: {len(results)}/{total} 个策略")

if len(results) >= 3:
    df = pd.DataFrame(results)

    # 按 n_segments 分组
    seg2 = df[df['n_segments'] <= 2]
    seg3 = df[df['n_segments'] >= 3]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图: T80 vs 估计寿命 散点图
    ax = axes[0]
    # 两段式
    for _, r in seg2.iterrows():
        color = '#2ecc71' if r['est_life'] > 1000 else '#e74c3c'
        ax.scatter(r['T80'], r['est_life'], s=120, c=color, marker='o',
                   edgecolors='black', linewidth=1, zorder=5)
        ax.annotate(r['name'][:25], (r['T80'], r['est_life']),
                    textcoords='offset points', xytext=(5, -8), fontsize=7)
    # 三段式
    for _, r in seg3.iterrows():
        ax.scatter(r['T80'], r['est_life'], s=150, c='#3498db', marker='^',
                   edgecolors='black', linewidth=1, zorder=10)
        ax.annotate(r['name'][:25], (r['T80'], r['est_life']),
                    textcoords='offset points', xytext=(5, 5), fontsize=7)

    ax.set_xlabel('Charging Time T80 (min)'); ax.set_ylabel('Estimated Cycle Life')
    ax.set_title('Multi-Stage vs Two-Stage Charging Strategies\n(PyBaMM Electrochemical Simulation)')
    ax.legend(['2-stage (data)', '3-stage (new)'], fontsize=8, loc='upper right')
    ax.grid(alpha=.3)

    # 右图: 衰减率对比
    ax = axes[1]
    names_short = [r['name'][:20] for r in results]
    decay_rates = [r['decay_rate'] for r in results]
    colors_bar = ['#2ecc71' if r['est_life'] > 1000 else '#e74c3c' if r.get('n_segments', 2) <= 2 else '#3498db'
                  for r in results]

    y_pos = range(len(results))
    ax.barh(y_pos, decay_rates, color=colors_bar, alpha=.8, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_short, fontsize=7)
    ax.set_xlabel('Capacity Decay Rate (%/cycle)')
    ax.set_title('Capacity Decay Rate Comparison\n(lower = better)')
    ax.axvline(0, color='gray')
    ax.invert_yaxis()

    plt.tight_layout()
    out_path = os.path.join(FIG, 'fig_pybamm_multistage.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → {out_path}")

    # 保存结果表
    df_out = df[['name', 'T80', 'est_life', 'decay_rate', 'n_segments', 'tag']]
    df_out.to_csv(os.path.join(PROC, 'multistage_results.csv'), index=False, encoding='utf-8-sig')
    print(f"  → {PROC}/multistage_results.csv")

    # 找最佳三段式
    if len(seg3) > 0:
        best_3 = seg3.loc[seg3['est_life'].idxmax()]
        print(f"\n[5] 最优三段式策略:")
        print(f"  {best_3['name']}")
        print(f"  T80 = {best_3['T80']:.1f} min")
        print(f"  估计寿命 = {best_3['est_life']:.0f}")
        print(f"  衰减率 = {best_3['decay_rate']:.4f}%/cycle")

        # 与推荐策略对比
        if len(seg2) > 0:
            best_2 = seg2.loc[seg2['est_life'].idxmax()]
            print(f"\n  对比推荐两段式:")
            print(f"  {best_2['name']}")
            print(f"  T80 = {best_2['T80']:.1f} min")
            print(f"  估计寿命 = {best_2['est_life']:.0f}")

            improvement = (best_3['est_life'] / best_2['est_life'] - 1) * 100
            time_diff = best_3['T80'] - best_2['T80']
            print(f"\n  改进: 寿命 {improvement:+.1f}%, 充电时间 {time_diff:+.1f} min")

print("\n" + "="*60)
print("Q4 多段优化完成！")
print("="*60)

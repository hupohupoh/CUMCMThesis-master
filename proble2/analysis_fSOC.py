"""
Analyze f(SOC) - the damage weighting function for Problem 2.
Determines how damage per unit charge varies with SOC.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from scipy.optimize import curve_fit

df = pd.read_csv('../problem1_results/battery_data_clean.csv')

# Use standard protocol only (batch1 + batch2)
std = df[df['batch'].isin(['batch1','batch2'])].copy()
print(f'Standard batteries: {len(std)}')

std['logL'] = np.log10(std['cycle_life'])
std['m1'] = std['C1'] * std['Q1'] / 100
std['m2'] = std['C2'] * (80 - std['Q1']) / 100

# ========================================
# 1. REGRESSION EQUATION
# ========================================
print('\n' + '='*60)
print('1. DOSE MODEL REGRESSION')
print('='*60)

X_dose = std[['m1','m2']]
y = std['logL']
lr_dose = LinearRegression()
lr_dose.fit(X_dose, y)
r2_dose = lr_dose.score(X_dose, y)
print(f'log10(L) = {lr_dose.intercept_:.4f} + ({lr_dose.coef_[0]:.4f})*m1 + ({lr_dose.coef_[1]:.4f})*m2')
print(f'R2 = {r2_dose:.4f}')
print(f'm1 covers low SOC [0 to Q1%], m2 covers mid-high SOC [Q1% to 80%]')
print(f'Damage ratio (high/low SOC) = {abs(lr_dose.coef_[1]/lr_dose.coef_[0]):.3f}')
print(f'Meaning: charging at high SOC is {abs(lr_dose.coef_[1]/lr_dose.coef_[0]):.1f}x more damaging per unit charge')

# ========================================
# 2. ESTIMATE f(SOC) FROM Q1-BINNED REGRESSIONS
# ========================================
print('\n' + '='*60)
print('2. Q1-BIN ANALYSIS: infer f(SOC) shape')
print('='*60)

bins = [0, 15, 25, 35, 45, 55, 65, 80]
labels = ['0-15%','15-25%','25-35%','35-45%','45-55%','55-65%','65-80%']
std['Q1_bin'] = pd.cut(std['Q1'], bins=bins, labels=labels)

# For each Q1 bin, regress logL ~ m1 + m2
# The coefficient -c1 is avg f over [0, Q1_mid]
# The coefficient -c2 is avg f over [Q1_mid, 80]
print(f'\n{"Q1_bin":<12s} {"n":>5s} {"Q1_mid":>8s} {"f_low":>10s} {"f_high":>10s} {"ratio":>8s} {"R2":>8s}')
print('-'*70)

bin_records = []
for label in labels:
    sub = std[std['Q1_bin'] == label].copy()
    if len(sub) < 5:
        continue
    X = sub[['m1','m2']]
    y_sub = sub['logL']
    if X['m1'].std() < 0.01 or X['m2'].std() < 0.01:
        continue
    lr = LinearRegression()
    lr.fit(X, y_sub)
    r2 = lr.score(X, y_sub)
    f_low = -lr.coef_[0]   # damage per unit m1
    f_high = -lr.coef_[1]  # damage per unit m2
    ratio = f_high / f_low if f_low > 0.01 else float('nan')
    bin_records.append({
        'Q1_range': label,
        'Q1_mid': sub['Q1'].mean(),
        'n': len(sub),
        'f_low': f_low,
        'f_high': f_high,
        'ratio': ratio,
        'R2': r2
    })
    print(f'{label:<12s} {len(sub):>5d} {sub["Q1"].mean():>8.1f} {f_low:>10.4f} {f_high:>10.4f} {ratio:>8.2f} {r2:>8.3f}')

bin_df = pd.DataFrame(bin_records)

# Check if f_low increases with Q1 (would indicate increasing f(SOC))
if len(bin_df) >= 3:
    sr, sp = spearmanr(bin_df['Q1_mid'], bin_df['f_low'])
    print(f'\nSpearman r(Q1_mid, f_low) = {sr:.4f}, p = {sp:.4f}')
    if sr > 0.3 and sp < 0.1:
        print('=> f(SOC) INCREASES with SOC (damage per unit charge grows at higher SOC)')
    elif abs(sr) < 0.3:
        print('=> f(SOC) is roughly constant (no strong SOC dependence in this data)')
    else:
        print(f'=> Unexpected pattern (r={sr:.3f})')

# ========================================
# 3. FIT CANDIDATE f(SOC) FUNCTIONS
# ========================================
print('\n' + '='*60)
print('3. FITTING f(SOC) FUNCTIONAL FORMS')
print('='*60)

# We have aggregate constraints:
# For each battery: log10(L) = const - [C1*int_0^Q1 f(s)ds + C2*int_Q1^80 f(s)ds]
# The dose model gives us: int_0^Q1 f(s)ds / Q1 = f_low_effective
# and int_Q1^80 f(s)ds / (80-Q1) = f_high_effective
# With ratio f_high/f_low ~ 1.15-1.16 from pooled model

# Candidate functions normalized so f(0) = base_level:
# Form A: f(s) = a + b*s                    (linear)
# Form B: f(s) = a + b*ln(1 + s/c)          (logarithmic saturating)
# Form C: f(s) = a * (1 + b*s)              (linear multiplicative)

# From the pooled dose model:
# avg f over [0, Q1_mean] proportional to |coef_m1|
# avg f over [Q1_mean, 80] proportional to |coef_m2|
# ratio ~ 1.15

# For the full dataset, mean Q1 ~ 45%
q1_mean = std['Q1'].mean()
print(f'\nMean Q1 across standard batteries: {q1_mean:.1f}%')
print(f'f_low (avg over [0,{q1_mean:.0f}%]) / f_high (avg over [{q1_mean:.0f}%,80%]) ratio = {abs(lr_dose.coef_[1]/lr_dose.coef_[0]):.3f}')

# Let's try to fit from the Q1-bin data
# For each bin j, we have:
# avg_f_low_j = (1/Q1_j) * integral_0^{Q1_j} f(s) ds
# avg_f_high_j = (1/(80-Q1_j)) * integral_{Q1_j}^{80} f(s) ds

print('\nUsing Q1-bin results to fit f(SOC) functional forms...\n')

# Prepare data for fitting
valid = bin_df[bin_df['n'] >= 5].copy()
if len(valid) >= 3:
    Q1_vals = valid['Q1_mid'].values
    f_low_vals = valid['f_low'].values
    f_high_vals = valid['f_high'].values

    # For each Q1, we have:
    # area_low(Q1) = f_low * Q1 = integral_0^{Q1} f(s) ds
    # area_high(Q1) = f_high * (80-Q1) = integral_{Q1}^{80} f(s) ds
    # total_area = area_low + area_high = integral_0^{80} f(s) ds
    area_low = f_low_vals * Q1_vals
    area_high = f_high_vals * (80 - Q1_vals)
    total_area = area_low + area_high

    print(f'{"Q1":>8s} {"f_low":>10s} {"f_high":>10s} {"area_low":>10s} {"area_high":>10s} {"total":>10s}')
    for i in range(len(Q1_vals)):
        print(f'{Q1_vals[i]:>8.1f} {f_low_vals[i]:>10.4f} {f_high_vals[i]:>10.4f} {area_low[i]:>10.4f} {area_high[i]:>10.4f} {total_area[i]:>10.4f}')

    # If f(SOC) is constant, then f_low = f_high and total_area is constant
    # Check variability of total_area
    print(f'\nTotal area mean: {total_area.mean():.4f}, std: {total_area.std():.4f}')
    print(f'CV of total area: {total_area.std()/total_area.mean()*100:.1f}%')

    # If f increases with SOC, area_low/Q1 (=f_low) < area_high/(80-Q1) (=f_high)
    # Check if f_low systematically < f_high
    n_higher = np.sum(f_high_vals > f_low_vals)
    print(f'f_high > f_low in {n_higher}/{len(Q1_vals)} bins')

# ========================================
# 4. DIRECT FIT: estimate f(SOC) as piecewise linear from data
# ========================================
print('\n' + '='*60)
print('4. FITTING PIECEWISE LINEAR f(SOC) DIRECTLY')
print('='*60)

# Model: log10(L) = a0 - [C1 * F(Q1) + C2 * (F(80) - F(Q1))]
# where F(s) = integral_0^s f(t) dt is the cumulative damage function
# f(s) = dF/ds is the marginal damage at SOC = s

# If f(s) = alpha + beta*s, then F(s) = alpha*s + beta*s^2/2
# So: C1 * (alpha*Q1 + beta*Q1^2/2) + C2 * [alpha*80 + beta*80^2/2 - alpha*Q1 - beta*Q1^2/2]
# Rearranging by collecting alpha and beta terms...

# Actually, let me just do a grid search or nonlinear fit on a simpler model
# The dose model already shows f varies by ~16% between low and high SOC

# Let's try a model with explicit f(SOC) functional forms
# Model: log10(L) = b0 - b1 * [C1*int_0^{Q1} (1 + gamma*s) ds + C2*int_{Q1}^{80} (1 + gamma*s) ds]
# where f(s) = 1 + gamma*s is the linear damage function
# int_0^{Q1} (1+gamma*s) ds = Q1 + gamma*Q1^2/2
# int_{Q1}^{80} (1+gamma*s) ds = (80-Q1) + gamma*(80^2 - Q1^2)/2

# Build the transformed features
Q1 = std['Q1'].values
C1 = std['C1'].values
C2 = std['C2'].values

# For linear f(s) = 1 + gamma*s:
# effective dose = C1*(Q1 + gamma*Q1^2/2) + C2*((80-Q1) + gamma*(80^2 - Q1^2)/2)
# = m1 + m2 + gamma * [C1*Q1^2/2 + C2*(80^2 - Q1^2)/2]

# Normalize Q1 to 0-1 range (SOC as fraction 0 to 0.8)
soc1 = Q1 / 100  # Q1 as fraction
soc_max = 0.8

# Base dose: m1 + m2 = C1*Q1/100 + C2*(80-Q1)/100
base_dose = std['m1'].values + std['m2'].values

# Gamma dose: C1*Q1^2/(2*100^2) + C2*((80/100)^2 - Q1^2/100^2)/2
# = C1*soc1^2/2 + C2*(soc_max^2 - soc1^2)/2
gamma_dose = C1 * soc1**2 / 2 + C2 * (soc_max**2 - soc1**2) / 2

print('\nFitting: log10(L) = b0 - b1*[m1+m2] - b2*[gamma_term]')
print('where gamma_term captures SOC-dependent extra damage')
print()

from sklearn.linear_model import LinearRegression
X_f = np.column_stack([base_dose, gamma_dose])
lr_f = LinearRegression()
lr_f.fit(X_f, y)
r2_f = lr_f.score(X_f, y)

print(f'log10(L) = {lr_f.intercept_:.4f} - {abs(lr_f.coef_[0]):.4f}*(m1+m2) - {abs(lr_f.coef_[1]):.4f}*gamma')
print(f'R2 = {r2_f:.4f}')
print(f'Compare to base dose model R2 = {r2_dose:.4f}')

# The effective f(SOC) = b1 + b2*SOC (in normalized SOC fraction units)
# Marginal damage at SOC=s: f(s) = |coef_0| + |coef_1| * s
# At s=0: f(0) = |coef_0|
# At s=0.8 (80%): f(0.8) = |coef_0| + 0.8*|coef_1|
b0_f = abs(lr_f.coef_[0])
b1_f = abs(lr_f.coef_[1])
print(f'\nImplied f(SOC) = {b0_f:.4f} + {b1_f:.4f} * SOC (SOC as fraction 0-0.8)')
print(f'f(0) = {b0_f:.4f}')
print(f'f(0.8) = {b0_f + 0.8*b1_f:.4f}')
print(f'Ratio f(0.8)/f(0) = {(b0_f + 0.8*b1_f)/b0_f:.3f}')

# ========================================
# 5. LOGARITHMIC f(SOC) fit
# ========================================
print('\n' + '='*60)
print('5. LOGARITHMIC f(SOC) FIT')
print('='*60)
print('f(SOC) = a + b*ln(1 + SOC/c)')

# For logarithmic: F(s) = a*s + b*[(s+c)*ln(1+s/c) - s]
# This is complex. Let's use a simpler approach:
# Assume f(s) = a * (1 + k*ln(1 + s/s0))
# where s0 is a reference SOC level

# Simpler: try a few gamma values for logarithmic weighting
# The dose = C1*int_0^Q1 (1 + gamma*ln(1+s/s0)) ds + C2*int_Q1^80 (...)

# Let me just try a simple grid search for the best logarithmic parameter
print('\nTrying different SOC-weighting schemes...')

# Compare: uniform vs linear vs log weighting
candidates = {}

# Model 0: uniform (no SOC dependence)
# logL = b0 + b1*(m1 + m2)
X0 = (std['m1'] + std['m2']).values.reshape(-1,1)
lr0 = LinearRegression(); lr0.fit(X0, y)
candidates['uniform'] = {'R2': lr0.score(X0, y), 'coef': lr0.coef_[0]}

# Model 1: linear f(SOC) = 1 + k*SOC
best_r2_lin = 0; best_k_lin = 0
for k in np.linspace(0, 5, 101):
    dose_k = C1 * (soc1 + k*soc1**2/2) + C2 * ((soc_max - soc1) + k*(soc_max**2 - soc1**2)/2)
    Xk = dose_k.reshape(-1,1)
    lrk = LinearRegression(); lrk.fit(Xk, y)
    r2k = lrk.score(Xk, y)
    if r2k > best_r2_lin:
        best_r2_lin = r2k; best_k_lin = k
candidates['linear f(SOC)'] = {'R2': best_r2_lin, 'k_best': best_k_lin}

# Model 2: logarithmic f(SOC) = 1 + k*ln(1+SOC/s0)
# int_0^x ln(1+t/a) dt = (a+x)*ln(1+x/a) - x
# Let s0 = 0.1 (10% SOC as reference)
s0 = 0.1
best_r2_log = 0; best_k_log = 0
for k in np.linspace(0, 3, 61):
    # int_0^soc1 ln(1+t/s0) dt = (s0+soc1)*ln(1+soc1/s0) - soc1
    int1 = (s0 + soc1) * np.log(1 + soc1/s0) - soc1
    int_total = (s0 + soc_max) * np.log(1 + soc_max/s0) - soc_max
    int2 = int_total - int1
    dose_k = C1*(soc1 + k*int1) + C2*((soc_max - soc1) + k*int2)
    Xk = dose_k.reshape(-1,1)
    lrk = LinearRegression(); lrk.fit(Xk, y)
    r2k = lrk.score(Xk, y)
    if r2k > best_r2_log:
        best_r2_log = r2k; best_k_log = k
candidates['log f(SOC)'] = {'R2': best_r2_log, 'k_best': best_k_log}

# Model 3: dose model (separate m1, m2) - the benchmark
candidates['dose (m1,m2 separate)'] = {'R2': r2_dose}

print('\nModel comparison:')
for name, res in candidates.items():
    print(f'  {name:<25s}: R2={res["R2"]:.4f}', end='')
    if 'k_best' in res:
        print(f', k_best={res["k_best"]:.3f}', end='')
    if 'coef' in res:
        print(f', coef={res["coef"]:.4f}', end='')
    print()

# ========================================
# 6. SUMMARY: RECOMMENDED f(SOC)
# ========================================
print('\n' + '='*60)
print('6. RECOMMENDED f(SOC) FUNCTION')
print('='*60)

# From the dose model directly:
# f(SOC) is approximately piecewise constant:
#   f_low ~ 0.367 (for SOC 0 to ~45%)
#   f_high ~ 0.424 (for SOC ~45% to 80%)
# Or continuously:
#   f(SOC) increases by about 15-20% from 0% to 80% SOC

print(f'''
Based on the analysis:

1. The dose model shows that the damage coefficient for high-SOC charging
   (m2 coef = {abs(lr_dose.coef_[1]):.4f}) is {abs(lr_dose.coef_[1]/lr_dose.coef_[0]):.1f}x the coefficient
   for low-SOC charging (m1 coef = {abs(lr_dose.coef_[0]):.4f}).

2. This implies f(SOC) INCREASES with SOC - charging at higher SOC
   causes more damage per unit of charge passed.

3. Recommended functional form:
   f(SOC) = a + b*SOC   (linear, with small positive b)
   or equivalently, normalize so f(0) = 1:
   f(SOC) = 1 + gamma*SOC
   where gamma ~ {best_k_lin:.2f} (from grid search optimization)
   This gives f(0.8)/f(0) ~ {1+0.8*best_k_lin:.2f}

4. The logarithmic (saturating) form also works well:
   f(SOC) = 1 + k*ln(1 + SOC/0.1)
   with k ~ {best_k_log:.2f}

5. Key physical interpretation:
   As SOC increases, the anode potential drops closer to the lithium
   plating potential. Each unit of charge current becomes increasingly
   likely to cause lithium plating side reactions rather than intercalation.
   The damage rate increases but may saturate at very high SOC.
''')

# Save for LaTeX
print('\n=== KEY NUMBERS FOR LATEX ===')
print(f'Dose model R2: {r2_dose:.4f}')
print(f'm1 coefficient: {lr_dose.coef_[0]:.4f}')
print(f'm2 coefficient: {lr_dose.coef_[1]:.4f}')
print(f'Damage ratio m2/m1: {abs(lr_dose.coef_[1]/lr_dose.coef_[0]):.3f}')
print(f'Linear f(SOC) gamma: {best_k_lin:.3f}')
print(f'Log f(SOC) k: {best_k_log:.3f}')
print(f'f(0.8)/f(0) ratio: {1+0.8*best_k_lin:.2f}')

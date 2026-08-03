"""
Physical model analysis for Problem 2.
Model: D_eff = C1*F(Q1) + C2*[F(80)-F(Q1)],  f(SOC) linear = 1 + alpha*SOC/100
log10(L) = beta0 - beta*D_eff
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

df = pd.read_csv('../problem1_results/battery_data_clean.csv')
std = df[df['batch'].isin(['batch1','batch2'])].copy()
print(f'standard batteries: n={len(std)}')

std['logL'] = np.log10(std['cycle_life'])
std['m1'] = std['C1']*std['Q1']/100
std['m2'] = std['C2']*(80-std['Q1'])/100

print('='*65)
print('STEP 1: dose model log10(L) ~ m1 + m2  (diagnostic)')
print('='*65)
Xd = std[['m1','m2']]
lr_dose = LinearRegression().fit(Xd, std['logL'])
print(f'  log10(L) = {lr_dose.intercept_:.4f} + ({lr_dose.coef_[0]:.4f})*m1 + ({lr_dose.coef_[1]:.4f})*m2')
print(f'  R2 = {lr_dose.score(Xd,std.logL):.4f}')
r1, r2 = lr_dose.coef_[0], lr_dose.coef_[1]
print(f'  |b2|/|b1| = {abs(r2/r1):.4f}')

# fit ratio from coefficients
ratio = abs(r2/r1)

# Calibrate alpha for linear f(SOC) using ratio constraint at mean Q1
s1_mean = std['Q1'].mean()/100
alpha = 2*(ratio-1) / ((0.8+s1_mean) - ratio*s1_mean)
print(f'\n  mean Q1 = {s1_mean*100:.1f}%, calibrate alpha = {alpha:.4f}')
print(f'  f(SOC) = 1 + {alpha:.3f}*SOC/100   f(0)=1, f(80%)={1+alpha*0.8:.3f}')

print('='*65)
print('STEP 2: physical model  log10(L) = beta0 - beta*D_eff')
print('='*65)
C1=std['C1'].values; C2=std['C2'].values; Q1=std['Q1'].values
s1=Q1/100  # fraction 0..0.8
# F(s) for linear f: F(s)=s+alpha*s^2/2  (s in fraction)
F1 = s1 + alpha*s1**2/2
Ft = (0.8 + alpha*0.8**2/2)
Deff = C1*F1 + C2*(Ft-F1)
Xp = Deff.reshape(-1,1)
lr_phys = LinearRegression().fit(Xp, std['logL'])
beta0, beta = lr_phys.intercept_, -lr_phys.coef_[0]
print(f'  log10(L) = {beta0:.4f} - {beta:.4f} * D_eff')
print(f'  R2 = {lr_phys.score(Xp,std.logL):.4f}')

# Expand D_eff in terms of C1,Q1,C2 (SOC in %)
# F(Q1) = Q1 + alpha*Q1^2/20000  ... derive carefully:
# s in %,  f(SOC)=1+alpha*SOC/100 => F(q)=q + alpha*q^2/200  (q in %)
# check: F(q) = q + alpha*q^2/(2*100) = q + alpha*q^2/200
# For q=80: F(80)=80+alpha*6400/200=80+alpha*32
print('\n  Expanded in C1,Q1,C2 (Q1 in %):')
print(f'  F(Q1) = Q1 + {alpha:.4f}*Q1^2/200')
print(f'  F(80) = 80 + {alpha*32:.4f}')
print(f'  D_eff = C1*[Q1 + {alpha:.4f}*Q1^2/200] + C2*[80-Q1 + {alpha:.4f}*(6400-Q1^2)/200]')

print('='*65)
print('STEP 3: marginal effects (partial derivatives)')
print('='*65)
print('  d(D_eff)/dC1 = F(Q1) > 0    -> C1 up shortens life')
print('  d(D_eff)/dC2 = F(80)-F(Q1) > 0  -> C2 up shortens life')
print('  d(D_eff)/dQ1 = (C1-C2)*f(Q1)  -> sign depends on C1 vs C2')
# numeric marginal effect at mean values
q1m = std['Q1'].mean(); c1m=std['C1'].mean(); c2m=std['C2'].mean()
dC1 = F1.mean()  # ~ avg F(Q1) but use mean of F1
print(f'\n  At mean C1={c1m:.2f}, Q1={q1m:.1f}%, C2={c2m:.2f}:')
print(f'    F(Q1) mean = {F1.mean():.4f}   (C1 marginal, small)')
print(f'    F(80)-F(Q1) mean = {(Ft-F1).mean():.4f}  (C2 marginal, larger)')
print(f'    ratio (C2 marg / C1 marg) = {(Ft-F1).mean()/F1.mean():.3f}')
print(f'    dD_eff/dQ1 = (C1-C2)*f(Q1_mean) = ({c1m}-{c2m})*{1+alpha*q1m/100:.3f} = {(c1m-c2m)*(1+alpha*q1m/100):.3f}')

print('='*65)
print('STEP 4: effect on log10(L) per unit change (beta * marginal)')
print('='*65)
print(f'  beta = {beta:.4f}')
print(f'  C1: dlog10(L)/dC1 = -beta*F(Q1) = -{beta:.4f}*{F1.mean():.4f} = {-beta*F1.mean():.4f}')
print(f'  C2: dlog10(L)/dC2 = -beta*[F(80)-F(Q1)] = -{beta:.4f}*{(Ft-F1).mean():.4f} = {-beta*(Ft-F1).mean():.4f}')
print(f'  Q1: dlog10(L)/dQ1 = -beta*(C1-C2)*f(Q1) = -{beta:.4f}*{(c1m-c2m):.3f}*{1+alpha*q1m/100:.3f} = {-beta*(c1m-c2m)*(1+alpha*q1m/100):.4f}')

# Which factor dominates in observed data range?
print('\n  Observed ranges:')
print(f'    C1: {std.C1.min():.1f}~{std.C1.max():.1f} (range {std.C1.max()-std.C1.min():.1f})')
print(f'    Q1: {std.Q1.min():.0f}~{std.Q1.max():.0f} (range {std.Q1.max()-std.Q1.min():.0f})')
print(f'    C2: {std.C2.min():.1f}~{std.C2.max():.1f} (range {std.C2.max()-std.C2.min():.1f})')
delta_logL_C1 = -beta*F1.mean()*(std.C1.max()-std.C1.min())
delta_logL_C2 = -beta*(Ft-F1).mean()*(std.C2.max()-std.C2.min())
print(f'  Total log10(L) swing across observed range:')
print(f'    C1: {delta_logL_C1:+.3f}  (寿命×{10**delta_logL_C1:.2f})')
print(f'    C2: {delta_logL_C2:+.3f}  (寿命×{10**delta_logL_C2:.2f})')

print('='*65)
print('STEP 5: compare with uniform-charge benchmark')
print('='*65)
# uniform: D0 = C1*Q1/100 + C2*(80-Q1)/100 = m1+m2
D0 = std['m1']+std['m2']
X0 = D0.values.reshape(-1,1)
lr0 = LinearRegression().fit(X0, std['logL'])
print(f'  uniform: log10(L) = {lr0.intercept_:.4f} - {abs(lr0.coef_[0]):.4f}*(m1+m2), R2={lr0.score(X0,std.logL):.4f}')
print(f'  dose(m1,m2): R2={lr_dose.score(Xd,std.logL):.4f}')
print(f'  physical(linear f): R2={lr_phys.score(Xp,std.logL):.4f}')

print('\n=== KEY NUMBERS FOR LATEX ===')
print(f'dose b1={r1:.4f}, b2={r2:.4f}, ratio={ratio:.3f}, R2_dose={lr_dose.score(Xd,std.logL):.4f}')
print(f'alpha={alpha:.4f}, f(0)=1, f(40%)={1+alpha*0.4:.3f}, f(80%)={1+alpha*0.8:.3f}')
print(f'beta0={beta0:.4f}, beta={beta:.4f}, R2_phys={lr_phys.score(Xp,std.logL):.4f}')
print(f'F(Q1)_mean={F1.mean():.4f}, [F80-F1]_mean={(Ft-F1).mean():.4f}')
print(f'dlogL/dC1={-beta*F1.mean():.4f}, dlogL/dC2={-beta*(Ft-F1).mean():.4f}')

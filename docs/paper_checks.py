"""Численные проверки к рецензии статьи (docs/paper_review.md).

Запуск из корня репозитория:  PYTHONPATH=. python docs/paper_checks.py  (~3 мин)
"""
import time, numpy as np
from scipy.optimize import least_squares
from cathode0d import CathodeConfig, solve, evaluate, make_transform, RESIDUAL_NAMES, VAR_NAMES

KW = dict(n_solutions=10, n_particles=400, n_iter=300)
print("=== 1. Ширина полосы зависит от границ поиска (2 А, 0.4 мг/с) ===")
base = CathodeConfig(Id=2.0, mdot=0.4e-6)
wide = CathodeConfig(Id=2.0, mdot=0.4e-6)
wide.bounds = dict(base.bounds, Ts=(1500.0, 2100.0), Tw_o=(1500.0, 2100.0), Vp=(1.0, 80.0))
narrow = CathodeConfig(Id=2.0, mdot=0.4e-6)
narrow.bounds = dict(base.bounds, Ts=(1750.0, 1800.0), Vp=(10.0, 20.0))
for name, cfg in [("таблица 1", base), ("шире", wide), ("уже", narrow)]:
    s = solve(cfg, seed=3, **KW)
    print(f"{name:10s} V_tot={s.mean_std('V_tot')[0]:.1f}±{s.mean_std('V_tot')[1]:.1f}  "
          f"Ts={s.mean_std('Ts')[0]:.0f}±{s.mean_std('Ts')[1]:.0f}  Vp={s.mean_std('Vp')[0]:.1f}±{s.mean_std('Vp')[1]:.1f}  "
          f"Jmax={s.J.max():.0e}  Te_o={s.mean_std('Te_o')[0]:.3f}±{s.mean_std('Te_o')[1]:.3f}")

print("\n=== 2. Свободные параметры: при любой Ts из диапазона находится точное решение ===")
for Ts in [1700, 1775, 1850, 1950]:
    cfg = CathodeConfig(Id=2.0, mdot=0.4e-6)
    cfg.bounds = dict(cfg.bounds, Ts=(Ts - 1e-6, Ts + 1e-6))
    s = solve(cfg, seed=4, n_solutions=3, n_particles=400, n_iter=300)
    print(f"Ts={Ts}  Jmax={s.J.max():.0e}  V_tot={s.mean_std('V_tot')[0]:.1f}  Vp={s.mean_std('Vp')[0]:.1f}  Leff={1e3*s.mean_std('Leff')[0]:.2f} мм")

print("\n=== 3. Обычный least_squares (Trust Region) с 20 случайных стартов ===")
cfg = CathodeConfig(Id=2.0, mdot=0.4e-6)
to_phys, _ = make_transform(cfg)
def res(u):
    o = evaluate(cfg, to_phys(u[None, :]))
    return np.array([o[n][0] for n in RESIDUAL_NAMES])
rng = np.random.default_rng(0); ok = 0; t = time.time(); Vs = []
for _ in range(20):
    r = least_squares(res, rng.random(10), bounds=(0, 1), xtol=1e-14, ftol=1e-14)
    J = np.abs(r.fun).sum()
    if J < 1e-6:
        ok += 1; Vs.append(evaluate(cfg, to_phys(r.x[None]))["V_tot"][0])
print(f"сошлось {ok}/20 за {time.time()-t:.1f} с; V_tot = {np.mean(Vs):.1f}±{np.std(Vs):.1f} В")
t = time.time(); s = solve(cfg, seed=0, **KW); print(f"PSO: 10 решений за {time.time()-t:.1f} с ({s.n_runs} прогонов)")

print("\n=== 4. n_ne определяется формулой Таунея алгебраически ===")
o = s.details
print("отношение p_e/p_taunay:", np.round(o['p_e']/o['p_taunay'], 6))

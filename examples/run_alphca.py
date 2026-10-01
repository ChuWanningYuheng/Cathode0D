"""Воспроизведение раздела 4 статьи: катод AlphCA, токи 1-3 А, расход 0.08-1 мг/с.

Запуск:  python examples/run_alphca.py [--quick]
Результаты: examples/out/results.csv и графики examples/out/*.png
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cathode0d import CathodeConfig, solve, residual_report, VAR_NAMES, VAR_UNITS  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out")

# Экспериментальное напряжение разряда (серия test3 из [27]), оцифровано
# приблизительно с рис. 4 и 9 статьи: {ток, А: [(мг/с, В), ...]}
EXPERIMENT = {
    1.0: [(0.08, 27.5), (0.1, 25.0), (0.4, 19.5), (1.0, 17.0)],
    2.0: [(0.08, 22.0), (0.1, 20.6), (0.4, 16.2), (1.0, 14.5)],
    3.0: [(0.08, 19.0), (0.1, 17.5), (0.4, 14.2), (1.0, 13.8)],
}

# Категориальные цвета (слоты 1-3 эталонной палитры) — по одному на ток
COLORS = {1.0: "#2a78d6", 2.0: "#eb6834", 3.0: "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

KEYS = ["V_tot", "n_ee", "n_ne", "Te_e", "Ts", "n_eo", "n_no", "Te_o", "Tw_o",
        "n_ek", "n_nk", "Te_k", "Leff", "Vp", "V_ds", "J"]


def run(currents, flows, pso_kw):
    rows = []
    for Id in currents:
        for m in flows:
            t = time.time()
            sol = solve(CathodeConfig(Id=Id, mdot=m * 1e-6), **pso_kw)
            row = {"Id": Id, "mdot_mg_s": m, "n_accepted": len(sol.J), "n_runs": sol.n_runs}
            for k in KEYS:
                row[k + "_mean"], row[k + "_std"] = sol.mean_std(k)
            rows.append(row)
            print(f"I={Id:.0f} A  mdot={m:.2f} mg/s  V={row['V_tot_mean']:.1f}±{row['V_tot_std']:.1f} V"
                  f"  Te_e={row['Te_e_mean']:.2f} eV  max J={sol.J.max():.1e}"
                  f"  ({len(sol.J)}/{sol.n_runs} прогонов, {time.time()-t:.0f} с)")
    return rows


def style(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_xlabel("Расход, мг/с", color=INK2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot(rows, currents):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Рис. 4: полоса x̄ ± σ напряжения разряда против эксперимента — по панели на ток
    fig, axes = plt.subplots(1, len(currents), figsize=(4.2 * len(currents), 3.6),
                             sharey=True, facecolor="#fcfcfb")
    for ax, Id in zip(np.atleast_1d(axes), currents):
        r = [x for x in rows if x["Id"] == Id]
        m = np.array([x["mdot_mg_s"] for x in r])
        v = np.array([x["V_tot_mean"] for x in r])
        s = np.array([x["V_tot_std"] for x in r])
        c = COLORS.get(Id, "#2a78d6")
        ax.fill_between(m, v - s, v + s, color=c, alpha=0.25, linewidth=0, label="модель, x̄ ± σ")
        ax.plot(m, v, color=c, linewidth=2, label="модель, среднее")
        if Id in EXPERIMENT:
            e = np.array(EXPERIMENT[Id])
            ax.plot(e[:, 0], e[:, 1], "o", color=INK, markersize=6, markeredgecolor="#fcfcfb",
                    label="эксперимент (≈ рис. 4)")
        style(ax, f"Ток разряда {Id:.0f} А", "Напряжение разряда, В")
        ax.set_facecolor("#fcfcfb")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "discharge_voltage.png"), dpi=130)

    # Рис. 6-8: тренды электронной температуры по областям (одна ось — эВ)
    regions = [("Te_e", "Эмиттер"), ("Te_o", "Диафрагма"), ("Te_k", "Keeper")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), sharey=True, facecolor="#fcfcfb")
    for ax, (key, name) in zip(axes, regions):
        for Id in currents:
            r = [x for x in rows if x["Id"] == Id]
            m = np.array([x["mdot_mg_s"] for x in r])
            ax.errorbar(m, [x[key + "_mean"] for x in r], yerr=[x[key + "_std"] for x in r],
                        color=COLORS.get(Id), linewidth=2, marker="o", markersize=5,
                        capsize=3, label=f"{Id:.0f} А")
        style(ax, f"{name}: электронная температура", "Te, эВ")
        ax.set_facecolor("#fcfcfb")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "electron_temperature.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="меньше точек и прогонов")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.quick:
        currents, flows = [2.0], [0.08, 0.4, 1.0]
        pso_kw = dict(n_solutions=5, n_particles=300, n_iter=300)
    else:
        currents, flows = [1.0, 2.0, 3.0], [0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        pso_kw = dict(n_solutions=10, n_particles=500, n_iter=400)

    rows = run(currents, flows, pso_kw)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, currents)

    # Режим "целевое напряжение" (ур. 29) для одной точки — как рис. 9-10 статьи
    print("\nРежим с целевым напряжением (ур. 29), 2 А, 0.4 мг/с, V_target = 16.2 В:")
    sol = solve(CathodeConfig(Id=2.0, mdot=0.4e-6, V_target=16.2), **pso_kw)
    for name, unit in zip(list(VAR_NAMES) + ["V_tot", "n_ek", "Te_k", "n_nk"],
                          list(VAR_UNITS) + ["V", "m^-3", "eV", "m^-3"]):
        mu, sd = sol.mean_std(name)
        print(f"  {name:6s} = {mu:10.4g} ± {sd:.2g} {unit}")
    print("  средние невязки, %:", {k: round(v, 4) for k, v in residual_report(sol).items()})
    print(f"\nГотово. Результаты в {OUT}")


if __name__ == "__main__":
    main()

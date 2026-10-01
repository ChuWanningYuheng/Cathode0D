"""Связка модели катода и PSO: решение системы уравнений для рабочей точки."""
from dataclasses import dataclass

import numpy as np

from .model import CathodeConfig, VAR_NAMES, LOG_VARS, RESIDUAL_NAMES, evaluate
from .pso import pso_minimize


def make_transform(cfg: CathodeConfig):
    """Возвращает пару функций: нормированные u in [0,1] <-> физические X."""
    lo = np.array([cfg.bounds[n][0] for n in VAR_NAMES], float)
    hi = np.array([cfg.bounds[n][1] for n in VAR_NAMES], float)
    is_log = np.array([n in LOG_VARS for n in VAR_NAMES])
    a = np.where(is_log, np.log10(lo), lo)
    b = np.where(is_log, np.log10(hi), hi)

    def to_phys(U):  # ур. 19 (линейно) и ур. 20 (логарифмически)
        Y = a + (b - a) * np.asarray(U, float)
        return np.where(is_log, 10.0 ** np.where(is_log, Y, 0.0), Y)

    def to_unit(X):
        Y = np.where(is_log, np.log10(X), X)
        return (Y - a) / (b - a)

    return to_phys, to_unit


@dataclass
class Solution:
    """Результат решения: лучшие найденные состояния и их статистика."""
    cfg: CathodeConfig
    X: np.ndarray            # (n, 10) принятые решения, по возрастанию J
    J: np.ndarray            # (n,) их функции стоимости
    details: dict            # все величины модели для каждого из лучших решений
    histories: list          # история J_best по каждому прогону PSO
    n_runs: int = 0          # сколько прогонов PSO было сделано

    def best(self):
        """Словарь величин лучшего решения (скаляры)."""
        i = int(np.argmin(self.J))
        return {k: float(v[i]) for k, v in self.details.items()}

    def mean_std(self, key):
        """Среднее и стандартное отклонение величины по лучшим решениям (ур. 30-31)."""
        v = np.asarray(self.details[key])
        return float(v.mean()), float(v.std())


def solve(cfg: CathodeConfig, n_solutions=10, max_runs=40, n_particles=500, n_iter=400,
          J_accept=0.05, tol=1e-10, seed=0, **pso_kw):
    """Решает модель для рабочей точки ``cfg``.

    Система (7 уравнений, 10 неизвестных) недоопределена, поэтому решений много.
    Как в статье, PSO запускается многократно с независимыми случайными роями.
    Лучшее решение прогона принимается, если его J < ``J_accept`` (прогоны,
    застрявшие в локальном минимуме, отбрасываются). Прогоны повторяются, пока
    не набрано ``n_solutions`` решений (или не исчерпан ``max_runs``). По набору
    считаются среднее и разброс (ур. 30-31) — "доверительная полоса".
    """
    to_phys, _ = make_transform(cfg)

    def cost(U):
        return evaluate(cfg, to_phys(U))["J"]

    rng = np.random.default_rng(seed)
    runs = []  # (J, u, history) всех прогонов
    for _ in range(max_runs):
        u, j, hist, _, _ = pso_minimize(cost, len(VAR_NAMES), n_particles, n_iter,
                                        rng=rng, tol=tol, **pso_kw)
        runs.append((j, u, hist))
        if sum(r[0] < J_accept for r in runs) >= n_solutions:
            break
    accepted = [r for r in runs if r[0] < J_accept]
    if not accepted:  # ничего не сошлось — возвращаем лучшее, что есть
        accepted = [min(runs, key=lambda r: r[0])]
    accepted.sort(key=lambda r: r[0])
    X = to_phys(np.array([r[1] for r in accepted]))
    details = evaluate(cfg, X)
    return Solution(cfg=cfg, X=X, J=details["J"], details=details,
                    histories=[r[2] for r in runs], n_runs=len(runs))


def residual_report(sol: Solution):
    """Средние значения невязок лучших решений, % (как рис. 5 статьи)."""
    return {n: 100 * float(np.mean(np.abs(sol.details[n]))) for n in RESIDUAL_NAMES}

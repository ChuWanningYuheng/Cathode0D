"""Оптимизация роем частиц (Particle Swarm Optimization, раздел 3 статьи).

Реализация работает в нормированных координатах u in [0, 1]^d. Перевод в
физические величины делает вызывающий код (линейно — ур. 19, логарифмически —
ур. 20). Благодаря этому скорости всех переменных имеют один масштаб, даже если
сама величина меняется на 3 порядка (иначе ур. 21/25 в линейном масштабе
почти не исследуют малые значения плотностей).
"""
import numpy as np


def pso_minimize(cost, dim, n_particles=500, n_iter=300, c1=0.7, c2=1.5, c3=1.5,
                 rng=None, tol=None):
    """Минимизирует ``cost(U) -> J`` (U формы (N, dim) в [0, 1]) методом PSO.

    c1 — инерция, c2 — "ностальгия" (тяга к личному рекорду PB),
    c3 — "социальная" тяга к глобальному рекорду GB (ур. 25).
    Возвращает (u_best, J_best, история J_best по итерациям, финальные PB и их J).
    """
    rng = np.random.default_rng(rng)
    # Инициализация роя (ур. 19-21 в нормированных координатах)
    X = rng.random((n_particles, dim))
    V = rng.random((n_particles, dim)) - 0.5
    J = cost(X)
    pb_x, pb_j = X.copy(), J.copy()                         # ур. 23
    g = np.argmin(pb_j)
    gb_x, gb_j = pb_x[g].copy(), pb_j[g]                    # ур. 24
    history = [gb_j]

    for _ in range(n_iter):
        r2 = rng.random((n_particles, dim))
        r3 = rng.random((n_particles, dim))
        V = c1 * V + c2 * r2 * (pb_x - X) + c3 * r3 * (gb_x - X)   # ур. 25
        X = X + V                                                  # ур. 26
        # Вышедшие за область частицы ставятся на границу; скорость по этой
        # координате гасится, чтобы частица не "липла" к стенке.
        out = (X < 0.0) | (X > 1.0)
        X = np.clip(X, 0.0, 1.0)
        V[out] = 0.0
        J = cost(X)
        better = J < pb_j                                          # ур. 27
        pb_x[better], pb_j[better] = X[better], J[better]
        g = np.argmin(pb_j)                                        # ур. 28
        if pb_j[g] < gb_j:
            gb_x, gb_j = pb_x[g].copy(), pb_j[g]
        history.append(gb_j)
        if tol is not None and gb_j < tol:
            break
    return gb_x, gb_j, np.array(history), pb_x, pb_j

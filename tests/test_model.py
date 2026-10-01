"""Проверки модели катода, оптимизатора PSO и решателя целиком."""
import numpy as np
import pytest

from cathode0d import (CathodeConfig, VAR_NAMES, RESIDUAL_NAMES, evaluate, solve,
                       solve_keeper, pso_minimize, make_transform)
from cathode0d import physics as ph

FAST = dict(n_solutions=3, max_runs=12, n_particles=300, n_iter=300)


# ---------------- PSO на известной функции ----------------
def test_pso_finds_minimum_of_shifted_sphere():
    target = np.array([0.2, 0.7, 0.5, 0.9])
    u, j, hist, _, _ = pso_minimize(lambda U: np.sum((U - target) ** 2, axis=1), 4,
                                    n_particles=100, n_iter=200, rng=0)
    assert j < 1e-10
    assert np.allclose(u, target, atol=1e-5)
    assert np.all(np.diff(hist) <= 0)  # глобальный рекорд не ухудшается


def test_pso_respects_bounds():
    # Минимум за пределами [0,1] -> оптимум на границе
    u, _, _, pb, _ = pso_minimize(lambda U: np.sum((U - 2.0) ** 2, axis=1), 3,
                                  n_particles=50, n_iter=50, rng=1)
    assert np.allclose(u, 1.0)
    assert pb.min() >= 0 and pb.max() <= 1


# ---------------- Преобразование координат ----------------
def test_transform_roundtrip_and_bounds():
    cfg = CathodeConfig()
    to_phys, to_unit = make_transform(cfg)
    U = np.random.default_rng(0).random((20, 10))
    X = to_phys(U)
    assert np.allclose(to_unit(X), U)
    for i, n in enumerate(VAR_NAMES):
        lo, hi = cfg.bounds[n]
        assert np.all((X[:, i] >= lo * (1 - 1e-12)) & (X[:, i] <= hi * (1 + 1e-12)))
    # логарифмическая переменная: середина диапазона = среднее геометрическое
    assert to_phys(np.full((1, 10), 0.5))[0, 0] == pytest.approx(np.sqrt(1e21 * 1e24))


# ---------------- Keeper ----------------
def test_keeper_satisfies_its_equations():
    cfg = CathodeConfig(Id=2.0, mdot=0.4e-6)
    ne_o, Te_o = np.array([3e20, 7e20, 2e21]), np.array([2.8, 2.4, 2.1])
    ne_k, Te_k, nn_k, ok = solve_keeper(cfg, ne_o, Te_o)
    assert np.all(ok)
    A_o, A_k = np.pi * cfg.r_o**2, np.pi * cfg.r_k**2
    # 1a: непрерывность потока Бома
    assert np.allclose(ne_k, ne_o * A_o / A_k * np.sqrt(Te_o / Te_k))
    # 1b: ионизация = потери ионов
    prod = ph.Q * np.pi * cfg.r_k**2 * cfg.L_k * nn_k * ne_k * ph.ionization_rate(Te_k)
    loss = (ph.j_bohm(ne_k, Te_k) * 2 * np.pi * cfg.r_k * cfg.L_k
            + 2 * ph.j_thermal(ne_k, Te_k) * np.pi * cfg.r_k**2)
    assert np.allclose(prod, loss, rtol=1e-8)
    # 1c: давление запертого потока
    x = 1 + ne_k / (ne_k + nn_k) * Te_k * ph.EV_TO_K / cfg.Tw_k
    p = (ne_k + nn_k) * ph.KB * cfg.Tw_k * x
    assert np.allclose(p, cfg.mdot / A_k * np.sqrt(ph.XE_RG * cfg.Tw_k / ph.XE_GAMMA * x))
    assert np.all(nn_k > 0)


# ---------------- Модель ----------------
def test_evaluate_shapes_and_finiteness():
    cfg = CathodeConfig()
    to_phys, _ = make_transform(cfg)
    X = to_phys(np.random.default_rng(3).random((256, 10)))
    out = evaluate(cfg, X)
    for n in RESIDUAL_NAMES + ("J", "V_tot", "n_ek", "Te_k", "n_nk"):
        assert out[n].shape == (256,)
        assert np.all(np.isfinite(out[n]))
    # невязки нормированы: |eps| <= 2 (при одинаковых знаках сторон <= 1)
    for n in RESIDUAL_NAMES:
        assert np.all(np.abs(out[n]) <= 2)


def test_evaluate_single_state_equals_batch():
    cfg = CathodeConfig()
    to_phys, _ = make_transform(cfg)
    X = to_phys(np.random.default_rng(4).random((5, 10)))
    batch = evaluate(cfg, X)["J"]
    single = [evaluate(cfg, x)["J"][0] for x in X]
    assert np.allclose(batch, single)


# ---------------- Решатель целиком ----------------
@pytest.fixture(scope="module")
def sol_2A_04():
    return solve(CathodeConfig(Id=2.0, mdot=0.4e-6), seed=0, **FAST)


def test_solution_converges(sol_2A_04):
    # Все 7 уравнений выполнены: сумма относительных невязок < 0.1 %
    assert len(sol_2A_04.J) == FAST["n_solutions"]
    assert sol_2A_04.J.max() < 1e-3


def test_solution_inside_domain(sol_2A_04):
    cfg = sol_2A_04.cfg
    for i, n in enumerate(VAR_NAMES):
        lo, hi = cfg.bounds[n]
        assert np.all(sol_2A_04.X[:, i] >= lo * 0.999) and np.all(sol_2A_04.X[:, i] <= hi * 1.001)


def test_solution_physically_plausible(sol_2A_04):
    b = sol_2A_04.best()
    # Порядки величин как на рис. 6-8 статьи для 2 А
    assert 1e19 < b["n_ee"] < 1e21
    assert 1e22 < b["n_ne"] < 1e24
    assert 1e20 < b["n_eo"] < 1e22
    assert b["Te_o"] > b["Te_e"]           # в диафрагме электроны горячее
    assert b["n_ne"] > b["n_no"]           # давление падает вниз по потоку
    assert 1.0 < b["Te_k"] < 6.0
    assert 5.0 < b["V_tot"] < 40.0          # напряжение разряда — десятки вольт
    # ток на эмиттере сходится с током разряда (ур. 9b)
    assert b["Leff"] * 2 * np.pi * sol_2A_04.cfg.r_e * (b["j_i_e"] + b["j_em"] - b["j_er"]) == \
        pytest.approx(2.0, rel=1e-3)


def test_trends_with_mass_flow():
    """Тренды из статьи: Te падает, плотности растут с ростом расхода."""
    lo = solve(CathodeConfig(Id=2.0, mdot=0.1e-6), seed=1, **FAST)
    hi = solve(CathodeConfig(Id=2.0, mdot=1.0e-6), seed=1, **FAST)
    for key in ("Te_e", "Te_o", "Te_k"):
        assert hi.mean_std(key)[0] < lo.mean_std(key)[0], key
    for key in ("n_ne", "n_no", "n_eo", "n_nk", "n_ek"):
        assert hi.mean_std(key)[0] > lo.mean_std(key)[0], key


def test_target_voltage_mode():
    sol = solve(CathodeConfig(Id=2.0, mdot=0.4e-6, V_target=16.2), seed=2, **FAST)
    assert sol.J.max() < 1e-3
    assert np.allclose(sol.details["V_tot"], 16.2, rtol=1e-3)

"""Проверки замыкающих соотношений (ур. 4-18) на известных значениях."""
import numpy as np
import pytest

from cathode0d import physics as ph


def test_cross_section_fit_at_1eV():
    # При Te = 1 эВ: log10(sigma) = p6/q2 = -9.9614/0.5227
    assert ph.sigma_en(1.0) == pytest.approx(10 ** (-9.9614 / 0.5227), rel=1e-12)


def test_cross_section_physical_range():
    s = ph.sigma_en(np.array([0.5, 1, 2, 4]))
    assert np.all((s > 1e-21) & (s < 1e-18))


def test_viscosity_reference_point():
    # При T = 289.7 K (Tr = 1) формула даёт ровно 2.3e-5 Па*с
    assert ph.xe_viscosity(289.7) == pytest.approx(2.3e-5)
    assert ph.xe_viscosity(1800.0) > ph.xe_viscosity(1000.0)


def test_ionization_rate_monotonic_and_magnitude():
    Te = np.linspace(0.5, 5, 50)
    K = ph.ionization_rate(Te)
    assert np.all(np.diff(K) > 0)
    # Гёбель и Кац: для Xe при 2 эВ K_i ~ 1e-16 м^3/с
    assert 5e-17 < ph.ionization_rate(2.0) < 3e-16


def test_bohm_current():
    ne, Te = 1e20, 2.0
    expected = 0.61 * ph.Q * ne * np.sqrt(ph.Q * Te / ph.XE_MASS)
    assert ph.j_bohm(ne, Te) == pytest.approx(expected)


def test_richardson_without_schottky():
    # При Vp, не дающем поля (1 + 2Vp/Te < 4), Шоттки выключен -> чистый Ричардсон
    Ts = 1800.0
    j = ph.j_emission(Ts, 2.66, 29e4, 1e20, 2.0, 1.0)
    assert j == pytest.approx(29e4 * Ts**2 * np.exp(-2.66 * ph.EV_TO_K / Ts))


def test_schottky_increases_emission():
    j0 = ph.j_emission(1800.0, 2.66, 29e4, 1e20, 2.0, 1.0)
    j1 = ph.j_emission(1800.0, 2.66, 29e4, 1e20, 2.0, 20.0)
    assert j1 > j0


def test_backstream_decreases_with_sheath():
    assert ph.j_backstream(1e20, 2.0, 20.0) < ph.j_backstream(1e20, 2.0, 5.0)


def test_coulomb_log():
    # ур. 7: 23 - 0.5*ln(1e-6*1e20 / 2^3) = 7.92
    assert ph.coulomb_log(1e20, 2.0) == pytest.approx(23 - 0.5 * np.log(1e14 / 8))
    assert ph.coulomb_log(1e19, 2.0) > ph.coulomb_log(1e20, 2.0)


def test_choked_density_satisfies_state_equation():
    mdot, A, Tw, ne, Te = 0.4e-6, np.pi * (0.2e-3) ** 2, 1800.0, 5e20, 2.5
    N = ph.choked_total_density(mdot, A, Tw, ne, Te)
    x = 1 + ne / N * Te * ph.EV_TO_K / Tw
    lhs = N * ph.KB * Tw * x
    rhs = mdot / A * np.sqrt(ph.XE_RG * Tw / ph.XE_GAMMA * x)
    assert lhs == pytest.approx(rhs, rel=1e-10)

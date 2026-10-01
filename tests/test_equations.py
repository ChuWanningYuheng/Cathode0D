"""Независимая проверка всех уравнений модели в фиксированной точке.

Здесь уравнения статьи переписаны заново, скалярно, на модуле ``math``, со своими
константами. Код пакета не используется (кроме ``evaluate`` для сравнения).
Тест ловит любую подмену формулы в ``model.py``/``physics.py``: убранный член,
неверный множитель, не та температура или площадь. Для таких ошибок решатель всё
равно сходится к J ~ 0, поэтому тесты решателя их не видят.
"""
import math

import pytest

from cathode0d import CathodeConfig, evaluate

# Константы (CODATA 2018), записаны независимо от physics.py
q, kB, me, eps0 = 1.602176634e-19, 1.380649e-23, 9.1093837015e-31, 8.8541878128e-12
mi = 131.293 * 1.66053906660e-27
Rg, gam = kB / mi, 5 / 3

# Фиксированная точка: типичное решение при 2 А, 0.4 мг/с (Te в эВ)
X = dict(n_ne=6.5e22, n_ee=3.0e20, Vp=10.0, Te_e=1.6, n_no=2.7e22, n_eo=7.5e20,
         Te_o=2.45, Ts=1780.0, Leff=3.0e-3, Tw_o=1800.0)


def Ki(Te):
    return 1e-20 * (3.97 + 0.643*Te - 0.0368*Te**2) * math.exp(-12.127/Te) \
        * math.sqrt(8*q*Te/(math.pi*me))


def sigma_en(Te):
    t = math.log10(Te)
    p = (0.0350, -0.3057, -0.6698, -18.5416, -12.8821, -9.9614)
    return 10 ** ((p[0]*t**5 + p[1]*t**4 + p[2]*t**3 + p[3]*t**2 + p[4]*t + p[5])
                  / (t**2 + 0.7093*t + 0.5227))


def eta(ne, nn, Te):                                     # ур. 4-7
    lnL = 23 - 0.5 * math.log(1e-6 * ne / Te**3)
    nu_ei = 2.9e-12 * ne * lnL / Te**1.5
    nu_en = sigma_en(Te) * nn * math.sqrt(8*q*Te/(math.pi*me))
    return (nu_ei + nu_en) * me / (ne * q**2)


def ji(ne, Te):                                          # ур. 13
    return 0.61 * q * ne * math.sqrt(q*Te/mi)


def jth(ne, Te):                                         # ур. 14
    return 0.25 * q * ne * math.sqrt(8*q*Te/(math.pi*mi))


def rel(lhs, rhs):
    return (rhs - lhs) / max(abs(lhs), abs(rhs))


def reference(cfg, x):
    Id, md, ei = cfg.Id, cfg.mdot, cfg.eps_i
    ro, Lo, re = cfg.r_o, cfg.L_o, cfg.r_e
    Ao, Ae, Aeff = math.pi*ro**2, math.pi*re**2, 2*math.pi*re*x["Leff"]
    Te_e, Te_o, Ts, Vp = x["Te_e"], x["Te_o"], x["Ts"], x["Vp"]
    ne_e, nn_e, ne_o, nn_o = x["n_ee"], x["n_ne"], x["n_eo"], x["n_no"]
    kTs = kB * Ts / q
    r = {}

    # --- Диафрагма ---
    ndot_o = q * math.pi*ro**2*Lo * nn_o * ne_o * Ki(Te_o)
    rhs_2a = ji(ne_o, Te_o) * (2*math.pi*ro*Lo + math.pi*ro**2*(1 - 0.25/0.61*math.sqrt(8/math.pi))) \
        + jth(ne_o, Te_o) * math.pi*ro**2
    r["eps_ion_o"] = rel(ndot_o, rhs_2a)
    R_o = eta(ne_o, nn_o, Te_o) * Lo / (math.pi*ro**2)              # ур. 3
    r["eps_Pow_o"] = rel(R_o*Id**2, ndot_o*ei + 2.5*Id*(Te_o - Te_e))
    a_o = ne_o / (ne_o + nn_o)
    x_o = 1 + a_o * Te_o*q/kB / x["Tw_o"]
    r["eps_press_o"] = rel((ne_o+nn_o)*kB*x["Tw_o"]*x_o,
                           md/(math.pi*ro**2) * math.sqrt(Rg*x["Tw_o"]/gam * x_o))

    # --- Эмиттер ---
    Ec = math.sqrt(ne_e*q*Te_e/eps0) * math.sqrt(max(2*math.sqrt(1 + 2*Vp/Te_e) - 4, 0))
    phi = cfg.phi0 - math.sqrt(q*Ec/(4*math.pi*eps0))
    jem = cfg.D * Ts**2 * math.exp(-q*phi/(kB*Ts))                  # ур. 15-16
    jer = 0.25*q*ne_e*math.exp(-Vp/Te_e)*math.sqrt(8*q*Te_e/(math.pi*me))  # ур. 18 (с Te)
    Vds = (9*Id*q*Te_o/(7.5*math.pi*ne_o*ro**2*q**2) * math.sqrt(me/(2*q))) ** (2/3)
    R_e = 3 * eta(ne_e, nn_e, Te_e) / (4*math.pi*x["Leff"])        # ур. 10
    ndot_out = ji(ne_e, Te_e)*(Aeff + Ae - Ao) + jth(ne_e, Te_e)*Ae
    r["eps_ion_e"] = rel(q*math.pi*re**2*x["Leff"]*nn_e*ne_e*Ki(Te_e) + ji(ne_o, Te_o)*Ao, ndot_out)
    r["eps_I"] = rel(Id, (ji(ne_e, Te_e) + jem - jer) * Aeff)
    P_in = ji(ne_o, Te_o)*(Vds + 2*kTs)*Ao + R_e*Id**2 + jem*(Vp + 1.5*kTs)*Aeff
    P_out = ndot_out*(ei + 2*kTs) + jer*2*Te_e*Aeff + 2.5*Te_e*Id
    r["eps_Pow_e"] = rel(P_in, P_out)
    Tr = Ts / 289.7
    mu = 2.3e-5 * Tr ** (0.71 + 0.29/Tr)
    p_t = 1.36e-9 * Id**0.3 * Ts**0.95 * md**0.57 * Lo**0.15 \
        / (mi**0.10 * ei**0.20 * mu**0.35 * (2*re)**1.43 * (2*ro)**1.71)
    x_e = 1 + ne_e/(ne_e + nn_e) * Te_e*q/kB / Ts
    r["eps_press_e"] = rel((ne_e+nn_e)*kB*Ts*x_e, p_t)

    r.update(R_o=R_o, R_e=R_e, V_ds=Vds, j_em=jem, j_er=jer, p_taunay=p_t,
             V_tot=(P_in + R_o*Id**2) / Id)
    r["J"] = sum(abs(r[k]) for k in ("eps_I", "eps_Pow_e", "eps_ion_e", "eps_press_e",
                                     "eps_Pow_o", "eps_ion_o", "eps_press_o"))
    return r


@pytest.mark.parametrize("Id,mdot", [(2.0, 0.4e-6), (1.0, 0.1e-6), (3.0, 1.0e-6)])
def test_all_equations_match_independent_reference(Id, mdot):
    cfg = CathodeConfig(Id=Id, mdot=mdot)
    ref = reference(cfg, X)
    out = evaluate(cfg, [X[n] for n in X])
    for key, val in ref.items():
        assert out[key][0] == pytest.approx(val, rel=1e-9, abs=1e-12), key


def test_reference_point_is_not_trivial():
    """Точка выбрана так, что ни один член не пренебрежимо мал (иначе тест слеп к нему)."""
    cfg = CathodeConfig()
    r = reference(cfg, X)
    Aeff = 2*math.pi*cfg.r_e*X["Leff"]
    assert r["j_er"] * Aeff > 0.01 * cfg.Id          # обратные электроны заметны
    assert r["V_ds"] > 0.1                            # двойной слой заметен
    assert r["j_em"] * Aeff > 0.1 * cfg.Id           # эмиссия заметна
    # Шоттки включён (Vp > 1.5 Te)
    assert X["Vp"] > 1.5 * X["Te_e"]

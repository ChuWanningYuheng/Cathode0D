"""Нульмерная (0D) модель плазмы катода с диафрагмой (раздел 2.3 статьи).

Объём катода разбит на три области:
  * эмиттер (insert, индекс ``e``)  — уравнения 9a-9d;
  * диафрагма (orifice, индекс ``o``) — уравнения 2a-2c;
  * поджигной электрод (keeper, индекс ``k``) — уравнения 1a-1c.

Неизвестные, которые подбирает оптимизатор (таблица 1 статьи), — вектор X из 10 чисел:
  n_ne, n_ee, Vp, Te_e, n_no, n_eo, Te_o, Ts, Leff, Tw_o.
Параметры области keeper (n_ek, Te_k, n_nk) выражаются через орифисные величины
и вычисляются внутри модели точно (функция :func:`solve_keeper`).

Главная функция — :func:`evaluate`: по вектору X возвращает невязки 7 уравнений
и все производные величины. Всё векторизовано по первой оси (весь рой сразу).
"""
from dataclasses import dataclass, field

import numpy as np

from . import physics as ph

# Порядок и имена переменных состояния (как в таблице 1 статьи)
VAR_NAMES = ("n_ne", "n_ee", "Vp", "Te_e", "n_no", "n_eo", "Te_o", "Ts", "Leff", "Tw_o")
VAR_UNITS = ("m^-3", "m^-3", "V", "eV", "m^-3", "m^-3", "eV", "K", "m", "K")
# Переменные, которые разыгрываются в логарифмическом масштабе (ур. 20)
LOG_VARS = ("n_ne", "n_ee", "n_no", "n_eo")

RESIDUAL_NAMES = ("eps_I", "eps_Pow_e", "eps_ion_e", "eps_press_e",
                  "eps_Pow_o", "eps_ion_o", "eps_press_o")


@dataclass
class CathodeConfig:
    """Входные данные модели: геометрия, материал эмиттера и рабочая точка.

    Значения по умолчанию соответствуют катоду SITAEL AlphCA (раздел 4 статьи).
    """
    # --- рабочая точка ---
    Id: float = 2.0                 # ток разряда, А
    mdot: float = 0.4e-6            # расход ксенона, кг/с (0.4 мг/с)
    # --- геометрия, м ---
    r_e: float = 1.5e-3             # внутренний радиус эмиттера (диаметр 3 мм)
    L_e: float = 6.0e-3             # длина эмиттера
    r_o: float = 0.2e-3             # радиус диафрагмы (диаметр 0.4 мм)
    L_o: float = 0.36e-3            # длина канала диафрагмы
    r_k: float = 0.3e-3             # радиус отверстия keeper'а (диаметр 0.6 мм)
    L_k: float = 2.3e-3             # длина области keeper: зазор 2 мм + отверстие 0.3 мм
    # --- стенки ---
    Tw_k: float = 800.0             # температура keeper'а, К (задана, как в статье)
    # --- материал эмиттера (LaB6, Гёбель и Кац) ---
    phi0: float = 2.66              # работа выхода, эВ
    D: float = 29.0e4               # постоянная Ричардсона-Дэшмана, А/(м^2 К^2)
    # --- газ ---
    eps_i: float = ph.XE_EPS_I      # средняя энергия ионизации, эВ
    # Множитель к эмпирической формуле давления Таунея (ур. 9d). В статье единицы
    # не указаны; принята интерпретация "СИ, но eps_i в эВ" (см. README).
    taunay_coef: float = 1.36e-9
    # Целевое напряжение разряда (ур. 29). None -> обычная функция стоимости (ур. 22)
    V_target: float | None = None
    # Границы области поиска PSO (таблица 1 статьи)
    bounds: dict = field(default_factory=lambda: {
        "n_ne": (1e21, 1e24), "n_ee": (1e18, 1e21), "Vp": (5.0, 50.0),
        "Te_e": (1.0, 4.0), "n_no": (1e21, 1e23), "n_eo": (1e18, 1e22),
        "Te_o": (1.0, 4.0), "Ts": (1700.0, 1950.0), "Leff": (1e-3, 6e-3),
        "Tw_o": (1700.0, 1950.0),
    })


# ---------------------------------------------------------------------------
# Keeper (ур. 1a-1c)
# ---------------------------------------------------------------------------
def q_volume(r, L):
    """q * pi r^2 L — множитель, переводящий частоту ионизаций в объёме в ток (А*м^3)."""
    return ph.Q * np.pi * r**2 * L


def _keeper_fields(cfg, ne_o, Te_o, Te_k):
    """n_ek и n_nk при заданной Te_k (из ур. 1a и 1c)."""
    A_o, A_k = np.pi * cfg.r_o**2, np.pi * cfg.r_k**2
    ne_k = ne_o * A_o / A_k * np.sqrt(Te_o / Te_k)                      # ур. 1a
    n_tot = ph.choked_total_density(cfg.mdot, A_k, cfg.Tw_k, ne_k, Te_k)  # ур. 1c
    nn_k = n_tot - ne_k
    return ne_k, nn_k


def _keeper_balance(cfg, ne_o, Te_o, Te_k):
    """Относительная невязка ионного баланса в keeper'е (ур. 1b)."""
    ne_k, nn_k = _keeper_fields(cfg, ne_o, Te_o, Te_k)
    prod = q_volume(cfg.r_k, cfg.L_k) * np.maximum(nn_k, 0.0) * ne_k * ph.ionization_rate(Te_k)
    loss = (ph.j_bohm(ne_k, Te_k) * 2*np.pi*cfg.r_k*cfg.L_k
            + 2.0 * ph.j_thermal(ne_k, Te_k) * np.pi*cfg.r_k**2)
    return (prod - loss) / (prod + loss)


def solve_keeper(cfg, ne_o, Te_o, Te_lo=0.2, Te_hi=20.0, n_iter=60):
    """Решает систему (1a-1c) для keeper'а векторной бисекцией по ln(Te_k).

    Невязка 1b монотонно растёт с Te_k (ионизация растёт экспоненциально),
    поэтому корень единственен. Если корня в [Te_lo, Te_hi] нет, берётся граница.
    Возвращает (ne_k, Te_k, nn_k).
    """
    ne_o, Te_o = np.broadcast_arrays(np.asarray(ne_o, float), np.asarray(Te_o, float))
    lo = np.full(ne_o.shape, np.log(Te_lo))
    hi = np.full(ne_o.shape, np.log(Te_hi))
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f = _keeper_balance(cfg, ne_o, Te_o, np.exp(mid))
        neg = f < 0  # ионизации не хватает -> нужна более высокая Te_k
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)
    Te_k = np.exp(0.5 * (lo + hi))
    ne_k, nn_k = _keeper_fields(cfg, ne_o, Te_o, Te_k)
    return ne_k, Te_k, nn_k


# ---------------------------------------------------------------------------
# Основная функция модели
# ---------------------------------------------------------------------------
def _rel(lhs, rhs):
    """Обезразмеренная невязка (RHS - LHS), отнесённая к масштабу уравнения."""
    return (rhs - lhs) / np.maximum(np.abs(rhs), np.abs(lhs))


def unpack(X):
    """Массив (N, 10) -> словарь столбцов по именам переменных."""
    X = np.atleast_2d(np.asarray(X, float))
    return {name: X[:, i] for i, name in enumerate(VAR_NAMES)}


def evaluate(cfg: CathodeConfig, X):
    """Вычисляет все уравнения модели для набора состояний X (форма (N, 10)).

    Возвращает словарь: невязки ``eps_*`` (7 шт.), функцию стоимости ``J`` и
    все производные величины (сопротивления, токи, напряжения, параметры keeper'а).
    """
    s = unpack(X)
    Id, mdot, eps_i = cfg.Id, cfg.mdot, cfg.eps_i
    nn_e, ne_e, Vp, Te_e = s["n_ne"], s["n_ee"], s["Vp"], s["Te_e"]
    nn_o, ne_o, Te_o = s["n_no"], s["n_eo"], s["Te_o"]
    Ts, Leff, Tw_o = s["Ts"], s["Leff"], s["Tw_o"]

    A_o = np.pi * cfg.r_o**2              # сечение диафрагмы
    A_e = np.pi * cfg.r_e**2              # сечение эмиттера
    A_eff = 2*np.pi * cfg.r_e * Leff      # площадь эмитирующей поверхности
    kTs = Ts / ph.EV_TO_K                 # k_B*Ts/q, В

    # ---------------- Диафрагма (orifice), ур. 2-3 ----------------
    ji_o = ph.j_bohm(ne_o, Te_o)
    jth_o = ph.j_thermal(ne_o, Te_o)
    ndot_ion_o = q_volume(cfg.r_o, cfg.L_o) * nn_o * ne_o * ph.ionization_rate(Te_o)  # ток ионизации, А
    loss_o = (ji_o * (2*np.pi*cfg.r_o*cfg.L_o + A_o*(1 - 0.25/0.61*np.sqrt(8/np.pi)))
              + jth_o * A_o)
    eta_o = ph.resistivity(ne_o, nn_o, Te_o)
    R_o = eta_o * cfg.L_o / A_o                                                 # ур. 3
    P_ohm_o = R_o * Id**2
    P_loss_o = ndot_ion_o * eps_i + 2.5 * Id * (Te_o - Te_e)                    # правая часть 2b
    x_o = 1 + ne_o/(ne_o+nn_o) * Te_o*ph.EV_TO_K/Tw_o     # 1 + alpha*Te/Tw
    p_o = (ne_o + nn_o) * ph.KB * Tw_o * x_o
    p_o_choked = mdot / A_o * np.sqrt(ph.XE_RG * Tw_o / ph.XE_GAMMA * x_o)

    eps_ion_o = _rel(ndot_ion_o, loss_o)        # 2a
    eps_Pow_o = _rel(P_ohm_o, P_loss_o)         # 2b
    eps_press_o = _rel(p_o, p_o_choked)         # 2c

    # ---------------- Эмиттер (insert), ур. 9-18 ----------------
    ji_e = ph.j_bohm(ne_e, Te_e)
    jth_e = ph.j_thermal(ne_e, Te_e)
    j_em = ph.j_emission(Ts, cfg.phi0, cfg.D, ne_e, Te_e, Vp)
    j_er = ph.j_backstream(ne_e, Te_e, Vp)
    V_ds = ph.double_sheath_voltage(Id, Te_o, ne_o, cfg.r_o)
    eta_e = ph.resistivity(ne_e, nn_e, Te_e)
    R_e = 3.0 * eta_e / (4.0 * np.pi * Leff)                                    # ур. 10

    # 9a: рождение ионов в объёме + приток из диафрагмы = уход на стенки
    ion_src_e = q_volume(cfg.r_e, Leff) * nn_e * ne_e * ph.ionization_rate(Te_e) + ji_o * A_o
    ndot_out_e = ji_e * (A_eff + A_e - A_o) + jth_e * A_e
    # 9b: сохранение тока на эмиттере
    I_wall = (ji_e + j_em - j_er) * A_eff
    # 9c: баланс мощности
    P_in_e = ji_o * (V_ds + 2*kTs) * A_o + R_e * Id**2 + j_em * (Vp + 1.5*kTs) * A_eff
    P_out_e = ndot_out_e * (eps_i + 2*kTs) + j_er * 2*Te_e * A_eff + 2.5 * Te_e * Id
    # 9d: давление в эмиттере = эмпирическая формула Таунея
    x_e = 1 + ne_e/(ne_e+nn_e) * Te_e*ph.EV_TO_K/Ts
    p_e = (ne_e + nn_e) * ph.KB * Ts * x_e
    mu = ph.xe_viscosity(Ts)
    p_taunay = (cfg.taunay_coef * Id**0.3 * Ts**0.95 * mdot**0.57 * cfg.L_o**0.15
                / (ph.XE_MASS**0.10 * eps_i**0.20 * mu**0.35
                   * (2*cfg.r_e)**1.43 * (2*cfg.r_o)**1.71))

    eps_ion_e = _rel(ion_src_e, ndot_out_e)     # 9a
    eps_I = _rel(Id, I_wall)                    # 9b (обе части умножены на A_eff)
    eps_Pow_e = _rel(P_in_e, P_out_e)           # 9c
    eps_press_e = _rel(p_e, p_taunay)           # 9d

    # ---------------- Keeper, ур. 1 ----------------
    ne_k, Te_k, nn_k = solve_keeper(cfg, ne_o, Te_o)

    # ---------------- Напряжение разряда ----------------
    # "отношение поглощённой мощности к току разряда" (раздел 3.3): вся мощность,
    # вводимая в плазму эмиттера (левая часть 9c) + омический нагрев диафрагмы (2b).
    P_abs = P_in_e + P_ohm_o
    V_tot = P_abs / Id

    eps = np.stack([eps_I, eps_Pow_e, eps_ion_e, eps_press_e,
                    eps_Pow_o, eps_ion_o, eps_press_o])
    eps = np.where(np.isfinite(eps), eps, 1.0)
    J = np.sum(np.abs(eps), axis=0)                                             # ур. 22
    if cfg.V_target is not None:
        J = J + np.abs(V_tot - cfg.V_target) / cfg.V_target                    # ур. 29
    # Эффективная длина эмиссии не может превышать длину эмиттера
    J = J + np.maximum(Leff - cfg.L_e, 0.0) / cfg.L_e

    out = dict(zip(RESIDUAL_NAMES, eps))
    out.update(J=J, V_tot=V_tot, P_abs=P_abs, V_ds=V_ds, R_e=R_e, R_o=R_o,
               j_em=j_em, j_er=j_er, j_i_e=ji_e, I_em=j_em*A_eff, I_ion_o=ndot_ion_o,
               p_e=p_e, p_taunay=p_taunay, p_o=p_o,
               n_ek=ne_k, Te_k=Te_k, n_nk=nn_k, Tw_k=np.full_like(J, cfg.Tw_k))
    out.update(s)
    return out

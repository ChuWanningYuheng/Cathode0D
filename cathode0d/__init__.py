"""0D-модель плазмы полого катода с диафрагмой, решаемая методом роя частиц (PSO).

Реализация статьи: G. Coppola, M. Panelli, F. Battista, "Solution of Orifice Hollow
Cathode Plasma Model Equations by Means of Particle Swarm Optimization", 2024,
doi:10.20944/preprints202406.1487.v1
"""
from .model import CathodeConfig, VAR_NAMES, VAR_UNITS, RESIDUAL_NAMES, evaluate, solve_keeper
from .pso import pso_minimize
from .solver import Solution, solve, make_transform, residual_report

__all__ = ["CathodeConfig", "VAR_NAMES", "VAR_UNITS", "RESIDUAL_NAMES", "evaluate",
           "solve_keeper", "pso_minimize", "Solution", "solve", "make_transform",
           "residual_report"]

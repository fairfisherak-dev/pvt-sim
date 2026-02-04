"""
Property correlations for petroleum fluid characterization.

This module provides correlations for estimating thermodynamic properties
of petroleum pseudo-components (SCN fractions, plus fractions) from basic
physical properties like molecular weight (MW), specific gravity (SG),
and normal boiling point (Tb).

Available correlation families:
- Critical properties (Tc, Pc, Vc): Riazi-Daubert, Kesler-Lee, Cavett
- Acentric factor (omega): Edmister, Kesler-Lee
- Boiling point (Tb): Soreide
- Parachor: Fanchi

References
----------
Riazi, M.R. and Daubert, T.E. (1987). "Characterization Parameters for
    Petroleum Fractions." Ind. Eng. Chem. Res., 26(4), 755-759.
Kesler, M.G. and Lee, B.I. (1976). "Improve Prediction of Enthalpy of
    Fractions." Hydrocarbon Processing, 55(3), 153-158.
Cavett, R.H. (1962). "Physical Data for Distillation Calculations,
    Vapor-Liquid Equilibria." Proc. 27th API Meeting, San Francisco.
Edmister, W.C. (1958). "Applied Hydrocarbon Thermodynamics, Part 4:
    Compressibility Factors and Equations of State." Petroleum Refiner, 37(4), 173-179.
Soreide, I. (1989). "Improved Phase Behavior Predictions of Petroleum
    Reservoir Fluids from a Cubic Equation of State." Dr.Ing. Thesis, NTNU.
Fanchi, J.R. (1985). "Calculation of Parachors for Compositional
    Simulation: An Update." SPE Reservoir Engineering, 1(4), 405-406.
"""

from .critical_props import (
    CriticalPropsResult,
    riazi_daubert_Tc,
    riazi_daubert_Pc,
    riazi_daubert_Vc,
    riazi_daubert_critical_props,
    kesler_lee_Tc,
    kesler_lee_Pc,
    kesler_lee_critical_props,
    cavett_Tc,
    cavett_Pc,
    cavett_critical_props,
    estimate_critical_props,
    CriticalPropsMethod,
)

from .acentric import (
    edmister_omega,
    kesler_lee_omega,
    estimate_omega,
    AcentricMethod,
)

from .boiling_point import (
    soreide_Tb,
    riazi_daubert_Tb,
    estimate_Tb,
    BoilingPointMethod,
)

from .parachor import (
    fanchi_parachor,
    estimate_parachor,
)

__all__ = [
    # Critical properties
    "CriticalPropsResult",
    "riazi_daubert_Tc",
    "riazi_daubert_Pc",
    "riazi_daubert_Vc",
    "riazi_daubert_critical_props",
    "kesler_lee_Tc",
    "kesler_lee_Pc",
    "kesler_lee_critical_props",
    "cavett_Tc",
    "cavett_Pc",
    "cavett_critical_props",
    "estimate_critical_props",
    "CriticalPropsMethod",
    # Acentric factor
    "edmister_omega",
    "kesler_lee_omega",
    "estimate_omega",
    "AcentricMethod",
    # Boiling point
    "soreide_Tb",
    "riazi_daubert_Tb",
    "estimate_Tb",
    "BoilingPointMethod",
    # Parachor
    "fanchi_parachor",
    "estimate_parachor",
]

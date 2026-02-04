"""
Katz & Firoozabadi (1978) generalized properties for petroleum hexane-plus groups.

This module encodes Table 1 ("GENERALIZED PROPERTIES OF PETROLEUM HEXANE PLUS GROUPS")
from Katz & Firoozabadi, JPT 1978 (SPE 6721-PA), which provides average normal boiling
point, density at 60/60°F, and molecular weight for SCN groups C6–C45.

Source PDF (project-mounted):
  /mnt/data/KatzFlroozabadi-1978_GeneralizedSCNProperties.pdf

Notes
-----
- The paper lists:
    * Average boiling point (°C) and (°F)
    * Density (g/mL) at 60°F, which is effectively specific gravity at 60/60°F.
    * Molecular weight (g/mol)
- The values are intended as "average properties for groups of compounds" (petroleum fractions),
  not necessarily pure n-paraffins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class KatzFiroozabadiRow:
    scn: int
    tb_c: float     # average normal boiling point, °C
    tb_f: float     # average normal boiling point, °F
    sg_6060: float  # density g/mL at 60°F ~ specific gravity 60/60°F
    mw: float       # g/mol


# Hard-coded Table 1 values (C6–C45).
_TABLE: Dict[int, KatzFiroozabadiRow] = {
    6:  KatzFiroozabadiRow(6,   63.9, 147.0, 0.685, 84.0),
    7:  KatzFiroozabadiRow(7,   91.9, 197.5, 0.722, 96.0),
    8:  KatzFiroozabadiRow(8,  116.7, 242.0, 0.745, 107.0),
    9:  KatzFiroozabadiRow(9,  142.2, 288.0, 0.764, 121.0),
    10: KatzFiroozabadiRow(10, 165.8, 330.5, 0.778, 134.0),
    11: KatzFiroozabadiRow(11, 187.2, 369.0, 0.789, 147.0),
    12: KatzFiroozabadiRow(12, 208.3, 407.0, 0.800, 161.0),
    13: KatzFiroozabadiRow(13, 227.2, 441.0, 0.811, 175.0),
    14: KatzFiroozabadiRow(14, 246.4, 475.5, 0.822, 190.0),
    15: KatzFiroozabadiRow(15, 266.0, 511.0, 0.832, 206.0),
    16: KatzFiroozabadiRow(16, 283.0, 542.0, 0.839, 222.0),
    17: KatzFiroozabadiRow(17, 300.0, 572.0, 0.847, 237.0),
    18: KatzFiroozabadiRow(18, 313.0, 595.0, 0.852, 251.0),
    19: KatzFiroozabadiRow(19, 325.0, 617.0, 0.857, 263.0),
    20: KatzFiroozabadiRow(20, 338.0, 640.5, 0.862, 275.0),
    21: KatzFiroozabadiRow(21, 351.0, 664.0, 0.867, 291.0),
    22: KatzFiroozabadiRow(22, 363.0, 686.0, 0.872, 305.0),
    23: KatzFiroozabadiRow(23, 375.0, 707.0, 0.877, 318.0),
    24: KatzFiroozabadiRow(24, 386.0, 727.0, 0.881, 331.0),
    25: KatzFiroozabadiRow(25, 397.0, 747.0, 0.885, 345.0),
    26: KatzFiroozabadiRow(26, 408.0, 766.0, 0.889, 359.0),
    27: KatzFiroozabadiRow(27, 419.0, 784.0, 0.893, 374.0),
    28: KatzFiroozabadiRow(28, 429.0, 802.0, 0.896, 388.0),
    29: KatzFiroozabadiRow(29, 438.0, 817.0, 0.899, 402.0),
    30: KatzFiroozabadiRow(30, 446.0, 834.0, 0.902, 416.0),
    31: KatzFiroozabadiRow(31, 455.0, 850.0, 0.906, 430.0),
    32: KatzFiroozabadiRow(32, 463.0, 866.0, 0.909, 444.0),
    33: KatzFiroozabadiRow(33, 471.0, 881.0, 0.912, 458.0),
    34: KatzFiroozabadiRow(34, 478.0, 895.0, 0.914, 472.0),
    35: KatzFiroozabadiRow(35, 486.0, 908.0, 0.917, 486.0),
    36: KatzFiroozabadiRow(36, 493.0, 922.0, 0.919, 500.0),
    37: KatzFiroozabadiRow(37, 500.0, 934.0, 0.922, 514.0),
    38: KatzFiroozabadiRow(38, 508.0, 947.0, 0.924, 528.0),
    39: KatzFiroozabadiRow(39, 515.0, 959.0, 0.926, 542.0),
    40: KatzFiroozabadiRow(40, 522.0, 972.0, 0.928, 556.0),
    41: KatzFiroozabadiRow(41, 528.0, 982.0, 0.930, 570.0),
    42: KatzFiroozabadiRow(42, 534.0, 993.0, 0.931, 584.0),
    43: KatzFiroozabadiRow(43, 540.0, 1004.0, 0.933, 598.0),
    44: KatzFiroozabadiRow(44, 547.0, 1017.0, 0.935, 612.0),
    45: KatzFiroozabadiRow(45, 553.0, 1027.0, 0.937, 626.0),
}


def get_katz_firoozabadi_table() -> Dict[int, KatzFiroozabadiRow]:
    """Return a copy of the C6–C45 generalized property table."""
    return dict(_TABLE)

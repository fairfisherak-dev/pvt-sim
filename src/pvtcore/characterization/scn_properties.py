"""
SCN property generation for characterization.

Implements docs/file_structure.md and docs/technical_notes.md §2:
- Provides generalized SCN properties (MW, SG, Tb) for C6–C45 using Katz & Firoozabadi (1978) Table 1.
- Supports extrapolation beyond C45 using a Pedersen-style ln(SG)=C + D*MW fit and Pedersen Tb(MW,SG).

Primary sources:
- Katz & Firoozabadi (1978): /mnt/data/KatzFlroozabadi-1978_GeneralizedSCNProperties.pdf
- Pedersen (1984): /mnt/data/Pedersen-1984_PlusFractionSplitting.pdf (Tb correlation + ln(SG) form)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .scn_tables.katz_firoozabadi_1978 import get_katz_firoozabadi_table


@dataclass(frozen=True)
class SCNProperties:
    n: np.ndarray        # SCN indices, shape (N,)
    mw: np.ndarray       # g/mol
    sg_6060: np.ndarray  # specific gravity at 60/60°F (dimensionless)
    tb_k: np.ndarray     # K


def _tb_c_to_k(tb_c: np.ndarray) -> np.ndarray:
    return tb_c + 273.15


def _mw_paraffin_baseline(n: np.ndarray) -> np.ndarray:
    # docs/technical_notes.md §2.2
    return 14.0 * n.astype(float) - 4.0


def _fit_ln_sg_vs_mw(mw: np.ndarray, sg: np.ndarray) -> Tuple[float, float]:
    """
    Fit ln(SG) = C + D*MW via least squares.

    Returns
    -------
    (C, D)
    """
    y = np.log(sg)
    A = np.column_stack([np.ones_like(mw), mw])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    C, D = float(coef[0]), float(coef[1])
    return C, D


def _tb_pedersen_rankine(mw: np.ndarray, sg: np.ndarray) -> np.ndarray:
    """
    Pedersen (1984) Tb correlation used for heavy extrapolation.

    Historically given in °R:
        Tb = 1928.3 - 1.695e5 * exp(-0.03522 * MW * SG)

    Returns
    -------
    Tb in °R.
    """
    return 1928.3 - 1.695e5 * np.exp(-0.03522 * mw * sg)


def _rankine_to_k(tb_r: np.ndarray) -> np.ndarray:
    return tb_r * (5.0 / 9.0)


def get_scn_properties(
    *,
    n_start: int = 6,
    n_end: int = 45,
    extrapolate: bool = True,
    extrapolation_fit_window: Tuple[int, int] = (35, 45),
) -> SCNProperties:
    """
    Get generalized SCN properties (MW, SG, Tb) for a requested SCN range.

    Parameters
    ----------
    n_start, n_end
        Inclusive SCN range.
    extrapolate
        If True and n_end > 45, extrapolate MW/SG/Tb beyond the Katz table.
        If False, raises ValueError when n_end > 45.
    extrapolation_fit_window
        (n_lo, n_hi) range within the table used to fit ln(SG) vs MW for extrapolation.

    Returns
    -------
    SCNProperties
        Arrays sized (n_end - n_start + 1) with MW (g/mol), SG(60/60), and Tb (K).
    """
    if n_end < n_start:
        raise ValueError(f"n_end must be >= n_start, got {n_start=}, {n_end=}")

    table = get_katz_firoozabadi_table()
    n = np.arange(n_start, n_end + 1, dtype=int)

    mw = np.empty_like(n, dtype=float)
    sg = np.empty_like(n, dtype=float)
    tb_k = np.empty_like(n, dtype=float)

    # Fill from table where available
    for idx, ni in enumerate(n):
        row = table.get(int(ni))
        if row is not None:
            mw[idx] = row.mw
            sg[idx] = row.sg_6060
            tb_k[idx] = float(_tb_c_to_k(np.array(row.tb_c, dtype=float)))
        else:
            if not extrapolate:
                raise ValueError(f"No generalized SCN properties for C{ni}; set extrapolate=True.")
            mw[idx] = np.nan
            sg[idx] = np.nan
            tb_k[idx] = np.nan

    if np.isnan(mw).any():
        # Extrapolate beyond table max (45)
        n_lo, n_hi = extrapolation_fit_window
        if not (6 <= n_lo <= n_hi <= 45):
            raise ValueError("extrapolation_fit_window must be within [6,45]")

        fit_ns = np.arange(n_lo, n_hi + 1, dtype=int)
        fit_mw = np.array([table[int(k)].mw for k in fit_ns], dtype=float)
        fit_sg = np.array([table[int(k)].sg_6060 for k in fit_ns], dtype=float)
        C, D = _fit_ln_sg_vs_mw(fit_mw, fit_sg)

        mask = np.isnan(mw)
        n_ext = n[mask]
        mw_ext = _mw_paraffin_baseline(n_ext)
        sg_ext = np.exp(C + D * mw_ext)
        tb_r = _tb_pedersen_rankine(mw_ext, sg_ext)
        tb_ext = _rankine_to_k(tb_r)

        mw[mask] = mw_ext
        sg[mask] = sg_ext
        tb_k[mask] = tb_ext

    # Basic sanity
    if not (np.isfinite(mw).all() and np.isfinite(sg).all() and np.isfinite(tb_k).all()):
        raise ValueError("Non-finite SCN properties produced.")
    if np.any(mw <= 0.0) or np.any(tb_k <= 0.0) or np.any(sg <= 0.0):
        raise ValueError("Non-physical SCN properties produced.")

    return SCNProperties(n=n, mw=mw, sg_6060=sg, tb_k=tb_k)

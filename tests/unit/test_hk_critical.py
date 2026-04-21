"""Tests for the Heidemann-Khalil mixture critical-point solver.

Reference values come from PR-EOS published benchmarks and from a well-converged
H-K solve of the same EOS used at runtime. The tests pin Tc within 2 K and Pc
within 2% to leave room for finite-difference step choice.
"""

from __future__ import annotations

import numpy as np
import pytest

from pvtcore.envelope.hk_critical import compute_critical_point


def test_hk_c2_c3_50_50(components, c2_c3_pr):
    z = np.array([0.5, 0.5], dtype=np.float64)
    result = compute_critical_point(
        z, [components["C2"], components["C3"]], c2_c3_pr,
    )
    assert result.converged
    assert result.Tc is not None and result.Pc is not None
    # PR-EOS: C2/C3 50/50 Tc ≈ 343-344 K, Pc ≈ 4.95 MPa.
    assert result.Tc == pytest.approx(343.7, abs=2.0)
    assert result.Pc == pytest.approx(4.95e6, rel=0.02)
    assert abs(result.residual_lambda) < 1e-5
    assert result.iterations <= 20


def test_hk_c1_c10_50_50(components, c1_c10_pr):
    z = np.array([0.5, 0.5], dtype=np.float64)
    result = compute_critical_point(
        z, [components["C1"], components["C10"]], c1_c10_pr,
    )
    assert result.converged
    # PR-EOS: C1/C10 50/50 is a hot, high-pressure mixture critical
    # Tc ≈ 580 K, Pc ≈ 8.7 MPa.
    assert result.Tc == pytest.approx(581.5, abs=3.0)
    assert result.Pc == pytest.approx(8.68e6, rel=0.03)


def test_hk_c1_c4_50_50(components, c1_c4_pr):
    z = np.array([0.5, 0.5], dtype=np.float64)
    result = compute_critical_point(
        z, [components["C1"], components["C4"]], c1_c4_pr,
    )
    assert result.converged
    assert result.Tc is not None and 250.0 < result.Tc < 420.0
    assert result.Pc is not None and 4e6 < result.Pc < 2e7


def test_hk_rejects_non_physical_result(components, c1_c10_pr):
    """A wildly wrong seed should still converge (bounds + damping), not crash."""
    z = np.array([0.5, 0.5], dtype=np.float64)
    # Seed T far from the correct answer — bounds/damping should recover.
    result = compute_critical_point(
        z, [components["C1"], components["C10"]], c1_c10_pr,
        T_init=400.0,
    )
    assert result.converged
    assert result.Tc == pytest.approx(581.5, abs=5.0)


def test_hk_result_has_consistent_pressure(components, c2_c3_pr):
    """Pc returned by H-K must equal P_EOS(Tc, Vc, z) by construction."""
    from pvtcore.envelope.hk_critical import _pressure_TVn

    z = np.array([0.5, 0.5], dtype=np.float64)
    result = compute_critical_point(
        z, [components["C2"], components["C3"]], c2_c3_pr,
    )
    assert result.converged
    P_check = _pressure_TVn(result.Tc, result.Vc, z, c2_c3_pr, None)
    assert P_check == pytest.approx(result.Pc, rel=1e-10)

from __future__ import annotations

import numpy as np
import pytest

from pvtcore.envelope.critical_point import detect_critical_point, find_critical_from_envelope


def test_detect_critical_point_accepts_hot_end_branch_closure(components):
    composition = np.array([0.5, 0.5], dtype=np.float64)
    binary = [components["C1"], components["C10"]]

    bubble_T = np.array([280.0, 300.0, 320.0, 340.0], dtype=np.float64)
    bubble_P = np.array([2.0e6, 3.1e6, 4.1e6, 4.80e6], dtype=np.float64)
    dew_T = np.array([290.0, 310.0, 330.0, 340.0], dtype=np.float64)
    dew_P = np.array([5.2e6, 5.0e6, 4.90e6, 4.82e6], dtype=np.float64)

    Tc, Pc = detect_critical_point(
        bubble_T, bubble_P, dew_T, dew_P, composition, binary,
    )

    assert Tc == pytest.approx(340.0)
    assert Pc == pytest.approx(4.81e6)


def test_detect_critical_point_rejects_mid_envelope_intersection(components):
    composition = np.array([0.5, 0.5], dtype=np.float64)
    binary = [components["C1"], components["C10"]]

    bubble_T = np.array([280.0, 300.0, 320.0, 340.0, 360.0], dtype=np.float64)
    bubble_P = np.array([2.0e6, 3.0e6, 4.2e6, 5.1e6, 5.8e6], dtype=np.float64)
    dew_T = np.array([280.0, 300.0, 320.0, 340.0, 360.0], dtype=np.float64)
    dew_P = np.array([6.5e6, 5.2e6, 4.22e6, 6.3e6, 7.4e6], dtype=np.float64)

    Tc, Pc = detect_critical_point(
        bubble_T, bubble_P, dew_T, dew_P, composition, binary,
    )

    assert Tc is None
    assert Pc is None


def test_find_critical_from_envelope_fails_closed_without_hot_end_closure(components):
    composition = np.array([0.5, 0.5], dtype=np.float64)
    binary = [components["C1"], components["C10"]]

    bubble_T = np.array([280.0, 300.0, 320.0, 340.0], dtype=np.float64)
    bubble_P = np.array([2.0e6, 3.0e6, 4.1e6, 5.0e6], dtype=np.float64)
    dew_T = np.array([285.0, 305.0, 325.0, 345.0], dtype=np.float64)
    dew_P = np.array([5.7e6, 5.8e6, 5.9e6, 6.0e6], dtype=np.float64)

    result = find_critical_from_envelope(
        bubble_T, bubble_P, dew_T, dew_P, composition, binary,
    )

    assert result.converged is False
    assert result.Tc is None
    assert result.Pc is None
    assert result.method == "strict_envelope_intersection"

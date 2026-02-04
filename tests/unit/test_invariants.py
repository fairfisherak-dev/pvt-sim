"""
Cross-cutting invariants for the current core (no plus-fractions yet).

These tests are intended to catch "impossible" states early during the
bottom-up rebuild.

They intentionally avoid asserting specific numerical targets (those belong
in validation/regression tests). Instead they assert algebraic/physical
constraints that should *always* hold when the public APIs succeed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pvtcore.models.component import load_components
from pvtcore.eos.peng_robinson import PengRobinsonEOS
from pvtcore.flash.pt_flash import pt_flash
from pvtcore.flash.bubble_point import calculate_bubble_point
from pvtcore.flash.dew_point import calculate_dew_point


def _assert_simplex(
    x: np.ndarray,
    *,
    tol_sum: float = 1e-10,
    allow_all_zero: bool = False,
) -> None:
    """Assert a vector is on the unit simplex.

    If allow_all_zero is True, the all-zero vector is also accepted (used for
    the absent phase in single-phase flash results).
    """
    x = np.asarray(x, dtype=float)
    assert x.ndim == 1
    assert np.all(np.isfinite(x))

    if allow_all_zero and np.allclose(x, 0.0, atol=1e-15):
        return

    # Mildly permissive lower bound to avoid failing on tiny negative roundoff.
    assert np.all(x >= -1e-12)
    assert np.all(x <= 1.0 + 1e-12)
    assert abs(float(x.sum()) - 1.0) < tol_sum


def _assert_positive_finite(arr: np.ndarray) -> None:
    """Assert array is finite and strictly positive."""
    arr = np.asarray(arr, dtype=float)
    assert np.all(np.isfinite(arr))
    assert np.all(arr > 0.0)


@pytest.fixture(scope="module")
def components():
    """Load component database."""
    return load_components()


@pytest.fixture(scope="module")
def binary_c1_c10(components):
    """Methane-decane binary mixture."""
    return [components["C1"], components["C10"]]


@pytest.fixture(scope="module")
def binary_c1_c4(components):
    """Methane-n-butane binary mixture."""
    return [components["C1"], components["C4"]]


@pytest.fixture(scope="module")
def eos_c1_c10(binary_c1_c10):
    """PR EOS for methane-decane."""
    return PengRobinsonEOS(binary_c1_c10)


@pytest.fixture(scope="module")
def eos_c1_c4(binary_c1_c4):
    """PR EOS for methane-butane."""
    return PengRobinsonEOS(binary_c1_c4)


class TestComponentDatabaseInvariants:
    def test_all_components_have_physical_parameters(self, components):
        """Basic physical sanity for the built-in pure-component dataset."""
        for comp_id, c in components.items():
            assert np.isfinite(c.Tc) and c.Tc > 0.0, comp_id
            assert np.isfinite(c.Pc) and c.Pc > 0.0, comp_id
            assert np.isfinite(c.Vc) and c.Vc > 0.0, comp_id
            assert np.isfinite(c.MW) and c.MW > 0.0, comp_id
            assert np.isfinite(c.Tb) and c.Tb > 0.0, comp_id
            assert np.isfinite(c.omega), comp_id

            # Very loose plausibility bounds (units are part of the invariant).
            assert 1e3 < c.Pc < 1e9, comp_id  # Pa
            assert 1.0 < c.Tc < 2_000.0, comp_id  # K (allow cryogenic species like He)
            assert 1.0 < c.MW < 1_000.0, comp_id  # g/mol


            # For normal fluids, Tc should exceed Tb.
            assert c.Tc > c.Tb, comp_id


class TestEOSInvariants:
    def test_pr_eos_respects_covolume_bound(self, eos_c1_c10):
        """All EOS roots should satisfy Z >= B (covolume constraint)."""
        T = 300.0
        P = 3.0e6
        z = np.array([0.6, 0.4])

        result = eos_c1_c10.calculate(P, T, z, phase="auto")

        assert np.isfinite(result.A)
        assert np.isfinite(result.B) and result.B > 0.0
        assert np.isfinite(result.a_mix) and result.a_mix >= 0.0
        assert np.isfinite(result.b_mix) and result.b_mix > 0.0

        for r in result.roots:
            assert np.isfinite(r)
            assert r >= result.B - 1e-14

        # Compressibility value(s) must also be >= B.
        if isinstance(result.Z, np.ndarray):
            assert np.all(result.Z >= result.B - 1e-14)
        else:
            assert result.Z >= result.B - 1e-14

    def test_pr_eos_fugacity_coefficients_are_positive(self, eos_c1_c10):
        """φ_i must be finite and strictly positive."""
        T = 300.0
        P = 3.0e6
        z = np.array([0.6, 0.4])

        # Test both phases explicitly.
        phi_L = eos_c1_c10.fugacity_coefficient(P, T, z, phase="liquid")
        phi_V = eos_c1_c10.fugacity_coefficient(P, T, z, phase="vapor")

        _assert_positive_finite(phi_L)
        _assert_positive_finite(phi_V)


class TestFlashInvariants:
    @pytest.mark.parametrize(
        "pressure",
        [1.0e6, 3.0e6, 1.0e7],
        ids=["10bar", "30bar", "100bar"],
    )
    def test_pt_flash_mass_balance_and_bounds(self, pressure, binary_c1_c10, eos_c1_c10):
        """PT flash results must obey mass balance and bounds for all phases."""
        T = 300.0
        z = np.array([0.6, 0.4])

        result = pt_flash(pressure, T, z, binary_c1_c10, eos_c1_c10)

        assert result.converged is True
        assert result.phase in {"two-phase", "vapor", "liquid"}

        assert np.isfinite(result.vapor_fraction)
        assert 0.0 <= result.vapor_fraction <= 1.0

        # Feed composition must be preserved.
        _assert_simplex(result.feed_composition)
        np.testing.assert_allclose(result.feed_composition, z, rtol=0, atol=1e-12)

        if result.phase == "two-phase":
            _assert_simplex(result.liquid_composition)
            _assert_simplex(result.vapor_composition)

            # Component-wise bounds.
            assert np.all(result.liquid_composition >= -1e-12)
            assert np.all(result.vapor_composition >= -1e-12)

            # K-values must be positive and consistent with y = Kx (where x>0).
            assert np.all(np.isfinite(result.K_values))
            assert np.all(result.K_values > 0.0)

            x = result.liquid_composition
            y = result.vapor_composition
            for i in range(len(x)):
                if x[i] > 1e-14:
                    assert y[i] / x[i] == pytest.approx(result.K_values[i], rel=2e-5, abs=1e-10)

            _assert_positive_finite(result.liquid_fugacity)
            _assert_positive_finite(result.vapor_fugacity)

        elif result.phase == "liquid":
            assert result.vapor_fraction == pytest.approx(0.0, abs=0.0)
            _assert_simplex(result.liquid_composition)
            _assert_simplex(result.vapor_composition, allow_all_zero=True)
            np.testing.assert_allclose(result.liquid_composition, z, atol=1e-12)

        else:  # vapor
            assert result.vapor_fraction == pytest.approx(1.0, abs=0.0)
            _assert_simplex(result.vapor_composition)
            _assert_simplex(result.liquid_composition, allow_all_zero=True)
            np.testing.assert_allclose(result.vapor_composition, z, atol=1e-12)

    def test_flash_extreme_conditions_return_expected_single_phases(self, binary_c1_c10, eos_c1_c10):
        """Choose conditions that should be unambiguously single-phase.

        This is a *sanity* check on the stability gate + single-phase return
        paths inside `pt_flash`, not a VLE accuracy test.
        """
        z = np.array([0.8, 0.2])

        # Very hot + low pressure: should be vapor-like.
        res_v = pt_flash(1.0e5, 600.0, z, binary_c1_c10, eos_c1_c10)
        assert res_v.converged is True
        assert res_v.phase == "vapor"
        assert res_v.vapor_fraction == pytest.approx(1.0, abs=0.0)
        _assert_simplex(res_v.vapor_composition)
        _assert_simplex(res_v.liquid_composition, allow_all_zero=True)

        # Cold + very high pressure: should be liquid-like.
        res_l = pt_flash(5.0e7, 250.0, z, binary_c1_c10, eos_c1_c10)
        assert res_l.converged is True
        assert res_l.phase == "liquid"
        assert res_l.vapor_fraction == pytest.approx(0.0, abs=0.0)
        _assert_simplex(res_l.liquid_composition)
        _assert_simplex(res_l.vapor_composition, allow_all_zero=True)


class TestSaturationInvariants:
    def test_bubble_point_outputs_are_well_formed(self, binary_c1_c10, eos_c1_c10):
        T = 300.0
        z = np.array([0.6, 0.4])

        bp = calculate_bubble_point(T, z, binary_c1_c10, eos_c1_c10)

        assert bp.converged is True
        assert np.isfinite(bp.pressure) and bp.pressure > 0.0
        _assert_simplex(bp.liquid_composition)
        _assert_simplex(bp.vapor_composition)
        assert np.all(np.isfinite(bp.K_values))
        assert np.all(bp.K_values > 0.0)

    def test_dew_point_outputs_are_well_formed(self, binary_c1_c4, eos_c1_c4):
        # C1-C4 behaves more nicely for dew point at sub-ambient temperatures.
        T = 250.0
        z = np.array([0.5, 0.5])

        dp = calculate_dew_point(T, z, binary_c1_c4, eos_c1_c4)

        assert dp.converged is True
        assert np.isfinite(dp.pressure) and dp.pressure > 0.0
        _assert_simplex(dp.vapor_composition)
        _assert_simplex(dp.liquid_composition)
        assert np.all(np.isfinite(dp.K_values))
        assert np.all(dp.K_values > 0.0)

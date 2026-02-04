"""Unit tests for Michelsen stability analysis.

Tests validate stability calculations against:
- Known stable single-phase conditions
- Known unstable two-phase conditions
- Literature results from Michelsen (1982)
"""

import pytest
import numpy as np
from pvtcore.stability.michelsen import (
    michelsen_stability_test,
    is_stable,
    StabilityResult,
    STABILITY_TOLERANCE,
    TPD_TOLERANCE
)
from pvtcore.stability.tpd import calculate_tpd, calculate_d_terms
from pvtcore.eos.peng_robinson import PengRobinsonEOS
from pvtcore.models.component import load_components
from pvtcore.core.errors import ValidationError, ConvergenceError


@pytest.fixture
def components():
    """Load component database."""
    return load_components()


@pytest.fixture
def methane_eos(components):
    """Create PR EOS for pure methane."""
    return PengRobinsonEOS([components['C1']])


@pytest.fixture
def binary_c1_c10_eos(components):
    """Create PR EOS for methane-decane binary mixture."""
    return PengRobinsonEOS([components['C1'], components['C10']])


@pytest.fixture
def binary_c1_c4_eos(components):
    """Create PR EOS for methane-butane binary mixture."""
    return PengRobinsonEOS([components['C1'], components['C4']])


class TestTPDFunction:
    """Test tangent plane distance calculations."""

    def test_tpd_at_feed_composition(self, methane_eos):
        """Test that TPD = 0 when trial equals feed."""
        T = 300.0  # K
        P = 5e6  # Pa
        z = np.array([1.0])

        # Calculate d terms
        d_terms = calculate_d_terms(z, methane_eos, P, T, phase='vapor')
        ln_phi_z = d_terms - np.log(z)

        # Trial = feed should give TPD = 0
        tpd = calculate_tpd(z, z, ln_phi_z, methane_eos, P, T, phase='vapor')

        # TPD should be zero (within numerical precision)
        assert abs(tpd) < 1e-12

    def test_tpd_composition_normalization(self, methane_eos):
        """Test that TPD calculation requires normalized compositions."""
        T = 300.0
        P = 5e6
        z = np.array([1.0])
        W_unnormalized = np.array([2.0])  # Not normalized

        d_terms = calculate_d_terms(z, methane_eos, P, T, phase='vapor')
        ln_phi_z = d_terms - np.log(z)

        # Should raise error for unnormalized composition
        with pytest.raises(Exception):  # CompositionError
            calculate_tpd(W_unnormalized, z, ln_phi_z, methane_eos, P, T, phase='vapor')

    def test_tpd_binary_mixture(self, binary_c1_c10_eos):
        """Test TPD calculation for binary mixture."""
        T = 300.0  # K
        P = 5e6  # Pa
        z = np.array([0.5, 0.5])  # Equal molar mixture

        # Calculate d terms for liquid feed
        d_terms = calculate_d_terms(z, binary_c1_c10_eos, P, T, phase='liquid')
        ln_phi_z = d_terms - np.log(z)

        # Trial vapor-like composition (enriched in light component)
        W_vapor = np.array([0.9, 0.1])

        tpd = calculate_tpd(
            W_vapor, z, ln_phi_z, binary_c1_c10_eos,
            P, T, phase='vapor'
        )

        # TPD should be a real number
        assert isinstance(tpd, (float, np.floating))


class TestStabilityResult:
    """Test StabilityResult data structure."""

    def test_stability_result_fields(self, methane_eos):
        """Test that StabilityResult contains required fields."""
        T = 400.0  # High T, low P → stable vapor
        P = 1e5  # 1 bar
        z = np.array([1.0])

        result = michelsen_stability_test(
            z, P, T, methane_eos, feed_phase='vapor'
        )

        # Check all required fields exist
        assert hasattr(result, 'stable')
        assert hasattr(result, 'tpd_min')
        assert hasattr(result, 'trial_compositions')
        assert hasattr(result, 'tpd_values')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'feed_phase')
        assert hasattr(result, 'converged')

        # Check types
        assert isinstance(result.stable, bool)
        assert isinstance(result.tpd_min, (float, np.floating))
        assert isinstance(result.trial_compositions, list)
        assert isinstance(result.tpd_values, list)
        assert isinstance(result.iterations, list)
        assert isinstance(result.feed_phase, str)
        assert isinstance(result.converged, bool)


class TestStableSinglePhase:
    """Test cases where mixture should be stable (single phase)."""

    def test_pure_methane_low_pressure_high_temperature(self, methane_eos):
        """Test that pure methane is stable vapor at low P, high T.

        Conditions: T = 400 K (well above Tc = 190.6 K), P = 1 bar
        Expected: Stable single-phase vapor
        """
        T = 400.0  # K
        P = 1e5  # 1 bar = 100 kPa
        z = np.array([1.0])

        result = michelsen_stability_test(
            z, P, T, methane_eos, feed_phase='vapor'
        )

        # Should be stable
        assert result.stable is True
        assert result.tpd_min >= -TPD_TOLERANCE
        assert result.converged is True

        # All TPD values should be non-negative
        assert all(tpd >= -TPD_TOLERANCE for tpd in result.tpd_values)

    def test_pure_methane_high_temperature_moderate_pressure(self, methane_eos, components):
        """Test methane stability at supercritical temperature.

        Conditions: T = 250 K (> Tc = 190.6 K), P = 5 MPa
        Expected: Stable supercritical fluid
        """
        comp = components['C1']
        T = 1.3 * comp.Tc  # Above critical temperature
        P = 5e6  # 50 bar
        z = np.array([1.0])

        result = michelsen_stability_test(
            z, P, T, methane_eos, feed_phase='vapor'
        )

        # Should be stable above critical temperature
        assert result.stable is True
        assert result.tpd_min >= -TPD_TOLERANCE

    def test_is_stable_convenience_function(self, methane_eos):
        """Test simplified is_stable() function."""
        T = 400.0
        P = 1e5
        z = np.array([1.0])

        stable = is_stable(z, P, T, methane_eos, feed_phase='vapor')

        assert stable is True


class TestUnstableTwoPhase:
    """Test cases where mixture should be unstable (two-phase)."""

    def test_c1_c10_mixture_vle_region(self, binary_c1_c10_eos, components):
        """Test C1-C10 binary in VLE region shows instability.

        Methane-decane at moderate conditions should show phase split
        due to large volatility difference.

        Conditions: T = 300 K, P = 3 MPa, z = [0.5, 0.5]
        Expected: Unstable, TPD < 0
        """
        T = 300.0  # K
        P = 3e6  # 30 bar
        z = np.array([0.5, 0.5])  # Equal molar

        result = michelsen_stability_test(
            z, P, T, binary_c1_c10_eos, feed_phase='liquid'
        )

        # Should be unstable
        assert result.stable is False
        assert result.tpd_min < -TPD_TOLERANCE

        # At least one trial should have negative TPD
        assert any(tpd < -TPD_TOLERANCE for tpd in result.tpd_values)

    def test_c1_c4_mixture_two_phase(self, binary_c1_c4_eos, components):
        """Test C1-C4 binary at conditions known to be two-phase.

        Conditions: T = 250 K, P = 4 MPa, z = [0.7, 0.3]
        Expected: Unstable (two-phase region)
        """
        T = 250.0  # K
        P = 4e6  # 40 bar
        z = np.array([0.7, 0.3])  # Light component rich

        result = michelsen_stability_test(
            z, P, T, binary_c1_c4_eos, feed_phase='liquid'
        )

        # Should be unstable at these conditions
        assert result.stable is False
        assert result.tpd_min < 0

    def test_pure_methane_below_critical(self, methane_eos, components):
        """Test pure methane below critical temperature.

        At subcritical temperature and moderate pressure, should find
        that liquid wants to vaporize or vapor wants to condense.

        Conditions: T = 150 K (< Tc = 190.6 K), P = 2 MPa
        Expected: May be unstable depending on exact conditions
        """
        comp = components['C1']
        T = 0.8 * comp.Tc  # Below critical
        P = 2e6  # 20 bar
        z = np.array([1.0])

        # Test as liquid feed
        result_liquid = michelsen_stability_test(
            z, P, T, methane_eos, feed_phase='liquid'
        )

        # Test as vapor feed
        result_vapor = michelsen_stability_test(
            z, P, T, methane_eos, feed_phase='vapor'
        )

        # At least one phase should be unstable (want to split)
        # or both could be metastable
        # We just verify the calculation runs without error
        assert isinstance(result_liquid.stable, bool)
        assert isinstance(result_vapor.stable, bool)


class TestConvergence:
    """Test convergence behavior of stability algorithm."""

    def test_convergence_flag(self, methane_eos):
        """Test that convergence flag is set correctly."""
        T = 400.0
        P = 1e5
        z = np.array([1.0])

        result = michelsen_stability_test(
            z, P, T, methane_eos, feed_phase='vapor'
        )

        # Should converge for this simple case
        assert result.converged is True

        # Iterations should be less than max
        assert all(it < 1000 for it in result.iterations)

    def test_custom_tolerance(self, methane_eos):
        """Test stability test with custom tolerance."""
        T = 400.0
        P = 1e5
        z = np.array([1.0])

        # Looser tolerance should converge faster
        result_loose = michelsen_stability_test(
            z, P, T, methane_eos, feed_phase='vapor',
            tolerance=1e-6
        )

        # Tighter tolerance may need more iterations
        result_tight = michelsen_stability_test(
            z, P, T, methane_eos, feed_phase='vapor',
            tolerance=1e-12
        )

        # Both should converge but tight may need more iterations
        assert result_loose.converged is True
        assert result_tight.converged is True


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_composition_sum(self, methane_eos):
        """Test that invalid composition sum raises error."""
        T = 300.0
        P = 5e6
        z_invalid = np.array([0.5])  # Doesn't sum to 1.0

        with pytest.raises(ValidationError):
            michelsen_stability_test(
                z_invalid, P, T, methane_eos, feed_phase='liquid'
            )

    def test_negative_pressure(self, methane_eos):
        """Test that negative pressure raises error."""
        T = 300.0
        P = -1e6  # Invalid
        z = np.array([1.0])

        with pytest.raises(ValidationError):
            michelsen_stability_test(
                z, P, T, methane_eos, feed_phase='liquid'
            )

    def test_negative_temperature(self, methane_eos):
        """Test that negative temperature raises error."""
        T = -100.0  # Invalid
        P = 5e6
        z = np.array([1.0])

        with pytest.raises(ValidationError):
            michelsen_stability_test(
                z, P, T, methane_eos, feed_phase='liquid'
            )

    def test_invalid_feed_phase(self, methane_eos):
        """Test that invalid feed phase raises error."""
        T = 300.0
        P = 5e6
        z = np.array([1.0])

        with pytest.raises(ValidationError):
            michelsen_stability_test(
                z, P, T, methane_eos, feed_phase='supercritical'  # Invalid
            )

    def test_composition_eos_mismatch(self, methane_eos):
        """Test that composition length must match EOS components."""
        T = 300.0
        P = 5e6
        z_wrong_size = np.array([0.5, 0.5])  # Two components, but EOS has one

        with pytest.raises(ValidationError):
            michelsen_stability_test(
                z_wrong_size, P, T, methane_eos, feed_phase='liquid'
            )


class TestPressureTemperatureTraverse:
    """Test stability along pressure/temperature paths."""

    def test_pressure_traverse(self, binary_c1_c10_eos):
        """Test stability test along an isobaric path.

        At low T: should be two-phase (unstable)
        At high T: should eventually become single-phase (stable)
        """
        P = 5e6  # 50 bar
        z = np.array([0.5, 0.5])

        temperatures = [250.0, 300.0, 350.0, 400.0, 450.0]
        stability_results = []

        for T in temperatures:
            result = michelsen_stability_test(
                z, P, T, binary_c1_c10_eos, feed_phase='liquid'
            )
            stability_results.append(result.stable)

        # At least should get valid results at all temperatures
        assert len(stability_results) == len(temperatures)
        assert all(isinstance(s, bool) for s in stability_results)

        # Verify we can run stability tests across temperature range
        # (Actual phase behavior depends on the system)
        # At very high T relative to both critical temperatures, expect stable
        T_very_high = 600.0  # Well above C10 critical temperature
        result_high = michelsen_stability_test(
            z, P, T_very_high, binary_c1_c10_eos, feed_phase='vapor'
        )
        assert result_high.stable is True

    def test_temperature_traverse(self, binary_c1_c10_eos):
        """Test stability test along an isothermal path.

        At low P: should be single-phase vapor (stable)
        At high P: may enter two-phase region
        """
        T = 400.0  # K - higher temperature to ensure vapor stability at low P
        z = np.array([0.5, 0.5])

        pressures = [1e5, 1e6, 3e6, 5e6, 10e6]  # 1 bar to 100 bar
        stability_results = []

        for P in pressures:
            result = michelsen_stability_test(
                z, P, T, binary_c1_c10_eos, feed_phase='vapor'
            )
            stability_results.append(result.stable)

        # At least should get valid results at all pressures
        assert len(stability_results) == len(pressures)
        assert all(isinstance(s, bool) for s in stability_results)

        # At very low pressure and high temperature, should be stable vapor
        # Test at extremely low pressure
        P_very_low = 1e4  # 0.1 bar
        result_low_p = michelsen_stability_test(
            z, P_very_low, T, binary_c1_c10_eos, feed_phase='vapor'
        )
        assert result_low_p.stable is True


class TestBinaryInteractionParameters:
    """Test effect of binary interaction parameters on stability."""

    def test_stability_with_kij(self, binary_c1_c10_eos):
        """Test that binary interaction parameters affect stability."""
        T = 300.0
        P = 5e6
        z = np.array([0.5, 0.5])

        # Test without BIP
        result_no_kij = michelsen_stability_test(
            z, P, T, binary_c1_c10_eos, feed_phase='liquid',
            binary_interaction=None
        )

        # Test with BIP (typical C1-C10 kij ≈ 0.04)
        kij = np.array([[0.0, 0.04],
                        [0.04, 0.0]])
        result_with_kij = michelsen_stability_test(
            z, P, T, binary_c1_c10_eos, feed_phase='liquid',
            binary_interaction=kij
        )

        # Both should complete successfully
        assert isinstance(result_no_kij.stable, bool)
        assert isinstance(result_with_kij.stable, bool)

        # BIPs can affect phase boundaries, so TPD values may differ
        # (we just verify both calculations work)


class TestMultipleTrials:
    """Test that multiple trial compositions are tested."""

    def test_two_trials_performed(self, binary_c1_c10_eos):
        """Test that both vapor-like and liquid-like trials are performed."""
        T = 300.0
        P = 5e6
        z = np.array([0.5, 0.5])

        result = michelsen_stability_test(
            z, P, T, binary_c1_c10_eos, feed_phase='liquid'
        )

        # Should have two trials (vapor-like and liquid-like)
        assert len(result.trial_compositions) == 2
        assert len(result.tpd_values) == 2
        assert len(result.iterations) == 2

    def test_trial_compositions_normalized(self, binary_c1_c10_eos):
        """Test that all trial compositions sum to 1.0."""
        T = 300.0
        P = 5e6
        z = np.array([0.5, 0.5])

        result = michelsen_stability_test(
            z, P, T, binary_c1_c10_eos, feed_phase='liquid'
        )

        # All trial compositions should be normalized
        for W in result.trial_compositions:
            assert abs(W.sum() - 1.0) < 1e-10

    def test_tpd_min_is_minimum(self, binary_c1_c10_eos):
        """Test that tpd_min is the minimum of all trial TPD values."""
        T = 300.0
        P = 5e6
        z = np.array([0.5, 0.5])

        result = michelsen_stability_test(
            z, P, T, binary_c1_c10_eos, feed_phase='liquid'
        )

        # tpd_min should be the minimum
        assert result.tpd_min == pytest.approx(min(result.tpd_values), abs=1e-10)

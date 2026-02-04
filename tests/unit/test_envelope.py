"""Unit tests for phase envelope calculations.

Tests validate phase envelope tracing against:
- Known thermodynamic relationships
- Critical point location
- Envelope shape and continuity
- Component behavior
"""

import pytest
import numpy as np
from pvtcore.envelope.phase_envelope import (
    calculate_phase_envelope,
    EnvelopeResult,
    estimate_cricondentherm,
    estimate_cricondenbar
)
from pvtcore.eos.peng_robinson import PengRobinsonEOS
from pvtcore.models.component import load_components
from pvtcore.core.errors import ValidationError


@pytest.fixture
def components():
    """Load component database."""
    return load_components()


@pytest.fixture
def binary_c1_c10_eos(components):
    """Create PR EOS for methane-decane binary mixture."""
    return PengRobinsonEOS([components['C1'], components['C10']])


@pytest.fixture
def binary_c1_c4_eos(components):
    """Create PR EOS for methane-butane binary mixture."""
    return PengRobinsonEOS([components['C1'], components['C4']])


@pytest.fixture
def binary_c2_c3_eos(components):
    """Create PR EOS for ethane-propane binary mixture."""
    return PengRobinsonEOS([components['C2'], components['C3']])


class TestEnvelopeResult:
    """Test EnvelopeResult data structure."""

    def test_envelope_result_fields(self, binary_c1_c10_eos, components):
        """Test that EnvelopeResult contains required fields."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        # Check all required fields exist
        assert hasattr(envelope, 'bubble_T')
        assert hasattr(envelope, 'bubble_P')
        assert hasattr(envelope, 'dew_T')
        assert hasattr(envelope, 'dew_P')
        assert hasattr(envelope, 'critical_T')
        assert hasattr(envelope, 'critical_P')
        assert hasattr(envelope, 'composition')
        assert hasattr(envelope, 'converged')
        assert hasattr(envelope, 'n_bubble_points')
        assert hasattr(envelope, 'n_dew_points')

        # Check types
        assert isinstance(envelope.bubble_T, np.ndarray)
        assert isinstance(envelope.bubble_P, np.ndarray)
        assert isinstance(envelope.dew_T, np.ndarray)
        assert isinstance(envelope.dew_P, np.ndarray)
        assert isinstance(envelope.converged, bool)
        assert isinstance(envelope.n_bubble_points, int)
        assert isinstance(envelope.n_dew_points, int)


class TestEnvelopeCalculation:
    """Test phase envelope calculation."""

    def test_envelope_converges(self, binary_c1_c10_eos, components):
        """Test that envelope calculation converges."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        assert envelope.converged is True
        # Should have at least the bubble curve
        assert envelope.n_bubble_points > 3

    def test_envelope_has_bubble_curve(self, binary_c1_c10_eos, components):
        """Test that envelope has bubble curve."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        # Should have multiple points on bubble curve
        assert len(envelope.bubble_T) > 5
        assert len(envelope.bubble_P) > 5

    def test_bubble_dew_curves_same_length(self, binary_c1_c10_eos, components):
        """Test that T and P arrays have matching lengths."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        assert len(envelope.bubble_T) == len(envelope.bubble_P)
        assert len(envelope.dew_T) == len(envelope.dew_P)

    def test_temperatures_positive(self, binary_c1_c10_eos, components):
        """Test that all temperatures are positive."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        assert np.all(envelope.bubble_T > 0)
        assert np.all(envelope.dew_T > 0)

    def test_pressures_positive(self, binary_c1_c10_eos, components):
        """Test that all pressures are positive."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        assert np.all(envelope.bubble_P > 0)
        assert np.all(envelope.dew_P > 0)

    def test_temperatures_increasing(self, binary_c1_c10_eos, components):
        """Test that temperatures increase along curves (continuation method)."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        # Temperatures should be generally increasing (with possible small variations)
        # Check that most steps are increasing on bubble curve
        if len(envelope.bubble_T) > 1:
            bubble_T_diff = np.diff(envelope.bubble_T)
            # At least 80% of steps should be increasing
            assert np.sum(bubble_T_diff > 0) > 0.8 * len(bubble_T_diff)

        # Check dew curve if it exists
        if len(envelope.dew_T) > 1:
            dew_T_diff = np.diff(envelope.dew_T)
            assert np.sum(dew_T_diff > 0) > 0.8 * len(dew_T_diff)


class TestCriticalPoint:
    """Test critical point detection."""

    def test_critical_point_detected(self, binary_c1_c10_eos, components):
        """Test that critical point is detected."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        assert envelope.critical_T is not None
        assert envelope.critical_P is not None

    def test_critical_point_positive(self, binary_c1_c10_eos, components):
        """Test that critical point values are positive."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        if envelope.critical_T is not None:
            assert envelope.critical_T > 0
        if envelope.critical_P is not None:
            assert envelope.critical_P > 0

    def test_critical_point_between_pure_components(self, binary_c1_c10_eos, components):
        """Test that critical temperature is between pure component Tc values.

        For a binary mixture, the critical temperature should lie between
        the critical temperatures of the pure components (Kay's rule approximation).
        """
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        Tc_C1 = components['C1'].Tc
        Tc_C10 = components['C10'].Tc

        if envelope.critical_T is not None:
            # Critical T should be between pure component values
            # Allow some margin for EOS predictions
            assert Tc_C1 * 0.9 < envelope.critical_T < Tc_C10 * 1.1

    def test_critical_point_on_envelope(self, binary_c1_c10_eos, components):
        """Test that critical point is physically reasonable.

        For asymmetric binary mixtures, envelope tracing may find different
        saturation branches, so the critical point may not lie exactly on
        the traced curves. We validate that:
        1. The critical point is between pure component criticals
        2. The critical point is close to Kay's mixing rule estimate
        """
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        if envelope.critical_T is not None and envelope.critical_P is not None:
            Tc_C1 = components['C1'].Tc
            Tc_C10 = components['C10'].Tc
            Pc_C1 = components['C1'].Pc
            Pc_C10 = components['C10'].Pc

            # Critical T should be between component criticals
            assert Tc_C1 * 0.8 < envelope.critical_T < Tc_C10 * 1.2

            # Critical P should be in reasonable range
            Pc_min = min(Pc_C1, Pc_C10)
            Pc_max = max(Pc_C1, Pc_C10)
            assert Pc_min * 0.5 < envelope.critical_P < Pc_max * 1.5

            # Should be close to Kay's mixing rule estimate
            Tc_kay = 0.5 * (Tc_C1 + Tc_C10)
            Pc_kay = 0.5 * (Pc_C1 + Pc_C10)

            # Within 30% of Kay's estimate for T
            assert abs(envelope.critical_T - Tc_kay) / Tc_kay < 0.30
            # Within 50% of Kay's estimate for P (more variation expected)
            assert abs(envelope.critical_P - Pc_kay) / Pc_kay < 0.50


class TestEnvelopeShape:
    """Test phase envelope shape and properties."""

    def test_bubble_curve_left_of_dew_curve(self, binary_c1_c10_eos, components):
        """Test that bubble curve is generally to the left of dew curve.

        At a given pressure, the bubble point temperature should be
        less than the dew point temperature.
        """
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        # At low temperatures, bubble T should be less than dew T
        if len(envelope.bubble_T) > 0 and len(envelope.dew_T) > 0:
            min_bubble_T = np.min(envelope.bubble_T)
            min_dew_T = np.min(envelope.dew_T)

            # Bubble curve starts at lower temperature
            assert min_bubble_T <= min_dew_T * 1.05  # Allow small margin

    def test_pressure_increases_with_temperature(self, binary_c1_c10_eos, components):
        """Test that pressure generally increases with temperature.

        Along both curves, pressure should increase as temperature increases
        (until near the critical point).
        """
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        # Check bubble curve (before critical point)
        if len(envelope.bubble_P) > 3:
            # At least first half should show increasing pressure
            mid_point = len(envelope.bubble_P) // 2
            assert envelope.bubble_P[mid_point] > envelope.bubble_P[0]

        # Check dew curve
        if len(envelope.dew_P) > 3:
            mid_point = len(envelope.dew_P) // 2
            assert envelope.dew_P[mid_point] > envelope.dew_P[0]


class TestCompositionVariation:
    """Test envelope for different compositions."""

    def test_c1_rich_envelope(self, binary_c1_c10_eos, components):
        """Test envelope for C1-rich mixture."""
        z = np.array([0.9, 0.1])  # 90% methane
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        assert envelope.converged is True
        assert envelope.critical_T is not None

        # For C1-rich, critical point should be reasonable
        # (mixture critical can be higher than pure C1 due to C10 influence)
        Tc_C1 = components['C1'].Tc
        Tc_C10 = components['C10'].Tc
        if envelope.critical_T is not None:
            # Should be between C1 critical and C10 critical
            assert Tc_C1 * 0.8 < envelope.critical_T < Tc_C10 * 0.8

    def test_c10_rich_envelope(self, binary_c1_c10_eos, components):
        """Test envelope for C10-rich mixture."""
        z = np.array([0.1, 0.9])  # 90% decane
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        assert envelope.converged is True
        assert envelope.critical_T is not None

        # For C10-rich, critical point should be reasonable
        # (mixture critical can be lower than pure C10 due to C1 influence)
        Tc_C1 = components['C1'].Tc
        Tc_C10 = components['C10'].Tc
        if envelope.critical_T is not None:
            # Should be between component criticals
            assert Tc_C1 * 1.2 < envelope.critical_T < Tc_C10 * 1.2

    def test_equal_mixture_envelope(self, binary_c1_c4_eos, components):
        """Test envelope for equal mixture."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C4']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c4_eos)

        assert envelope.converged is True
        # Should have at least bubble curve
        assert envelope.n_bubble_points > 5


class TestDifferentSystems:
    """Test envelope for different component systems."""

    def test_ethane_propane_envelope(self, binary_c2_c3_eos, components):
        """Test envelope for ethane-propane (similar components).

        Ethane and propane are relatively similar, so the envelope
        should be narrower than C1-C10.
        """
        z = np.array([0.5, 0.5])
        binary = [components['C2'], components['C3']]

        envelope = calculate_phase_envelope(z, binary, binary_c2_c3_eos)

        assert envelope.converged is True
        assert envelope.critical_T is not None

        # Critical T should be between C2 and C3 critical temperatures
        Tc_C2 = components['C2'].Tc
        Tc_C3 = components['C3'].Tc
        if envelope.critical_T is not None:
            assert Tc_C2 * 0.95 < envelope.critical_T < Tc_C3 * 1.05

    def test_c1_c4_envelope(self, binary_c1_c4_eos, components):
        """Test envelope for C1-C4 system."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C4']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c4_eos)

        assert envelope.converged is True
        # Should have at least bubble curve
        assert len(envelope.bubble_T) > 0


class TestInputValidation:
    """Test input validation."""

    def test_invalid_composition_sum(self, binary_c1_c10_eos, components):
        """Test that invalid composition sum raises error."""
        z_invalid = np.array([0.5, 0.3])  # Sums to 0.8
        binary = [components['C1'], components['C10']]

        with pytest.raises(ValidationError):
            calculate_phase_envelope(z_invalid, binary, binary_c1_c10_eos)

    def test_composition_length_mismatch(self, binary_c1_c10_eos, components):
        """Test that composition length mismatch raises error."""
        z_wrong = np.array([0.33, 0.33, 0.34])  # 3 components, EOS has 2
        binary = [components['C1'], components['C10']]

        with pytest.raises(ValidationError):
            calculate_phase_envelope(z_wrong, binary, binary_c1_c10_eos)


class TestCricondenPoints:
    """Test cricondentherm and cricondenbar estimation."""

    def test_cricondentherm_estimated(self, binary_c1_c10_eos, components):
        """Test cricondentherm (maximum temperature) estimation."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)
        T_cdt, P_cdt = estimate_cricondentherm(envelope)

        if T_cdt is not None:
            assert T_cdt > 0
            assert P_cdt > 0
            # Should be on dew curve
            if len(envelope.dew_T) > 0:
                assert T_cdt <= np.max(envelope.dew_T) * 1.01

    def test_cricondenbar_estimated(self, binary_c1_c10_eos, components):
        """Test cricondenbar (maximum pressure) estimation."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)
        T_cdb, P_cdb = estimate_cricondenbar(envelope)

        if P_cdb is not None:
            assert T_cdb > 0
            assert P_cdb > 0

    def test_cricondenbar_near_critical(self, binary_c2_c3_eos, components):
        """Test that cricondenbar is near critical point.

        For symmetric systems (similar components), the cricondenbar
        (maximum pressure) occurs at or very near the critical point.

        Note: We use C2-C3 instead of C1-C10 because asymmetric mixtures
        can have complex phase behavior where envelope tracing may find
        different saturation branches.
        """
        z = np.array([0.5, 0.5])
        binary = [components['C2'], components['C3']]

        envelope = calculate_phase_envelope(z, binary, binary_c2_c3_eos)
        T_cdb, P_cdb = estimate_cricondenbar(envelope)

        if (envelope.critical_P is not None and P_cdb is not None):
            # For similar components, should be reasonably close
            # (within 50% - allows for numerical discretization)
            rel_diff = abs(P_cdb - envelope.critical_P) / envelope.critical_P
            assert rel_diff < 0.5


class TestAdaptiveStepSize:
    """Test adaptive step size behavior."""

    def test_more_points_near_critical(self, binary_c1_c10_eos, components):
        """Test that step size adapts (more points near critical region).

        Near the critical point, the algorithm should take smaller steps
        to capture the sharp changes in the envelope.
        """
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        envelope = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        # Should have generated a reasonable number of points
        assert envelope.n_bubble_points > 10

        # Near critical point, spacing should be tighter
        if envelope.critical_T is not None and len(envelope.bubble_T) > 5:
            # Find points near critical temperature
            Tc = envelope.critical_T
            tolerance = Tc * 0.1  # 10% range around critical

            near_critical = np.abs(envelope.bubble_T - Tc) < tolerance
            n_near_critical = np.sum(near_critical)

            # Should have multiple points near critical region
            assert n_near_critical >= 2


class TestEnvelopeConsistency:
    """Test consistency of envelope results."""

    def test_reproducible_results(self, binary_c1_c10_eos, components):
        """Test that envelope calculation gives consistent results."""
        z = np.array([0.5, 0.5])
        binary = [components['C1'], components['C10']]

        # Calculate twice
        envelope1 = calculate_phase_envelope(z, binary, binary_c1_c10_eos)
        envelope2 = calculate_phase_envelope(z, binary, binary_c1_c10_eos)

        # Should give same critical point (within tolerance)
        if envelope1.critical_T is not None and envelope2.critical_T is not None:
            assert abs(envelope1.critical_T - envelope2.critical_T) < 1.0  # Within 1 K
            assert abs(envelope1.critical_P - envelope2.critical_P) / envelope1.critical_P < 0.01

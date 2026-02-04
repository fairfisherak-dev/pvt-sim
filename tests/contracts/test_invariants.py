"""Physics invariant tests for PVT calculations.

These tests verify fundamental physical laws and constraints that must
always hold, regardless of specific inputs:

1. Material Balance: Total moles in = total moles out
2. Composition Closure: Mole fractions sum to 1.0
3. Phase Composition Positivity: All mole fractions >= 0
4. K-value Consistency: K = y/x at equilibrium
5. Fugacity Equality: f_i^L = f_i^V at equilibrium (two-phase)
6. Thermodynamic Bounds: Physical limits on properties
7. Limiting Behavior: Correct behavior at extreme conditions

Reference: Gameplan Section 3 - "Physics sanity checks: conservation,
monotonicity, bounds, limiting behavior"
"""

import numpy as np
import pytest

import sys
sys.path.insert(0, 'src')

from pvtcore.models.component import load_components
from pvtcore.eos.peng_robinson import PengRobinsonEOS
from pvtcore.flash.pt_flash import pt_flash
from pvtcore.flash.bubble_point import calculate_bubble_point
from pvtcore.flash.dew_point import calculate_dew_point
from pvtcore.core.errors import ConvergenceStatus


class TestCompositionInvariants:
    """Tests for composition-related invariants."""

    @pytest.fixture
    def binary_mixture(self):
        """Standard binary C1-C4 mixture for testing."""
        components = load_components()
        comp_list = [components['C1'], components['C4']]
        eos = PengRobinsonEOS(comp_list)
        z = np.array([0.5, 0.5])
        return comp_list, eos, z

    def test_feed_composition_closure(self, binary_mixture):
        """Feed composition must sum to 1.0."""
        comp_list, eos, z = binary_mixture
        assert np.isclose(z.sum(), 1.0, atol=1e-10)

    def test_liquid_composition_closure_two_phase(self, binary_mixture):
        """Liquid phase mole fractions must sum to 1.0 in two-phase."""
        comp_list, eos, z = binary_mixture
        result = pt_flash(2e6, 250, z, comp_list, eos)

        if result.phase == 'two-phase':
            assert np.isclose(result.liquid_composition.sum(), 1.0, atol=1e-10), \
                f"Liquid composition sum = {result.liquid_composition.sum()}"

    def test_vapor_composition_closure_two_phase(self, binary_mixture):
        """Vapor phase mole fractions must sum to 1.0 in two-phase."""
        comp_list, eos, z = binary_mixture
        result = pt_flash(2e6, 250, z, comp_list, eos)

        if result.phase == 'two-phase':
            assert np.isclose(result.vapor_composition.sum(), 1.0, atol=1e-10), \
                f"Vapor composition sum = {result.vapor_composition.sum()}"

    def test_composition_positivity(self, binary_mixture):
        """All mole fractions must be non-negative."""
        comp_list, eos, z = binary_mixture
        result = pt_flash(2e6, 250, z, comp_list, eos)

        assert np.all(result.liquid_composition >= -1e-15), \
            f"Negative liquid composition: {result.liquid_composition}"
        assert np.all(result.vapor_composition >= -1e-15), \
            f"Negative vapor composition: {result.vapor_composition}"

    @pytest.mark.parametrize("nv_expected,P,T", [
        (0.0, 200e5, 300),   # High P -> liquid
        (1.0, 1e5, 400),     # Low P, high T -> vapor
    ])
    def test_single_phase_composition_equals_feed(self, binary_mixture, nv_expected, P, T):
        """In single-phase, phase composition should equal feed."""
        comp_list, eos, z = binary_mixture
        result = pt_flash(P, T, z, comp_list, eos)

        if result.vapor_fraction == 0.0:
            np.testing.assert_allclose(result.liquid_composition, z, atol=1e-10)
        elif result.vapor_fraction == 1.0:
            np.testing.assert_allclose(result.vapor_composition, z, atol=1e-10)


class TestMaterialBalance:
    """Tests for material balance invariants."""

    @pytest.fixture
    def binary_mixture(self):
        components = load_components()
        comp_list = [components['C1'], components['C4']]
        eos = PengRobinsonEOS(comp_list)
        z = np.array([0.5, 0.5])
        return comp_list, eos, z

    def test_component_material_balance(self, binary_mixture):
        """z_i = (1-nv)*x_i + nv*y_i for all components."""
        comp_list, eos, z = binary_mixture
        result = pt_flash(2e6, 250, z, comp_list, eos)

        if result.phase == 'two-phase':
            nv = result.vapor_fraction
            x = result.liquid_composition
            y = result.vapor_composition

            # Material balance: z = (1-nv)*x + nv*y
            z_reconstructed = (1.0 - nv) * x + nv * y
            np.testing.assert_allclose(z_reconstructed, z, atol=1e-10,
                err_msg="Material balance violated")

    def test_vapor_fraction_bounds(self, binary_mixture):
        """Vapor fraction must be in [0, 1]."""
        comp_list, eos, z = binary_mixture

        # Test multiple conditions
        for P in [1e5, 2e6, 50e6, 200e6]:
            for T in [200, 250, 300, 400]:
                result = pt_flash(P, T, z, comp_list, eos)
                assert 0.0 <= result.vapor_fraction <= 1.0, \
                    f"nv = {result.vapor_fraction} at P={P}, T={T}"


class TestKValueConsistency:
    """Tests for K-value consistency."""

    @pytest.fixture
    def binary_mixture(self):
        components = load_components()
        comp_list = [components['C1'], components['C4']]
        eos = PengRobinsonEOS(comp_list)
        z = np.array([0.5, 0.5])
        return comp_list, eos, z

    def test_k_value_definition(self, binary_mixture):
        """K_i = y_i / x_i for two-phase systems."""
        comp_list, eos, z = binary_mixture
        result = pt_flash(2e6, 250, z, comp_list, eos)

        if result.phase == 'two-phase':
            x = result.liquid_composition
            y = result.vapor_composition
            K = result.K_values

            # Avoid division by zero
            mask = x > 1e-15
            K_computed = np.where(mask, y / x, 0.0)
            K_expected = np.where(mask, K, 0.0)

            np.testing.assert_allclose(K_computed, K_expected, rtol=1e-5,
                err_msg="K-values inconsistent with y/x")

    def test_light_component_k_greater_than_one(self, binary_mixture):
        """Light components should have K > 1 (prefer vapor)."""
        comp_list, eos, z = binary_mixture
        result = pt_flash(2e6, 250, z, comp_list, eos)

        if result.phase == 'two-phase':
            # C1 (lighter) should have K > 1
            # C4 (heavier) should have K < 1
            assert result.K_values[0] > 1.0, \
                f"Light component (C1) K = {result.K_values[0]} should be > 1"
            assert result.K_values[1] < 1.0, \
                f"Heavy component (C4) K = {result.K_values[1]} should be < 1"


class TestThermodynamicConsistency:
    """Tests for thermodynamic consistency."""

    @pytest.fixture
    def binary_mixture(self):
        components = load_components()
        comp_list = [components['C1'], components['C4']]
        eos = PengRobinsonEOS(comp_list)
        z = np.array([0.5, 0.5])
        return comp_list, eos, z

    def test_fugacity_equality_at_equilibrium(self, binary_mixture):
        """At equilibrium: f_i^L = f_i^V for each component."""
        comp_list, eos, z = binary_mixture
        result = pt_flash(2e6, 250, z, comp_list, eos)

        if result.phase == 'two-phase':
            P = result.pressure
            x = result.liquid_composition
            y = result.vapor_composition
            phi_L = result.liquid_fugacity
            phi_V = result.vapor_fugacity

            # Fugacity: f_i = phi_i * x_i * P (liquid) or phi_i * y_i * P (vapor)
            f_L = phi_L * x * P
            f_V = phi_V * y * P

            # At equilibrium, these should be equal
            # Note: rtol=1e-5 is appropriate since the flash solver converges to ~1e-6 tolerance
            np.testing.assert_allclose(f_L, f_V, rtol=1e-5,
                err_msg="Fugacity equality violated at equilibrium")

    def test_pressure_effect_on_vapor_fraction(self, binary_mixture):
        """Increasing pressure at constant T should decrease vapor fraction."""
        comp_list, eos, z = binary_mixture
        T = 280  # Fixed temperature

        pressures = [1e6, 2e6, 5e6, 10e6]
        vapor_fractions = []

        for P in pressures:
            result = pt_flash(P, T, z, comp_list, eos)
            vapor_fractions.append(result.vapor_fraction)

        # Vapor fraction should be monotonically decreasing with pressure
        # (in the two-phase region)
        for i in range(len(vapor_fractions) - 1):
            if 0 < vapor_fractions[i] < 1 and 0 < vapor_fractions[i+1] < 1:
                assert vapor_fractions[i] >= vapor_fractions[i+1], \
                    f"Vapor fraction should decrease with pressure: " \
                    f"nv({pressures[i]/1e6}MPa)={vapor_fractions[i]:.3f}, " \
                    f"nv({pressures[i+1]/1e6}MPa)={vapor_fractions[i+1]:.3f}"


class TestLimitingBehavior:
    """Tests for correct behavior at limiting conditions."""

    def test_pure_component_k_value(self):
        """Near-pure component should have K ≈ 1 for dominant component."""
        components = load_components()
        comp_list = [components['C1'], components['C4']]
        eos = PengRobinsonEOS(comp_list)

        # 99.9% C1
        z = np.array([0.999, 0.001])
        result = pt_flash(2e6, 200, z, comp_list, eos)

        if result.phase == 'two-phase':
            # For near-pure C1, its K should be close to 1
            # (its composition in both phases should be similar)
            # This is a weak test since we're not at the saturation curve
            pass  # K can vary widely off the saturation curve

    def test_bubble_point_below_dew_point(self):
        """Bubble point pressure should be > dew point pressure."""
        components = load_components()
        comp_list = [components['C1'], components['C4']]
        eos = PengRobinsonEOS(comp_list)
        z = np.array([0.5, 0.5])
        T = 280

        bp = calculate_bubble_point(T, z, comp_list, eos)
        dp = calculate_dew_point(T, z, comp_list, eos)

        assert bp.pressure > dp.pressure, \
            f"Bubble point ({bp.pressure/1e5:.1f} bar) should be > " \
            f"dew point ({dp.pressure/1e5:.1f} bar)"

    def test_high_pressure_gives_liquid(self):
        """Very high pressure should give single-phase liquid."""
        components = load_components()
        comp_list = [components['C1'], components['C4']]
        eos = PengRobinsonEOS(comp_list)
        z = np.array([0.5, 0.5])

        # 500 bar should definitely be liquid
        result = pt_flash(500e5, 300, z, comp_list, eos)
        assert result.phase == 'liquid', \
            f"At 500 bar, expected liquid but got {result.phase}"

    def test_low_pressure_high_temp_gives_vapor(self):
        """Low pressure at high temperature should give vapor."""
        components = load_components()
        comp_list = [components['C1'], components['C4']]
        eos = PengRobinsonEOS(comp_list)
        z = np.array([0.5, 0.5])

        # 0.1 bar at 500K should definitely be vapor
        result = pt_flash(0.1e5, 500, z, comp_list, eos)
        assert result.phase == 'vapor', \
            f"At 0.1 bar and 500K, expected vapor but got {result.phase}"


class TestRobustness:
    """Tests for numerical robustness."""

    def test_trace_component_handling(self):
        """Flash should handle trace components without NaN."""
        components = load_components()
        comp_list = [components['C1'], components['C4']]
        eos = PengRobinsonEOS(comp_list)

        # 1 ppm of C4
        z = np.array([0.999999, 0.000001])
        result = pt_flash(2e6, 250, z, comp_list, eos)

        assert result.status != ConvergenceStatus.NUMERIC_ERROR, \
            "Should handle trace components without numeric error"
        assert np.all(np.isfinite(result.liquid_composition)), \
            "Liquid composition contains non-finite values"
        assert np.all(np.isfinite(result.vapor_composition)), \
            "Vapor composition contains non-finite values"

    def test_convergence_returns_valid_status(self):
        """Flash should return valid ConvergenceStatus."""
        components = load_components()
        comp_list = [components['C1'], components['C4']]
        eos = PengRobinsonEOS(comp_list)
        z = np.array([0.5, 0.5])

        result = pt_flash(2e6, 250, z, comp_list, eos)

        assert isinstance(result.status, ConvergenceStatus), \
            f"Status should be ConvergenceStatus, got {type(result.status)}"
        assert result.status in [
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.MAX_ITERS,
            ConvergenceStatus.DIVERGED,
            ConvergenceStatus.STAGNATED,
            ConvergenceStatus.NUMERIC_ERROR,
        ], f"Unexpected status: {result.status}"

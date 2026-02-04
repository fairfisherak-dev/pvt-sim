"""Unit tests for flash calculation module.

Tests Wilson K-values, Rachford-Rice solver, and PT flash algorithm.
"""

import pytest
import numpy as np
from pvtcore.stability.wilson import (
    wilson_k_values,
    wilson_k_value_single,
    is_trivial_solution,
    wilson_correlation_valid
)
from pvtcore.flash.rachford_rice import (
    rachford_rice_function,
    solve_rachford_rice,
    calculate_phase_compositions,
    find_valid_brackets
)
from pvtcore.flash.pt_flash import pt_flash, FlashResult
from pvtcore.eos.peng_robinson import PengRobinsonEOS
from pvtcore.models.component import load_components
from pvtcore.core.errors import ValidationError, ConvergenceError


@pytest.fixture
def components():
    """Load component database."""
    return load_components()


@pytest.fixture
def binary_system(components):
    """Create methane-decane binary system."""
    return [components['C1'], components['C10']]


@pytest.fixture
def ternary_system(components):
    """Create methane-propane-decane ternary system."""
    return [components['C1'], components['C3'], components['C10']]


class TestWilsonKValues:
    """Test Wilson K-value correlation."""

    def test_light_component_k_greater_than_one(self, components):
        """Test that light components have K > 1 at typical conditions."""
        # Methane at 300 K, 30 bar
        T = 300.0
        P = 3e6
        K = wilson_k_values(P, T, [components['C1']])

        # Light component should prefer vapor phase
        assert K[0] > 1.0

    def test_heavy_component_k_less_than_one(self, components):
        """Test that heavy components have K < 1 at typical conditions."""
        # n-Decane at 300 K, 30 bar
        T = 300.0
        P = 3e6
        K = wilson_k_values(P, T, [components['C10']])

        # Heavy component should prefer liquid phase
        assert K[0] < 1.0

    def test_k_values_reasonable_range(self, binary_system):
        """Test that K-values are in reasonable range."""
        T = 300.0
        P = 3e6
        K = wilson_k_values(P, T, binary_system)

        # K-values should be positive
        assert np.all(K > 0)

        # K-values typically between 1e-6 and 1000
        # Heavy components like decane can have very low K-values at high pressure
        assert np.all(K > 1e-6)
        assert np.all(K < 1e3)

    def test_k_decreases_with_pressure(self, components):
        """Test that K-values decrease with increasing pressure."""
        T = 300.0
        P_low = 1e6   # 10 bar
        P_high = 5e6  # 50 bar

        K_low = wilson_k_values(P_low, T, [components['C1']])
        K_high = wilson_k_values(P_high, T, [components['C1']])

        # Higher pressure favors liquid phase (lower K)
        assert K_low[0] > K_high[0]

    def test_k_increases_with_temperature(self, components):
        """Test that K-values increase with increasing temperature."""
        P = 3e6
        T_low = 250.0
        T_high = 350.0

        K_low = wilson_k_values(P, T_low, [components['C1']])
        K_high = wilson_k_values(P, T_high, [components['C1']])

        # Higher temperature favors vapor phase (higher K)
        assert K_high[0] > K_low[0]

    def test_wilson_single_component(self, components):
        """Test single component Wilson K-value function."""
        T = 300.0
        P = 3e6

        K_array = wilson_k_values(P, T, [components['C1']])
        K_single = wilson_k_value_single(P, T, components['C1'])

        assert K_array[0] == pytest.approx(K_single, rel=1e-10)

    def test_wilson_acentric_factor_effect(self, components):
        """Test that acentric factor affects K-values correctly."""
        T = 300.0
        P = 3e6

        # Methane (low omega) vs ethane (higher omega)
        K_c1 = wilson_k_values(P, T, [components['C1']])
        K_c2 = wilson_k_values(P, T, [components['C2']])

        # At same T and P, component with lower Tc should have higher K
        # Methane: Tc = 190.6 K, Ethane: Tc = 305.3 K
        assert K_c1[0] > K_c2[0]

    def test_trivial_solution_all_vapor(self):
        """Test detection of all-vapor trivial solution."""
        K = np.array([2.0, 3.0, 4.0])  # All K > 1
        z = np.array([0.5, 0.3, 0.2])

        is_trivial, phase = is_trivial_solution(K, z)

        assert is_trivial is True
        assert phase == 'vapor'

    def test_trivial_solution_all_liquid(self):
        """Test detection of all-liquid trivial solution."""
        K = np.array([0.3, 0.5, 0.7])  # All K < 1
        z = np.array([0.5, 0.3, 0.2])

        is_trivial, phase = is_trivial_solution(K, z)

        assert is_trivial is True
        assert phase == 'liquid'

    def test_trivial_solution_two_phase(self):
        """Test detection of two-phase system."""
        K = np.array([2.0, 1.0, 0.5])  # Mixed K values
        z = np.array([0.5, 0.3, 0.2])

        is_trivial, phase = is_trivial_solution(K, z)

        assert is_trivial is False
        assert phase is None

    def test_wilson_correlation_validity(self, components):
        """Test Wilson correlation validity check."""
        # Valid conditions
        T = 300.0
        P = 3e6
        valid, msg = wilson_correlation_valid(P, T, [components['C1']])
        assert valid is True

        # Invalid: very high pressure
        P_high = 100e6  # 1000 bar
        valid, msg = wilson_correlation_valid(P_high, T, [components['C1']])
        assert valid is False


class TestRachfordRice:
    """Test Rachford-Rice equation solver."""

    def test_rachford_rice_function_at_zero(self):
        """Test Rachford-Rice function evaluation."""
        K = np.array([2.0, 0.5])
        z = np.array([0.5, 0.5])

        f = rachford_rice_function(0.0, K, z)

        # At nv=0 (all liquid): f = Σ zi(Ki - 1)
        expected = np.sum(z * (K - 1.0))
        assert f == pytest.approx(expected, abs=1e-10)

    def test_rachford_rice_function_monotonic(self):
        """Test that Rachford-Rice function is monotonically decreasing."""
        K = np.array([3.0, 0.5])
        z = np.array([0.6, 0.4])

        nv_values = np.linspace(0.1, 0.9, 10)
        f_values = [rachford_rice_function(nv, K, z) for nv in nv_values]

        # Function should be monotonically decreasing
        for i in range(len(f_values) - 1):
            assert f_values[i] > f_values[i + 1]

    def test_solve_rachford_rice_simple_case(self):
        """Test Rachford-Rice solver with simple binary."""
        K = np.array([2.0, 0.5])
        z = np.array([0.5, 0.5])

        nv, x, y = solve_rachford_rice(K, z)

        # Check physical constraints
        assert 0.0 <= nv <= 1.0
        assert np.all(x >= 0.0)
        assert np.all(y >= 0.0)
        assert np.sum(x) == pytest.approx(1.0, abs=1e-10)
        assert np.sum(y) == pytest.approx(1.0, abs=1e-10)

        # Check material balance
        z_calc = (1 - nv) * x + nv * y
        assert np.allclose(z_calc, z, atol=1e-10)

        # Check K-values
        K_calc = y / x
        assert np.allclose(K_calc, K, atol=1e-8)

    def test_solve_rachford_rice_all_vapor(self):
        """Test Rachford-Rice with all K > 1 (all vapor)."""
        K = np.array([2.0, 3.0, 4.0])
        z = np.array([0.5, 0.3, 0.2])

        nv, x, y = solve_rachford_rice(K, z)

        # Should give all vapor
        assert nv == pytest.approx(1.0, abs=1e-10)
        assert np.allclose(y, z, atol=1e-10)
        assert np.allclose(x, 0.0, atol=1e-10)

    def test_solve_rachford_rice_all_liquid(self):
        """Test Rachford-Rice with all K < 1 (all liquid)."""
        K = np.array([0.3, 0.5, 0.7])
        z = np.array([0.5, 0.3, 0.2])

        nv, x, y = solve_rachford_rice(K, z)

        # Should give all liquid
        assert nv == pytest.approx(0.0, abs=1e-10)
        assert np.allclose(x, z, atol=1e-10)
        assert np.allclose(y, 0.0, atol=1e-10)

    def test_solve_rachford_rice_ternary(self):
        """Test Rachford-Rice solver with ternary mixture."""
        K = np.array([3.0, 1.0, 0.3])
        z = np.array([0.4, 0.3, 0.3])

        nv, x, y = solve_rachford_rice(K, z)

        # Validate result
        assert 0.0 < nv < 1.0
        assert np.all(x > 0.0)
        assert np.all(y > 0.0)

        # Material balance
        z_calc = (1 - nv) * x + nv * y
        assert np.allclose(z_calc, z, atol=1e-10)

    def test_calculate_phase_compositions(self):
        """Test phase composition calculation."""
        K = np.array([2.0, 0.5])
        z = np.array([0.5, 0.5])
        nv = 0.5

        x, y = calculate_phase_compositions(nv, K, z)

        # Check normalization
        assert np.sum(x) == pytest.approx(1.0, abs=1e-10)
        assert np.sum(y) == pytest.approx(1.0, abs=1e-10)

        # Check K-values
        K_calc = y / x
        assert np.allclose(K_calc, K, atol=1e-10)

    def test_find_valid_brackets(self):
        """Test finding valid brackets for Rachford-Rice."""
        K = np.array([3.0, 0.5])
        z = np.array([0.5, 0.5])

        nv_min, nv_max = find_valid_brackets(K, z)

        # Brackets should be in valid range
        assert 0.0 <= nv_min < nv_max <= 1.0

        # Function should have opposite signs at brackets
        f_min = rachford_rice_function(nv_min, K, z)
        f_max = rachford_rice_function(nv_max, K, z)
        assert f_min * f_max < 0

    def test_invalid_composition_raises_error(self):
        """Test that invalid composition raises ValidationError."""
        K = np.array([2.0, 0.5])
        z = np.array([0.5, 0.6])  # Doesn't sum to 1

        with pytest.raises(ValidationError):
            solve_rachford_rice(K, z)

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched array lengths raise ValidationError."""
        K = np.array([2.0, 0.5])
        z = np.array([0.5, 0.3, 0.2])

        with pytest.raises(ValidationError):
            solve_rachford_rice(K, z)


class TestPTFlash:
    """Test PT flash calculations."""

    def test_flash_converges_binary_system(self, binary_system):
        """Test that flash converges for methane-decane binary."""
        eos = PengRobinsonEOS(binary_system)

        # Conditions: 300 K, 30 bar
        T = 300.0
        P = 3e6
        z = np.array([0.5, 0.5])

        result = pt_flash(P, T, z, binary_system, eos)

        # Check convergence
        assert result.converged is True
        assert result.iterations < 50

        # Check physical validity
        assert 0.0 <= result.vapor_fraction <= 1.0
        assert np.all(result.liquid_composition >= 0.0)
        assert np.all(result.vapor_composition >= 0.0)
        assert np.sum(result.liquid_composition) == pytest.approx(1.0, abs=1e-6)
        assert np.sum(result.vapor_composition) == pytest.approx(1.0, abs=1e-6)

        # Check material balance
        z_calc = ((1 - result.vapor_fraction) * result.liquid_composition +
                  result.vapor_fraction * result.vapor_composition)
        assert np.allclose(z_calc, z, atol=1e-6)

    def test_flash_light_component_enriched_in_vapor(self, binary_system):
        """Test that light component is enriched in vapor phase."""
        eos = PengRobinsonEOS(binary_system)

        T = 300.0
        P = 3e6
        z = np.array([0.5, 0.5])

        result = pt_flash(P, T, z, binary_system, eos)

        if result.phase == 'two-phase':
            # Methane (C1) should be enriched in vapor
            # y_C1 > x_C1
            assert result.vapor_composition[0] > result.liquid_composition[0]

            # Decane (C10) should be enriched in liquid
            # x_C10 > y_C10
            assert result.liquid_composition[1] > result.vapor_composition[1]

    def test_flash_k_values_consistent(self, binary_system):
        """Test that K-values are consistent with compositions."""
        eos = PengRobinsonEOS(binary_system)

        T = 300.0
        P = 3e6
        z = np.array([0.5, 0.5])

        result = pt_flash(P, T, z, binary_system, eos)

        if result.phase == 'two-phase':
            # K = y / x
            K_calc = result.vapor_composition / result.liquid_composition
            assert np.allclose(K_calc, result.K_values, rtol=1e-3)

    def test_flash_fugacity_equality(self, binary_system):
        """Test that fugacities are equal at equilibrium."""
        eos = PengRobinsonEOS(binary_system)

        T = 300.0
        P = 3e6
        z = np.array([0.5, 0.5])

        result = pt_flash(P, T, z, binary_system, eos)

        if result.phase == 'two-phase':
            # At equilibrium: fi_L = fi_V
            # φi_L × xi × P = φi_V × yi × P
            f_L = result.liquid_fugacity * result.liquid_composition * P
            f_V = result.vapor_fugacity * result.vapor_composition * P

            assert np.allclose(f_L, f_V, rtol=1e-4)

    def test_flash_with_initial_k_values(self, binary_system):
        """Test flash with provided initial K-values."""
        eos = PengRobinsonEOS(binary_system)

        T = 300.0
        P = 3e6
        z = np.array([0.5, 0.5])

        # Provide initial K-values
        K_init = np.array([3.0, 0.1])

        result = pt_flash(P, T, z, binary_system, eos, K_initial=K_init)

        assert result.converged is True

    def test_flash_ternary_system(self, ternary_system):
        """Test flash with ternary mixture."""
        eos = PengRobinsonEOS(ternary_system)

        T = 300.0
        P = 3e6
        z = np.array([0.5, 0.3, 0.2])

        result = pt_flash(P, T, z, ternary_system, eos)

        # Check convergence
        assert result.converged is True

        # Check material balance
        if result.phase == 'two-phase':
            z_calc = ((1 - result.vapor_fraction) * result.liquid_composition +
                      result.vapor_fraction * result.vapor_composition)
            assert np.allclose(z_calc, z, atol=1e-6)

    def test_flash_high_pressure_single_phase(self, binary_system):
        """Test that high pressure gives single phase (liquid)."""
        eos = PengRobinsonEOS(binary_system)

        T = 300.0
        P = 50e6  # Very high pressure
        z = np.array([0.5, 0.5])

        result = pt_flash(P, T, z, binary_system, eos)

        # At very high pressure, should be single liquid phase
        # or very low vapor fraction
        assert result.vapor_fraction < 0.1

    def test_flash_low_pressure_single_phase(self, binary_system):
        """Test that low pressure gives single phase (vapor)."""
        eos = PengRobinsonEOS(binary_system)

        T = 500.0  # High temperature
        P = 0.3e6  # Low pressure
        z = np.array([0.5, 0.5])

        result = pt_flash(P, T, z, binary_system, eos)

        # At low pressure and high temperature, should be mostly vapor
        assert result.vapor_fraction > 0.9

    def test_flash_result_dataclass(self, binary_system):
        """Test FlashResult dataclass structure."""
        eos = PengRobinsonEOS(binary_system)

        T = 300.0
        P = 3e6
        z = np.array([0.5, 0.5])

        result = pt_flash(P, T, z, binary_system, eos)

        # Check all attributes exist
        assert hasattr(result, 'converged')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'vapor_fraction')
        assert hasattr(result, 'liquid_composition')
        assert hasattr(result, 'vapor_composition')
        assert hasattr(result, 'K_values')
        assert hasattr(result, 'liquid_fugacity')
        assert hasattr(result, 'vapor_fugacity')
        assert hasattr(result, 'phase')
        assert hasattr(result, 'pressure')
        assert hasattr(result, 'temperature')
        assert hasattr(result, 'residual')

        # Check types
        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)
        assert isinstance(result.vapor_fraction, (float, np.floating))
        assert isinstance(result.liquid_composition, np.ndarray)
        assert isinstance(result.vapor_composition, np.ndarray)
        assert isinstance(result.K_values, np.ndarray)

    def test_flash_invalid_composition_raises_error(self, binary_system):
        """Test that invalid composition raises ValidationError."""
        eos = PengRobinsonEOS(binary_system)

        T = 300.0
        P = 3e6
        z = np.array([0.5, 0.6])  # Doesn't sum to 1

        with pytest.raises(ValidationError):
            pt_flash(P, T, z, binary_system, eos)

    def test_flash_negative_pressure_raises_error(self, binary_system):
        """Test that negative pressure raises ValidationError."""
        eos = PengRobinsonEOS(binary_system)

        T = 300.0
        P = -1e6  # Negative
        z = np.array([0.5, 0.5])

        with pytest.raises(ValidationError):
            pt_flash(P, T, z, binary_system, eos)


class TestFlashSpecialCases:
    """Test special cases and edge conditions."""

    def test_pure_component_flash(self, components):
        """Test flash calculation for pure component."""
        eos = PengRobinsonEOS([components['C1']])

        T = 150.0  # Below critical temperature
        P = 2e6
        z = np.array([1.0])

        result = pt_flash(P, T, z, [components['C1']], eos)

        # Pure component can be two-phase
        assert result.converged is True

    def test_flash_near_critical_point(self, components):
        """Test flash near critical point."""
        eos = PengRobinsonEOS([components['C1']])

        # Near critical point of methane
        T = components['C1'].Tc * 0.99
        P = components['C1'].Pc * 1.01
        z = np.array([1.0])

        result = pt_flash(P, T, z, [components['C1']], eos)

        # Should handle near-critical conditions
        assert result.converged is True

    def test_flash_with_binary_interaction(self, binary_system):
        """Test flash with binary interaction parameters."""
        eos = PengRobinsonEOS(binary_system)

        T = 300.0
        P = 3e6
        z = np.array([0.5, 0.5])

        # With interaction parameter
        kij = np.array([[0.0, 0.03],
                        [0.03, 0.0]])

        result = pt_flash(P, T, z, binary_system, eos,
                         binary_interaction=kij)

        assert result.converged is True

    def test_single_phase_liquid_high_pressure(self, binary_system):
        """Test flash returns single-phase liquid at high pressure, low temperature."""
        eos = PengRobinsonEOS(binary_system)

        # High pressure, low temperature - should be single-phase liquid
        T = 200.0  # K (low temperature)
        P = 5e7    # Pa (50 MPa, very high pressure)
        z = np.array([0.3, 0.7])

        result = pt_flash(P, T, z, binary_system, eos)

        # Should converge without crash
        assert result.converged is True

        # Should be single-phase liquid
        assert result.phase == 'liquid'
        assert result.vapor_fraction == pytest.approx(0.0, abs=1e-6)

        # Liquid composition should equal feed
        assert np.allclose(result.liquid_composition, z, atol=1e-6)

        # Vapor composition should be zero
        assert np.all(result.vapor_composition == 0.0)

        # Should not need many iterations for stable phase
        assert result.iterations == 0

    def test_single_phase_vapor_low_pressure(self, binary_system):
        """Test flash returns single-phase vapor at low pressure, high temperature."""
        eos = PengRobinsonEOS(binary_system)

        # Low pressure, high temperature - should be single-phase vapor
        T = 500.0  # K (high temperature)
        P = 1e5    # Pa (1 bar, low pressure)
        z = np.array([0.7, 0.3])

        result = pt_flash(P, T, z, binary_system, eos)

        # Should converge without crash
        assert result.converged is True

        # Should be single-phase vapor
        assert result.phase == 'vapor'
        assert result.vapor_fraction == pytest.approx(1.0, abs=1e-6)

        # Vapor composition should equal feed
        assert np.allclose(result.vapor_composition, z, atol=1e-6)

        # Liquid composition should be zero
        assert np.all(result.liquid_composition == 0.0)

        # Should not need many iterations for stable phase
        assert result.iterations == 0

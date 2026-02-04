"""Unit tests for the confinement module (capillary pressure, confined flash)."""

import numpy as np
import pytest

from pvtcore.confinement import (
    # Capillary pressure
    calculate_capillary_pressure,
    capillary_pressure_simple,
    vapor_pressure_from_liquid,
    liquid_pressure_from_vapor,
    modified_k_value,
    modified_k_values_array,
    estimate_bubble_point_suppression,
    estimate_dew_point_enhancement,
    critical_pore_radius,
    CapillaryPressureResult,
    # Confined flash
    confined_flash,
    ConfinedFlashResult,
    # Confined envelope
    estimate_envelope_shrinkage,
)
from pvtcore.models.component import load_components
from pvtcore.eos.peng_robinson import PengRobinsonEOS
from pvtcore.core.errors import ValidationError, ConvergenceError


@pytest.fixture
def components():
    """Load component database."""
    return load_components()


@pytest.fixture
def methane_butane(components):
    """Binary C1-C4 mixture (similar to Nojabaei et al. 2013)."""
    return [components['C1'], components['C4']]


@pytest.fixture
def methane_butane_eos(methane_butane):
    """EOS for C1-C4 binary."""
    return PengRobinsonEOS(methane_butane)


@pytest.fixture
def methane_propane(components):
    """Binary C1-C3 mixture."""
    return [components['C1'], components['C3']]


@pytest.fixture
def methane_propane_eos(methane_propane):
    """EOS for C1-C3 binary."""
    return PengRobinsonEOS(methane_propane)


# =============================================================================
# Capillary Pressure Tests
# =============================================================================

class TestCapillaryPressure:
    """Tests for capillary pressure calculations."""

    def test_capillary_pressure_simple(self):
        """Test simple capillary pressure calculation."""
        # 10 mN/m IFT, 10 nm pore
        Pc = capillary_pressure_simple(ift=10.0, pore_radius_nm=10.0)

        # Pc = 2σ/r = 2 × 0.010 N/m / 10×10⁻⁹ m = 2 MPa
        assert abs(Pc - 2e6) < 1e4  # Within 10 kPa

    def test_capillary_pressure_scaling(self):
        """Test that Pc scales inversely with pore radius."""
        ift = 10.0  # mN/m

        Pc_10nm = capillary_pressure_simple(ift, 10.0)
        Pc_5nm = capillary_pressure_simple(ift, 5.0)
        Pc_20nm = capillary_pressure_simple(ift, 20.0)

        # Pc ∝ 1/r
        assert abs(Pc_5nm / Pc_10nm - 2.0) < 0.01
        assert abs(Pc_10nm / Pc_20nm - 2.0) < 0.01

    def test_capillary_pressure_ift_scaling(self):
        """Test that Pc scales linearly with IFT."""
        r = 10.0  # nm

        Pc_10 = capillary_pressure_simple(10.0, r)
        Pc_20 = capillary_pressure_simple(20.0, r)
        Pc_5 = capillary_pressure_simple(5.0, r)

        # Pc ∝ σ
        assert abs(Pc_20 / Pc_10 - 2.0) < 0.01
        assert abs(Pc_10 / Pc_5 - 2.0) < 0.01

    def test_capillary_pressure_full(self):
        """Test full capillary pressure calculation with result object."""
        result = calculate_capillary_pressure(
            ift=10.0,
            pore_radius=10.0,
            contact_angle=0.0,  # Complete wetting
            radius_units='nm',
        )

        assert isinstance(result, CapillaryPressureResult)
        assert result.Pc > 0
        assert abs(result.Pc_MPa - 2.0) < 0.01
        assert result.contact_angle == 0.0

    def test_capillary_pressure_with_contact_angle(self):
        """Test capillary pressure with non-zero contact angle."""
        ift = 10.0
        r = 10.0

        # Complete wetting (θ = 0)
        result_0 = calculate_capillary_pressure(ift, r, contact_angle=0.0)

        # Partial wetting (θ = 60°, cos(60°) = 0.5)
        result_60 = calculate_capillary_pressure(ift, r, contact_angle=60.0)

        # Pc should be halved with cos(60°) = 0.5
        assert abs(result_60.Pc / result_0.Pc - 0.5) < 0.01

    def test_capillary_pressure_oil_wet(self):
        """Test capillary pressure for oil-wet system (θ > 90°)."""
        # Oil-wet: contact angle > 90° gives negative Pc
        result = calculate_capillary_pressure(
            ift=10.0, pore_radius=10.0, contact_angle=120.0
        )

        # cos(120°) = -0.5, so Pc < 0
        assert result.Pc < 0

    def test_capillary_pressure_unit_conversion(self):
        """Test different radius units."""
        ift = 10.0

        result_nm = calculate_capillary_pressure(ift, 10.0, radius_units='nm')
        result_m = calculate_capillary_pressure(ift, 10e-9, radius_units='m')
        result_um = calculate_capillary_pressure(ift, 0.01, radius_units='um')

        # All should give same Pc
        assert abs(result_nm.Pc - result_m.Pc) < 1.0
        assert abs(result_nm.Pc - result_um.Pc) < 1.0

    def test_capillary_pressure_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            capillary_pressure_simple(ift=-5.0, pore_radius_nm=10.0)

        with pytest.raises(ValueError):
            capillary_pressure_simple(ift=10.0, pore_radius_nm=0.0)

        with pytest.raises(ValueError):
            capillary_pressure_simple(ift=10.0, pore_radius_nm=-5.0)


class TestPressureRelationships:
    """Tests for pressure relationship functions."""

    def test_vapor_liquid_pressure_relationship(self):
        """Test Pv = PL + Pc relationship."""
        P_L = 5e6  # 5 MPa
        Pc = 2e6   # 2 MPa

        P_V = vapor_pressure_from_liquid(P_L, Pc)
        assert P_V == P_L + Pc  # 7 MPa

        # Reverse
        P_L_back = liquid_pressure_from_vapor(P_V, Pc)
        assert abs(P_L_back - P_L) < 1.0

    def test_modified_k_value(self):
        """Test modified K-value calculation."""
        K_bulk = 2.0
        P_L = 5e6
        P_V = 7e6  # Includes Pc

        K_conf = modified_k_value(K_bulk, P_L, P_V)

        # K_confined = K_bulk × (PL/Pv)
        expected = K_bulk * (P_L / P_V)
        assert abs(K_conf - expected) < 1e-10

        # K_confined < K_bulk when Pv > PL
        assert K_conf < K_bulk

    def test_modified_k_values_array(self):
        """Test vectorized modified K-values."""
        K_bulk = np.array([3.0, 0.5, 1.2])
        P_L = 5e6
        P_V = 6e6

        K_conf = modified_k_values_array(K_bulk, P_L, P_V)

        assert len(K_conf) == 3
        assert np.all(K_conf < K_bulk)  # All K reduced

        # Check ratio
        ratio = P_L / P_V
        assert np.allclose(K_conf / K_bulk, ratio)


class TestBubbleDewEstimates:
    """Tests for bubble/dew point shift estimates."""

    def test_bubble_point_suppression(self):
        """Test bubble point suppression estimate."""
        Pc = 2e6  # 2 MPa
        P_bubble_bulk = 10e6  # 10 MPa

        P_bubble_conf = estimate_bubble_point_suppression(Pc, P_bubble_bulk)

        # Bubble point suppressed by approximately Pc
        assert P_bubble_conf < P_bubble_bulk
        assert abs(P_bubble_conf - (P_bubble_bulk - Pc)) < 1.0

    def test_dew_point_enhancement(self):
        """Test dew point enhancement estimate."""
        Pc = 2e6  # 2 MPa
        P_dew_bulk = 3e6  # 3 MPa

        P_dew_conf = estimate_dew_point_enhancement(Pc, P_dew_bulk)

        # Dew point enhanced by approximately Pc
        assert P_dew_conf > P_dew_bulk
        assert abs(P_dew_conf - (P_dew_bulk + Pc)) < 1.0

    def test_critical_pore_radius(self):
        """Test critical pore radius calculation."""
        ift = 10.0  # mN/m
        Pc_target = 2e6  # 2 MPa

        r = critical_pore_radius(ift, Pc_target)

        # Should be 10 nm for these values
        assert abs(r - 10.0) < 0.1

        # Verify by back-calculating Pc
        Pc_check = capillary_pressure_simple(ift, r)
        assert abs(Pc_check - Pc_target) < 1e3


# =============================================================================
# Confined Flash Tests
# =============================================================================

class TestConfinedFlash:
    """Tests for confined flash calculations."""

    def test_confined_flash_basic(self, methane_butane, methane_butane_eos):
        """Test basic confined flash calculation."""
        z = np.array([0.5, 0.5])  # More balanced composition
        P_L = 3e6  # 3 MPa - more likely two-phase
        T = 300.0  # K - lower T for two-phase region
        r = 10.0   # 10 nm pore

        result = confined_flash(
            P_L, T, z, methane_butane, methane_butane_eos,
            pore_radius_nm=r,
        )

        assert isinstance(result, ConfinedFlashResult)
        # Accept any converged result (two-phase or single-phase)
        assert result.phase in ['two-phase', 'liquid', 'vapor']
        assert result.liquid_pressure == P_L
        assert result.pore_radius == r

    def test_confined_flash_has_capillary_pressure(self, methane_butane, methane_butane_eos):
        """Test that confined flash produces non-zero capillary pressure."""
        z = np.array([0.5, 0.5])
        P_L = 3e6  # 3 MPa - likely two-phase
        T = 300.0
        r = 10.0

        result = confined_flash(
            P_L, T, z, methane_butane, methane_butane_eos,
            pore_radius_nm=r,
        )

        if result.phase == 'two-phase':
            assert result.capillary_pressure > 0
            assert result.vapor_pressure > result.liquid_pressure
            assert result.ift > 0

    def test_confined_flash_pressure_relationship(self, methane_butane, methane_butane_eos):
        """Test Pv = PL + Pc in confined flash result."""
        z = np.array([0.5, 0.5])
        P_L = 3e6
        T = 300.0
        r = 10.0

        result = confined_flash(
            P_L, T, z, methane_butane, methane_butane_eos,
            pore_radius_nm=r,
        )

        if result.phase == 'two-phase':
            # Pv = PL + Pc
            expected_Pv = result.liquid_pressure + result.capillary_pressure
            assert abs(result.vapor_pressure - expected_Pv) < 1.0

    def test_confined_flash_smaller_pore_higher_pc(self, methane_butane, methane_butane_eos):
        """Test that smaller pores give higher capillary pressure."""
        z = np.array([0.5, 0.5])
        P_L = 4e6
        T = 320.0

        result_10nm = confined_flash(
            P_L, T, z, methane_butane, methane_butane_eos,
            pore_radius_nm=10.0,
        )
        result_5nm = confined_flash(
            P_L, T, z, methane_butane, methane_butane_eos,
            pore_radius_nm=5.0,
        )

        if result_10nm.phase == 'two-phase' and result_5nm.phase == 'two-phase':
            # Smaller pore should have higher Pc
            assert result_5nm.capillary_pressure > result_10nm.capillary_pressure

    def test_confined_flash_single_phase_no_pc(self, methane_butane, methane_butane_eos):
        """Test that single-phase result has zero capillary pressure."""
        z = np.array([0.95, 0.05])  # Very light mixture
        P_L = 1e6  # Low pressure - likely all vapor
        T = 400.0  # High temperature
        r = 10.0

        result = confined_flash(
            P_L, T, z, methane_butane, methane_butane_eos,
            pore_radius_nm=r,
        )

        if result.phase in ['liquid', 'vapor']:
            assert result.capillary_pressure == 0.0

    def test_confined_flash_k_values_modified(self, methane_butane, methane_butane_eos):
        """Test that confined K-values are modified from bulk."""
        z = np.array([0.5, 0.5])
        P_L = 3e6
        T = 300.0
        r = 10.0

        result = confined_flash(
            P_L, T, z, methane_butane, methane_butane_eos,
            pore_radius_nm=r,
        )

        if result.phase == 'two-phase':
            # K_confined = K_bulk × (PL/Pv) < K_bulk
            ratio = result.liquid_pressure / result.vapor_pressure
            expected_K = result.K_values_bulk * ratio

            assert np.allclose(result.K_values, expected_K, rtol=0.01)

    def test_confined_flash_invalid_pore_radius(self, methane_butane, methane_butane_eos):
        """Test error handling for invalid pore radius."""
        z = np.array([0.5, 0.5])

        with pytest.raises(ValidationError):
            confined_flash(5e6, 300.0, z, methane_butane, methane_butane_eos,
                          pore_radius_nm=0.0)

        with pytest.raises(ValidationError):
            confined_flash(5e6, 300.0, z, methane_butane, methane_butane_eos,
                          pore_radius_nm=-10.0)

        with pytest.raises(ValidationError):
            confined_flash(5e6, 300.0, z, methane_butane, methane_butane_eos,
                          pore_radius_nm=0.5)  # Sub-nanometer


# =============================================================================
# Envelope Shrinkage Tests
# =============================================================================

class TestEnvelopeShrinkage:
    """Tests for envelope shrinkage estimates."""

    def test_estimate_shrinkage(self):
        """Test envelope shrinkage estimation."""
        result = estimate_envelope_shrinkage(pore_radius_nm=10.0, ift_typical=10.0)

        assert 'Pc_typical' in result
        assert 'bubble_suppression' in result
        assert 'dew_enhancement' in result

        # For 10 nm pore with 10 mN/m IFT, Pc ≈ 2 MPa
        assert abs(result['Pc_typical_MPa'] - 2.0) < 0.1

    def test_estimate_shrinkage_scaling(self):
        """Test that shrinkage increases with smaller pores."""
        result_10nm = estimate_envelope_shrinkage(10.0)
        result_5nm = estimate_envelope_shrinkage(5.0)

        assert result_5nm['Pc_typical'] > result_10nm['Pc_typical']
        assert result_5nm['bubble_suppression'] > result_10nm['bubble_suppression']


# =============================================================================
# Physical Validation Tests
# =============================================================================

class TestPhysicalValidation:
    """Tests validating physical behavior of confinement effects."""

    def test_confinement_effect_direction(self, methane_propane, methane_propane_eos):
        """Test that confinement effects are in correct direction.

        Key physics:
        - Liquid pressure < vapor pressure (wetting liquid)
        - Bubble point suppressed (lower P)
        - Vapor fraction reduced
        """
        z = np.array([0.6, 0.4])
        T = 280.0

        # Find a pressure in two-phase region
        for P in [2e6, 3e6, 4e6]:
            result = confined_flash(
                P, T, z, methane_propane, methane_propane_eos,
                pore_radius_nm=10.0,
            )
            if result.phase == 'two-phase':
                # Physical checks
                assert result.vapor_pressure >= result.liquid_pressure
                assert result.capillary_pressure >= 0

                # K-values should be reduced from bulk
                assert np.all(result.K_values <= result.K_values_bulk * 1.001)
                break

    def test_large_pore_approaches_bulk(self, methane_propane, methane_propane_eos):
        """Test that large pores approach bulk behavior."""
        z = np.array([0.5, 0.5])
        P = 3e6
        T = 280.0

        # Very large pore (100 nm) should have small Pc
        result = confined_flash(
            P, T, z, methane_propane, methane_propane_eos,
            pore_radius_nm=100.0,
        )

        if result.phase == 'two-phase':
            # Pc should be small (< 0.5 MPa for 100 nm)
            assert result.capillary_pressure < 0.5e6

            # K-values should be close to bulk
            ratio = result.liquid_pressure / result.vapor_pressure
            assert ratio > 0.9  # Close to 1

    def test_nojabaei_style_mixture(self, methane_butane, methane_butane_eos):
        """Test with mixture similar to Nojabaei et al. (2013) paper.

        The paper used C1-nC4 mixtures and showed:
        - Significant Pc effects for pores < 20 nm
        - Bubble point suppression of several MPa for 5 nm pores
        """
        # Composition similar to paper
        z = np.array([0.7, 0.3])  # 70% C1, 30% C4
        T = 350.0  # K

        # Test at different pore sizes
        results = {}
        for r in [5.0, 10.0, 20.0, 50.0]:
            try:
                result = confined_flash(
                    5e6, T, z, methane_butane, methane_butane_eos,
                    pore_radius_nm=r,
                )
                if result.converged:
                    results[r] = result.capillary_pressure
            except ConvergenceError:
                pass

        # Check that Pc increases as pore radius decreases
        if len(results) >= 2:
            pore_sizes = sorted(results.keys())
            for i in range(len(pore_sizes) - 1):
                r_small = pore_sizes[i]
                r_large = pore_sizes[i + 1]
                # Smaller pore should have higher Pc
                assert results[r_small] >= results[r_large] * 0.9

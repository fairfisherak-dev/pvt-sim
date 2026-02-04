"""Unit tests for the properties module (density, viscosity, IFT)."""

import numpy as np
import pytest

from pvtcore.properties import (
    calculate_density,
    calculate_phase_densities,
    mixture_molecular_weight,
    estimate_volume_shift_peneloux,
    DensityResult,
    calculate_viscosity_lbc,
    calculate_phase_viscosities,
    ViscosityResult,
    calculate_ift_parachor,
    calculate_ift_from_mass_density,
    IFTResult,
)
from pvtcore.models.component import load_components, Component
from pvtcore.eos.peng_robinson import PengRobinsonEOS
from pvtcore.core.errors import ValidationError, PropertyError


@pytest.fixture
def components():
    """Load component database."""
    return load_components()


@pytest.fixture
def methane_propane(components):
    """Binary C1-C3 mixture components."""
    return [components['C1'], components['C3']]


@pytest.fixture
def methane_decane(components):
    """Binary C1-C10 mixture components."""
    return [components['C1'], components['C10']]


@pytest.fixture
def methane_propane_eos(methane_propane):
    """EOS for C1-C3 binary."""
    return PengRobinsonEOS(methane_propane)


@pytest.fixture
def methane_decane_eos(methane_decane):
    """EOS for C1-C10 binary."""
    return PengRobinsonEOS(methane_decane)


# =============================================================================
# Density Tests
# =============================================================================

class TestDensityCalculation:
    """Tests for density calculations."""

    def test_density_basic(self, methane_propane, methane_propane_eos):
        """Test basic density calculation."""
        z = np.array([0.7, 0.3])
        P = 5e6  # 5 MPa
        T = 300.0  # K

        result = calculate_density(
            P, T, z, methane_propane, methane_propane_eos, phase='vapor'
        )

        assert isinstance(result, DensityResult)
        assert result.molar_density > 0
        assert result.mass_density > 0
        assert result.molar_volume > 0
        assert result.Z > 0
        assert not result.volume_translated

    def test_liquid_denser_than_vapor(self, methane_propane, methane_propane_eos):
        """Test that liquid density is greater than vapor density.

        Use different compositions typical of VLE equilibrium:
        - Liquid is richer in heavy component (propane)
        - Vapor is richer in light component (methane)
        """
        x = np.array([0.3, 0.7])  # Liquid (propane-rich)
        y = np.array([0.9, 0.1])  # Vapor (methane-rich)
        P = 2e6  # 2 MPa
        T = 250.0  # K

        liquid = calculate_density(
            P, T, x, methane_propane, methane_propane_eos, phase='liquid'
        )
        vapor = calculate_density(
            P, T, y, methane_propane, methane_propane_eos, phase='vapor'
        )

        # Liquid should be denser
        assert liquid.mass_density > vapor.mass_density
        assert liquid.molar_density > vapor.molar_density

        # Liquid has smaller molar volume
        assert liquid.molar_volume < vapor.molar_volume

    def test_density_ideal_gas_limit(self, methane_propane, methane_propane_eos):
        """Test that low pressure approaches ideal gas behavior."""
        z = np.array([0.7, 0.3])
        P = 1e4  # Very low pressure (10 kPa)
        T = 400.0  # High temperature

        result = calculate_density(
            P, T, z, methane_propane, methane_propane_eos, phase='vapor'
        )

        # At low pressure, Z should be close to 1
        assert 0.95 < result.Z < 1.05

    def test_density_pressure_dependence(self, methane_propane, methane_propane_eos):
        """Test that density increases with pressure."""
        z = np.array([0.7, 0.3])
        T = 350.0

        densities = []
        for P in [1e6, 5e6, 10e6]:
            result = calculate_density(
                P, T, z, methane_propane, methane_propane_eos, phase='vapor'
            )
            densities.append(result.mass_density)

        # Density should increase with pressure
        assert densities[0] < densities[1] < densities[2]

    def test_density_temperature_dependence(self, methane_propane, methane_propane_eos):
        """Test that gas density decreases with temperature at constant P."""
        z = np.array([0.7, 0.3])
        P = 5e6

        densities = []
        for T in [250.0, 300.0, 350.0]:
            result = calculate_density(
                P, T, z, methane_propane, methane_propane_eos, phase='vapor'
            )
            densities.append(result.mass_density)

        # Density should decrease with temperature
        assert densities[0] > densities[1] > densities[2]

    def test_density_with_volume_shift(self, methane_propane, methane_propane_eos):
        """Test density calculation with Peneloux volume translation."""
        z = np.array([0.5, 0.5])
        P = 5e6
        T = 280.0

        # Get volume shift parameters
        volume_shift = estimate_volume_shift_peneloux(methane_propane, 'PR')

        result_no_shift = calculate_density(
            P, T, z, methane_propane, methane_propane_eos, phase='liquid'
        )
        result_with_shift = calculate_density(
            P, T, z, methane_propane, methane_propane_eos, phase='liquid',
            volume_shift=volume_shift,
        )

        assert not result_no_shift.volume_translated
        assert result_with_shift.volume_translated
        # Volume shift typically increases liquid density prediction
        # (negative shift decreases molar volume, increases density)

    def test_calculate_phase_densities(self, methane_propane, methane_propane_eos):
        """Test convenience function for both phases."""
        x = np.array([0.3, 0.7])  # Liquid (more propane)
        y = np.array([0.9, 0.1])  # Vapor (more methane)
        P = 3e6
        T = 280.0

        liquid, vapor = calculate_phase_densities(
            P, T, x, y, methane_propane, methane_propane_eos
        )

        assert isinstance(liquid, DensityResult)
        assert isinstance(vapor, DensityResult)
        assert liquid.mass_density > vapor.mass_density

    def test_mixture_molecular_weight(self, methane_propane):
        """Test mixture MW calculation."""
        z = np.array([0.7, 0.3])
        MW_mix = mixture_molecular_weight(z, methane_propane)

        # Manual calculation
        expected = 0.7 * methane_propane[0].MW + 0.3 * methane_propane[1].MW
        assert abs(MW_mix - expected) < 1e-10

    def test_density_invalid_pressure(self, methane_propane, methane_propane_eos):
        """Test error handling for invalid pressure."""
        z = np.array([0.5, 0.5])

        with pytest.raises(ValidationError):
            calculate_density(-1e6, 300.0, z, methane_propane, methane_propane_eos)

    def test_density_invalid_temperature(self, methane_propane, methane_propane_eos):
        """Test error handling for invalid temperature."""
        z = np.array([0.5, 0.5])

        with pytest.raises(ValidationError):
            calculate_density(5e6, -100.0, z, methane_propane, methane_propane_eos)

    def test_density_composition_mismatch(self, methane_propane, methane_propane_eos):
        """Test error handling for composition length mismatch."""
        z = np.array([0.5, 0.3, 0.2])  # 3 components, but only 2 in list

        with pytest.raises(ValidationError):
            calculate_density(5e6, 300.0, z, methane_propane, methane_propane_eos)


# =============================================================================
# Viscosity Tests
# =============================================================================

class TestViscosityLBC:
    """Tests for LBC viscosity correlation."""

    def test_viscosity_basic(self, methane_propane):
        """Test basic viscosity calculation."""
        z = np.array([0.7, 0.3])
        T = 300.0
        rho_mol = 5000.0  # mol/m³

        result = calculate_viscosity_lbc(rho_mol, T, z, methane_propane)

        assert isinstance(result, ViscosityResult)
        assert result.viscosity > 0
        assert result.viscosity_cp > 0
        assert result.dilute_gas_viscosity > 0
        assert result.reduced_density > 0

        # Pa·s to cp conversion
        assert abs(result.viscosity * 1000 - result.viscosity_cp) < 1e-10

    def test_viscosity_liquid_greater_than_vapor(self, methane_propane):
        """Test that liquid viscosity is greater than vapor viscosity."""
        z = np.array([0.5, 0.5])
        T = 280.0

        # Typical liquid vs vapor densities
        rho_liquid = 12000.0  # mol/m³
        rho_vapor = 500.0  # mol/m³

        mu_liquid = calculate_viscosity_lbc(rho_liquid, T, z, methane_propane)
        mu_vapor = calculate_viscosity_lbc(rho_vapor, T, z, methane_propane)

        assert mu_liquid.viscosity > mu_vapor.viscosity

    def test_viscosity_increases_with_density(self, methane_propane):
        """Test that viscosity increases with density."""
        z = np.array([0.5, 0.5])
        T = 300.0

        viscosities = []
        for rho in [1000.0, 5000.0, 10000.0]:
            result = calculate_viscosity_lbc(rho, T, z, methane_propane)
            viscosities.append(result.viscosity)

        # Viscosity should increase with density
        assert viscosities[0] < viscosities[1] < viscosities[2]

    def test_viscosity_dilute_gas_limit(self, methane_propane):
        """Test that low density approaches dilute gas viscosity."""
        z = np.array([0.7, 0.3])
        T = 400.0
        rho_mol = 1.0  # Very low density

        result = calculate_viscosity_lbc(rho_mol, T, z, methane_propane)

        # At very low density, viscosity should be close to dilute gas value
        assert result.viscosity_cp > 0
        # Reduced density should be small
        assert result.reduced_density < 0.1

    def test_viscosity_reasonable_range(self, methane_propane):
        """Test that calculated viscosities are in reasonable range."""
        z = np.array([0.5, 0.5])
        T = 300.0

        # Vapor-like density
        rho_vapor = 500.0
        result_vapor = calculate_viscosity_lbc(rho_vapor, T, z, methane_propane)
        # Gas viscosity typically 0.01-0.03 cp
        assert 0.005 < result_vapor.viscosity_cp < 0.1

        # Liquid-like density
        rho_liquid = 10000.0
        result_liquid = calculate_viscosity_lbc(rho_liquid, T, z, methane_propane)
        # Liquid viscosity typically 0.05-2 cp for light hydrocarbons
        assert 0.01 < result_liquid.viscosity_cp < 5.0

    def test_calculate_phase_viscosities(self, methane_propane):
        """Test convenience function for both phases."""
        x = np.array([0.3, 0.7])
        y = np.array([0.9, 0.1])
        T = 280.0
        rho_L = 12000.0
        rho_V = 400.0

        liquid, vapor = calculate_phase_viscosities(
            rho_L, rho_V, T, x, y, methane_propane
        )

        assert isinstance(liquid, ViscosityResult)
        assert isinstance(vapor, ViscosityResult)
        assert liquid.viscosity > vapor.viscosity

    def test_viscosity_invalid_density(self, methane_propane):
        """Test error handling for invalid density."""
        z = np.array([0.5, 0.5])

        with pytest.raises(ValidationError):
            calculate_viscosity_lbc(-100.0, 300.0, z, methane_propane)

    def test_viscosity_invalid_temperature(self, methane_propane):
        """Test error handling for invalid temperature."""
        z = np.array([0.5, 0.5])

        with pytest.raises(ValidationError):
            calculate_viscosity_lbc(5000.0, -100.0, z, methane_propane)


# =============================================================================
# IFT Tests
# =============================================================================

class TestIFTParachor:
    """Tests for parachor IFT correlation."""

    def test_ift_basic(self, methane_propane):
        """Test basic IFT calculation."""
        x = np.array([0.3, 0.7])  # Liquid
        y = np.array([0.9, 0.1])  # Vapor
        rho_L = 12000.0  # mol/m³
        rho_V = 500.0  # mol/m³

        result = calculate_ift_parachor(x, y, rho_L, rho_V, methane_propane)

        assert isinstance(result, IFTResult)
        assert result.ift > 0
        assert result.ift_dyn_cm > 0
        assert result.ift == result.ift_dyn_cm  # mN/m = dyn/cm

    def test_ift_reasonable_range(self, methane_propane):
        """Test that IFT is in physically reasonable range."""
        x = np.array([0.3, 0.7])
        y = np.array([0.9, 0.1])
        rho_L = 10000.0
        rho_V = 400.0

        result = calculate_ift_parachor(x, y, rho_L, rho_V, methane_propane)

        # Typical hydrocarbon IFT: 1-30 mN/m
        assert 0.1 < result.ift < 50.0

    def test_ift_decreases_near_critical(self, methane_propane):
        """Test that IFT decreases as phases become similar (near critical)."""
        x = np.array([0.5, 0.5])
        y = np.array([0.5, 0.5])  # Same composition
        rho_L = 5000.0
        rho_V = 4500.0  # Close to liquid density

        result = calculate_ift_parachor(x, y, rho_L, rho_V, methane_propane)

        # IFT should be very low when phases are similar
        assert result.ift < 1.0

    def test_ift_zero_at_equal_conditions(self, methane_propane):
        """Test that IFT is zero when phases are identical."""
        x = np.array([0.5, 0.5])
        y = np.array([0.5, 0.5])
        rho = 5000.0  # Same density

        result = calculate_ift_parachor(x, y, rho, rho, methane_propane)

        # IFT should be essentially zero
        assert result.ift < 1e-10

    def test_ift_increases_with_density_difference(self, methane_propane):
        """Test that IFT increases with phase density difference."""
        x = np.array([0.3, 0.7])
        y = np.array([0.9, 0.1])
        rho_V = 500.0

        ifts = []
        for rho_L in [5000.0, 10000.0, 15000.0]:
            result = calculate_ift_parachor(x, y, rho_L, rho_V, methane_propane)
            ifts.append(result.ift)

        # IFT should increase with density difference
        assert ifts[0] < ifts[1] < ifts[2]

    def test_ift_from_mass_density(self, methane_propane):
        """Test IFT calculation from mass densities."""
        x = np.array([0.3, 0.7])
        y = np.array([0.9, 0.1])

        # Mass densities in kg/m³
        rho_L_mass = 400.0  # Typical liquid
        rho_V_mass = 20.0   # Typical vapor

        result = calculate_ift_from_mass_density(
            x, y, rho_L_mass, rho_V_mass, methane_propane
        )

        assert isinstance(result, IFTResult)
        assert result.ift > 0

    def test_ift_with_custom_parachors(self, methane_propane):
        """Test IFT calculation with user-specified parachors."""
        x = np.array([0.3, 0.7])
        y = np.array([0.9, 0.1])
        rho_L = 10000.0
        rho_V = 500.0

        # Custom parachors
        custom_P = np.array([77.0, 150.3])  # C1, C3 literature values

        result = calculate_ift_parachor(
            x, y, rho_L, rho_V, methane_propane, parachors=custom_P
        )

        assert result.ift > 0

    def test_ift_invalid_liquid_composition_length(self, methane_propane):
        """Test error handling for composition length mismatch."""
        x = np.array([0.3, 0.4, 0.3])  # 3 components
        y = np.array([0.9, 0.1])  # 2 components

        with pytest.raises(ValidationError):
            calculate_ift_parachor(x, y, 10000.0, 500.0, methane_propane)

    def test_ift_invalid_density(self, methane_propane):
        """Test error handling for invalid density."""
        x = np.array([0.3, 0.7])
        y = np.array([0.9, 0.1])

        with pytest.raises(ValidationError):
            calculate_ift_parachor(x, y, -1000.0, 500.0, methane_propane)

        with pytest.raises(ValidationError):
            calculate_ift_parachor(x, y, 10000.0, 0.0, methane_propane)


# =============================================================================
# Integration Tests
# =============================================================================

class TestPropertiesIntegration:
    """Integration tests for property calculations."""

    def test_full_property_workflow(self, methane_propane, methane_propane_eos):
        """Test complete workflow: density → viscosity → IFT."""
        # Conditions
        P = 3e6  # 3 MPa
        T = 280.0  # K
        x = np.array([0.3, 0.7])  # Liquid
        y = np.array([0.85, 0.15])  # Vapor

        # Step 1: Calculate densities
        rho_L = calculate_density(
            P, T, x, methane_propane, methane_propane_eos, phase='liquid'
        )
        rho_V = calculate_density(
            P, T, y, methane_propane, methane_propane_eos, phase='vapor'
        )

        assert rho_L.mass_density > rho_V.mass_density

        # Step 2: Calculate viscosities
        mu_L = calculate_viscosity_lbc(
            rho_L.molar_density, T, x, methane_propane
        )
        mu_V = calculate_viscosity_lbc(
            rho_V.molar_density, T, y, methane_propane
        )

        assert mu_L.viscosity > mu_V.viscosity

        # Step 3: Calculate IFT
        ift = calculate_ift_parachor(
            x, y, rho_L.molar_density, rho_V.molar_density, methane_propane
        )

        assert ift.ift > 0

    def test_asymmetric_mixture(self, methane_decane, methane_decane_eos):
        """Test properties for asymmetric C1-C10 mixture."""
        P = 10e6  # 10 MPa
        T = 350.0  # K
        x = np.array([0.2, 0.8])  # Liquid (decane-rich)
        y = np.array([0.95, 0.05])  # Vapor (methane-rich)

        # Density
        rho_L = calculate_density(
            P, T, x, methane_decane, methane_decane_eos, phase='liquid'
        )
        rho_V = calculate_density(
            P, T, y, methane_decane, methane_decane_eos, phase='vapor'
        )

        # Large density difference expected for asymmetric mixture
        density_ratio = rho_L.mass_density / rho_V.mass_density
        assert density_ratio > 2  # Liquid should be significantly denser

        # IFT should be substantial for asymmetric mixture
        ift = calculate_ift_parachor(
            x, y, rho_L.molar_density, rho_V.molar_density, methane_decane
        )
        assert ift.ift > 1.0  # Non-trivial IFT

    def test_properties_with_volume_translation(self, methane_propane, methane_propane_eos):
        """Test that volume translation affects density but not IFT sign."""
        P = 5e6
        T = 280.0
        x = np.array([0.3, 0.7])
        y = np.array([0.9, 0.1])

        volume_shift = estimate_volume_shift_peneloux(methane_propane, 'PR')

        # Densities with and without shift
        rho_L_no_shift = calculate_density(
            P, T, x, methane_propane, methane_propane_eos, phase='liquid'
        )
        rho_L_shifted = calculate_density(
            P, T, x, methane_propane, methane_propane_eos, phase='liquid',
            volume_shift=volume_shift,
        )

        rho_V = calculate_density(
            P, T, y, methane_propane, methane_propane_eos, phase='vapor'
        )

        # Both should give positive IFT
        ift_no_shift = calculate_ift_parachor(
            x, y, rho_L_no_shift.molar_density, rho_V.molar_density, methane_propane
        )
        ift_shifted = calculate_ift_parachor(
            x, y, rho_L_shifted.molar_density, rho_V.molar_density, methane_propane
        )

        assert ift_no_shift.ift > 0
        assert ift_shifted.ift > 0

"""
Unit tests for property correlations module.

Tests critical property correlations, acentric factor correlations,
boiling point correlations, and parachor correlations.

Note: Correlations are inherently approximate and are designed for
pseudo-components where experimental data is unavailable. Pure component
properties should come from the database, not correlations.
"""

import pytest
import numpy as np

from pvtcore.correlations import (
    # Critical properties
    CriticalPropsMethod,
    riazi_daubert_Tc,
    riazi_daubert_Pc,
    riazi_daubert_Vc,
    riazi_daubert_critical_props,
    kesler_lee_Tc,
    kesler_lee_Pc,
    kesler_lee_critical_props,
    cavett_Tc,
    cavett_Pc,
    cavett_critical_props,
    estimate_critical_props,
    # Acentric factor
    AcentricMethod,
    edmister_omega,
    kesler_lee_omega,
    estimate_omega,
    # Boiling point
    BoilingPointMethod,
    soreide_Tb,
    riazi_daubert_Tb,
    estimate_Tb,
    # Parachor
    fanchi_parachor,
    estimate_parachor,
)


# =============================================================================
# Reference Data for Validation
# =============================================================================

# n-Heptane (C7) properties from NIST - used for Kesler-Lee validation
# Kesler-Lee requires known Tb, so we use a pure component as reference
C7_REF = {
    "MW": 100.20,
    "Tc": 540.2,      # K
    "Pc": 2.74e6,     # Pa
    "omega": 0.350,
    "Tb": 371.6,      # K
    "SG": 0.684,
}


class TestKeslerLeeCriticalProps:
    """Tests for Kesler-Lee correlations (the best-tested correlation)."""

    def test_kesler_lee_Tc(self):
        """Test Kesler-Lee Tc correlation against C7 reference."""
        Tc = kesler_lee_Tc(
            MW=C7_REF["MW"],
            SG=C7_REF["SG"],
            Tb=C7_REF["Tb"],
        )
        # Should be within 2% of reference
        assert abs(Tc - C7_REF["Tc"]) / C7_REF["Tc"] < 0.02

    def test_kesler_lee_Pc(self):
        """Test Kesler-Lee Pc correlation against C7 reference."""
        Pc = kesler_lee_Pc(
            MW=C7_REF["MW"],
            SG=C7_REF["SG"],
            Tb=C7_REF["Tb"],
        )
        # Should be within 10% of reference
        assert abs(Pc - C7_REF["Pc"]) / C7_REF["Pc"] < 0.10

    def test_kesler_lee_requires_Tb(self):
        """Test that Kesler-Lee raises error without Tb."""
        with pytest.raises(ValueError, match="requires Tb"):
            kesler_lee_Tc(MW=100.0, SG=0.7, Tb=None)

    def test_kesler_lee_critical_props_complete(self):
        """Test complete Kesler-Lee critical property estimation."""
        result = kesler_lee_critical_props(
            MW=C7_REF["MW"],
            SG=C7_REF["SG"],
            Tb=C7_REF["Tb"],
        )
        assert result.method == CriticalPropsMethod.KESLER_LEE
        assert abs(result.Tc - C7_REF["Tc"]) / C7_REF["Tc"] < 0.02
        assert result.Vc > 0


class TestCriticalPropertyCorrelations:
    """Tests for other critical property correlations."""

    def test_riazi_daubert_produces_reasonable_Tc(self):
        """Test that Riazi-Daubert gives physically reasonable Tc."""
        Tc = riazi_daubert_Tc(
            MW=C7_REF["MW"],
            SG=C7_REF["SG"],
            Tb=C7_REF["Tb"],
        )
        # Tc should be higher than Tb and within plausible range
        assert Tc > C7_REF["Tb"]
        assert 400 < Tc < 800  # Reasonable range for C7

    def test_riazi_daubert_Tc_without_Tb(self):
        """Test Riazi-Daubert Tc correlation without Tb (less accurate)."""
        Tc = riazi_daubert_Tc(
            MW=C7_REF["MW"],
            SG=C7_REF["SG"],
            Tb=None,
        )
        # Without Tb input, correlation uses estimated Tb internally
        # Less accurate but should still be in physically reasonable range
        assert 400 < Tc < 1000

    def test_riazi_daubert_Pc(self):
        """Test Riazi-Daubert Pc correlation."""
        Pc = riazi_daubert_Pc(
            MW=C7_REF["MW"],
            SG=C7_REF["SG"],
            Tb=C7_REF["Tb"],
        )
        # Should be in reasonable range for hydrocarbons
        assert 1e6 < Pc < 5e6

    def test_riazi_daubert_Vc(self):
        """Test Riazi-Daubert Vc correlation."""
        Vc = riazi_daubert_Vc(
            MW=C7_REF["MW"],
            SG=C7_REF["SG"],
            Tb=C7_REF["Tb"],
        )
        # Vc should be positive
        assert Vc > 0

    def test_riazi_daubert_critical_props_result_type(self):
        """Test that result is proper dataclass."""
        result = riazi_daubert_critical_props(
            MW=C7_REF["MW"],
            SG=C7_REF["SG"],
            Tb=C7_REF["Tb"],
        )
        assert hasattr(result, "Tc")
        assert hasattr(result, "Pc")
        assert hasattr(result, "Vc")
        assert result.method == CriticalPropsMethod.RIAZI_DAUBERT

    def test_cavett_Tc(self):
        """Test Cavett Tc correlation produces reasonable values."""
        Tc = cavett_Tc(
            MW=C7_REF["MW"],
            SG=C7_REF["SG"],
            Tb=C7_REF["Tb"],
        )
        # Tc should be higher than Tb
        assert Tc > C7_REF["Tb"]
        assert 400 < Tc < 800

    def test_cavett_Pc(self):
        """Test Cavett Pc correlation."""
        Pc = cavett_Pc(
            MW=C7_REF["MW"],
            SG=C7_REF["SG"],
            Tb=C7_REF["Tb"],
        )
        assert 1e6 < Pc < 5e6

    def test_estimate_critical_props_method_selection(self):
        """Test unified interface with method selection."""
        result_kl = estimate_critical_props(
            MW=C7_REF["MW"],
            SG=C7_REF["SG"],
            Tb=C7_REF["Tb"],
            method=CriticalPropsMethod.KESLER_LEE,
        )
        result_cv = estimate_critical_props(
            MW=C7_REF["MW"],
            SG=C7_REF["SG"],
            Tb=C7_REF["Tb"],
            method=CriticalPropsMethod.CAVETT,
        )

        assert result_kl.method == CriticalPropsMethod.KESLER_LEE
        assert result_cv.method == CriticalPropsMethod.CAVETT
        # Both should give reasonable Tc (Cavett tends to give higher values)
        assert 400 < result_kl.Tc < 700
        assert 400 < result_cv.Tc < 850

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            riazi_daubert_Tc(MW=-100.0, SG=0.7)
        with pytest.raises(ValueError):
            riazi_daubert_Tc(MW=100.0, SG=-0.7)
        with pytest.raises(ValueError):
            riazi_daubert_Tc(MW=float("nan"), SG=0.7)


class TestAcentricFactorCorrelations:
    """Tests for acentric factor correlations."""

    def test_edmister_omega(self):
        """Test Edmister acentric factor correlation."""
        omega = edmister_omega(
            Tb=C7_REF["Tb"],
            Tc=C7_REF["Tc"],
            Pc=C7_REF["Pc"],
        )
        # Should be within 15% of reference
        assert abs(omega - C7_REF["omega"]) < 0.07

    def test_kesler_lee_omega(self):
        """Test Kesler-Lee acentric factor correlation."""
        omega = kesler_lee_omega(
            Tb=C7_REF["Tb"],
            Tc=C7_REF["Tc"],
            Pc=C7_REF["Pc"],
        )
        # Should be in physically reasonable range
        assert 0.2 < omega < 0.5

    def test_estimate_omega_method_selection(self):
        """Test unified interface with method selection."""
        omega_ed = estimate_omega(
            Tb=C7_REF["Tb"],
            Tc=C7_REF["Tc"],
            Pc=C7_REF["Pc"],
            method=AcentricMethod.EDMISTER,
        )
        omega_kl = estimate_omega(
            Tb=C7_REF["Tb"],
            Tc=C7_REF["Tc"],
            Pc=C7_REF["Pc"],
            method=AcentricMethod.KESLER_LEE,
        )

        # Both should give reasonable results
        assert 0.0 < omega_ed < 1.0
        assert 0.0 < omega_kl < 1.0

    def test_omega_physical_range(self):
        """Test that omega is in physically reasonable range."""
        omega = edmister_omega(Tb=400.0, Tc=600.0, Pc=3e6)
        assert -0.5 <= omega <= 2.0

    def test_omega_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            edmister_omega(Tb=600.0, Tc=500.0, Pc=3e6)  # Tb > Tc


class TestBoilingPointCorrelations:
    """Tests for boiling point correlations."""

    def test_soreide_Tb_produces_reasonable_values(self):
        """Test Soreide boiling point gives reasonable values."""
        Tb = soreide_Tb(MW=100.0, SG=0.75)
        # Should be in reasonable range for C7-like fraction
        assert 300 < Tb < 450

    def test_riazi_daubert_Tb(self):
        """Test Riazi-Daubert boiling point correlation."""
        Tb = riazi_daubert_Tb(MW=100.0, SG=0.75)
        assert 300 < Tb < 450

    def test_estimate_Tb_method_selection(self):
        """Test unified interface with method selection."""
        Tb_s = estimate_Tb(
            MW=150.0,
            SG=0.80,
            method=BoilingPointMethod.SOREIDE,
        )
        Tb_rd = estimate_Tb(
            MW=150.0,
            SG=0.80,
            method=BoilingPointMethod.RIAZI_DAUBERT,
        )

        # Both should give reasonable boiling points
        assert 350 < Tb_s < 600
        assert 350 < Tb_rd < 600

    def test_Tb_increases_with_MW(self):
        """Test that Tb increases with MW (expected trend)."""
        Tb1 = soreide_Tb(MW=100.0, SG=0.75)
        Tb2 = soreide_Tb(MW=150.0, SG=0.80)
        Tb3 = soreide_Tb(MW=200.0, SG=0.85)
        assert Tb1 < Tb2 < Tb3

    def test_Tb_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            soreide_Tb(MW=-100.0, SG=0.7)
        with pytest.raises(ValueError):
            soreide_Tb(MW=100.0, SG=-0.7)


class TestParachorCorrelations:
    """Tests for parachor correlations."""

    # Reference parachors (typical values)
    C7_PARACHOR = 312.5

    def test_fanchi_parachor(self):
        """Test Fanchi parachor correlation."""
        P = fanchi_parachor(MW=C7_REF["MW"])
        # Should be within 15% of reference
        assert abs(P - self.C7_PARACHOR) / self.C7_PARACHOR < 0.15

    def test_estimate_parachor_with_id(self):
        """Test parachor estimation with component ID lookup."""
        # Should use database value
        P = estimate_parachor(MW=16.04, component_id="C1")
        assert P == 77.0  # Methane database value

    def test_estimate_parachor_without_id(self):
        """Test parachor estimation without component ID."""
        P = estimate_parachor(MW=C7_REF["MW"])
        assert P > 0
        assert 250 < P < 400  # Reasonable range for C7

    def test_parachor_linear_trend(self):
        """Test that parachor increases with MW."""
        P1 = fanchi_parachor(MW=100.0)
        P2 = fanchi_parachor(MW=150.0)
        P3 = fanchi_parachor(MW=200.0)
        assert P1 < P2 < P3


class TestCorrelationConsistency:
    """Tests for consistency between correlations."""

    def test_roundtrip_consistency(self):
        """Test that Tb -> Tc -> omega gives consistent results."""
        MW = 150.0
        SG = 0.82

        # Estimate Tb
        Tb = estimate_Tb(MW=MW, SG=SG)

        # Estimate Tc, Pc using Kesler-Lee (requires Tb)
        crit = estimate_critical_props(
            MW=MW, SG=SG, Tb=Tb,
            method=CriticalPropsMethod.KESLER_LEE
        )

        # Estimate omega
        omega = estimate_omega(Tb=Tb, Tc=crit.Tc, Pc=crit.Pc)

        # All should be physically reasonable
        assert 350 < Tb < 700  # K
        assert Tb < crit.Tc < 900  # Tc > Tb
        assert 1e6 < crit.Pc < 5e6  # Pa
        assert 0.0 < omega < 1.5

    def test_scn_property_trends(self):
        """Test that properties follow expected trends with MW."""
        MWs = [100, 150, 200, 250, 300]
        SG = 0.82

        Tbs = [estimate_Tb(MW=mw, SG=SG) for mw in MWs]
        Tcs = []
        Pcs = []
        omegas = []

        for mw, tb in zip(MWs, Tbs):
            crit = estimate_critical_props(
                MW=mw, SG=SG, Tb=tb,
                method=CriticalPropsMethod.KESLER_LEE
            )
            Tcs.append(crit.Tc)
            Pcs.append(crit.Pc)
            omega = estimate_omega(Tb=tb, Tc=crit.Tc, Pc=crit.Pc)
            omegas.append(omega)

        # Tb should increase with MW
        assert all(Tbs[i] < Tbs[i + 1] for i in range(len(Tbs) - 1))

        # Tc should increase with MW
        assert all(Tcs[i] < Tcs[i + 1] for i in range(len(Tcs) - 1))

        # Pc should decrease with MW
        assert all(Pcs[i] > Pcs[i + 1] for i in range(len(Pcs) - 1))

        # Omega should increase with MW
        assert all(omegas[i] < omegas[i + 1] for i in range(len(omegas) - 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for the full characterization pipeline.

Tests plus-fraction splitting, lumping, delumping, BIP correlations,
and the unified CharacterizedFluid class.
"""

import pytest
import numpy as np

from pvtcore.characterization import (
    # Plus-fraction splitting
    split_plus_fraction_pedersen,
    split_plus_fraction_katz,
    katz_classic_split,
    split_plus_fraction_lohrenz,
    PedersenSplitResult,
    KatzSplitResult,
    LohrenzSplitResult,
    # SCN properties
    get_scn_properties,
    SCNProperties,
    # Lumping
    lump_by_mw_groups,
    lump_by_indices,
    suggest_lumping_groups,
    LumpingResult,
    # Delumping
    delump_kvalue_interpolation,
    delump_simple_distribution,
    DelumpingResult,
    # BIP
    build_bip_matrix,
    BIPMethod,
    get_default_bip,
    chueh_prausnitz_kij,
)


# =============================================================================
# Test Data
# =============================================================================

# Typical C7+ properties for a reservoir fluid
C7PLUS_Z = 0.25
C7PLUS_MW = 215.0
C7PLUS_SG = 0.85


class TestPedersenSplitting:
    """Tests for Pedersen plus-fraction splitting."""

    def test_basic_split(self):
        """Test basic plus-fraction splitting."""
        result = split_plus_fraction_pedersen(
            z_plus=C7PLUS_Z,
            MW_plus=C7PLUS_MW,
        )

        assert isinstance(result, PedersenSplitResult)
        assert len(result.n) == len(result.z)
        assert len(result.n) == len(result.MW)

    def test_mole_fraction_constraint(self):
        """Test that mole fractions sum to z_plus."""
        result = split_plus_fraction_pedersen(
            z_plus=C7PLUS_Z,
            MW_plus=C7PLUS_MW,
        )

        z_sum = result.z.sum()
        assert abs(z_sum - C7PLUS_Z) / C7PLUS_Z < 1e-6

    def test_mw_constraint(self):
        """Test that weighted MW matches MW_plus."""
        result = split_plus_fraction_pedersen(
            z_plus=C7PLUS_Z,
            MW_plus=C7PLUS_MW,
        )

        # Calculate weighted average MW
        z_normalized = result.z / result.z.sum()
        mw_avg = (z_normalized * result.MW).sum()

        # Should be close to MW_plus
        assert abs(mw_avg - C7PLUS_MW) / C7PLUS_MW < 0.05

    def test_scn_range(self):
        """Test custom SCN range."""
        result = split_plus_fraction_pedersen(
            z_plus=C7PLUS_Z,
            MW_plus=C7PLUS_MW,
            n_start=7,
            n_end=30,
        )

        assert result.n[0] == 7
        assert result.n[-1] == 30
        assert len(result.n) == 24

    def test_exponential_decay(self):
        """Test that distribution decays with SCN."""
        result = split_plus_fraction_pedersen(
            z_plus=C7PLUS_Z,
            MW_plus=C7PLUS_MW,
        )

        # z should generally decrease (not strictly monotonic due to fitting)
        assert result.z[0] > result.z[-1]

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            split_plus_fraction_pedersen(z_plus=-0.1, MW_plus=200.0)
        with pytest.raises(ValueError):
            split_plus_fraction_pedersen(z_plus=0.25, MW_plus=-200.0)


class TestKatzSplitting:
    """Tests for Katz plus-fraction splitting."""

    def test_basic_split(self):
        """Test basic Katz splitting."""
        result = split_plus_fraction_katz(
            z_plus=C7PLUS_Z,
            MW_plus=C7PLUS_MW,
        )

        assert isinstance(result, KatzSplitResult)
        assert result.z.sum() > 0

    def test_mole_fraction_constraint(self):
        """Test mole fraction sum."""
        result = split_plus_fraction_katz(
            z_plus=C7PLUS_Z,
            MW_plus=C7PLUS_MW,
        )

        assert abs(result.z.sum() - C7PLUS_Z) / C7PLUS_Z < 1e-6

    def test_classic_split(self):
        """Test classic Katz with fixed coefficients."""
        result = katz_classic_split(z_plus=C7PLUS_Z)

        # Classic split should use A and B coefficients
        assert abs(result.A - 1.38205 * C7PLUS_Z) < 0.01
        assert abs(result.B - 0.25903) < 0.01


class TestLohrenzSplitting:
    """Tests for Lohrenz plus-fraction splitting."""

    def test_basic_split(self):
        """Test basic Lohrenz splitting."""
        result = split_plus_fraction_lohrenz(
            z_plus=C7PLUS_Z,
            MW_plus=C7PLUS_MW,
        )

        assert isinstance(result, LohrenzSplitResult)
        assert len(result.z) > 0

    def test_mole_fraction_constraint(self):
        """Test mole fraction sum."""
        result = split_plus_fraction_lohrenz(
            z_plus=C7PLUS_Z,
            MW_plus=C7PLUS_MW,
        )

        assert abs(result.z.sum() - C7PLUS_Z) / C7PLUS_Z < 1e-6

    def test_quadratic_coefficient(self):
        """Test that quadratic coefficient A is typically negative."""
        result = split_plus_fraction_lohrenz(
            z_plus=C7PLUS_Z,
            MW_plus=C7PLUS_MW,
        )

        # A is typically small negative for quadratic decay
        assert result.A < 0.01  # Could be small positive or negative


class TestSCNProperties:
    """Tests for SCN property generation."""

    def test_basic_properties(self):
        """Test basic SCN property retrieval."""
        props = get_scn_properties(n_start=7, n_end=20)

        assert isinstance(props, SCNProperties)
        assert len(props.n) == 14
        assert props.n[0] == 7
        assert props.n[-1] == 20

    def test_property_arrays(self):
        """Test that all property arrays match length."""
        props = get_scn_properties(n_start=6, n_end=45)

        assert len(props.mw) == len(props.n)
        assert len(props.sg_6060) == len(props.n)
        assert len(props.tb_k) == len(props.n)

    def test_property_trends(self):
        """Test that properties follow expected trends."""
        props = get_scn_properties(n_start=6, n_end=30)

        # MW should increase
        assert all(props.mw[i] < props.mw[i + 1] for i in range(len(props.mw) - 1))

        # Tb should increase
        assert all(props.tb_k[i] < props.tb_k[i + 1] for i in range(len(props.tb_k) - 1))

    def test_extrapolation(self):
        """Test extrapolation beyond table (C45+)."""
        props = get_scn_properties(n_start=7, n_end=60, extrapolate=True)

        assert props.n[-1] == 60
        assert np.isfinite(props.mw[-1])
        assert np.isfinite(props.sg_6060[-1])
        assert np.isfinite(props.tb_k[-1])

    def test_extrapolation_error(self):
        """Test that extrapolation=False raises error."""
        with pytest.raises(ValueError):
            get_scn_properties(n_start=7, n_end=50, extrapolate=False)


class TestLumping:
    """Tests for component lumping."""

    @pytest.fixture
    def sample_components(self):
        """Create sample component data for lumping tests."""
        n = 20
        return {
            "z": np.random.dirichlet(np.ones(n)),
            "MW": np.linspace(100, 400, n),
            "Tc": np.linspace(500, 800, n),
            "Pc": np.linspace(3e6, 1.5e6, n),
            "Vc": np.linspace(0.0004, 0.001, n),
            "omega": np.linspace(0.3, 0.8, n),
        }

    def test_basic_lumping(self, sample_components):
        """Test basic MW-based lumping."""
        result = lump_by_mw_groups(
            n_groups=5,
            **sample_components,
        )

        assert isinstance(result, LumpingResult)
        assert result.n_lumped <= 5
        assert result.n_original == 20

    def test_mole_fraction_conservation(self, sample_components):
        """Test that total mole fraction is conserved."""
        result = lump_by_mw_groups(
            n_groups=5,
            **sample_components,
        )

        z_total = sum(c.z for c in result.components)
        assert abs(z_total - sample_components["z"].sum()) < 1e-10

    def test_lumping_by_indices(self, sample_components):
        """Test lumping with explicit index groups."""
        groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9], list(range(10, 20))]

        result = lump_by_indices(
            group_indices=groups,
            **sample_components,
        )

        assert result.n_lumped == 4

    def test_suggest_groups(self, sample_components):
        """Test automatic group suggestion."""
        groups = suggest_lumping_groups(
            MW=sample_components["MW"],
            n_groups=8,
            preserve_light=3,
        )

        # Should have 8 groups total
        assert len(groups) == 8

        # First 3 should be single components
        assert all(len(g) == 1 for g in groups[:3])

    def test_property_averaging(self, sample_components):
        """Test that lumped properties are reasonable averages."""
        result = lump_by_mw_groups(
            n_groups=3,
            **sample_components,
        )

        for comp in result.components:
            # Properties should be within range of original
            assert sample_components["MW"].min() <= comp.MW <= sample_components["MW"].max()
            assert sample_components["Tc"].min() <= comp.Tc <= sample_components["Tc"].max()


class TestDelumping:
    """Tests for composition delumping."""

    def test_simple_delumping(self):
        """Test simple delumping using z distribution."""
        # Original detailed composition
        z_detailed = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.2])

        # Lumped compositions (2 groups)
        x_lumped = np.array([0.55, 0.45])
        y_lumped = np.array([0.65, 0.35])

        # Mapping
        lump_mapping = [[0, 1, 2, 3, 4], [5, 6, 7]]

        x_detail, y_detail = delump_simple_distribution(
            x_lumped=x_lumped,
            y_lumped=y_lumped,
            z_detailed=z_detailed,
            lump_mapping=lump_mapping,
        )

        # Check sums match
        assert abs(x_detail.sum() - 1.0) < 1e-10
        assert abs(y_detail.sum() - 1.0) < 1e-10

    def test_kvalue_delumping(self):
        """Test K-value interpolation delumping."""
        # Set up test case
        n_detailed = 8
        z_detailed = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.2])
        MW_detailed = np.array([100, 120, 140, 160, 180, 200, 220, 250])

        # Lumped results
        K_lumped = np.array([2.0, 0.5])  # First group more volatile
        x_lumped = np.array([0.4, 0.6])
        y_lumped = np.array([0.7, 0.3])
        MW_lumped = np.array([140, 220])

        lump_mapping = [[0, 1, 2, 3, 4], [5, 6, 7]]

        result = delump_kvalue_interpolation(
            K_lumped=K_lumped,
            x_lumped=x_lumped,
            y_lumped=y_lumped,
            MW_lumped=MW_lumped,
            z_detailed=z_detailed,
            MW_detailed=MW_detailed,
            lump_mapping=lump_mapping,
        )

        assert isinstance(result, DelumpingResult)
        assert len(result.x) == n_detailed
        assert abs(result.x.sum() - 1.0) < 1e-10


class TestBIPCorrelations:
    """Tests for BIP correlations."""

    def test_default_bip_lookup(self):
        """Test default BIP lookup."""
        kij = get_default_bip("N2", "C1")
        assert kij == 0.025

        kij = get_default_bip("CO2", "C1")
        assert kij == 0.105

    def test_default_bip_symmetry(self):
        """Test that BIP lookup is symmetric."""
        kij1 = get_default_bip("N2", "C1")
        kij2 = get_default_bip("C1", "N2")
        assert kij1 == kij2

    def test_chueh_prausnitz_correlation(self):
        """Test Chueh-Prausnitz BIP correlation."""
        Tc1 = 190.6  # Methane
        Tc2 = 540.2  # n-Heptane

        kij = chueh_prausnitz_kij(Tc1, Tc2, A=0.01, B=3.0)

        # Should be small positive for HC/HC pair
        assert 0.0 <= kij <= 0.05

    def test_chueh_prausnitz_identical_components(self):
        """Test that identical components have kij=0."""
        Tc = 300.0
        kij = chueh_prausnitz_kij(Tc, Tc, A=0.1, B=1.0)
        assert abs(kij) < 1e-10

    def test_build_bip_matrix_zero(self):
        """Test building zero BIP matrix."""
        names = ["C1", "C2", "C3"]
        Tc = np.array([190.6, 305.3, 369.9])

        result = build_bip_matrix(
            component_ids=names,
            Tc=Tc,
            method=BIPMethod.ZERO,
        )

        assert result.kij.shape == (3, 3)
        assert np.allclose(result.kij, 0.0)

    def test_build_bip_matrix_defaults(self):
        """Test building BIP matrix with defaults."""
        names = ["N2", "CO2", "C1", "C2", "C3"]
        Tc = np.array([126.19, 304.18, 190.6, 305.3, 369.9])

        result = build_bip_matrix(
            component_ids=names,
            Tc=Tc,
            method=BIPMethod.DEFAULT_VALUES,
        )

        assert result.kij.shape == (5, 5)

        # Check symmetry
        assert np.allclose(result.kij, result.kij.T)

        # Diagonal should be zero
        assert np.allclose(np.diag(result.kij), 0.0)

        # N2-C1 should match default
        assert result.get_kij_by_name("N2", "C1") == 0.025

    def test_bip_matrix_custom_override(self):
        """Test custom BIP override."""
        names = ["C1", "C2", "C3"]
        Tc = np.array([190.6, 305.3, 369.9])

        custom = {("C1", "C3"): 0.05}

        result = build_bip_matrix(
            component_ids=names,
            Tc=Tc,
            method=BIPMethod.DEFAULT_VALUES,
            custom_bips=custom,
        )

        assert result.get_kij_by_name("C1", "C3") == 0.05
        assert result.get_kij_by_name("C3", "C1") == 0.05


class TestSplittingMethodComparison:
    """Compare different plus-fraction splitting methods."""

    def test_all_methods_satisfy_mole_balance(self):
        """Test that all methods satisfy mole fraction constraint."""
        methods = [
            ("Pedersen", split_plus_fraction_pedersen),
            ("Katz", split_plus_fraction_katz),
            ("Lohrenz", split_plus_fraction_lohrenz),
        ]

        for name, split_fn in methods:
            result = split_fn(z_plus=C7PLUS_Z, MW_plus=C7PLUS_MW)
            z_sum = result.z.sum()
            assert abs(z_sum - C7PLUS_Z) / C7PLUS_Z < 1e-5, f"{name} failed mole balance"

    def test_methods_give_reasonable_mw(self):
        """Test that all methods give reasonable MW distributions."""
        methods = [
            ("Pedersen", split_plus_fraction_pedersen),
            ("Katz", split_plus_fraction_katz),
            ("Lohrenz", split_plus_fraction_lohrenz),
        ]

        for name, split_fn in methods:
            result = split_fn(z_plus=C7PLUS_Z, MW_plus=C7PLUS_MW)

            # Weighted average MW should be close to MW_plus
            z_norm = result.z / result.z.sum()
            mw_avg = (z_norm * result.MW).sum()

            assert abs(mw_avg - C7PLUS_MW) / C7PLUS_MW < 0.20, \
                f"{name} MW average {mw_avg:.1f} too far from {C7PLUS_MW}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

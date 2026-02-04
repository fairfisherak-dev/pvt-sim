"""Unit tests for PPR78 group contribution k_ij(T) module.

Tests cover:
- Group decomposition from built-in database
- Group decomposition from explicit groups dict
- PPR78 k_ij calculation at reference temperature (298.15 K)
- Temperature dependence of k_ij
- Symmetry property (k_ij = k_ji)
- Diagonal property (k_ii = 0)
"""

import numpy as np
import pytest

from pvtcore.eos.groups import (
    PPR78Group,
    GroupDecomposer,
    parse_group_name,
    get_n_alkane_groups,
    BUILTIN_GROUPS,
)
from pvtcore.eos.ppr78 import (
    PPR78Calculator,
    PPR78_T_REF,
    StaticBIPProvider,
)


class TestGroupDefinitions:
    """Tests for PPR78 group definitions."""

    def test_parse_group_name_standard(self):
        """Standard group names should parse correctly."""
        assert parse_group_name("CH3") == PPR78Group.CH3
        assert parse_group_name("CH2") == PPR78Group.CH2
        assert parse_group_name("CO2") == PPR78Group.CO2
        assert parse_group_name("N2") == PPR78Group.N2

    def test_parse_group_name_case_insensitive(self):
        """Group names should be case-insensitive."""
        assert parse_group_name("ch3") == PPR78Group.CH3
        assert parse_group_name("Ch3") == PPR78Group.CH3
        assert parse_group_name("n2") == PPR78Group.N2

    def test_parse_group_name_aliases(self):
        """Common aliases should be recognized."""
        assert parse_group_name("ACH") == PPR78Group.CHaro
        assert parse_group_name("AC") == PPR78Group.Caro

    def test_parse_group_name_invalid(self):
        """Invalid group names should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown PPR78 group"):
            parse_group_name("INVALID_GROUP")


class TestGroupDecomposer:
    """Tests for GroupDecomposer class."""

    @pytest.fixture
    def decomposer(self):
        """Create a GroupDecomposer instance."""
        return GroupDecomposer(use_rdkit=False)

    def test_decompose_methane(self, decomposer):
        """Methane should decompose to CH4 group."""
        groups = decomposer.decompose(component_id="C1")
        assert groups == {PPR78Group.CH4: 1}

    def test_decompose_ethane(self, decomposer):
        """Ethane should decompose to CH3: 2 (structural basis for consistency)."""
        groups = decomposer.decompose(component_id="C2")
        assert groups == {PPR78Group.CH3: 2}

    def test_decompose_propane(self, decomposer):
        """Propane: 2 CH3 + 1 CH2."""
        groups = decomposer.decompose(component_id="C3")
        assert groups == {PPR78Group.CH3: 2, PPR78Group.CH2: 1}

    def test_decompose_isobutane(self, decomposer):
        """Isobutane: 3 CH3 + 1 CH."""
        groups = decomposer.decompose(component_id="iC4")
        assert groups == {PPR78Group.CH3: 3, PPR78Group.CH: 1}

    def test_decompose_n_butane(self, decomposer):
        """n-Butane: 2 CH3 + 2 CH2."""
        groups = decomposer.decompose(component_id="C4")
        assert groups == {PPR78Group.CH3: 2, PPR78Group.CH2: 2}

    def test_decompose_co2(self, decomposer):
        """CO2 as whole-molecule group."""
        groups = decomposer.decompose(component_id="CO2")
        assert groups == {PPR78Group.CO2: 1}

    def test_decompose_n2(self, decomposer):
        """N2 as whole-molecule group."""
        groups = decomposer.decompose(component_id="N2")
        assert groups == {PPR78Group.N2: 1}

    def test_decompose_h2s(self, decomposer):
        """H2S as whole-molecule group."""
        groups = decomposer.decompose(component_id="H2S")
        assert groups == {PPR78Group.H2S: 1}

    def test_decompose_benzene(self, decomposer):
        """Benzene: 6 aromatic CH."""
        groups = decomposer.decompose(component_id="BENZENE")
        assert groups == {PPR78Group.CHaro: 6}

    def test_decompose_cyclohexane(self, decomposer):
        """Cyclohexane: 6 cyclic CH2."""
        groups = decomposer.decompose(component_id="CYCLOHEXANE")
        assert groups == {PPR78Group.CH2_cyclic: 6}

    def test_decompose_from_groups_dict(self, decomposer):
        """Should accept explicit groups dictionary."""
        groups = decomposer.decompose(groups_dict={"CH3": 2, "CH2": 4})
        assert groups == {PPR78Group.CH3: 2, PPR78Group.CH2: 4}

    def test_decompose_missing_component(self, decomposer):
        """Unknown component without groups should raise ValueError."""
        with pytest.raises(ValueError):
            decomposer.decompose(component_id="UNKNOWN_COMPONENT")


class TestGetNAlkaneGroups:
    """Tests for n-alkane group generation."""

    def test_methane(self):
        """C1 -> {CH4: 1}."""
        assert get_n_alkane_groups(1) == {PPR78Group.CH4: 1}

    def test_ethane(self):
        """C2 -> {C2H6: 1}."""
        assert get_n_alkane_groups(2) == {PPR78Group.C2H6: 1}

    def test_propane(self):
        """C3 -> {CH3: 2, CH2: 1}."""
        assert get_n_alkane_groups(3) == {PPR78Group.CH3: 2, PPR78Group.CH2: 1}

    def test_decane(self):
        """C10 -> {CH3: 2, CH2: 8}."""
        assert get_n_alkane_groups(10) == {PPR78Group.CH3: 2, PPR78Group.CH2: 8}

    def test_invalid_carbon_number(self):
        """Carbon number < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            get_n_alkane_groups(0)


class TestPPR78Calculator:
    """Tests for PPR78Calculator class."""

    @pytest.fixture
    def calculator(self):
        """Create and configure a PPR78Calculator."""
        calc = PPR78Calculator(use_rdkit=False)
        # Register common components
        calc.register_component("C1")
        calc.register_component("C2")
        calc.register_component("C3")
        calc.register_component("C4")
        calc.register_component("CO2")
        calc.register_component("N2")
        calc.register_component("H2S")
        return calc

    def test_register_component(self, calculator):
        """Components should be registered successfully."""
        assert calculator.n_components == 7
        assert "C1" in calculator.component_ids
        assert "CO2" in calculator.component_ids

    def test_diagonal_zero(self, calculator):
        """k_ii should always be zero."""
        for comp in calculator.component_ids:
            kij = calculator.calculate_kij(comp, comp, 300.0)
            assert kij == 0.0

    def test_symmetry(self, calculator):
        """k_ij should equal k_ji."""
        pairs = [("C1", "CO2"), ("C1", "N2"), ("C2", "H2S"), ("C3", "C4")]
        for comp_i, comp_j in pairs:
            kij = calculator.calculate_kij(comp_i, comp_j, 300.0)
            kji = calculator.calculate_kij(comp_j, comp_i, 300.0)
            assert kij == pytest.approx(kji, rel=1e-10)

    def test_temperature_dependence(self, calculator):
        """k_ij should vary with temperature."""
        # Use C1-C3 pair which has more moderate k_ij values
        kij_200 = calculator.calculate_kij("C1", "C3", 200.0)
        kij_300 = calculator.calculate_kij("C1", "C3", 300.0)
        kij_400 = calculator.calculate_kij("C1", "C3", 400.0)

        # Should be different at different temperatures
        assert not np.isclose(kij_200, kij_300, rtol=0.001)
        assert not np.isclose(kij_300, kij_400, rtol=0.001)

    def test_reasonable_kij_range(self, calculator):
        """k_ij values should be in reasonable range [-0.5, 0.5]."""
        pairs = [("C1", "CO2"), ("C1", "N2"), ("C1", "H2S"), ("C1", "C3")]
        for comp_i, comp_j in pairs:
            kij = calculator.calculate_kij(comp_i, comp_j, 300.0)
            assert -0.5 <= kij <= 0.5

    def test_get_kij_matrix(self, calculator):
        """Matrix should be symmetric with zero diagonal."""
        kij = calculator.get_kij_matrix(300.0)

        n = calculator.n_components
        assert kij.shape == (n, n)

        # Diagonal should be zero
        assert np.allclose(np.diag(kij), 0.0)

        # Should be symmetric
        assert np.allclose(kij, kij.T)

    def test_kij_by_index(self, calculator):
        """get_kij should match calculate_kij."""
        i = calculator.component_ids.index("C1")
        j = calculator.component_ids.index("CO2")

        kij_by_index = calculator.get_kij(i, j, 300.0)
        kij_by_name = calculator.calculate_kij("C1", "CO2", 300.0)

        assert kij_by_index == pytest.approx(kij_by_name, rel=1e-10)

    def test_unregistered_component_error(self, calculator):
        """Using unregistered component should raise ValueError."""
        with pytest.raises(ValueError, match="not registered"):
            calculator.calculate_kij("C1", "UNKNOWN", 300.0)


class TestStaticBIPProvider:
    """Tests for StaticBIPProvider class."""

    def test_get_kij(self):
        """Should return values from static matrix."""
        kij_matrix = np.array([
            [0.0, 0.1, 0.2],
            [0.1, 0.0, 0.15],
            [0.2, 0.15, 0.0],
        ])
        provider = StaticBIPProvider(kij_matrix)

        # Temperature should be ignored
        assert provider.get_kij(0, 1, 300.0) == pytest.approx(0.1)
        assert provider.get_kij(0, 1, 400.0) == pytest.approx(0.1)
        assert provider.get_kij(1, 2, 300.0) == pytest.approx(0.15)

    def test_get_kij_matrix(self):
        """Should return a copy of the static matrix."""
        kij_matrix = np.array([
            [0.0, 0.1],
            [0.1, 0.0],
        ])
        provider = StaticBIPProvider(kij_matrix)

        result = provider.get_kij_matrix(300.0)
        assert np.allclose(result, kij_matrix)

        # Should be a copy
        result[0, 1] = 999.0
        assert provider.get_kij(0, 1, 300.0) == pytest.approx(0.1)


class TestComponentDatabaseIntegration:
    """Tests for PPR78 with component database."""

    def test_load_components_with_groups(self):
        """Components should load with group decompositions."""
        from pvtcore.models import load_components

        components = load_components()

        # Check a few components have groups
        assert components["C1"].groups == {"CH4": 1}
        assert components["C3"].groups == {"CH3": 2, "CH2": 1}
        assert components["BENZENE"].groups == {"CHaro": 6}

    def test_ppr78_with_database_components(self):
        """PPR78 should work with database component groups."""
        from pvtcore.models import load_components

        components = load_components()
        calc = PPR78Calculator(use_rdkit=False)

        # Register components using their stored groups
        for comp_id in ["C1", "C3", "CO2"]:
            comp = components[comp_id]
            calc.register_component(comp_id, groups=comp.groups)

        # Calculate k_ij
        kij = calc.calculate_kij("C1", "CO2", 300.0)
        assert -0.5 <= kij <= 0.5

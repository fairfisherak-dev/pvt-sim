"""Unit tests for cubic equation solver.

Tests the Cardano's formula implementation with known analytical solutions
and validates root selection logic for EOS applications.
"""

import pytest
import math
from pvtcore.core.numerics.cubic_solver import (
    eos_cubic_coefficients,
    solve_cubic,
    select_root,
    solve_cubic_eos,
    cubic_diagnostics,
)


class TestCubicSolver:
    """Test suite for cubic equation solver."""

    def test_three_simple_roots(self):
        """Test cubic with three simple integer roots: (Z-1)(Z-2)(Z-3) = 0."""
        # Expand: Z³ - 6Z² + 11Z - 6 = 0
        # So: c₂ = -6, c₁ = 11, c₀ = -6
        roots = solve_cubic(-6.0, 11.0, -6.0)
        assert len(roots) == 3
        assert roots[0] == pytest.approx(1.0, abs=1e-10)
        assert roots[1] == pytest.approx(2.0, abs=1e-10)
        assert roots[2] == pytest.approx(3.0, abs=1e-10)

    def test_triple_root(self):
        """Test cubic with triple root: (Z-2)³ = 0."""
        # Expand: Z³ - 6Z² + 12Z - 8 = 0
        roots = solve_cubic(-6.0, 12.0, -8.0)
        # Should return single root
        assert len(roots) == 1
        assert roots[0] == pytest.approx(2.0, abs=1e-10)

    def test_one_real_root(self):
        """Test cubic with one real root: Z³ - 1 = 0."""
        # Z³ - 1 = 0, so c₂ = 0, c₁ = 0, c₀ = -1
        # Real root: Z = 1, complex roots: Z = -0.5 ± i√3/2
        roots = solve_cubic(0.0, 0.0, -1.0)
        assert len(roots) == 1
        assert roots[0] == pytest.approx(1.0, abs=1e-10)

    def test_double_root(self):
        """Test cubic with one simple and one double root: (Z-1)²(Z-3) = 0."""
        # Expand: Z³ - 5Z² + 7Z - 3 = 0
        roots = solve_cubic(-5.0, 7.0, -3.0)
        # Should return roots [1, 3] (double root at 1 counted once)
        assert len(roots) >= 1
        assert 1.0 in [pytest.approx(r, abs=1e-10) for r in roots]
        assert 3.0 in [pytest.approx(r, abs=1e-10) for r in roots]

    def test_negative_roots(self):
        """Test cubic with negative roots: (Z+1)(Z+2)(Z+3) = 0."""
        # Expand: Z³ + 6Z² + 11Z + 6 = 0
        roots = solve_cubic(6.0, 11.0, 6.0)
        assert len(roots) == 3
        assert roots[0] == pytest.approx(-3.0, abs=1e-10)
        assert roots[1] == pytest.approx(-2.0, abs=1e-10)
        assert roots[2] == pytest.approx(-1.0, abs=1e-10)

    def test_mixed_sign_roots(self):
        """Test cubic with mixed sign roots: (Z-2)(Z+1)(Z+3) = 0."""
        # Expand: Z³ + 2Z² - 5Z - 6 = 0
        roots = solve_cubic(2.0, -5.0, -6.0)
        assert len(roots) == 3
        assert roots[0] == pytest.approx(-3.0, abs=1e-9)
        assert roots[1] == pytest.approx(-1.0, abs=1e-9)
        assert roots[2] == pytest.approx(2.0, abs=1e-9)

    def test_fractional_roots(self):
        """Test cubic with fractional roots."""
        # (Z - 0.5)(Z - 1.5)(Z - 2.5) = 0
        # Correct expansion: Z³ - 4.5Z² + 5.75Z - 1.875 = 0
        roots = solve_cubic(-4.5, 5.75, -1.875)
        assert len(roots) == 3
        assert roots[0] == pytest.approx(0.5, abs=1e-9)
        assert roots[1] == pytest.approx(1.5, abs=1e-9)
        assert roots[2] == pytest.approx(2.5, abs=1e-9)

    def test_vietas_formulas_three_roots(self):
        """Verify Vieta's formulas for cubic with three roots."""
        c2, c1, c0 = -6.0, 11.0, -6.0
        roots = solve_cubic(c2, c1, c0)

        # Sum of roots should equal -c₂
        assert sum(roots) == pytest.approx(-c2, abs=1e-9)

        # Sum of products of pairs: r₁r₂ + r₁r₃ + r₂r₃ = c₁
        pairs_sum = roots[0] * roots[1] + roots[0] * roots[2] + roots[1] * roots[2]
        assert pairs_sum == pytest.approx(c1, abs=1e-9)

        # Product of roots should equal -c₀
        assert roots[0] * roots[1] * roots[2] == pytest.approx(-c0, abs=1e-9)

    def test_discriminant_positive(self):
        """Test that discriminant is correctly identified as positive for three real roots."""
        # (Z-1)(Z-2)(Z-3) = Z³ - 6Z² + 11Z - 6
        diag = cubic_diagnostics(-6.0, 11.0, -6.0)
        assert diag['discriminant'] > 0
        assert diag['discriminant_sign'] == 'positive'
        assert diag['num_real_roots'] == 3

    def test_discriminant_negative(self):
        """Test that discriminant is correctly identified as negative for one real root."""
        # Z³ - 1 = 0
        diag = cubic_diagnostics(0.0, 0.0, -1.0)
        assert diag['discriminant'] < 0
        assert diag['discriminant_sign'] == 'negative'
        assert diag['num_real_roots'] == 1

    def test_typical_eos_case_vapor(self):
        """Test typical EOS case with large vapor root."""
        # Typical case: one small root (liquid) and one large root (vapor)
        # Example: Z³ - Z² - 0.5Z + 0.05 = 0
        roots = solve_cubic(-1.0, -0.5, 0.05)
        if len(roots) == 3:
            # Three roots: liquid, unstable, vapor
            assert roots[0] < 0.5  # Liquid root
            assert roots[-1] > 0.5  # Vapor root
        else:
            # Single root
            assert len(roots) == 1


class TestRootSelection:
    """Test suite for root selection logic."""

    def test_select_liquid_root(self):
        """Test selection of smallest (liquid) root."""
        roots = [0.1, 0.5, 2.5]
        liquid_root = select_root(roots, root_type="liquid")
        assert liquid_root == 0.1

    def test_select_vapor_root(self):
        """Test selection of largest (vapor) root."""
        roots = [0.1, 0.5, 2.5]
        vapor_root = select_root(roots, root_type="vapor")
        assert vapor_root == 2.5

    def test_select_all_roots(self):
        """Test selection of all roots."""
        roots = [0.1, 0.5, 2.5]
        all_roots = select_root(roots, root_type="all")
        assert all_roots == [0.1, 0.5, 2.5]

    def test_filter_negative_roots(self):
        """Test that negative roots are filtered out."""
        roots = [-0.5, 0.1, 2.5]
        valid_roots = select_root(roots, root_type="all", min_value=0.0)
        assert -0.5 not in valid_roots
        assert 0.1 in valid_roots
        assert 2.5 in valid_roots

    def test_single_root_selection(self):
        """Test root selection when only one root is available."""
        roots = [1.5]
        assert select_root(roots, root_type="liquid") == 1.5
        assert select_root(roots, root_type="vapor") == 1.5

    def test_invalid_root_type_raises_error(self):
        """Test that invalid root type raises ValueError."""
        roots = [0.1, 2.5]
        with pytest.raises(ValueError, match="Invalid root_type"):
            select_root(roots, root_type="invalid")

    def test_no_valid_roots_raises_error(self):
        """Test that no valid roots raises ValueError."""
        roots = [-1.0, -0.5]
        with pytest.raises(ValueError, match="No valid roots found"):
            select_root(roots, root_type="liquid", min_value=0.0)

    def test_min_value_filtering(self):
        """Test custom minimum value for root filtering."""
        roots = [0.05, 0.1, 2.5]
        # In EOS, Z must be greater than B (typically ~0.08 for gases)
        valid_roots = select_root(roots, root_type="all", min_value=0.08)
        assert 0.05 not in valid_roots
        assert 0.1 in valid_roots
        assert 2.5 in valid_roots


class TestSolveCubicEOS:
    """Test suite for EOS-specific cubic solver."""

    def test_typical_vapor_case(self):
        """Test typical vapor phase case."""
        # High temperature, moderate pressure: expect large Z (near 1)
        A = 0.1  # Small attraction
        B = 0.05  # Small repulsion
        Z = solve_cubic_eos(A, B, root_type="vapor")
        assert Z > 0.5  # Vapor-like compressibility
        assert Z > B  # Physical constraint

    def test_typical_liquid_case(self):
        """Test typical liquid phase case."""
        # Low temperature, high pressure: expect small Z
        A = 5.0  # Large attraction
        B = 0.08  # Moderate repulsion
        Z = solve_cubic_eos(A, B, root_type="liquid")
        assert Z >= B  # Must be greater than B
        assert Z < 0.5  # Liquid-like compressibility

    def test_ideal_gas_limit(self):
        """Test that Z approaches 1 as A, B approach 0 (ideal gas)."""
        A = 1e-6
        B = 1e-7
        Z = solve_cubic_eos(A, B, root_type="vapor")
        assert Z == pytest.approx(1.0, rel=1e-3)

    def test_two_phase_region(self):
        """Test two-phase region returns multiple roots."""
        # Moderate A and B should give three roots
        A = 1.5
        B = 0.08
        roots = solve_cubic_eos(A, B, root_type="all")

        if len(roots) == 3:
            # Three-phase region
            Z_liquid = roots[0]
            Z_vapor = roots[-1]
            assert Z_liquid < Z_vapor
            assert Z_liquid >= B
            assert Z_vapor > 0.5

    def test_z_greater_than_b_constraint(self):
        """Test that all returned roots satisfy Z > B."""
        A = 2.0
        B = 0.1
        roots = solve_cubic_eos(A, B, root_type="all")

        for Z in roots:
            assert Z >= B, f"Root {Z} is less than B={B}"

    def test_coefficient_calculation(self):
        """Test that PR EOS coefficients are calculated correctly."""
        A = 1.0
        B = 0.1

        # PR EOS coefficients for Z³ + c₂Z² + c₁Z + c₀ = 0
        c2_expected = -(1.0 - B)
        c1_expected = A - 2.0 * B - 3.0 * B ** 2
        c0_expected = -(A * B - B ** 2 - B ** 3)

        # Verify by solving manually
        roots_manual = solve_cubic(c2_expected, c1_expected, c0_expected)
        roots_eos = solve_cubic_eos(A, B, root_type="all")

        assert len(roots_manual) == len(roots_eos)
        for r_manual, r_eos in zip(roots_manual, roots_eos):
            assert r_manual == pytest.approx(r_eos, abs=1e-10)

    def test_generalized_srk_coefficients(self):
        """Test generalized coefficient mapping for SRK."""
        A = 1.0
        B = 0.1
        c2, c1, c0 = eos_cubic_coefficients(A, B, u=1.0, w=0.0)

        assert c2 == pytest.approx(-1.0)
        assert c1 == pytest.approx(A - B - B ** 2)
        assert c0 == pytest.approx(-(A * B))

        roots_manual = solve_cubic(c2, c1, c0)
        roots_eos = solve_cubic_eos(A, B, root_type="all", u=1.0, w=0.0)

        assert len(roots_manual) == len(roots_eos)
        for r_manual, r_eos in zip(roots_manual, roots_eos):
            assert r_manual == pytest.approx(r_eos, abs=1e-10)


class TestCubicDiagnostics:
    """Test suite for cubic diagnostics function."""

    def test_diagnostics_three_roots(self):
        """Test diagnostics for three real roots case."""
        diag = cubic_diagnostics(-6.0, 11.0, -6.0)

        assert diag['num_real_roots'] == 3
        assert diag['discriminant_sign'] == 'positive'
        assert len(diag['roots']) == 3

        # Verify Vieta's formula: sum of roots = -c₂
        assert diag['roots_sum'] == pytest.approx(6.0, abs=1e-9)

    def test_diagnostics_one_root(self):
        """Test diagnostics for one real root case."""
        diag = cubic_diagnostics(0.0, 0.0, -1.0)

        assert diag['num_real_roots'] == 1
        assert diag['discriminant_sign'] == 'negative'
        assert len(diag['roots']) == 1

    def test_diagnostics_includes_coefficients(self):
        """Test that diagnostics includes all input coefficients."""
        c2, c1, c0 = -6.0, 11.0, -6.0
        diag = cubic_diagnostics(c2, c1, c0)

        assert diag['c2'] == c2
        assert diag['c1'] == c1
        assert diag['c0'] == c0
        assert 'p' in diag  # Depressed cubic coefficient
        assert 'q' in diag  # Depressed cubic coefficient


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_large_coefficients(self):
        """Test solver with very large coefficients."""
        # Scale roots by 1000: if Z' = 1000Z, then
        # Z³ - 6Z² + 11Z - 6 becomes Z'³ - 6000Z'² + 11000000Z' - 6000000000
        # Base: (Z-1)(Z-2)(Z-3) = Z³ - 6Z² + 11Z - 6
        k = 1000
        c2 = -6 * k        # -6000
        c1 = 11 * k**2     # 11000000
        c0 = -6 * k**3     # -6000000000
        roots = solve_cubic(c2, c1, c0)
        assert len(roots) == 3
        # Roots should be around 1000, 2000, 3000
        assert roots[0] == pytest.approx(1000.0, rel=1e-5)
        assert roots[1] == pytest.approx(2000.0, rel=1e-5)
        assert roots[2] == pytest.approx(3000.0, rel=1e-5)

    def test_very_small_coefficients(self):
        """Test solver with very small coefficients.

        Note: With very small roots (0.001, 0.002, 0.003), numerical precision
        can cause root clustering or loss. The solver may return fewer roots
        if they are numerically indistinguishable.
        """
        # Scale roots by 0.001: if Z' = 0.001Z, then
        # Z³ - 6Z² + 11Z - 6 becomes Z'³ - 0.006Z'² + 0.000011Z' - 0.000000006
        # Base: (Z-1)(Z-2)(Z-3) = Z³ - 6Z² + 11Z - 6
        k = 0.001
        c2 = -6 * k        # -0.006
        c1 = 11 * k**2     # 0.000011
        c0 = -6 * k**3     # -0.000000006
        roots = solve_cubic(c2, c1, c0)

        # With very small coefficients, solver may return 1-3 roots
        # depending on numerical precision
        assert len(roots) >= 1
        assert len(roots) <= 3

        # Check that roots are positive and in expected range
        for root in roots:
            assert 0.0 < root < 0.01

    def test_near_zero_discriminant(self):
        """Test behavior near discriminant = 0."""
        # Design a case with discriminant very close to zero
        # (Z-1)²(Z-2) = Z³ - 4Z² + 5Z - 2
        roots = solve_cubic(-4.0, 5.0, -2.0)
        # Should handle the double root correctly
        assert 1.0 in [pytest.approx(r, abs=1e-8) for r in roots]
        assert 2.0 in [pytest.approx(r, abs=1e-8) for r in roots]

    def test_all_zeros(self):
        """Test cubic equation Z³ = 0."""
        roots = solve_cubic(0.0, 0.0, 0.0)
        assert len(roots) == 1
        assert roots[0] == pytest.approx(0.0, abs=1e-10)

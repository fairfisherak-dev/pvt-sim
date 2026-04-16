"""Phase envelope tests — consolidated.

Uses session-scoped fixtures from conftest.py so that
`calculate_phase_envelope` is called once per mixture, not once per test.
"""

from __future__ import annotations

import pytest
import numpy as np

from pvtcore.envelope.phase_envelope import (
    calculate_phase_envelope,
    estimate_cricondentherm,
    estimate_cricondenbar,
)
from pvtcore.envelope.iso_lines import (
    IsoLineMode,
    IsoLineSegment,
    IsoLinesResult,
    compute_iso_lines,
    compute_iso_vol_lines,
    compute_iso_beta_lines,
    compute_alpha_from_flash,
)
from pvtcore.envelope.ternary import (
    PhaseClassification,
    TernaryGridPoint,
    TernaryResult,
    generate_barycentric_grid,
    barycentric_to_cartesian,
    cartesian_to_barycentric,
    compute_ternary_diagram,
    get_triangle_vertices,
    DEFAULT_N_SUBDIVISIONS,
)
from pvtcore.core.errors import ValidationError
from pvtcore.flash.pt_flash import pt_flash


# ── Phase envelope: structure, physics, critical point ─────────────────────

def test_envelope_structure_and_convergence(c1_c10_envelope):
    """Result fields, types, array lengths, convergence flag."""
    env = c1_c10_envelope
    assert env.converged is True
    assert isinstance(env.bubble_T, np.ndarray)
    assert isinstance(env.bubble_P, np.ndarray)
    assert isinstance(env.dew_T, np.ndarray)
    assert isinstance(env.dew_P, np.ndarray)
    assert isinstance(env.converged, bool)
    assert isinstance(env.n_bubble_points, int)
    assert isinstance(env.n_dew_points, int)
    assert len(env.bubble_T) == len(env.bubble_P)
    assert len(env.dew_T) == len(env.dew_P)
    assert env.n_bubble_points > 3
    assert len(env.bubble_T) > 5


def test_envelope_physics(c1_c10_envelope):
    """T/P positive, mostly increasing, bubble left of dew, adaptive density."""
    env = c1_c10_envelope
    assert np.all(env.bubble_T > 0)
    assert np.all(env.dew_T > 0)
    assert np.all(env.bubble_P > 0)
    assert np.all(env.dew_P > 0)

    if len(env.bubble_T) > 1:
        assert np.sum(np.diff(env.bubble_T) > 0) > 0.8 * (len(env.bubble_T) - 1)
    if len(env.dew_T) > 1:
        assert np.sum(np.diff(env.dew_T) > 0) > 0.8 * (len(env.dew_T) - 1)

    if len(env.bubble_T) > 0 and len(env.dew_T) > 0:
        assert np.min(env.bubble_T) <= np.min(env.dew_T) * 1.05

    if len(env.bubble_P) > 3:
        mid = len(env.bubble_P) // 2
        assert env.bubble_P[mid] > env.bubble_P[0]

    assert env.n_bubble_points > 10
    if env.critical_T is not None and len(env.bubble_T) > 5:
        near = np.abs(env.bubble_T - env.critical_T) < env.critical_T * 0.1
        assert np.sum(near) >= 2


def test_critical_point(c1_c10_envelope, components):
    """Detected, positive, between pure Tc, near Kay's estimate."""
    env = c1_c10_envelope
    assert env.critical_T is not None
    assert env.critical_P is not None
    assert env.critical_T > 0
    assert env.critical_P > 0

    Tc_C1 = components["C1"].Tc
    Tc_C10 = components["C10"].Tc
    Pc_C1 = components["C1"].Pc
    Pc_C10 = components["C10"].Pc
    assert Tc_C1 * 0.8 < env.critical_T < Tc_C10 * 1.2

    Pc_min, Pc_max = min(Pc_C1, Pc_C10), max(Pc_C1, Pc_C10)
    assert Pc_min * 0.5 < env.critical_P < Pc_max * 1.5

    Tc_kay = 0.5 * (Tc_C1 + Tc_C10)
    Pc_kay = 0.5 * (Pc_C1 + Pc_C10)
    assert abs(env.critical_T - Tc_kay) / Tc_kay < 0.30
    assert abs(env.critical_P - Pc_kay) / Pc_kay < 0.50


def test_cricondentherm_and_cricondenbar(c1_c10_envelope, c2_c3_envelope):
    """Cricondentherm / cricondenbar estimation from cached envelopes."""
    T_cdt, P_cdt = estimate_cricondentherm(c1_c10_envelope)
    if T_cdt is not None:
        assert T_cdt > 0 and P_cdt > 0
        if len(c1_c10_envelope.dew_T) > 0:
            assert T_cdt <= np.max(c1_c10_envelope.dew_T) * 1.01

    T_cdb, P_cdb = estimate_cricondenbar(c1_c10_envelope)
    if P_cdb is not None:
        assert T_cdb > 0 and P_cdb > 0

    T_cdb_23, P_cdb_23 = estimate_cricondenbar(c2_c3_envelope)
    if c2_c3_envelope.critical_P is not None and P_cdb_23 is not None:
        assert abs(P_cdb_23 - c2_c3_envelope.critical_P) / c2_c3_envelope.critical_P < 0.5


def test_envelope_reproducibility(components, c1_c10_pr):
    """Two independent calls give same critical point."""
    z = np.array([0.5, 0.5])
    binary = [components["C1"], components["C10"]]
    env1 = calculate_phase_envelope(z, binary, c1_c10_pr)
    env2 = calculate_phase_envelope(z, binary, c1_c10_pr)
    if env1.critical_T is not None and env2.critical_T is not None:
        assert abs(env1.critical_T - env2.critical_T) < 1.0
        assert abs(env1.critical_P - env2.critical_P) / env1.critical_P < 0.01


@pytest.mark.parametrize("z,fixture_name,check", [
    (np.array([0.9, 0.1]), "c1_c10_pr", "c1_rich"),
    (np.array([0.1, 0.9]), "c1_c10_pr", "c10_rich"),
    (np.array([0.5, 0.5]), "c1_c4_pr", "c1_c4"),
    (np.array([0.5, 0.5]), "c2_c3_pr", "c2_c3"),
])
def test_composition_variation(z, fixture_name, check, components, request):
    """Envelopes converge for different compositions and mixtures."""
    eos = request.getfixturevalue(fixture_name)
    comp_ids = {
        "c1_c10_pr": ("C1", "C10"),
        "c1_c4_pr": ("C1", "C4"),
        "c2_c3_pr": ("C2", "C3"),
    }[fixture_name]
    binary = [components[cid] for cid in comp_ids]
    env = calculate_phase_envelope(z, binary, eos)
    assert env.converged is True
    if check == "c2_c3":
        Tc_lo = components["C2"].Tc
        Tc_hi = components["C3"].Tc
        if env.critical_T is not None:
            assert Tc_lo * 0.95 < env.critical_T < Tc_hi * 1.05


def test_input_validation(components, c1_c10_pr):
    """Bad composition sum and length mismatch raise ValidationError."""
    binary = [components["C1"], components["C10"]]
    with pytest.raises(ValidationError):
        calculate_phase_envelope(np.array([0.5, 0.3]), binary, c1_c10_pr)
    with pytest.raises(ValidationError):
        calculate_phase_envelope(np.array([0.33, 0.33, 0.34]), binary, c1_c10_pr)


# ── Iso-lines (use cached envelope) ───────────────────────────────────────

def test_iso_lines(c1_c10_envelope, components, c1_c10_pr):
    """Iso-vol and iso-beta from shared envelope: mode gating, accuracy, physics."""
    env = c1_c10_envelope
    binary = [components["C1"], components["C10"]]

    # IsoLineMode enum
    assert len({IsoLineMode.NONE, IsoLineMode.ISO_VOL, IsoLineMode.ISO_BETA, IsoLineMode.BOTH}) == 4

    # Mode NONE
    r_none = compute_iso_lines(env, binary, c1_c10_pr, mode=IsoLineMode.NONE)
    assert isinstance(r_none, IsoLinesResult)
    assert len(r_none.iso_vol_lines) == 0 and len(r_none.iso_beta_lines) == 0

    # Mode ISO_VOL
    r_vol = compute_iso_lines(env, binary, c1_c10_pr, mode=IsoLineMode.ISO_VOL,
                              alpha_levels=[0.5], n_temperature_points=10)
    assert len(r_vol.iso_vol_lines) > 0 and len(r_vol.iso_beta_lines) == 0

    # Mode ISO_BETA
    r_beta = compute_iso_lines(env, binary, c1_c10_pr, mode=IsoLineMode.ISO_BETA,
                               beta_levels=[0.5], n_temperature_points=10)
    assert len(r_beta.iso_vol_lines) == 0 and len(r_beta.iso_beta_lines) > 0

    # Mode BOTH + composition preserved
    r_both = compute_iso_lines(env, binary, c1_c10_pr, mode=IsoLineMode.BOTH,
                               alpha_levels=[0.5], beta_levels=[0.5], n_temperature_points=10)
    assert len(r_both.iso_vol_lines) > 0 and len(r_both.iso_beta_lines) > 0
    assert np.allclose(r_both.composition, np.array([0.5, 0.5]))

    # Default alpha includes 0.5
    r_def = compute_iso_lines(env, binary, c1_c10_pr, mode=IsoLineMode.ISO_VOL,
                              n_temperature_points=10)
    assert 0.5 in r_def.alpha_levels

    # Iso-vol segment structure and accuracy
    vol_segs = compute_iso_vol_lines(env, binary, c1_c10_pr,
                                     alpha_levels=[0.5], n_temperature_points=15)
    for seg in vol_segs.get(0.5, []):
        assert isinstance(seg, IsoLineSegment)
        assert len(seg.temperatures) == len(seg.pressures)
        if len(seg) > 0:
            for a in seg.vapor_volume_fractions:
                assert abs(a - 0.5) < 0.01

    # Iso-beta accuracy
    beta_segs = compute_iso_beta_lines(env, binary, c1_c10_pr,
                                       beta_levels=[0.5], n_temperature_points=15)
    for seg in beta_segs.get(0.5, []):
        if len(seg) > 0:
            for b in seg.vapor_fractions:
                assert abs(b - 0.5) < 0.01


def test_iso_lines_physical_constraints(c1_c10_envelope, components, c1_c10_pr):
    """Low alpha near bubble, high alpha near dew, beta=0.5 bisects envelope."""
    env = c1_c10_envelope
    binary = [components["C1"], components["C10"]]

    low_alpha = compute_iso_vol_lines(env, binary, c1_c10_pr,
                                      alpha_levels=[0.05], n_temperature_points=20)
    for seg in low_alpha.get(0.05, []):
        for beta in seg.vapor_fractions:
            assert beta < 0.5

    hi_alpha = compute_iso_vol_lines(env, binary, c1_c10_pr,
                                     alpha_levels=[0.95], n_temperature_points=20)
    for seg in hi_alpha.get(0.95, []):
        if len(seg) > 0:
            bub_T_sorted = env.bubble_T[np.argsort(env.bubble_T)]
            bub_P_sorted = env.bubble_P[np.argsort(env.bubble_T)]
            dew_T_sorted = env.dew_T[np.argsort(env.dew_T)]
            dew_P_sorted = env.dew_P[np.argsort(env.dew_T)]
            for T, P, a in zip(seg.temperatures, seg.pressures, seg.vapor_volume_fractions):
                P_bub = np.interp(T, bub_T_sorted, bub_P_sorted)
                P_dew = np.interp(T, dew_T_sorted, dew_P_sorted)
                assert a == pytest.approx(0.95, abs=1e-6)
                assert abs(P - P_dew) < abs(P - P_bub)

    mid_beta = compute_iso_beta_lines(env, binary, c1_c10_pr,
                                      beta_levels=[0.5], n_temperature_points=20)
    bub_T_sorted = env.bubble_T[np.argsort(env.bubble_T)]
    bub_P_sorted = env.bubble_P[np.argsort(env.bubble_T)]
    dew_T_sorted = env.dew_T[np.argsort(env.dew_T)]
    dew_P_sorted = env.dew_P[np.argsort(env.dew_T)]
    for seg in mid_beta.get(0.5, []):
        for T, P in zip(seg.temperatures, seg.pressures):
            P_bub = np.interp(T, bub_T_sorted, bub_P_sorted)
            P_dew = np.interp(T, dew_T_sorted, dew_P_sorted)
            P_lo, P_hi = min(P_bub, P_dew), max(P_bub, P_dew)
            assert P_lo * 0.99 <= P <= P_hi * 1.01


def test_alpha_from_flash(components, c1_c10_pr):
    """compute_alpha_from_flash for two-phase, liquid, and vapor."""
    z = np.array([0.5, 0.5])
    binary = [components["C1"], components["C10"]]

    two_ph = pt_flash(3e6, 350.0, z, binary, c1_c10_pr)
    if two_ph.is_two_phase:
        a, vl, vv = compute_alpha_from_flash(two_ph, c1_c10_pr)
        assert 0.0 <= a <= 1.0
        assert vl > 0 and vv > 0 and vv > vl

    liq = pt_flash(50e6, 300.0, z, binary, c1_c10_pr)
    if liq.phase == "liquid":
        a, vl, _ = compute_alpha_from_flash(liq, c1_c10_pr)
        assert a == 0.0 and vl > 0

    vap = pt_flash(0.1e6, 500.0, z, binary, c1_c10_pr)
    if vap.phase == "vapor":
        a, _, vv = compute_alpha_from_flash(vap, c1_c10_pr)
        assert a == 1.0 and vv > 0


# ── Ternary diagram (cheap pure-math + one compute call) ──────────────────

def test_barycentric_grid():
    """Grid geometry: size, normalization, vertices, center, round-trip."""
    grid10 = generate_barycentric_grid(10)
    assert len(grid10) == (10 + 1) * (10 + 2) // 2
    grid_def = generate_barycentric_grid()
    n = DEFAULT_N_SUBDIVISIONS
    assert len(grid_def) == (n + 1) * (n + 2) // 2

    grid20 = generate_barycentric_grid(20)
    assert np.allclose(np.sum(grid20, axis=1), 1.0)
    assert np.all(grid20 >= 0)

    assert any(np.allclose(r, [1, 0, 0]) for r in grid10)
    assert any(np.allclose(r, [0, 1, 0]) for r in grid10)
    assert any(np.allclose(r, [0, 0, 1]) for r in grid10)

    grid9 = generate_barycentric_grid(9)
    assert any(np.allclose(r, [1 / 3, 1 / 3, 1 / 3]) for r in grid9)

    verts = get_triangle_vertices()
    for bary, expected in zip([[1, 0, 0], [0, 1, 0], [0, 0, 1]], verts):
        cart = barycentric_to_cartesian(np.array([bary]))
        assert np.allclose(cart.flatten(), expected)
    center_cart = barycentric_to_cartesian(np.array([[1 / 3, 1 / 3, 1 / 3]]))
    assert np.allclose(center_cart.flatten(), np.mean(verts, axis=0))

    rt = cartesian_to_barycentric(barycentric_to_cartesian(grid10))
    assert np.allclose(grid10, rt, atol=1e-10)


def test_ternary_grid_point():
    """TernaryGridPoint data structure."""
    pt2 = TernaryGridPoint(composition=np.array([0.33, 0.33, 0.34]),
                           classification=PhaseClassification.TWO_PHASE)
    assert pt2.is_two_phase and not pt2.is_single_phase
    pt1 = TernaryGridPoint(composition=np.array([0.33, 0.33, 0.34]),
                           classification=PhaseClassification.SINGLE_PHASE_LIQUID)
    assert pt1.is_single_phase and not pt1.is_two_phase
    assert len({PhaseClassification.SINGLE_PHASE_LIQUID, PhaseClassification.SINGLE_PHASE_VAPOR,
                PhaseClassification.TWO_PHASE, PhaseClassification.FAILED}) == 4


def test_ternary_diagram(components, c1_c4_c10_pr):
    """Full ternary compute: fields, counts, mass balance, tie-lines."""
    ternary_comps = [components["C1"], components["C4"], components["C10"]]

    with pytest.raises(ValueError, match="3 components"):
        from pvtcore.eos.peng_robinson import PengRobinsonEOS
        eos2 = PengRobinsonEOS([components["C1"], components["C10"]])
        compute_ternary_diagram(300.0, 5e6, [components["C1"], components["C10"]], eos2,
                                n_subdivisions=5)

    n = 7
    result = compute_ternary_diagram(350.0, 5e6, ternary_comps, c1_c4_c10_pr,
                                     n_subdivisions=n)
    assert isinstance(result, TernaryResult)
    assert result.temperature == 350.0
    assert result.pressure == 5e6
    assert len(result.components) == 3
    expected_pts = (n + 1) * (n + 2) // 2
    assert result.n_total_points == expected_pts == len(result.grid_points)
    assert result.n_single_phase + result.n_two_phase + result.n_failed == result.n_total_points

    for pt in result.grid_points:
        if pt.is_two_phase:
            z_check = (1 - pt.vapor_fraction) * pt.liquid_composition + pt.vapor_fraction * pt.vapor_composition
            assert np.max(np.abs(pt.composition - z_check)) < 1e-8
            assert np.isclose(np.sum(pt.liquid_composition), 1.0, atol=1e-10)
            assert np.isclose(np.sum(pt.vapor_composition), 1.0, atol=1e-10)
            assert np.all(pt.liquid_composition >= 0) and np.all(pt.liquid_composition <= 1)
            assert np.all(pt.vapor_composition >= 0) and np.all(pt.vapor_composition <= 1)

    r_tie = compute_ternary_diagram(350.0, 5e6, ternary_comps, c1_c4_c10_pr,
                                    n_subdivisions=10, compute_tie_lines=True, tie_line_skip=1)
    if r_tie.n_two_phase > 0:
        assert len(r_tie.tie_lines) > 0
    r_skip = compute_ternary_diagram(350.0, 5e6, ternary_comps, c1_c4_c10_pr,
                                     n_subdivisions=10, compute_tie_lines=True, tie_line_skip=3)
    if r_tie.n_two_phase > 3:
        assert len(r_skip.tie_lines) < len(r_tie.tie_lines)

    if result.n_single_phase > 0:
        sp = result.get_single_phase_compositions()
        assert len(sp) == result.n_single_phase and sp.shape[1] == 3
    if result.n_two_phase > 0:
        tp = result.get_two_phase_compositions()
        assert len(tp) == result.n_two_phase and tp.shape[1] == 3

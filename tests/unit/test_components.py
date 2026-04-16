"""Component database validation against NIST reference values.

One parametrized test covers all species/property spot checks.
Structural, trend, and function tests cover the rest.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from pvtcore.models.component import (
    build_component_alias_index,
    Component,
    get_component,
    get_components_cached,
    load_components,
    resolve_component_id,
)


# ---------------------------------------------------------------------------
# NIST reference table: (component_id, attribute, expected, tolerance_kwargs)
# tolerance_kwargs is passed to pytest.approx as **kwargs.
# ---------------------------------------------------------------------------

NIST_REFERENCE = [
    # Nitrogen
    ("N2", "Tc", 126.19, {"rel": 0.001}),
    ("N2", "Pc_MPa", 3.3978, {"rel": 0.001}),
    ("N2", "MW", 28.0134, {"rel": 0.0001}),
    ("N2", "Tb", 77.34, {"rel": 0.001}),
    ("N2", "omega", 0.039, {"abs": 0.002}),
    # Carbon dioxide
    ("CO2", "Tc", 304.18, {"rel": 0.001}),
    ("CO2", "Pc_MPa", 7.38, {"rel": 0.01}),
    ("CO2", "Vc_cm3_per_mol", 91.9, {"rel": 0.01}),
    ("CO2", "MW", 44.0095, {"rel": 0.0001}),
    ("CO2", "omega", 0.239, {"abs": 0.005}),
    # Hydrogen sulfide
    ("H2S", "Tc", 373.3, {"rel": 0.001}),
    ("H2S", "Pc_MPa", 8.96, {"rel": 0.01}),
    ("H2S", "MW", 34.081, {"rel": 0.0001}),
    ("H2S", "Tb", 212.87, {"rel": 0.001}),
    ("H2S", "omega", 0.081, {"abs": 0.005}),
    # Methane
    ("C1", "Tc", 190.6, {"rel": 0.002}),
    ("C1", "Pc_MPa", 4.61, {"rel": 0.01}),
    ("C1", "Vc_cm3_per_mol", 98.52, {"rel": 0.01}),
    ("C1", "MW", 16.0425, {"rel": 0.0001}),
    ("C1", "Tb", 111.0, {"rel": 0.02}),
    ("C1", "omega", 0.011, {"abs": 0.002}),
    # Ethane
    ("C2", "Tc", 305.3, {"rel": 0.001}),
    ("C2", "Pc_MPa", 4.9, {"rel": 0.02}),
    ("C2", "Vc_cm3_per_mol", 147, {"rel": 0.01}),
    ("C2", "MW", 30.069, {"rel": 0.0001}),
    ("C2", "omega", 0.099, {"abs": 0.002}),
    # Propane
    ("C3", "Tc", 369.9, {"rel": 0.001}),
    ("C3", "Pc_MPa", 4.25, {"rel": 0.01}),
    ("C3", "MW", 44.0956, {"rel": 0.0001}),
    ("C3", "omega", 0.153, {"abs": 0.002}),
    # n-Butane
    ("C4", "Tc", 425.0, {"rel": 0.003}),
    ("C4", "Pc_MPa", 3.8, {"rel": 0.01}),
    ("C4", "MW", 58.1222, {"rel": 0.0001}),
    ("C4", "omega", 0.199, {"abs": 0.002}),
    # Isobutane
    ("iC4", "Tc", 407.7, {"rel": 0.002}),
    ("iC4", "Pc_MPa", 3.65, {"rel": 0.015}),
    ("iC4", "omega", 0.183, {"abs": 0.002}),
    # n-Decane
    ("C10", "Tc", 617.8, {"rel": 0.002}),
    ("C10", "Pc_MPa", 2.11, {"rel": 0.04}),
    ("C10", "MW", 142.2817, {"rel": 0.0001}),
    ("C10", "Tb", 447.2, {"rel": 0.001}),
    ("C10", "omega", 0.4884, {"abs": 0.005}),
]


@pytest.mark.parametrize(
    "comp_id,attr,expected,tol",
    NIST_REFERENCE,
    ids=[f"{c[0]}.{c[1]}" for c in NIST_REFERENCE],
)
def test_component_property_vs_nist(components, comp_id, attr, expected, tol):
    """Spot-check individual component properties against NIST reference."""
    assert getattr(components[comp_id], attr) == pytest.approx(expected, **tol)


def test_database_structure(components):
    """DB loads, contains all expected species, correct dataclass shape."""
    assert isinstance(components, dict)
    assert len(components) >= 16

    expected = [
        "N2", "CO2", "H2S",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
        "iC4", "iC5", "neoC5",
    ]
    for cid in expected:
        assert cid in components, f"{cid} missing"

    c1 = components["C1"]
    assert isinstance(c1, Component)
    for attr in ("name", "formula", "Tc", "Pc", "Vc", "omega", "MW", "Tb"):
        assert hasattr(c1, attr)


def test_physical_property_ranges(components):
    """All properties within physically reasonable bounds, Tb < Tc."""
    for cid, c in components.items():
        assert c.Tc > 0, f"{cid} Tc"
        assert c.Pc > 0, f"{cid} Pc"
        assert c.Vc > 0, f"{cid} Vc"
        assert c.MW > 0, f"{cid} MW"
        assert c.Tb > 0, f"{cid} Tb"
        assert -0.5 <= c.omega <= 2.0, f"{cid} omega={c.omega}"
        assert c.Tb < c.Tc, f"{cid} Tb >= Tc"


def test_alkane_trends_and_isomers(components):
    """n-Alkane Tc/MW/ω increase with carbon number; isomer MW equality."""
    alkanes = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]
    for i in range(len(alkanes) - 1):
        a, b = components[alkanes[i]], components[alkanes[i + 1]]
        assert b.Tc > a.Tc, f"Tc: {alkanes[i]} vs {alkanes[i+1]}"
        assert b.MW > a.MW, f"MW: {alkanes[i]} vs {alkanes[i+1]}"
        assert b.omega > a.omega, f"omega: {alkanes[i]} vs {alkanes[i+1]}"

    assert components["C4"].MW == pytest.approx(components["iC4"].MW, rel=1e-10)
    assert components["C5"].MW == pytest.approx(components["iC5"].MW, rel=1e-10)
    assert components["C5"].MW == pytest.approx(components["neoC5"].MW, rel=1e-10)
    assert components["C4"].Tc > components["iC4"].Tc
    assert components["C5"].Tc > components["iC5"].Tc > components["neoC5"].Tc


def test_unit_conversions_and_api(components):
    """Pressure/volume unit conversions, get_component, aliases, caching."""
    c1 = components["C1"]
    assert c1.Pc_bar == pytest.approx(c1.Pc / 1e5, rel=1e-10)
    assert c1.Pc_MPa == pytest.approx(c1.Pc / 1e6, rel=1e-10)
    assert c1.Vc_cm3_per_mol == pytest.approx(c1.Vc * 1e6, rel=1e-10)
    assert c1.Vc_L_per_mol == pytest.approx(c1.Vc * 1e3, rel=1e-10)

    methane = get_component("C1")
    assert methane.name == "Methane"
    assert methane.formula == "CH4"

    n_butane = get_component("nC4")
    assert n_butane.id == "C4"

    with pytest.raises(KeyError):
        get_component("INVALID")

    assert resolve_component_id("nC4", components) == "C4"
    assert resolve_component_id("methane", components) == "C1"
    assert resolve_component_id("CH4", components) == "C1"

    alias_index = build_component_alias_index()
    assert alias_index["nc4"] == "C4"
    assert alias_index["methane"] == "C1"

    c1a = get_components_cached()
    c1b = get_components_cached()
    assert c1a is c1b

    assert "Methane" in repr(c1)
    assert "Methane" in str(c1)


def test_database_file_integrity():
    """JSON file exists and loads from explicit path."""
    project_root = Path(__file__).resolve().parents[2]
    json_path = project_root / "data" / "pure_components" / "components.json"
    assert json_path.exists()
    comps = load_components(json_path)
    assert len(comps) >= 16

    with pytest.raises(FileNotFoundError):
        load_components(Path("/nonexistent/path/components.json"))

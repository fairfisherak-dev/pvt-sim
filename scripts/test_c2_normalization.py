"""Test C2 (ethane) normalization to structural basis.

Verify that C2 now uses CH3: 2 instead of C2H6: 1 and that
k_ij calculations still work correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pvtcore.models.component import load_components
from pvtcore.eos.groups.decomposition import GroupDecomposer
from pvtcore.eos.ppr78 import PPR78Calculator


def main():
    print("=" * 60)
    print("C2 (Ethane) Normalization Test")
    print("=" * 60)

    # Load components
    components = load_components()
    c2 = components["C2"]

    # Check groups in JSON
    print("\n[1] C2 groups from JSON:")
    print(f"    {c2.groups}")
    if c2.groups == {"CH3": 2}:
        print("    OK - Using structural basis (CH3: 2)")
    elif c2.groups == {"C2H6": 1}:
        print("    WARNING - Still using C2H6 (old basis)")
    else:
        print(f"    ERROR - Unexpected groups: {c2.groups}")

    # Check group decomposition
    print("\n[2] Group decomposition:")
    decomposer = GroupDecomposer(use_rdkit=False)
    groups = decomposer.decompose(component_id="C2")
    print(f"    Decomposed groups: {groups}")

    # Check k_ij calculations with C2
    print("\n[3] Testing k_ij calculations involving C2:")
    calc = PPR78Calculator(use_rdkit=False)

    # Register all components
    for comp_id, comp in components.items():
        calc.register_component(comp_id, groups=comp.groups)

    test_pairs = [
        ("C1", "C2"),   # Methane-Ethane
        ("C2", "C3"),   # Ethane-Propane
        ("C2", "CO2"),  # Ethane-CO2
        ("C2", "N2"),   # Ethane-N2
        ("C2", "H2S"),  # Ethane-H2S
    ]

    print("    k_ij values at T=298.15 K:")
    for comp_i, comp_j in test_pairs:
        try:
            kij = calc.calculate_kij(comp_i, comp_j, 298.15)
            print(f"      k_{{{comp_i}-{comp_j}}} = {kij:8.5f}")
        except Exception as e:
            print(f"      k_{{{comp_i}-{comp_j}}} = ERROR: {e}")

    # Compare with different temperatures
    print("\n[4] Temperature dependence for C1-C2:")
    temps = [200.0, 250.0, 298.15, 350.0, 400.0]
    print("    Temperature (K)   k_ij")
    for T in temps:
        kij = calc.calculate_kij("C1", "C2", T)
        print(f"    {T:>14.2f}   {kij:8.5f}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

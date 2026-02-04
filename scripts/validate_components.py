"""Validate the expanded component database.

This script checks:
1. All components load correctly
2. New fields (InChI, InChIKey, isomer_group) are accessible
3. PPR78 group decomposition works for all components
4. PPR78Calculator can compute k_ij for new components
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pvtcore.models.component import load_components, ComponentFamily
from pvtcore.eos.groups.decomposition import GroupDecomposer
from pvtcore.eos.ppr78 import PPR78Calculator


def main():
    """Run validation checks."""
    print("=" * 60)
    print("Component Database Validation")
    print("=" * 60)

    # Test 1: Load all components
    print("\n[1] Loading components...")
    components = load_components()
    print(f"    Loaded {len(components)} components")

    expected_count = 46
    if len(components) == expected_count:
        print(f"    OK - Expected {expected_count} components")
    else:
        print(f"    WARNING - Expected {expected_count}, got {len(components)}")

    # Test 2: Check new components exist
    print("\n[2] Checking new Tier 1-4 components...")
    new_components = [
        "O2", "MeSH", "EtSH",  # Tier 1
        "CYCLOPENTANE", "MCYCLOPENTANE", "NAPHTHALENE", "CUMENE",  # Tier 2
        "C11", "C12", "C13", "C14", "C15", "C16",  # Tier 3
        "METHANOL", "ETHANOL",  # Tier 4
    ]

    missing = []
    for comp_id in new_components:
        if comp_id not in components:
            missing.append(comp_id)

    if missing:
        print(f"    ERROR - Missing components: {missing}")
    else:
        print(f"    OK - All {len(new_components)} new components present")

    # Test 3: Check extended fields
    print("\n[3] Checking extended fields...")
    test_comp = components["C1"]

    fields_ok = True
    if not hasattr(test_comp, "inchi"):
        print("    ERROR - 'inchi' field missing")
        fields_ok = False
    if not hasattr(test_comp, "inchikey"):
        print("    ERROR - 'inchikey' field missing")
        fields_ok = False
    if not hasattr(test_comp, "isomer_group"):
        print("    ERROR - 'isomer_group' field missing")
        fields_ok = False

    if fields_ok:
        print("    OK - Extended fields present")

    # Test 4: Check families
    print("\n[4] Checking component families...")
    family_counts = {}
    no_family = []

    for comp_id, comp in components.items():
        if comp.family is None:
            no_family.append(comp_id)
        else:
            family_name = comp.family.name
            family_counts[family_name] = family_counts.get(family_name, 0) + 1

    print("    Family distribution:")
    for family, count in sorted(family_counts.items()):
        print(f"      {family}: {count}")

    if no_family:
        print(f"    Components without family: {no_family}")

    # Test 5: Check PPR78 groups
    print("\n[5] Checking PPR78 group decomposition...")
    decomposer = GroupDecomposer(use_rdkit=False)

    failed_decomp = []
    empty_groups = []
    for comp_id in components.keys():
        try:
            groups = decomposer.decompose(component_id=comp_id)
            if groups is not None and len(groups) == 0:
                empty_groups.append(comp_id)
        except Exception as e:
            failed_decomp.append(f"{comp_id} ({e})")

    if failed_decomp:
        print(f"    ERROR - Failed to decompose: {failed_decomp}")
    elif empty_groups:
        print(f"    OK - All {len(components)} components decompose successfully")
        print(f"    NOTE - {len(empty_groups)} components have no PPR78 groups (polar): {empty_groups}")
    else:
        print(f"    OK - All {len(components)} components decompose successfully")

    # Test 6: Check PPR78 k_ij calculation for new components
    print("\n[6] Testing PPR78 k_ij calculation...")
    calc = PPR78Calculator(use_rdkit=False)

    # Register all components
    for comp_id, comp in components.items():
        calc.register_component(comp_id, groups=comp.groups)

    # Test k_ij for some new component pairs
    test_pairs = [
        ("C1", "O2"),
        ("C1", "MeSH"),
        ("BENZENE", "NAPHTHALENE"),
        ("C11", "C16"),
        ("CO2", "METHANOL"),
    ]

    print("    Sample k_ij values at T=298.15 K:")
    for comp_i, comp_j in test_pairs:
        if comp_i in components and comp_j in components:
            try:
                kij = calc.calculate_kij(comp_i, comp_j, 298.15)
                print(f"      k_{{{comp_i}-{comp_j}}} = {kij:8.5f}")
            except Exception as e:
                print(f"      k_{{{comp_i}-{comp_j}}} = ERROR: {e}")

    # Test 7: Verify specific component properties
    print("\n[7] Verifying specific new components...")

    # Check NAPHTHALENE (fused aromatic)
    naphthalene = components.get("NAPHTHALENE")
    if naphthalene:
        print(f"    NAPHTHALENE: {naphthalene.name}")
        print(f"      Formula: {naphthalene.formula}")
        print(f"      Family: {naphthalene.family}")
        print(f"      Groups: {naphthalene.groups}")
        print(f"      Tc = {naphthalene.Tc:.2f} K")

    # Check C16 (generated alkane)
    c16 = components.get("C16")
    if c16:
        print(f"    C16: {c16.name}")
        print(f"      Formula: {c16.formula}")
        print(f"      MW = {c16.MW:.2f} g/mol")
        print(f"      Tb = {c16.Tb:.2f} K")

    # Check METHANOL (polar)
    methanol = components.get("METHANOL")
    if methanol:
        print(f"    METHANOL: {methanol.name}")
        print(f"      Note: {methanol.note}")

    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

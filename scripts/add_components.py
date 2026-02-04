"""Script to add new components to the database.

This script adds Tier 1-4 components with proper thermodynamic properties,
SMILES, families, and PPR78 group decompositions.
"""

import json
from pathlib import Path

# Define new components to add
NEW_COMPONENTS = {
    # === Tier 1: Additional permanent gases ===
    "O2": {
        "id": "O2",
        "name": "Oxygen",
        "formula": "O2",
        "cas": "7782-44-7",
        "smiles": "O=O",
        "family": "DIATOMIC",
        "groups": {"O2": 1},  # Would need PPR78 extension
        "Tc": 154.58,
        "Tc_unit": "K",
        "Pc": 5043000,
        "Pc_unit": "Pa",
        "Vc": 7.34e-5,
        "Vc_unit": "m3/mol",
        "omega": 0.022,
        "MW": 31.9988,
        "MW_unit": "g/mol",
        "Tb": 90.19,
        "Tb_unit": "K"
    },

    # === Tier 1: Mercaptans ===
    "MeSH": {
        "id": "MeSH",
        "name": "Methanethiol",
        "aliases": ["Methyl mercaptan", "Thiomethanol"],
        "formula": "CH4S",
        "cas": "74-93-1",
        "smiles": "CS",
        "family": "SULFUR_ORGANIC",
        "groups": {"CH3": 1, "SH": 1},
        "Tc": 469.95,
        "Tc_unit": "K",
        "Pc": 7230000,
        "Pc_unit": "Pa",
        "Vc": 1.45e-4,
        "Vc_unit": "m3/mol",
        "omega": 0.158,
        "MW": 48.108,
        "MW_unit": "g/mol",
        "Tb": 279.1,
        "Tb_unit": "K"
    },

    "EtSH": {
        "id": "EtSH",
        "name": "Ethanethiol",
        "aliases": ["Ethyl mercaptan", "Thioethanol"],
        "formula": "C2H6S",
        "cas": "75-08-1",
        "smiles": "CCS",
        "family": "SULFUR_ORGANIC",
        "groups": {"CH3": 1, "CH2": 1, "SH": 1},
        "Tc": 499.15,
        "Tc_unit": "K",
        "Pc": 5490000,
        "Pc_unit": "Pa",
        "Vc": 2.07e-4,
        "Vc_unit": "m3/mol",
        "omega": 0.189,
        "MW": 62.134,
        "MW_unit": "g/mol",
        "Tb": 308.15,
        "Tb_unit": "K"
    },

    # === Tier 2: Additional naphthenes ===
    "CYCLOPENTANE": {
        "id": "CYCLOPENTANE",
        "name": "Cyclopentane",
        "formula": "C5H10",
        "cas": "287-92-3",
        "smiles": "C1CCCC1",
        "family": "CYCLOALKANE",
        "groups": {"CH2_cyclic": 5},
        "Tc": 511.76,
        "Tc_unit": "K",
        "Pc": 4515000,
        "Pc_unit": "Pa",
        "Vc": 2.58e-4,
        "Vc_unit": "m3/mol",
        "omega": 0.196,
        "MW": 70.1329,
        "MW_unit": "g/mol",
        "Tb": 322.4,
        "Tb_unit": "K"
    },

    "MCYCLOPENTANE": {
        "id": "MCYCLOPENTANE",
        "name": "Methylcyclopentane",
        "formula": "C6H12",
        "cas": "96-37-7",
        "smiles": "CC1CCCC1",
        "family": "CYCLOALKANE",
        "groups": {"CH2_cyclic": 4, "CHcyclic": 1, "CH3": 1},
        "Tc": 532.79,
        "Tc_unit": "K",
        "Pc": 3790000,
        "Pc_unit": "Pa",
        "Vc": 3.19e-4,
        "Vc_unit": "m3/mol",
        "omega": 0.230,
        "MW": 84.1595,
        "MW_unit": "g/mol",
        "Tb": 345.0,
        "Tb_unit": "K"
    },

    # === Tier 2: Additional aromatics ===
    "NAPHTHALENE": {
        "id": "NAPHTHALENE",
        "name": "Naphthalene",
        "formula": "C10H8",
        "cas": "91-20-3",
        "smiles": "c1ccc2ccccc2c1",
        "family": "AROMATIC",
        "groups": {"CHaro": 8, "Caro": 2},  # Fused aromatic
        "Tc": 748.4,
        "Tc_unit": "K",
        "Pc": 4051000,
        "Pc_unit": "Pa",
        "Vc": 4.07e-4,
        "Vc_unit": "m3/mol",
        "omega": 0.302,
        "MW": 128.1705,
        "MW_unit": "g/mol",
        "Tb": 491.14,
        "Tb_unit": "K"
    },

    "CUMENE": {
        "id": "CUMENE",
        "name": "Cumene",
        "aliases": ["Isopropylbenzene"],
        "formula": "C9H12",
        "cas": "98-82-8",
        "smiles": "CC(C)c1ccccc1",
        "family": "AROMATIC",
        "groups": {"CHaro": 5, "Caro": 1, "CH3": 2, "CH": 1},
        "Tc": 631.15,
        "Tc_unit": "K",
        "Pc": 3209000,
        "Pc_unit": "Pa",
        "Vc": 4.35e-4,
        "Vc_unit": "m3/mol",
        "omega": 0.326,
        "MW": 120.191,
        "MW_unit": "g/mol",
        "Tb": 425.56,
        "Tb_unit": "K"
    },
}

# Generate C11-C16 n-alkanes
for n in range(11, 17):
    comp_id = f"C{n}"
    # Correlation for n-alkanes (Riazi-Daubert)
    # Tc = 191.106 + 37.978*n + 0.20947*n^2
    # Pc = (1.6071 + 2.108*n)^-2 * 1e6 Pa
    # Tb = 94.84 + 30.04*n
    Tc = 191.106 + 37.978 * n + 0.20947 * n**2
    Pc_MPa = (1.6071 + 2.108 * n) ** -2
    Pc = Pc_MPa * 1e6
    Tb = 94.84 + 30.04 * n
    MW = 14.0266 * n + 2.0158
    # omega correlation: omega = 0.5899*(n-1) / (n+1)
    omega = 0.2905 + 0.027 * n
    # Vc correlation
    Vc = (2.16e-5) * n + 4.5e-5

    NEW_COMPONENTS[comp_id] = {
        "id": comp_id,
        "name": f"n-{['', '', '', '', '', '', '', '', '', '', '', 'Undecane', 'Dodecane', 'Tridecane', 'Tetradecane', 'Pentadecane', 'Hexadecane'][n]}",
        "aliases": [f"nC{n}", f"n-C{n}"],
        "formula": f"C{n}H{2*n+2}",
        "smiles": "C" * n,
        "family": "ALKANE",
        "groups": {"CH3": 2, "CH2": n - 2},
        "Tc": round(Tc, 2),
        "Tc_unit": "K",
        "Pc": int(Pc),
        "Pc_unit": "Pa",
        "Vc": round(Vc, 6),
        "Vc_unit": "m3/mol",
        "omega": round(omega, 4),
        "MW": round(MW, 4),
        "MW_unit": "g/mol",
        "Tb": round(Tb, 2),
        "Tb_unit": "K"
    }

# Tier 4: EOR solvents
NEW_COMPONENTS.update({
    "METHANOL": {
        "id": "METHANOL",
        "name": "Methanol",
        "aliases": ["Methyl alcohol", "MeOH"],
        "formula": "CH4O",
        "cas": "67-56-1",
        "smiles": "CO",
        "family": "UNDEFINED",  # Polar, outside typical families
        "groups": {},  # No standard PPR78 groups for alcohols
        "Tc": 512.6,
        "Tc_unit": "K",
        "Pc": 8084000,
        "Pc_unit": "Pa",
        "Vc": 1.18e-4,
        "Vc_unit": "m3/mol",
        "omega": 0.565,
        "MW": 32.042,
        "MW_unit": "g/mol",
        "Tb": 337.85,
        "Tb_unit": "K",
        "note": "Polar component; cubic EOS accuracy limited without association modeling"
    },

    "ETHANOL": {
        "id": "ETHANOL",
        "name": "Ethanol",
        "aliases": ["Ethyl alcohol", "EtOH"],
        "formula": "C2H6O",
        "cas": "64-17-5",
        "smiles": "CCO",
        "family": "UNDEFINED",
        "groups": {},
        "Tc": 513.9,
        "Tc_unit": "K",
        "Pc": 6137000,
        "Pc_unit": "Pa",
        "Vc": 1.68e-4,
        "Vc_unit": "m3/mol",
        "omega": 0.644,
        "MW": 46.069,
        "MW_unit": "g/mol",
        "Tb": 351.44,
        "Tb_unit": "K",
        "note": "Polar component; cubic EOS accuracy limited without association modeling"
    },
})


def main():
    """Add new components to the database."""
    db_path = Path(__file__).parent.parent / "data" / "pure_components" / "components.json"

    # Load existing database
    with open(db_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Current database has {len(data['components'])} components")

    # Add new components
    added_count = 0
    for comp_id, comp_data in NEW_COMPONENTS.items():
        if comp_id not in data['components']:
            data['components'][comp_id] = comp_data
            added_count += 1
            print(f"Added: {comp_id} - {comp_data['name']}")
        else:
            print(f"Skipped (exists): {comp_id}")

    print(f"\nAdded {added_count} new components")
    print(f"New total: {len(data['components'])} components")

    # Save updated database
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDatabase saved to {db_path}")


if __name__ == "__main__":
    main()

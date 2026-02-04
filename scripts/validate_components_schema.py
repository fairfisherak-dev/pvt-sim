# validate_components.py
# Validates data/pure_components/components.json group-key consistency + basic thermo sanity.
#
# Run from repo root:
#   python validate_components.py
#
# Optional:
#   python validate_components.py --fix-ethane   (prints a JSON patch suggestion)

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Set, Tuple


CANONICAL_GROUP_KEYS: Set[str] = {
    # Structural (HC + rings + aromatics)
    "CH4", "CH3", "CH2", "CH", "C",
    "CH2_cyclic", "CHcyclic",
    "CHaro", "Caro",
    # Sulfur groups
    "SH", "S",
    # Special species (keep as-is for now)
    "N2", "CO2", "H2S", "H2", "CO", "He", "Ar", "SO2", "COS", "CS2", "H2O",
}

DEPRECATED_GROUP_KEYS: Set[str] = {"C2H6"}  # currently only seen on C2 in your repo


def _is_nonneg_int(x: Any) -> bool:
    return isinstance(x, int) and x >= 0


def _finite_pos(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x)) and float(x) > 0.0


def _finite_nonneg(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x)) and float(x) >= 0.0


def validate_components(db_path: Path) -> Tuple[int, int]:
    data = json.loads(db_path.read_text(encoding="utf-8"))
    comps: Dict[str, Dict[str, Any]] = data.get("components", {})
    errors = 0
    warnings = 0

    required_top = {"schema_version", "components"}
    missing_top = required_top - set(data.keys())
    if missing_top:
        errors += 1
        print(f"ERROR: missing top-level keys: {sorted(missing_top)}")

    for key, c in comps.items():
        prefix = f"[{key}]"

        # Basic required fields (adjust if your schema evolves)
        for req in ("id", "name", "aliases", "formula", "family", "groups", "MW", "Tc", "Pc", "Vc", "omega", "Tb"):
            if req not in c:
                errors += 1
                print(f"ERROR {prefix}: missing '{req}'")

        # groups validation
        groups = c.get("groups", {})
        if not isinstance(groups, dict):
            errors += 1
            print(f"ERROR {prefix}: 'groups' must be a dict")
            continue

        gkeys = set(groups.keys())
        unknown = gkeys - CANONICAL_GROUP_KEYS
        if unknown:
            errors += 1
            print(f"ERROR {prefix}: unknown group keys: {sorted(unknown)}")

        deprecated = gkeys & DEPRECATED_GROUP_KEYS
        if deprecated:
            errors += 1
            print(f"ERROR {prefix}: deprecated group keys present: {sorted(deprecated)}")

        for gk, gv in groups.items():
            if not _is_nonneg_int(gv):
                errors += 1
                print(f"ERROR {prefix}: group '{gk}' must be a non-negative int; got {gv!r}")

        # "mixed basis" heuristic: forbid formula-as-group patterns (like C2H6) generally
        for gk in gkeys:
            if any(ch.isdigit() for ch in gk) and gk not in CANONICAL_GROUP_KEYS:
                warnings += 1
                print(f"WARNING {prefix}: suspicious formula-like group key '{gk}' (not canonical)")

        # Thermo sanity (units assumed: Tc K, Pc Pa, Vc m^3/mol, Tb K, MW g/mol)
        if "MW" in c and not _finite_pos(c["MW"]):
            errors += 1
            print(f"ERROR {prefix}: MW must be finite > 0 (g/mol); got {c.get('MW')!r}")

        if "Tc" in c and not _finite_pos(c["Tc"]):
            errors += 1
            print(f"ERROR {prefix}: Tc must be finite > 0 (K); got {c.get('Tc')!r}")

        if "Pc" in c and not _finite_pos(c["Pc"]):
            errors += 1
            print(f"ERROR {prefix}: Pc must be finite > 0 (Pa); got {c.get('Pc')!r}")

        if "Vc" in c and not _finite_pos(c["Vc"]):
            errors += 1
            print(f"ERROR {prefix}: Vc must be finite > 0 (m^3/mol); got {c.get('Vc')!r}")

        if "Tb" in c and not _finite_pos(c["Tb"]):
            errors += 1
            print(f"ERROR {prefix}: Tb must be finite > 0 (K); got {c.get('Tb')!r}")

        # Omega validation: must be finite (negative allowed); warn on strongly negative values
        omega = c.get("Omega")
        if omega is not None:
            if not (isinstance(omega, (int, float)) and math.isfinite(float(omega))):
                errors += 1
                print(f"ERROR {prefix}: Omega must be finite; got {omega!r}")
            elif float(omega) < -0.5:
                warnings += 1
                print(f"WARNING {prefix}: Omega is strongly negative: {float(omega)}")



        # Optional consistency checks (warnings)
        # - Typical Pc range sanity (very loose): 0.1 MPa to 200 MPa
        Pc = c.get("Pc", None)
        if isinstance(Pc, (int, float)) and math.isfinite(float(Pc)):
            if not (1e5 <= float(Pc) <= 2e8):
                warnings += 1
                print(f"WARNING {prefix}: Pc looks out of expected Pa range (1e5..2e8): {float(Pc):.6g}")

        # - Tb should generally be < Tc (not always true for pseudo-components; warn only)
        Tb = c.get("Tb", None)
        Tc = c.get("Tc", None)
        if isinstance(Tb, (int, float)) and isinstance(Tc, (int, float)) and math.isfinite(float(Tb)) and math.isfinite(float(Tc)):
            if float(Tb) >= float(Tc):
                warnings += 1
                print(f"WARNING {prefix}: Tb >= Tc (may be ok for pseudo-components). Tb={float(Tb):.3f}, Tc={float(Tc):.3f}")

        # Aliases sanity
        aliases = c.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            errors += 1
            print(f"ERROR {prefix}: aliases must be a list[str]")
        else:
            if len(set(a.lower() for a in aliases)) != len(aliases):
                warnings += 1
                print(f"WARNING {prefix}: duplicate aliases by case-folding")

    return errors, warnings


def print_ethane_fix(db_path: Path) -> None:
    data = json.loads(db_path.read_text(encoding="utf-8"))
    comps: Dict[str, Dict[str, Any]] = data.get("components", {})
    c2 = comps.get("C2")
    if not c2:
        print("No component 'C2' found; nothing to suggest.")
        return
    groups = c2.get("groups", {})
    if groups == {"C2H6": 1}:
        print("Suggested fix for [C2] groups:")
        print('  OLD: {"C2H6": 1}')
        print('  NEW: {"CH3": 2}')
    else:
        print("C2 groups not equal to {'C2H6': 1}; no suggestion printed.")
        print(f"Current C2 groups: {groups!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/pure_components/components.json", help="Path to components.json")
    ap.add_argument("--fix-ethane", action="store_true", help="Print suggested ethane group normalization if applicable")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB file not found: {db_path}")

    if args.fix_ethane:
        print_ethane_fix(db_path)
        return

    errors, warnings = validate_components(db_path)
    print()
    print(f"Done. Errors={errors}, Warnings={warnings}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()

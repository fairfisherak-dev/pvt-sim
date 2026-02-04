"""PPR78 group definitions for group contribution k_ij(T) calculations.

This module defines the functional groups used in the PPR78 (Predictive
Peng-Robinson 1978) model for calculating temperature-dependent binary
interaction parameters.

References
----------
[1] Jaubert, J.-N. and Mutelet, F. (2004). "VLE predictions with the
    Peng-Robinson equation of state and temperature dependent kij calculated
    through a group contribution method." Fluid Phase Equilibria, 224(2), 285-304.
[2] Jaubert, J.-N. et al. (2005). "Extension of the PPR78 model to systems
    containing aromatic compounds." Fluid Phase Equilibria, 237(1-2), 193-211.
"""

from enum import Enum, auto


class PPR78Group(Enum):
    """PPR78 functional groups for group contribution calculations.

    These groups are used to decompose molecules for the PPR78 model,
    which calculates temperature-dependent binary interaction parameters
    k_ij(T) from group contributions.

    The group definitions follow the original PPR78 paper (Jaubert & Mutelet,
    2004) with extensions for aromatics and additional compounds.

    Notes
    -----
    Groups are categorized as:
    - Paraffinic: CH3, CH2, CH, C (chain carbons)
    - Whole molecules: CH4, C2H6, CO2, N2, H2S, etc.
    - Aromatic: CHaro, Caro (aromatic ring carbons)
    - Cyclic: CH2_cyclic, CHcyclic, Ccyclic (ring carbons)
    - Sulfur: SH (thiol group)

    Group Basis Standardization
    ----------------------------
    For consistency in group-contribution calculations:
    - CH4 (methane): Use whole-molecule group CH4
    - C2H6 (ethane): Deprecated for new components - use structural basis CH3: 2
    - C3+ alkanes: Use structural basis (CH3, CH2, CH, C)

    C2H6 is retained in the enum for backward compatibility with published
    PPR78 interaction parameters, but new component definitions should use
    the structural basis (2 × CH3) for consistency across all alkanes.
    """

    # === Paraffinic (chain) groups ===
    CH3 = auto()  # Methyl group (-CH3)
    CH2 = auto()  # Methylene group (-CH2-)
    CH = auto()   # Methine group (>CH-)
    C = auto()    # Quaternary carbon (>C<)

    # === Whole molecule groups (special treatment) ===
    CH4 = auto()    # Methane
    C2H6 = auto()   # Ethane
    CO2 = auto()    # Carbon dioxide
    N2 = auto()     # Nitrogen
    H2S = auto()    # Hydrogen sulfide
    H2 = auto()     # Hydrogen
    CO = auto()     # Carbon monoxide
    He = auto()     # Helium
    Ar = auto()     # Argon
    H2O = auto()    # Water

    # === Aromatic groups ===
    CHaro = auto()  # Aromatic CH (also called ACH)
    Caro = auto()   # Substituted aromatic carbon (AC)

    # === Cyclic (naphthenic) groups ===
    CH2_cyclic = auto()  # Cyclic methylene
    CHcyclic = auto()    # Cyclic methine
    Ccyclic = auto()     # Cyclic quaternary carbon

    # === Sulfur groups ===
    SH = auto()     # Thiol/mercaptan group (-SH)
    S = auto()      # Sulfide (-S-)

    # === Additional groups for sulfur compounds ===
    CS2 = auto()    # Carbon disulfide (whole molecule)
    COS = auto()    # Carbonyl sulfide (whole molecule)
    SO2 = auto()    # Sulfur dioxide (whole molecule)


# Group name aliases for flexible input parsing
GROUP_ALIASES: dict[str, PPR78Group] = {
    # Standard names
    "CH3": PPR78Group.CH3,
    "CH2": PPR78Group.CH2,
    "CH": PPR78Group.CH,
    "C": PPR78Group.C,
    "CH4": PPR78Group.CH4,
    "C2H6": PPR78Group.C2H6,
    "CO2": PPR78Group.CO2,
    "N2": PPR78Group.N2,
    "H2S": PPR78Group.H2S,
    "H2": PPR78Group.H2,
    "CO": PPR78Group.CO,
    "HE": PPR78Group.He,
    "AR": PPR78Group.Ar,
    "H2O": PPR78Group.H2O,
    # Aromatic aliases
    "CHARO": PPR78Group.CHaro,
    "ACH": PPR78Group.CHaro,  # Common alias for aromatic CH
    "CARO": PPR78Group.Caro,
    "AC": PPR78Group.Caro,    # Common alias for aromatic C
    # Cyclic aliases
    "CH2_CYCLIC": PPR78Group.CH2_cyclic,
    "CYC-CH2": PPR78Group.CH2_cyclic,
    "CHCYCLIC": PPR78Group.CHcyclic,
    "CYC-CH": PPR78Group.CHcyclic,
    "CCYCLIC": PPR78Group.Ccyclic,
    "CYC-C": PPR78Group.Ccyclic,
    # Sulfur
    "SH": PPR78Group.SH,
    "S": PPR78Group.S,
    "CS2": PPR78Group.CS2,
    "COS": PPR78Group.COS,
    "SO2": PPR78Group.SO2,
}


def parse_group_name(name: str) -> PPR78Group:
    """Parse a group name string to PPR78Group enum.

    Parameters
    ----------
    name : str
        Group name (case-insensitive). Supports various aliases.

    Returns
    -------
    PPR78Group
        The corresponding group enum value.

    Raises
    ------
    ValueError
        If the group name is not recognized.

    Examples
    --------
    >>> parse_group_name("CH3")
    <PPR78Group.CH3: 1>
    >>> parse_group_name("ach")  # Case-insensitive alias
    <PPR78Group.CHaro: 15>
    """
    name_upper = name.upper().replace("-", "_").replace(" ", "_")

    # Try direct lookup in aliases
    if name_upper in GROUP_ALIASES:
        return GROUP_ALIASES[name_upper]

    # Try enum member lookup
    try:
        return PPR78Group[name_upper]
    except KeyError:
        pass

    # Try without underscores
    name_no_underscore = name_upper.replace("_", "")
    for member in PPR78Group:
        if member.name.replace("_", "") == name_no_underscore:
            return member

    available = ", ".join(sorted(GROUP_ALIASES.keys()))
    raise ValueError(
        f"Unknown PPR78 group: '{name}'. Available groups: {available}"
    )

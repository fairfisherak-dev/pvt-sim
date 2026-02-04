"""SMILES to PPR78 group decomposition.

This module provides functionality to decompose molecular structures
into PPR78 functional groups for k_ij(T) calculation.

The decomposition can be done via:
1. Pre-stored group counts from component database
2. Built-in lookup table for common petroleum components
3. SMARTS pattern matching using RDKit (optional dependency)
"""

from functools import lru_cache
from typing import Dict, Optional

from .definitions import PPR78Group, parse_group_name


# Built-in group decompositions for common petroleum components
# These are curated values that don't require RDKit
BUILTIN_GROUPS: Dict[str, Dict[PPR78Group, int]] = {
    # === Inorganic / Diatomic ===
    "N2": {PPR78Group.N2: 1},
    "CO2": {PPR78Group.CO2: 1},
    "H2S": {PPR78Group.H2S: 1},
    "H2": {PPR78Group.H2: 1},
    "CO": {PPR78Group.CO: 1},
    "HE": {PPR78Group.He: 1},
    "AR": {PPR78Group.Ar: 1},
    "H2O": {PPR78Group.H2O: 1},
    # Sulfur compounds
    "CS2": {PPR78Group.CS2: 1},
    "COS": {PPR78Group.COS: 1},
    "SO2": {PPR78Group.SO2: 1},

    # === Normal Paraffins ===
    "C1": {PPR78Group.CH4: 1},
    "C2": {PPR78Group.CH3: 2},
    "C3": {PPR78Group.CH3: 2, PPR78Group.CH2: 1},
    "C4": {PPR78Group.CH3: 2, PPR78Group.CH2: 2},
    "NC4": {PPR78Group.CH3: 2, PPR78Group.CH2: 2},
    "C5": {PPR78Group.CH3: 2, PPR78Group.CH2: 3},
    "NC5": {PPR78Group.CH3: 2, PPR78Group.CH2: 3},
    "C6": {PPR78Group.CH3: 2, PPR78Group.CH2: 4},
    "NC6": {PPR78Group.CH3: 2, PPR78Group.CH2: 4},
    "C7": {PPR78Group.CH3: 2, PPR78Group.CH2: 5},
    "NC7": {PPR78Group.CH3: 2, PPR78Group.CH2: 5},
    "C8": {PPR78Group.CH3: 2, PPR78Group.CH2: 6},
    "NC8": {PPR78Group.CH3: 2, PPR78Group.CH2: 6},
    "C9": {PPR78Group.CH3: 2, PPR78Group.CH2: 7},
    "NC9": {PPR78Group.CH3: 2, PPR78Group.CH2: 7},
    "C10": {PPR78Group.CH3: 2, PPR78Group.CH2: 8},
    "NC10": {PPR78Group.CH3: 2, PPR78Group.CH2: 8},

    # === Iso-paraffins ===
    "IC4": {PPR78Group.CH3: 3, PPR78Group.CH: 1},
    "IC5": {PPR78Group.CH3: 3, PPR78Group.CH2: 1, PPR78Group.CH: 1},
    "NEOC5": {PPR78Group.CH3: 4, PPR78Group.C: 1},

    # === Aromatics (BTEX) ===
    "BENZENE": {PPR78Group.CHaro: 6},
    "TOLUENE": {PPR78Group.CHaro: 5, PPR78Group.Caro: 1, PPR78Group.CH3: 1},
    "ETHYLBENZENE": {
        PPR78Group.CHaro: 5,
        PPR78Group.Caro: 1,
        PPR78Group.CH3: 1,
        PPR78Group.CH2: 1,
    },
    "O_XYLENE": {PPR78Group.CHaro: 4, PPR78Group.Caro: 2, PPR78Group.CH3: 2},
    "M_XYLENE": {PPR78Group.CHaro: 4, PPR78Group.Caro: 2, PPR78Group.CH3: 2},
    "P_XYLENE": {PPR78Group.CHaro: 4, PPR78Group.Caro: 2, PPR78Group.CH3: 2},
    "NAPHTHALENE": {PPR78Group.CHaro: 8, PPR78Group.Caro: 2},

    # === Cycloalkanes (Naphthenes) ===
    "CYCLOPENTANE": {PPR78Group.CH2_cyclic: 5},
    "CYCLOHEXANE": {PPR78Group.CH2_cyclic: 6},
    "MCYCLOPENTANE": {
        PPR78Group.CH2_cyclic: 4,
        PPR78Group.CHcyclic: 1,
        PPR78Group.CH3: 1,
    },
    "MCYCLOHEXANE": {
        PPR78Group.CH2_cyclic: 5,
        PPR78Group.CHcyclic: 1,
        PPR78Group.CH3: 1,
    },

    # === Additional components ===
    # Oxygen
    "O2": {PPR78Group.N2: 1},  # Placeholder - treat similar to N2

    # Mercaptans
    "MESH": {PPR78Group.CH3: 1, PPR78Group.SH: 1},
    "ETSH": {PPR78Group.CH3: 1, PPR78Group.CH2: 1, PPR78Group.SH: 1},

    # Additional aromatics
    "NAPHTHALENE": {PPR78Group.CHaro: 8, PPR78Group.Caro: 2},
    "CUMENE": {PPR78Group.CHaro: 5, PPR78Group.Caro: 1, PPR78Group.CH3: 2, PPR78Group.CH: 1},

    # C11-C20 n-alkanes
    "C11": {PPR78Group.CH3: 2, PPR78Group.CH2: 9},
    "C12": {PPR78Group.CH3: 2, PPR78Group.CH2: 10},
    "C13": {PPR78Group.CH3: 2, PPR78Group.CH2: 11},
    "C14": {PPR78Group.CH3: 2, PPR78Group.CH2: 12},
    "C15": {PPR78Group.CH3: 2, PPR78Group.CH2: 13},
    "C16": {PPR78Group.CH3: 2, PPR78Group.CH2: 14},
    "C17": {PPR78Group.CH3: 2, PPR78Group.CH2: 15},
    "C18": {PPR78Group.CH3: 2, PPR78Group.CH2: 16},
    "C19": {PPR78Group.CH3: 2, PPR78Group.CH2: 17},
    "C20": {PPR78Group.CH3: 2, PPR78Group.CH2: 18},

    # Polar components (no standard PPR78 groups - k_ij defaults to zero)
    "METHANOL": {},
    "ETHANOL": {},
}


class GroupDecomposer:
    """Decomposes molecules into PPR78 groups.

    This class provides multiple strategies for group decomposition:
    1. Lookup: Use pre-stored group counts from component database
    2. Built-in: Use curated decompositions for common components
    3. SMARTS: Parse SMILES and count groups using RDKit (if available)

    Attributes:
        use_rdkit: Whether RDKit-based SMARTS matching is available

    Example
    -------
    >>> decomposer = GroupDecomposer()
    >>> groups = decomposer.decompose(component_id="C1")
    >>> print(groups)
    {<PPR78Group.CH4: 5>: 1}

    >>> groups = decomposer.decompose(groups_dict={"CH3": 2, "CH2": 4})
    >>> print(groups)
    {<PPR78Group.CH3: 1>: 2, <PPR78Group.CH2: 2>: 4}
    """

    def __init__(self, use_rdkit: bool = True):
        """Initialize decomposer.

        Parameters
        ----------
        use_rdkit : bool
            Whether to attempt RDKit import for SMARTS matching.
            If False or RDKit unavailable, only lookup mode works.
        """
        self._rdkit_available = False
        self._Chem = None

        if use_rdkit:
            try:
                from rdkit import Chem
                self._Chem = Chem
                self._rdkit_available = True
            except ImportError:
                pass

    @property
    def rdkit_available(self) -> bool:
        """Check if RDKit is available for SMARTS matching."""
        return self._rdkit_available

    def decompose(
        self,
        smiles: Optional[str] = None,
        groups_dict: Optional[Dict[str, int]] = None,
        component_id: Optional[str] = None,
    ) -> Dict[PPR78Group, int]:
        """Decompose molecule into PPR78 groups.

        Parameters
        ----------
        smiles : str, optional
            SMILES string for the molecule.
        groups_dict : dict, optional
            Pre-computed group counts (e.g., {"CH3": 2, "CH2": 4}).
        component_id : str, optional
            Component ID for lookup in built-in database.

        Returns
        -------
        dict
            Mapping of PPR78Group enum to count.

        Raises
        ------
        ValueError
            If decomposition cannot be performed with given inputs.

        Notes
        -----
        Priority order:
        1. Use provided groups_dict (if given)
        2. Lookup from built-in database by component_id
        3. Parse SMILES with RDKit (if available)
        """
        # Priority 1: Use provided groups dict
        if groups_dict is not None:
            return self._parse_groups_dict(groups_dict)

        # Priority 2: Built-in database lookup
        if component_id is not None:
            builtin = self._lookup_builtin(component_id)
            if builtin is not None:
                return builtin.copy()

        # Priority 3: SMARTS matching from SMILES
        if smiles is not None:
            if self._rdkit_available:
                return self._decompose_smarts(smiles)
            else:
                raise ValueError(
                    f"Cannot decompose SMILES '{smiles}': RDKit not installed. "
                    "Install with: pip install rdkit"
                )

        raise ValueError(
            "Cannot decompose: provide groups_dict, valid component_id, "
            "or SMILES string (requires RDKit)"
        )

    def _parse_groups_dict(
        self, groups_dict: Dict[str, int]
    ) -> Dict[PPR78Group, int]:
        """Convert string group names to PPR78Group enum."""
        result: Dict[PPR78Group, int] = {}
        for name, count in groups_dict.items():
            group = parse_group_name(name)
            result[group] = count
        return result

    def _lookup_builtin(
        self, component_id: str
    ) -> Optional[Dict[PPR78Group, int]]:
        """Lookup groups for common components."""
        # Normalize ID (remove hyphens, keep underscores)
        normalized = component_id.upper().replace("-", "")

        # Direct lookup
        if normalized in BUILTIN_GROUPS:
            return BUILTIN_GROUPS[normalized]

        # Try common variations
        variations = [
            normalized,
            f"N{normalized}",  # e.g., "C4" -> "NC4"
            normalized.replace("N", ""),  # e.g., "NC4" -> "C4"
            normalized.replace("I", "IC"),  # e.g., "IC4" normalization
        ]

        for var in variations:
            if var in BUILTIN_GROUPS:
                return BUILTIN_GROUPS[var]

        return None

    def _decompose_smarts(self, smiles: str) -> Dict[PPR78Group, int]:
        """Decompose using RDKit SMARTS matching.

        This is a simplified implementation that handles common cases.
        For full accuracy, the component database groups should be used.
        """
        if self._Chem is None:
            raise RuntimeError("RDKit not available")

        Chem = self._Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Add explicit hydrogens for accurate counting
        mol = Chem.AddHs(mol)
        n_heavy = mol.GetNumHeavyAtoms()

        # Handle whole-molecule cases first
        smiles_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

        # Simple molecules
        whole_molecule_map = {
            "C": PPR78Group.CH4,
            "CC": PPR78Group.C2H6,
            "[H][H]": PPR78Group.H2,
            "N#N": PPR78Group.N2,
            "O=C=O": PPR78Group.CO2,
            "S": PPR78Group.H2S,
            "[C-]#[O+]": PPR78Group.CO,
            "[He]": PPR78Group.He,
            "[Ar]": PPR78Group.Ar,
            "O": PPR78Group.H2O,
            "S=C=S": PPR78Group.CS2,
            "O=C=S": PPR78Group.COS,
            "O=S=O": PPR78Group.SO2,
        }

        if smiles_canonical in whole_molecule_map:
            return {whole_molecule_map[smiles_canonical]: 1}

        # For more complex molecules, use SMARTS patterns
        result: Dict[PPR78Group, int] = {}

        # Check for aromatic carbons
        aromatic_ch_pattern = Chem.MolFromSmarts("[cH]")
        aromatic_c_pattern = Chem.MolFromSmarts("[c;H0]")

        if aromatic_ch_pattern:
            matches = mol.GetSubstructMatches(aromatic_ch_pattern)
            if matches:
                result[PPR78Group.CHaro] = len(matches)

        if aromatic_c_pattern:
            matches = mol.GetSubstructMatches(aromatic_c_pattern)
            if matches:
                result[PPR78Group.Caro] = len(matches)

        # Check for cyclic (non-aromatic ring) carbons
        cyc_ch2_pattern = Chem.MolFromSmarts("[CH2;R;!a]")
        cyc_ch_pattern = Chem.MolFromSmarts("[CH1;R;!a]")

        if cyc_ch2_pattern:
            matches = mol.GetSubstructMatches(cyc_ch2_pattern)
            if matches:
                result[PPR78Group.CH2_cyclic] = len(matches)

        if cyc_ch_pattern:
            matches = mol.GetSubstructMatches(cyc_ch_pattern)
            if matches:
                result[PPR78Group.CHcyclic] = len(matches)

        # Chain carbons (non-ring, non-aromatic)
        ch3_pattern = Chem.MolFromSmarts("[CH3;!R]")
        ch2_pattern = Chem.MolFromSmarts("[CH2;!R;!a]")
        ch_pattern = Chem.MolFromSmarts("[CH1;!R;!a]")
        c_pattern = Chem.MolFromSmarts("[CH0;!R;!a]")

        if ch3_pattern:
            matches = mol.GetSubstructMatches(ch3_pattern)
            if matches:
                result[PPR78Group.CH3] = len(matches)

        if ch2_pattern:
            matches = mol.GetSubstructMatches(ch2_pattern)
            if matches:
                result[PPR78Group.CH2] = len(matches)

        if ch_pattern:
            matches = mol.GetSubstructMatches(ch_pattern)
            if matches:
                result[PPR78Group.CH] = len(matches)

        if c_pattern:
            matches = mol.GetSubstructMatches(c_pattern)
            if matches:
                result[PPR78Group.C] = len(matches)

        # Check for thiol groups
        sh_pattern = Chem.MolFromSmarts("[SH]")
        if sh_pattern:
            matches = mol.GetSubstructMatches(sh_pattern)
            if matches:
                result[PPR78Group.SH] = len(matches)

        return result

    @lru_cache(maxsize=1000)
    def decompose_cached(
        self,
        component_id: str,
        smiles: Optional[str] = None,
    ) -> tuple:
        """Cached version of decompose for repeated calls.

        Returns a tuple of (group, count) pairs for hashability.
        """
        result = self.decompose(
            smiles=smiles,
            component_id=component_id,
        )
        return tuple(sorted((g.value, c) for g, c in result.items()))


def get_n_alkane_groups(carbon_number: int) -> Dict[PPR78Group, int]:
    """Get PPR78 groups for a normal alkane by carbon number.

    Parameters
    ----------
    carbon_number : int
        Number of carbon atoms (1 to any).

    Returns
    -------
    dict
        PPR78 group decomposition.

    Examples
    --------
    >>> get_n_alkane_groups(1)
    {<PPR78Group.CH4: 5>: 1}
    >>> get_n_alkane_groups(3)
    {<PPR78Group.CH3: 1>: 2, <PPR78Group.CH2: 2>: 1}
    """
    if carbon_number < 1:
        raise ValueError(f"Carbon number must be >= 1, got {carbon_number}")

    if carbon_number == 1:
        return {PPR78Group.CH4: 1}
    elif carbon_number == 2:
        return {PPR78Group.C2H6: 1}
    else:
        # CnH(2n+2) = 2*CH3 + (n-2)*CH2
        return {
            PPR78Group.CH3: 2,
            PPR78Group.CH2: carbon_number - 2,
        }

"""PPR78 group contribution module for k_ij(T) calculations.

This module provides functionality to decompose molecules into PPR78
functional groups, which are used to calculate temperature-dependent
binary interaction parameters k_ij(T).

Example
-------
>>> from pvtcore.eos.groups import PPR78Group, GroupDecomposer
>>>
>>> decomposer = GroupDecomposer()
>>>
>>> # From component ID
>>> groups = decomposer.decompose(component_id="C3")
>>> print(groups)
{<PPR78Group.CH3: 1>: 2, <PPR78Group.CH2: 2>: 1}
>>>
>>> # From explicit groups dict
>>> groups = decomposer.decompose(groups_dict={"CH3": 2, "CH2": 4})
"""

from .definitions import (
    GROUP_ALIASES,
    PPR78Group,
    parse_group_name,
)
from .decomposition import (
    BUILTIN_GROUPS,
    GroupDecomposer,
    get_n_alkane_groups,
)

__all__ = [
    # Enums and constants
    "PPR78Group",
    "GROUP_ALIASES",
    "BUILTIN_GROUPS",
    # Functions
    "parse_group_name",
    "get_n_alkane_groups",
    # Classes
    "GroupDecomposer",
]

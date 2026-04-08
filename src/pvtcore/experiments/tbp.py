"""True Boiling Point (TBP) experiment surface.

This module is intentionally present so TBP can mature into a first-class
experiment/workflow without pretending that the implementation already exists.

Current policy:
- TBP-backed cuts are supported through schema-driven characterization.
- TBP is *not yet* exposed as a ``pvtapp`` calculation type.
- Direct TBP experiment execution is reserved but not yet implemented.
"""

from __future__ import annotations


def simulate_tbp(*args, **kwargs):
    """Reserved entrypoint for future standalone TBP execution.

    Raises
    ------
    NotImplementedError
        Always, until the standalone TBP workflow is actually implemented and
        validated.
    """
    raise NotImplementedError(
        "Standalone TBP execution is not implemented yet. "
        "Use TBP cuts through schema-driven characterization for now."
    )

"""Pytest configuration.

The project uses a `src/` layout, but we want `pytest` to work without requiring
`pip install -e .` first. We therefore prepend `<repo>/src` to `sys.path` at
collection time.

This mirrors what the development helper scripts under `scripts/` do.
"""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    sys.path.insert(0, str(src_path))

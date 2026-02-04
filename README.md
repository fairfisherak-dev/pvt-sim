# PVT Simulator (pvt-simulator)

Core thermodynamics + PVT workflows:

- Plus-fraction characterization (starting with Pedersen-style split)
- PR EOS (with mixing rules) + fugacity
- Michelsen-style stability (TPD)
- PT flash (Rachford–Rice + successive substitution)

This repo uses a **src-layout** Python package: `pvtcore`.

## Install (development)

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

Run tests:

```bash
pytest
```

## Install (runtime only)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

> Long-term intent: publish wheels to an internal/public index once the thermo kernel stabilizes.

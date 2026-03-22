# PVT Simulator (pvt-simulator)

Core thermodynamics + PVT workflows:

- Plus-fraction characterization (starting with Pedersen-style split)
- PR EOS (with mixing rules) + fugacity
- Michelsen-style stability (TPD)
- PT flash (Rachford–Rice + successive substitution)
- Bubble-point, dew-point, and phase-envelope workflows
- Lab workflows: CCE, DL, CVD, and multi-stage separators
- Nano-confinement workflows: capillary pressure, confined flash, and confined envelopes

Dedicated TBP workflows are not implemented in this repo today. Phase-1 support is limited to schema-driven `pvtcore` characterization via `fluid.plus_fraction.tbp_data.cuts`; see [docs/tbp.md](docs/tbp.md).

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

Headless CLI:

```bash
pvtsim --help
```

GUI:

```bash
python -m pip install -e '.[gui]'
pvtsim-gui
```

> Long-term intent: publish wheels to an internal/public index once the thermo kernel stabilizes.

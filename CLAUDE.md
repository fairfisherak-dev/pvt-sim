# CLAUDE.md — PVT-SIM Canon Repo

This file is the Claude Code execution context for the PVT simulator repo.
Claude Code reads it on session start. Keep it current as the repo evolves.

This file is not the canonical home for simulator architecture, equations,
units, workflows, or coding standards. That lives in `README.md` and `docs/`.

## Precedence

- Direct user instructions for the current task take priority over this file,
  unless they conflict with higher-level safety constraints.
- `README.md` and `docs/` are the canonical software source of truth. Treat
  agent-orientation files (this one, `AGENTS.md`) as execution adapters, not
  simulator specs.
- Verified code, config, and runtime behavior beat stale docs. If drift is
  found, update the docs — do not silently change behavior to match them.
- `AGENTS.md` governs Codex in this repo. Claude Code does **not** run that
  lane/worktree model by default — see "Claude Code posture" below. The
  shared-surface and main-branch rules in `AGENTS.md` still apply.

## Startup reads

Before non-trivial work, read what's relevant from the canonical docs set:

- `docs/architecture.md` — module layout and data flow
- `docs/development.md` — stack, conventions, error handling, verification
- `docs/runtime_surface_standard.md` — app/runtime parity contract
- `docs/technical_notes.md` — equation-level contract and dependency ordering
- `docs/numerical_methods.md` — solver policy, tolerances, damping
- `docs/input_schema.md` — fluid/config data contracts
- `docs/units.md` — canonical internal units (SI)
- `docs/validation_plan.md` — validation targets and strategy

If the task involves concurrent or delegated work, also read
`PVTSIM_DEPENDENCY_MAP.md`.

## Repository shape

src-layout Python package, split into two layers:

- `src/pvtcore/` — computational kernel, no GUI dependencies. Groups:
  `core`, `models`, `characterization`, `correlations`, `eos`, `stability`,
  `flash`, `envelope`, `properties`, `experiments`, `confinement`, `tuning`,
  `io`, `validation`.
- `src/pvtapp/` — desktop GUI (PySide6), CLI entrypoints, schemas, runtime
  orchestration, result presentation.

Runtime EOS surface: `Peng-Robinson (1976)`, `Peng-Robinson (1978)`, `SRK`.

Runtime workflow surface (verified from `src/pvtapp/capabilities.py`):
PT Flash, Stability, Bubble Point, Dew Point, Phase Envelope, CCE,
Differential Liberation, CVD, Swelling Test, Separator.

Entrypoints: `pvtsim` (CLI), `pvtsim-cli` (CLI alias), `pvtsim-gui` (desktop).

## Environment

- Python `>=3.10`, venv at `.venv`.
- Windows is the primary run surface. Headless Linux supported for tests and
  CI; GUI and GUI-contract tests require a display (`xvfb-run -a` or `Xvfb :99`
  with `DISPLAY=:99`).
- Install for local Windows development:

```powershell
python.exe -m pip install -e .[full,dev]
```

- `.env.defaults` is the tracked repo-safe baseline. `.env` is machine-local
  and ignored. Keep the interpreter pointed at `.venv\Scripts\python.exe` and
  `python.envFile=${workspaceFolder}/.env.defaults`.

## Common commands

| Task | Command |
|---|---|
| Routine headless tests | `pytest` |
| Target a single file (fast iteration) | `pytest tests/unit/test_flash.py` |
| Integration-root / merge-gate baseline | `python scripts/run_premerge_checks.py --baseline-only` |
| Full lane pre-merge (baseline + touched-surface) | `python scripts/run_premerge_checks.py` |
| Long-form validation lane | `python scripts/run_full_validation.py` |
| GUI-contract tests (opt-in) | `pytest --run-gui-contracts` |
| Ad hoc config validation | `pvtsim validate examples/pt_flash_config.json` |
| Launch desktop app | `pvtsim-gui` |
| Format check | `black --check src/ tests/` |
| Lint | `flake8 src/ --max-line-length=120` |

Default `pytest` collects `tests/unit` plus `tests/contracts/test_invariants.py`
and deselects `gui_contract` and `nightly` markers. The full headless suite is
~1100 tests and ~13 minutes wall time — phase envelope tests dominate. For
iteration, target specific files or markers. To diagnose slow runs, profile
rather than re-running blindly:

```bash
python -m pytest tests/unit tests/contracts/test_invariants.py --durations=40 -q --tb=no
```

## Known test state

9 pre-existing failures on `main` as of 2026-04-15 (dew characterization,
thermopack envelope, stability robustness, flash fixture invariants, MI PVT
bubble point). These are known and do not block progress. Do not silently
reframe a new failure as pre-existing — confirm against the baseline before
dismissing.

## Conventions

- 4-space indentation.
- `pvtcore` stays free of GUI dependencies. Hard rule.
- Type hints on public functions, dataclasses, and schema-facing APIs.
- `from __future__ import annotations` in new Python modules.
- Physical quantities keep explicit units in docstrings and APIs. Unit
  conversions live at I/O boundaries only; solver code operates on canonical
  internal SI (Pa, K, mol). See `docs/units.md`.
- Prefer small, auditable edits over broad refactors unless correctness
  demands otherwise.
- Error model: `pvtcore` raises domain exceptions from `pvtcore.core.errors`
  (`ConvergenceError`, `ValidationError`, `PhaseError`, `EOSError`,
  `PropertyError`, etc.). `pvtapp` and schema boundaries raise `ValueError`,
  `FileNotFoundError`, or Pydantic validation errors on malformed input.
- Do not silently coerce physically invalid inputs into a run. Surface
  enough context (variable, phase, component, config key) to identify the
  failure.

## Runtime surface rule

Domain features in `pvtcore` must be **either** runtime-wired through the
canonical `pvtapp` path for their intended workflows, **or** explicitly marked
as experimental / not app-supported, **or** removed. "Present in code but not
wired" is not an acceptable steady state. GUI controls must drive actual
runtime behavior — display-only controls that imply runtime behavior are a
contract violation. See `docs/runtime_surface_standard.md` for the full rule
set and the current mandatory standards (characterization methods, heavy-end
lumping = Whitson, delumping, BIP surface).

## Claude Code posture

Claude Code is used in this repo for focused review and surgical
implementation, not lane orchestration.

- Read the actual source files before proposing changes. This repo is mature
  and its conventions differ from generic Python assumptions — training-data
  defaults will be wrong here.
- Propose the approach, get confirmation for anything non-trivial, then
  implement. One task at a time. If a fix spans multiple modules, flag the
  scope explicitly — do not silently expand.
- Run the smallest relevant verification for the touched surface first. Only
  reach for the full suite or `run_premerge_checks.py` when the change is
  broad or when preparing a merge gate.
- Do **not** invoke Codex's lane/worktree model (`gui`, `thermo`,
  `validation`, `scratch`, integration root) unless Ole explicitly asks.
  Claude Code defaults to the currently checked-out branch in the current
  working tree.
- Do **not** perform `main`-affecting Git operations (merge, rebase,
  cherry-pick, revert, push to `main`) unless explicitly instructed. Those
  are controller-level operations and are routed through the integration
  root per `AGENTS.md`.
- When reviewing, the literature is the source of truth for thermodynamic
  correctness. `pytest` passing is necessary but not sufficient. If code
  disagrees with the published algorithm, the code is wrong — surface it.

## Shared / serial-only surfaces

Do not edit these without explicit instruction, even on a scoped task:

- `AGENTS.md`, `CLAUDE.md` (this file), `PVTSIM_DEPENDENCY_MAP.md`
- `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`
- `.github/` workflows
- top-level packaging, install, and environment files
- repo-wide import, typing, or architectural changes that cut across multiple
  partitions

## Escalation

Stop and surface the issue to Ole when:

- the task expands beyond its original scope
- a shared or serial-only surface is required
- a Git operation affecting `main` is needed
- verification suggests cross-module impact
- a change would cause drift from a canonical doc and the doc is not being
  updated in the same slice

## Bottom line

Read `AGENTS.md` for execution boundaries.
Read `docs/` for simulator reality.
Read this file for the Claude Code posture.
When they conflict, verified code and direct user instruction win.
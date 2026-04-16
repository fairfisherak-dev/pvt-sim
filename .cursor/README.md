# Cursor local config

- `rules/` — repo rules for agent work (committed).
- `mcp.json` — optional local MCP servers; **not committed** (machine-specific paths). Copy `mcp.json.example` to `mcp.json` and edit paths.

## Recommended local setup

- Create the repo virtual environment and install the full desktop/dev stack:
  `python.exe -m venv .venv`, activate it, then `python.exe -m pip install -e .[full,dev]`.
- Use `.\.venv\Scripts\python.exe` as the interpreter in Cursor.
- Keep `python.envFile` pointed at `.env.defaults`.
- Put repo-safe shared defaults in `.env.defaults`. Keep secrets and
  machine-local overrides in `.env`, which stays ignored.
- Copy `mcp.json.example` to `mcp.json`, edit the local Python and server
  paths, and leave `mcp.json` uncommitted.

## Merge / absorb rule

- A clean merge is only mechanical. It is not proof that runtime behavior,
  tests, validation assets, schema surface, or tracked config survived.
- Do not call a `main`-affecting merge complete until shared and
  branch-owned runtime, GUI, schema, test, validation, script/doc, and tracked
  env/config surfaces are explicitly accounted for.
- Never delete or drop one side's behavior just because the other side lacks
  it. Silence is not retirement.

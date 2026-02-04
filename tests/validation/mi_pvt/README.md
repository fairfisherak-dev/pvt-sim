# MI-PVT Reference Cases

This folder contains reference cases exported from MI-PVT to validate PVT-SIM outputs.

## How to add a case

Create a JSON file under:

`tests/data/mi_pvt_reference/cases/<case_id>.json`

### Case schema (minimal)

- `case_id` (string)
- `task` (string): `"pt_flash" | "bubble_point" | "dew_point"`
- `temperature` (number K, or {"value":..., "unit":"K"})
- `pressure` (required for pt_flash; optional for bubble/dew if you only want saturation pressure)
  - either number (Pa), or {"value":..., "unit":"atm"|"bar"|"psi"|"Pa"}
- `composition`: list of {"id": "<MI label>", "z": <float>}

Supported MI labels right now:
CO2, C1, C2, C3, nC4, nC5, C6
(Heavy lump labels like C7-C12 are not supported yet in this repo snapshot.)

### Expected outputs

Put MI outputs in the `expected` object. Examples:

For pt_flash:
- `phase`: "vapor" | "liquid" | "two-phase"  (optional)
- `vapor_fraction`: float (optional)
- `x`: array of liquid composition aligned to input ordering (optional)
- `y`: array of vapor composition aligned to input ordering (optional)

For bubble/dew:
- `pressure_pa`: saturation pressure in Pa

### Tolerances

Optional `tolerances` object. Examples:
- `pressure_pa_atol`: absolute Pa tolerance (default 5e4 Pa ~ 0.5 bar)
- `composition_atol`: absolute mol fraction tolerance (default 1e-3)
- `vapor_fraction_atol`: absolute tolerance (default 1e-3)

## Notes on MI settings

MI-PVT must be configured to match:
- EOS: Peng-Robinson 1976 (PR)
- Binary interaction parameters (kij): ideally all zero unless explicitly set

Record MI settings in the case file under `mi_settings` if you can.

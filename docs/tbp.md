# TBP Status

A bounded standalone TBP kernel now exists in `pvtcore` at `pvtcore.experiments.tbp`.

Current supported surface:

- `pvtcore.experiments.tbp.simulate_tbp(...)` accepts phase-1 TBP cuts (`C<number>` names with `z` and `mw`, plus optional `sg`) and returns a standalone cut-resolved assay summary.
- The standalone result includes derived `z_plus` / `mw_plus_g_per_mol` plus cumulative mole- and mass-yield curves across the ordered cuts.
- Ordered single cuts, interval cuts such as `C7-C9`, and gapped but non-overlapping sequences are now accepted.
- Optional cut boiling points can be entered explicitly as `tb_k`; when `sg` is present and `tb_k` is omitted, the runtime preserves an estimated boiling-point curve for reporting.
- `fluid.plus_fraction.tbp_data.cuts` remains supported in the schema-driven `load_fluid_definition(...)` + `characterize_from_schema(...)` path.
- Pedersen heavy-end characterization now supports `solve_AB_from: fit_to_tbp` when TBP cuts are provided, so TBP data can drive the actual plus-fraction split used by downstream runtime workflows.
- The desktop plus-fraction editor now carries optional TBP cuts and the Pedersen `fit_to_tbp` solve mode through `RunConfig` for non-standalone runtime calculations.
- `pvtapp` now supports a bounded standalone TBP workflow end to end: desktop calculation selection, cut-table input, runtime/config execution via `RunConfig(calculation_type="tbp", tbp_config={...})`, and saved run artifacts/results rendering.
- TBP run results now also preserve an explicit aggregate heavy-end bridge context in the result artifact and reporting surfaces so the derived `C<n>+` intent remains visible for downstream EOS-backed reuse.

Current limitations:

- The first TBP cut must still start at `fluid.plus_fraction.cut_start`.
- The current `fit_to_tbp` implementation is a bounded Pedersen A/B fit. It does not yet drive broader TBP-specific property correlation fitting, BIP fitting, or a richer distillation-characterization workflow.
- The desktop GUI admission is still intentionally narrow: a standalone TBP assay screen plus optional TBP cuts inside the plus-fraction editor for Pedersen `fit_to_tbp`. It is not yet a full boiling-point / distillation / characterization workflow.
- The standalone TBP assay bridge is still aggregate-only. The new runtime-backed Pedersen fit path is separate from that bridge and does not automatically imply broader TBP-driven lumping, BIP, or EOS selection logic.

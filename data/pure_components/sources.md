# Pure Component Data Sources

This document describes the sources used for thermodynamic properties in the pure component database.

## Primary Source

**NIST Chemistry WebBook**
- URL: https://webbook.nist.gov/chemistry/
- Description: The NIST Chemistry WebBook provides access to data compiled and distributed by NIST under the Standard Reference Data Program.
- Properties obtained: Tc (critical temperature), Pc (critical pressure), Vc (critical volume), MW (molecular weight), Tb (normal boiling point)

## Component-Specific NIST Pages

- **Nitrogen (N2)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7727379&Mask=4
  - Reference: Jacobsen, Stewart, et al., 1986

- **Carbon Dioxide (CO2)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Mask=4
  - Critical properties reference: Suehiro et al., 1996
  - Note: CO2 has no liquid phase at atmospheric pressure; Tb value is sublimation temperature at 1 atm

- **Hydrogen Sulfide (H2S)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C7783064&Mask=4
  - References: Goodwin 1983; Cubitt et al. 1987

- **Methane (C1)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C74828&Mask=4

- **Ethane (C2)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C74840&Mask=4

- **Propane (C3)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C74986&Mask=4
  - Critical volume reference: Ambrose and Tsonopoulos, 1995

- **n-Butane (C4)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C106978&Mask=4

- **Isobutane (iC4)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C75285&Mask=4

- **n-Pentane (C5)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C109660&Mask=4

- **Isopentane (iC5)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C78784&Mask=4
  - Reference: Daubert (1996)

- **Neopentane (neoC5)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C463821&Mask=4

- **n-Hexane (C6)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C110543&Mask=4
  - Critical volume reference: Ambrose and Tsonopoulos, 1995

- **n-Heptane (C7)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C142825&Mask=4

- **n-Octane (C8)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C111659&Mask=4

- **n-Nonane (C9)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C111842&Mask=4

- **n-Decane (C10)**: https://webbook.nist.gov/cgi/cbook.cgi?ID=C124185&Mask=4

## Acentric Factor Sources

**Primary Source for Acentric Factors:**

1. **Critical Constants and Acentric Factors Table**
   - URL: https://www.kaylaiacovino.com/Petrology_Tools/Critical_Constants_and_Acentric_Factors.htm
   - Based on: Reid, R. C., J. M. Prausnitz, and B. E. Poling, 1987, "The Properties of Gases and Liquids, 4th Ed."
   - Components: N2, CO2, H2S, C1, C2, C3, C4, iC4, C5, iC5

2. **CoolProp Database** (v7.2.0)
   - URL: https://coolprop.org/
   - Documentation pages:
     - Neopentane: https://coolprop.org/fluid_properties/fluids/Neopentane.html
     - n-Hexane: https://coolprop.org/fluid_properties/fluids/n-Hexane.html
     - n-Heptane: https://coolprop.org/fluid_properties/fluids/n-Heptane.html
     - n-Octane: https://coolprop.org/fluid_properties/fluids/n-Octane.html
     - n-Nonane: https://coolprop.org/fluid_properties/fluids/n-Nonane.html
     - n-Decane: https://coolprop.org/fluid_properties/fluids/n-Decane.html
   - Note: CoolProp is a thermophysical property library that compiles data from various sources including NIST REFPROP

## Standard References

The data in this database is consistent with values from the following standard references:

1. **Reid, R. C., Prausnitz, J. M., and Poling, B. E.** (1987). *The Properties of Gases and Liquids, 4th Edition*. McGraw-Hill.

2. **Poling, B. E., Prausnitz, J. M., and O'Connell, J. P.** (2001). *The Properties of Gases and Liquids, 5th Edition*. McGraw-Hill.

3. **NIST Standard Reference Database 69: NIST Chemistry WebBook**. National Institute of Standards and Technology. https://doi.org/10.18434/T4D303

4. **Bell, I. H., et al.** (2014). "Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and the Open-Source Thermophysical Property Library CoolProp." *Industrial & Engineering Chemistry Research*, 53(6), 2498-2508.

## Data Quality and Uncertainty

All values are reported as compiled by the Thermodynamics Research Center (TRC) at NIST Boulder Laboratories. Where multiple experimental values were available, NIST reported averages with uncertainty ranges. The uncertainties vary by component and property but are typically:

- Critical temperature: ±0.1 to ±2 K
- Critical pressure: ±0.1 to ±0.8 bar
- Critical volume: ±0.001 to ±0.005 L/mol
- Normal boiling point: ±0.2 to ±0.6 K

Acentric factors are dimensionless parameters calculated from vapor pressure data and typically have uncertainties of ±0.001 to ±0.01.

## Data Compilation Date

Data compiled: January 31, 2026

## Notes

1. **Critical Volume Conversions**: All volumes were converted from L/mol to m³/mol (1 L/mol = 0.001 m³/mol)

2. **Pressure Conversions**: All pressures were converted from bar to Pa (1 bar = 100,000 Pa)

3. **Missing Values**:
   - N2 critical volume was estimated from literature values (89.2 cm³/mol) as it was not explicitly provided in the NIST phase change data
   - H2S critical volume was calculated from critical density and molecular weight

4. **Special Cases**:
   - CO2: The normal boiling point is not defined at 1 atm as CO2 sublimes directly from solid to gas. The value provided (194.7 K) is the sublimation temperature at 1 atm.

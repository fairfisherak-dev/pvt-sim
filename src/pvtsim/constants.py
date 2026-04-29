"""Physical constants and tolerances. Values cited to LECTURE_SLIDES_MERGED.md."""

# Gas constant in field units (slide 378)
R = 10.732  # psia·ft³/(lbmol·°R)

# Standard conditions (slide 281)
SC_P = 14.696   # psia
SC_T_F = 60.0   # °F
SC_T_R = 519.67 # °R

# Molar volume of gas at SC (slide 283)
SCF_PER_LBMOL = 379.6

# Barrel conversion (slide 428)
FT3_PER_BBL = 5.615

# PR EOS dimensionless constants (slide 376)
OMEGA_A = 0.45724
OMEGA_B = 0.07780

# Convergence
FLASH_TOL = 1e-10
RR_TOL = 1e-12
SAT_TOL = 1e-6
MAX_ITER = 100

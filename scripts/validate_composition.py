#!/usr/bin/env python3
"""
Validation script for comparing pvtcore results against MI-PVT or other simulators.

Usage:
    python scripts/validate_composition.py

Edit the COMPOSITION section below to test different mixtures.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from pvtcore.models import load_components
from pvtcore.eos import PengRobinsonEOS
from pvtcore.flash import pt_flash, calculate_bubble_point, calculate_dew_point
from pvtcore.stability import is_stable
from pvtcore.envelope import calculate_phase_envelope

# ============================================================
# COMPOSITION - EDIT THIS SECTION
# ============================================================
# Component IDs available: N2, CO2, H2S, C1, C2, C3, C4, iC4, C5, iC5, neoC5, C6, C7, C8, C9, C10

MIXTURE_NAME = "MI-PVT Gas 1"

COMPOSITION = {
    "CO2": 0.6498,
    "C1":  0.1057,
    "C2":  0.1058,
    "C3":  0.1235,
    "C4":  0.0152,
}

# Test conditions
T_BUBBLE_DEW = 250.0    # K - Temperature for bubble/dew point calculations
T_FLASH = 250.0         # K - Temperature for flash calculation
P_FLASH = 5e6        # Pa - Pressure for flash calculation (5 MPa)
T_ENVELOPE_START = 150.0  # K - Starting temperature for phase envelope

# ============================================================
# END OF USER CONFIGURATION
# ============================================================


def main():
    # Load component database
    all_comps = load_components()
    
    # Build component list and composition array
    components = []
    z = []
    for comp_id, mole_frac in COMPOSITION.items():
        if comp_id not in all_comps:
            print(f"ERROR: Component '{comp_id}' not in database.")
            print(f"Available: {', '.join(sorted(all_comps.keys()))}")
            sys.exit(1)
        components.append(all_comps[comp_id])
        z.append(mole_frac)
    
    z = np.array(z)
    
    # Validate composition
    if abs(z.sum() - 1.0) > 1e-4:
        print(f"WARNING: Composition sums to {z.sum():.6f}, not 1.0")
        print("Normalizing...")
        z = z / z.sum()
    
    # Create EOS
    eos = PengRobinsonEOS(components)
    
    # Header
    print("=" * 70)
    print(f"PVTCORE VALIDATION: {MIXTURE_NAME}")
    print("=" * 70)
    
    # Composition table
    print("\nCOMPOSITION:")
    print("-" * 40)
    for i, c in enumerate(components):
        print(f"  {c.name:20s} ({COMPOSITION.keys().__iter__().__next__() if i == 0 else list(COMPOSITION.keys())[i]:4s}): {z[i]:.4f}")
    print(f"  {'TOTAL':20s}       : {z.sum():.4f}")
    
    # Mixture properties
    MW_mix = sum(z[i] * c.MW for i, c in enumerate(components))
    Tc_mix = sum(z[i] * c.Tc for i, c in enumerate(components))  # Simple mixing rule
    Pc_mix = sum(z[i] * c.Pc for i, c in enumerate(components))  # Simple mixing rule
    print(f"\nMIXTURE PROPERTIES (linear mixing rules):")
    print(f"  MW_mix: {MW_mix:.2f} g/mol")
    print(f"  Tc_mix: {Tc_mix:.1f} K (pseudo-critical)")
    print(f"  Pc_mix: {Pc_mix/1e6:.2f} MPa (pseudo-critical)")
    
    # 1. Bubble Point
    print("\n" + "=" * 70)
    print(f"BUBBLE POINT at T = {T_BUBBLE_DEW} K ({T_BUBBLE_DEW - 273.15:.1f} °C)")
    print("=" * 70)
    try:
        bp = calculate_bubble_point(T_BUBBLE_DEW, z, components, eos)
        print(f"  Pressure:   {bp.pressure/1e6:.4f} MPa  ({bp.pressure/1e5:.2f} bar, {bp.pressure/6894.76:.1f} psia)")
        print(f"  Converged:  {bp.converged}")
        print(f"  Iterations: {bp.iterations}")
        print(f"\n  Incipient vapor composition:")
        for i, c in enumerate(components):
            print(f"    {c.name:15s}: {bp.vapor_composition[i]:.4f}")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # 2. Dew Point
    print("\n" + "=" * 70)
    print(f"DEW POINT at T = {T_BUBBLE_DEW} K ({T_BUBBLE_DEW - 273.15:.1f} °C)")
    print("=" * 70)
    try:
        dp = calculate_dew_point(T_BUBBLE_DEW, z, components, eos)
        print(f"  Pressure:   {dp.pressure/1e6:.4f} MPa  ({dp.pressure/1e5:.2f} bar, {dp.pressure/6894.76:.1f} psia)")
        print(f"  Converged:  {dp.converged}")
        print(f"  Iterations: {dp.iterations}")
        print(f"\n  Incipient liquid composition:")
        for i, c in enumerate(components):
            print(f"    {c.name:15s}: {dp.liquid_composition[i]:.4f}")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # 3. PT Flash
    print("\n" + "=" * 70)
    print(f"PT FLASH at T = {T_FLASH} K, P = {P_FLASH/1e6:.2f} MPa")
    print("=" * 70)
    try:
        flash = pt_flash(P_FLASH, T_FLASH, z, components, eos)
        print(f"  Phase:         {flash.phase}")
        print(f"  Vapor fraction: {flash.vapor_fraction:.4f}")
        print(f"  Converged:     {flash.converged}")
        print(f"  Iterations:    {flash.iterations}")
        
        if flash.phase == 'two-phase':
            print(f"\n  Liquid composition (x):")
            for i, c in enumerate(components):
                print(f"    {c.name:15s}: {flash.liquid_composition[i]:.4f}")
            print(f"\n  Vapor composition (y):")
            for i, c in enumerate(components):
                print(f"    {c.name:15s}: {flash.vapor_composition[i]:.4f}")
            print(f"\n  K-values (y/x):")
            for i, c in enumerate(components):
                print(f"    {c.name:15s}: {flash.K_values[i]:.4f}")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # 4. Stability
    print("\n" + "=" * 70)
    print(f"STABILITY at T = {T_FLASH} K, P = {P_FLASH/1e6:.2f} MPa")
    print("=" * 70)
    try:
        stable_L = is_stable(z, P_FLASH, T_FLASH, eos, feed_phase='liquid')
        stable_V = is_stable(z, P_FLASH, T_FLASH, eos, feed_phase='vapor')
        print(f"  Stable as liquid: {stable_L}")
        print(f"  Stable as vapor:  {stable_V}")
        if stable_L or stable_V:
            print(f"  --> Single phase")
        else:
            print(f"  --> Two-phase region")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # 5. Phase Envelope
    print("\n" + "=" * 70)
    print("PHASE ENVELOPE")
    print("=" * 70)
    try:
        envelope = calculate_phase_envelope(z, components, eos, T_start=T_ENVELOPE_START)
        print(f"  Bubble curve points: {envelope.n_bubble_points}")
        print(f"  Dew curve points:    {envelope.n_dew_points}")
        print(f"  Converged:           {envelope.converged}")
        
        if envelope.critical_P is not None:
            print(f"\n  Cricondenbar (max P on envelope):")
            print(f"    T = {envelope.critical_T:.1f} K ({envelope.critical_T - 273.15:.1f} °C)")
            print(f"    P = {envelope.critical_P/1e6:.3f} MPa ({envelope.critical_P/1e5:.1f} bar)")
        
        # Print bubble curve
        print(f"\n  BUBBLE CURVE (first 15 points):")
        print(f"  {'T (K)':>10s} {'T (°C)':>10s} {'P (MPa)':>10s} {'P (bar)':>10s}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for i in range(min(15, len(envelope.bubble_T))):
            T_K = envelope.bubble_T[i]
            P_Pa = envelope.bubble_P[i]
            print(f"  {T_K:10.1f} {T_K-273.15:10.1f} {P_Pa/1e6:10.3f} {P_Pa/1e5:10.1f}")
        if len(envelope.bubble_T) > 15:
            print(f"  ... ({len(envelope.bubble_T)} total points)")
        
        # Print dew curve
        print(f"\n  DEW CURVE (first 15 points):")
        print(f"  {'T (K)':>10s} {'T (°C)':>10s} {'P (MPa)':>10s} {'P (bar)':>10s}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for i in range(min(15, len(envelope.dew_T))):
            T_K = envelope.dew_T[i]
            P_Pa = envelope.dew_P[i]
            print(f"  {T_K:10.1f} {T_K-273.15:10.1f} {P_Pa/1e6:10.3f} {P_Pa/1e5:10.1f}")
        if len(envelope.dew_T) > 15:
            print(f"  ... ({len(envelope.dew_T)} total points)")
            
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # 6. Z-factor at a few pressures (vapor phase)
    print("\n" + "=" * 70)
    print(f"Z-FACTOR (compressibility) at T = {T_FLASH} K")
    print("=" * 70)
    print(f"  {'P (MPa)':>10s} {'Z':>10s} {'ρ (kg/m³)':>12s} {'Vm (L/mol)':>12s}")
    print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
    
    for P_test in [1e6, 2e6, 3e6, 5e6, 7e6, 10e6]:
        try:
            # Check stability first
            stable = is_stable(z, P_test, T_FLASH, eos, feed_phase='vapor')
            if stable:
                result = eos.calculate(P_test, T_FLASH, z, phase='vapor')
                print(f"  {P_test/1e6:10.1f} {result.Z:10.4f} {result.density:12.2f} {result.molar_volume*1e3:12.4f}")
            else:
                print(f"  {P_test/1e6:10.1f} {'(two-phase)':>10s}")
        except Exception as e:
            print(f"  {P_test/1e6:10.1f} {'ERROR':>10s}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
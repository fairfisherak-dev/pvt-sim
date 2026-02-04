#!/usr/bin/env python3
"""Validation script for phase envelope calculation.

Generates and plots a phase envelope for a C1-C10 binary mixture
to verify the envelope tracing algorithm works correctly.

Example
-------
$ python scripts/validate_envelope.py
"""

import sys
from pathlib import Path

# Add src to path so we can import pvtcore
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import matplotlib.pyplot as plt

from pvtcore.models.component import load_components
from pvtcore.eos import PengRobinsonEOS
from pvtcore.envelope import calculate_phase_envelope


def main():
    """Generate and plot C1-C10 phase envelope."""

    # Load component database
    print("Loading component database...")
    components = load_components()

    # Select C1 (methane) and C10 (n-decane)
    c1 = components['C1']
    c10 = components['C10']
    component_list = [c1, c10]

    print(f"Components:")
    print(f"  {c1.name}: Tc = {c1.Tc:.2f} K, Pc = {c1.Pc_MPa:.4f} MPa")
    print(f"  {c10.name}: Tc = {c10.Tc:.2f} K, Pc = {c10.Pc_MPa:.4f} MPa")

    # Define composition (70% C1, 30% C10)
    composition = np.array([0.7, 0.3])
    print(f"\nComposition: {composition[0]:.1f} {c1.name}, {composition[1]:.1f} {c10.name}")

    # Create Peng-Robinson EOS
    print("\nInitializing Peng-Robinson EOS...")
    eos = PengRobinsonEOS(component_list)

    # Calculate phase envelope
    print("Calculating phase envelope (this may take a minute)...")
    envelope = calculate_phase_envelope(
        composition=composition,
        components=component_list,
        eos=eos,
        T_start=150.0,  # Start at 150 K
        T_step_initial=5.0,  # 5 K initial step
        max_points=500,
        detect_critical=True
    )

    # Print results
    print(f"\nPhase Envelope Results:")
    print(f"  Converged: {envelope.converged}")
    print(f"  Bubble points: {envelope.n_bubble_points}")
    print(f"  Dew points: {envelope.n_dew_points}")

    if envelope.critical_T is not None:
        print(f"  Critical point: T = {envelope.critical_T:.2f} K, P = {envelope.critical_P/1e6:.4f} MPa")
    else:
        print(f"  Critical point: Not detected")

    # Create plot
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot bubble point curve
    if len(envelope.bubble_T) > 0:
        ax.plot(
            envelope.bubble_T,
            envelope.bubble_P / 1e6,  # Convert Pa to MPa
            'b-',
            linewidth=2,
            label='Bubble Point Curve',
            marker='o',
            markersize=3,
            markevery=10
        )

    # Plot dew point curve (if available)
    if len(envelope.dew_T) > 0:
        ax.plot(
            envelope.dew_T,
            envelope.dew_P / 1e6,  # Convert Pa to MPa
            'r-',
            linewidth=2,
            label='Dew Point Curve',
            marker='s',
            markersize=3,
            markevery=10
        )

    # Mark critical point
    if envelope.critical_T is not None and envelope.critical_P is not None:
        ax.plot(
            envelope.critical_T,
            envelope.critical_P / 1e6,
            'k*',
            markersize=20,
            label=f'Critical Point ({envelope.critical_T:.1f} K, {envelope.critical_P/1e6:.2f} MPa)',
            markeredgecolor='gold',
            markeredgewidth=1.5
        )

    # Format plot
    ax.set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pressure (MPa)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Phase Envelope: {composition[0]:.1f} {c1.name} + {composition[1]:.1f} {c10.name}',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set reasonable axis limits
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Add two-phase region label
    if len(envelope.bubble_T) > 0 and len(envelope.dew_T) > 0:
        # Place label in the middle of the envelope
        mid_idx_bubble = len(envelope.bubble_T) // 2
        mid_T = envelope.bubble_T[mid_idx_bubble]
        mid_P = envelope.bubble_P[mid_idx_bubble] / 1e6
        ax.text(
            mid_T, mid_P * 0.5,
            'Two-Phase\nRegion',
            fontsize=11,
            ha='center',
            va='center',
            style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    plt.tight_layout()

    # Save plot
    output_path = project_root / "outputs" / "c1_c10_envelope.png"
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully!")

    # Display plot
    print("\nValidation complete! Check outputs/c1_c10_envelope.png for the result.")

    return envelope


if __name__ == "__main__":
    try:
        envelope = main()
    except Exception as e:
        print(f"\nError during validation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

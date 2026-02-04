#!/usr/bin/env python3
"""Interactive Python shell for PVT simulator development.

Preloads common imports and provides convenient access to core functionality.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Core imports
import numpy as np

# PVT Core modules
from pvtcore.models.component import load_components, get_component, Component
from pvtcore.eos import PengRobinsonEOS, CubicEOS
from pvtcore.flash import pt_flash, FlashResult
from pvtcore.flash.bubble_point import calculate_bubble_point, BubblePointResult
from pvtcore.flash.dew_point import calculate_dew_point, DewPointResult
from pvtcore.stability import michelsen_stability_test, is_stable, StabilityResult
from pvtcore.envelope import calculate_phase_envelope, EnvelopeResult

# Helper functions
def quick_flash_example():
    """Run a quick flash calculation example."""
    print("\n=== Quick Flash Example ===")

    # Load components
    components = load_components()
    c1 = components['C1']
    c4 = components['C4']

    # Create mixture
    comp_list = [c1, c4]
    z = np.array([0.5, 0.5])

    # Create EOS
    eos = PengRobinsonEOS(comp_list)

    # Flash at 250 K, 2 MPa
    T = 250.0  # K
    P = 2e6    # Pa

    print(f"\nFlashing {z[0]:.1f} {c1.name} + {z[1]:.1f} {c4.name}")
    print(f"at T = {T:.1f} K, P = {P/1e6:.2f} MPa")

    result = pt_flash(P, T, z, comp_list, eos)

    print(f"\nResults:")
    print(f"  Phases: {result.phase}")
    print(f"  Vapor fraction: {result.vapor_fraction:.4f}")
    if result.phase == 'two-phase':
        print(f"  Liquid composition: {result.liquid_composition}")
        print(f"  Vapor composition: {result.vapor_composition}")

    return result


def quick_stability_example():
    """Run a quick stability test example."""
    print("\n=== Quick Stability Test Example ===")

    # Load components
    components = load_components()
    c1 = components['C1']
    c10 = components['C10']

    # Create mixture
    comp_list = [c1, c10]
    z = np.array([0.5, 0.5])

    # Create EOS
    eos = PengRobinsonEOS(comp_list)

    # Test stability at 300 K, 1 MPa
    T = 300.0  # K
    P = 1e6    # Pa

    print(f"\nTesting stability of {z[0]:.1f} {c1.name} + {z[1]:.1f} {c10.name}")
    print(f"at T = {T:.1f} K, P = {P/1e6:.2f} MPa")

    stable = is_stable(z, P, T, eos)

    print(f"\nResult: {'STABLE' if stable else 'UNSTABLE'}")
    print(f"  (Expect unstable - this is in the two-phase region)")

    return stable


def main():
    """Start interactive shell with preloaded imports."""

    # Banner
    print("=" * 70)
    print("PVT Simulator - Interactive Python Shell")
    print("=" * 70)
    print()
    print("Preloaded imports:")
    print("  numpy as np")
    print()
    print("Models:")
    print("  Component, load_components(), get_component()")
    print()
    print("EOS:")
    print("  PengRobinsonEOS, CubicEOS")
    print()
    print("Flash:")
    print("  pt_flash(), FlashResult")
    print("  calculate_bubble_point(), BubblePointResult")
    print("  calculate_dew_point(), DewPointResult")
    print()
    print("Stability:")
    print("  michelsen_stability_test(), is_stable(), StabilityResult")
    print()
    print("Envelope:")
    print("  calculate_phase_envelope(), EnvelopeResult")
    print()
    print("Helper functions:")
    print("  quick_flash_example() - Run example flash calculation")
    print("  quick_stability_example() - Run example stability test")
    print()
    print("=" * 70)
    print()

    # Example: Load components for convenience
    print("Loading component database...")
    components = load_components()
    print(f"Loaded {len(components)} components: {', '.join(sorted(components.keys()))}")
    print()
    print("Tip: Access components with components['C1'], components['C10'], etc.")
    print("=" * 70)
    print()

    # Start interactive shell
    try:
        import code

        # Create namespace with all preloaded objects
        namespace = {
            'np': np,
            'Component': Component,
            'load_components': load_components,
            'get_component': get_component,
            'components': components,
            'PengRobinsonEOS': PengRobinsonEOS,
            'CubicEOS': CubicEOS,
            'pt_flash': pt_flash,
            'FlashResult': FlashResult,
            'calculate_bubble_point': calculate_bubble_point,
            'BubblePointResult': BubblePointResult,
            'calculate_dew_point': calculate_dew_point,
            'DewPointResult': DewPointResult,
            'michelsen_stability_test': michelsen_stability_test,
            'is_stable': is_stable,
            'StabilityResult': StabilityResult,
            'calculate_phase_envelope': calculate_phase_envelope,
            'EnvelopeResult': EnvelopeResult,
            'quick_flash_example': quick_flash_example,
            'quick_stability_example': quick_stability_example,
        }

        code.interact(local=namespace, banner="")

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

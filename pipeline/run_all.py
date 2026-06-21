# ============================================================================
# run_all.py — Master Pipeline Runner
# ============================================================================
# Runs all 7 phases in sequence. Can also be used to run individual phases.
#
# Usage:
#   python pipeline/run_all.py              # Run everything
#   python pipeline/run_all.py --phase 3    # Run only Phase 3
# ============================================================================

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_phase(phase_num, phase_name, module_name):
    """Run a single pipeline phase and time it."""
    print(f"\n{'#' * 80}")
    print(f"# PHASE {phase_num}: {phase_name}")
    print(f"{'#' * 80}\n")

    start = time.time()

    try:
        module = __import__(f"pipeline.{module_name}", fromlist=[module_name])
        elapsed = time.time() - start
        print(f"\n✅ Phase {phase_num} completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ Phase {phase_num} failed after {elapsed:.1f} seconds: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the traffic analysis pipeline")
    parser.add_argument("--phase", type=int, default=None,
                        help="Run only a specific phase (1-7)")
    parser.add_argument("--start-from", type=int, default=1,
                        help="Start from a specific phase (1-7)")
    args = parser.parse_args()

    phases = [
        (1, "Data Cleaning & Feature Engineering", "data_cleaning"),
        (2, "OSM Road Context Enrichment", "osm_enrichment"),
        (3, "Dual Hotspot Detection", "hotspot_detection"),
        (4, "Dwell-Time & Queueing Model", "queueing_model"),
        (5, "Congestion Impact Score", "cis_calculation"),
        (6, "Forecasting", "forecasting"),
        (7, "Patrol Prioritization", "prioritization"),
    ]

    total_start = time.time()

    print("=" * 80)
    print("BENGALURU PARKING CONGESTION INTELLIGENCE SYSTEM")
    print("Full Pipeline Runner")
    print("=" * 80)

    if args.phase:
        # Run single phase
        phase = [p for p in phases if p[0] == args.phase]
        if phase:
            run_phase(*phase[0])
        else:
            print(f"Invalid phase number: {args.phase}. Valid: 1-7")
    else:
        # Run all phases from start_from
        results = {}
        for phase_num, phase_name, module_name in phases:
            if phase_num < args.start_from:
                continue
            success = run_phase(phase_num, phase_name, module_name)
            results[phase_num] = success
            if not success:
                print(f"\n⚠️ Pipeline stopped at Phase {phase_num}.")
                print(f"   Fix the issue and re-run with: --start-from {phase_num}")
                break

        total_elapsed = time.time() - total_start
        print(f"\n{'=' * 80}")
        print(f"PIPELINE COMPLETE — Total time: {total_elapsed:.1f} seconds "
              f"({total_elapsed/60:.1f} minutes)")
        print(f"{'=' * 80}")

        for num, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} Phase {num}: {phases[num-1][1]}")


if __name__ == "__main__":
    main()

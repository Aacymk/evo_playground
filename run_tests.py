"""
run_tests.py — EVO-SIM test runner
====================================
Run all tests, or pick a specific suite.

Usage:
    python run_tests.py              # fast suite only (~60s)
    python run_tests.py --all        # every test including slow ones (~5-10 min)
    python run_tests.py --suite brain
    python run_tests.py --suite lifecycle
    python run_tests.py --suite sensors
    python run_tests.py --suite conservation
    python run_tests.py --suite determinism
    python run_tests.py --suite logging

Suites marked [SLOW] can take several minutes because they run the full
simulation loop for many frames to verify logging and reproducibility.
"""

import argparse
import subprocess
import sys
import time

SUITES = {
    "brain":        ("tests/test_brain_evolution.py", "Brain & Evolution",    False),
    "lifecycle":    ("tests/test_lifecycle.py",        "Lifecycle & Deaths",   False),
    "sensors":      ("tests/test_sensors_world.py",    "Sensors & World",      False),
    "conservation": ("tests/test_conservation.py",     "Conservation & Cohort",False),
    "determinism":  ("tests/test_determinism.py",      "Determinism & Seeds",  True),
    "logging":      ("tests/test_logging.py",          "Logging Consistency",  True),
}

FAST_SUITES = [k for k, (_, _, slow) in SUITES.items() if not slow]
ALL_SUITES  = list(SUITES.keys())


def run_suite(name):
    path, label, slow = SUITES[name]
    tag = " [SLOW]" if slow else ""
    print(f"\n{'─'*55}")
    print(f"  {label}{tag}")
    print(f"{'─'*55}")

    start  = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-v", "--tb=short", "--no-header"],
        capture_output=False,
    )
    elapsed = time.perf_counter() - start

    status = "PASSED" if result.returncode == 0 else "FAILED"
    print(f"\n  → {status} in {elapsed:.1f}s")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="EVO-SIM test runner")
    parser.add_argument("--all",   action="store_true",
                        help="Run all suites including slow ones")
    parser.add_argument("--suite", choices=ALL_SUITES, default=None,
                        help="Run a single named suite")
    args = parser.parse_args()

    if args.suite:
        to_run = [args.suite]
    elif args.all:
        to_run = ALL_SUITES
    else:
        to_run = FAST_SUITES

    slow_count = sum(1 for k in to_run if SUITES[k][2])
    print(f"\nEVO-SIM Test Runner")
    print(f"Suites to run: {', '.join(to_run)}")
    if slow_count:
        print(f"Note: {slow_count} slow suite(s) included — may take several minutes.")
    if not args.all and not args.suite:
        print("Tip: use --all to include slow suites (determinism, logging).")

    results = {}
    total_start = time.perf_counter()

    for name in to_run:
        results[name] = run_suite(name)

    total_elapsed = time.perf_counter() - total_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  RESULTS  ({total_elapsed:.1f}s total)")
    print(f"{'═'*55}")
    all_passed = True
    for name in to_run:
        _, label, slow = SUITES[name]
        icon   = "✓" if results[name] else "✗"
        status = "PASSED" if results[name] else "FAILED"
        tag    = " [SLOW]" if slow else ""
        print(f"  {icon}  {label}{tag}  —  {status}")
        if not results[name]:
            all_passed = False

    print(f"{'═'*55}")
    if all_passed:
        print(f"  All {len(to_run)} suite(s) passed.")
    else:
        failed = [n for n in to_run if not results[n]]
        print(f"  {len(failed)} suite(s) FAILED: {', '.join(failed)}")
        print(f"  Run a specific suite for details: python run_tests.py --suite {failed[0]}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

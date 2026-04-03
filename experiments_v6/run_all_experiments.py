"""
SubstanceNet v4 — Run All Experiments
======================================
Master script that runs all experiments sequentially.
Generates all JSON results and PNG/PDF figures.

Usage: python experiments/run_all_experiments.py
Time: ~5 minutes on GPU
"""
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

EXPERIMENTS = [
    ('01_mnist_backprop', '01: MNIST Backpropagation'),
    ('02_cognitive_battery', '02: Cognitive Task Battery'),
    ('03_recognition_paradigm', '03: Recognition Paradigm'),
    ('04_velocity_tuning', '04: Velocity Tuning Curve'),
    ('05_hebbian_maturation', '05: Hebbian Maturation'),
    ('06_kappa_analysis', '06: κ ≈ 1 Emergence Analysis'),
]


def main():
    print()
    print('=' * 60)
    print('  SubstanceNet v4 — Running All Experiments')
    print('=' * 60)

    total_start = time.time()
    results_summary = {}

    for module_name, title in EXPERIMENTS:
        print(f'\n{"─" * 60}')
        print(f'  Starting: {title}')
        print(f'{"─" * 60}')

        exp_start = time.time()
        try:
            mod = __import__(module_name)
            mod.run()
            elapsed = time.time() - exp_start
            results_summary[module_name] = {
                'status': 'OK',
                'time': f'{elapsed:.1f}s',
            }
            print(f'\n  [{title}] completed in {elapsed:.1f}s')
        except Exception as e:
            elapsed = time.time() - exp_start
            results_summary[module_name] = {
                'status': f'FAILED: {e}',
                'time': f'{elapsed:.1f}s',
            }
            print(f'\n  [{title}] FAILED in {elapsed:.1f}s: {e}')

    total_elapsed = time.time() - total_start

    # Summary
    print(f'\n{"=" * 60}')
    print(f'  SUMMARY')
    print(f'{"=" * 60}')
    print(f'  {"Experiment":<35} {"Status":<10} {"Time":>8}')
    print(f'  {"─" * 35} {"─" * 10} {"─" * 8}')

    all_ok = True
    for module_name, title in EXPERIMENTS:
        r = results_summary.get(module_name, {})
        status = r.get('status', 'UNKNOWN')
        t = r.get('time', '?')
        mark = '✓' if status == 'OK' else '✗'
        if status != 'OK':
            all_ok = False
        print(f'  {title:<35} {mark} {status:<8} {t:>8}')

    print(f'  {"─" * 35} {"─" * 10} {"─" * 8}')
    print(f'  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)')
    print()

    # Check outputs
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    figures_dir = os.path.join(os.path.dirname(__file__), 'figures')

    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    png_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]

    print(f'  Results: {len(json_files)} JSON files in experiments/results/')
    print(f'  Figures: {len(png_files)} PNG files in figures/')
    print()

    if all_ok:
        print('  All experiments PASSED ✓')
    else:
        print('  Some experiments FAILED ✗')
        sys.exit(1)

    print()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Calculate reproducible human baseline for zero-crossings rate.

Uses random sampling with a fixed seed to select one human solution per problem,
simulating "one human solving all available problems once."
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_human_baseline(csv_path='marupudi_data/tsp-singles.csv', n_iterations=1000, seed=42):
    """
    Calculate human baseline for zero-crossings rate using bootstrap sampling.

    Args:
        csv_path: Path to human data CSV file
        n_iterations: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        dict with baseline statistics
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Load human data
    df = pd.read_csv(csv_path)

    # Group by problem (base_uuid)
    problems = df.groupby('base_uuid')['number_of_crossings'].apply(list).to_dict()

    # Bootstrap sampling: randomly select one human per problem
    zero_crossing_rates = []

    for iteration in range(n_iterations):
        zero_count = 0
        total_count = 0

        for problem_id, crossings_list in problems.items():
            # Randomly select one human solution for this problem
            selected_crossings = np.random.choice(crossings_list)
            total_count += 1
            if selected_crossings == 0:
                zero_count += 1

        if total_count > 0:
            zero_crossing_rates.append(zero_count / total_count * 100)

    # Calculate statistics
    mean_rate = np.mean(zero_crossing_rates)
    std_rate = np.std(zero_crossing_rates)
    median_rate = np.median(zero_crossing_rates)
    ci_95_lower = np.percentile(zero_crossing_rates, 2.5)
    ci_95_upper = np.percentile(zero_crossing_rates, 97.5)

    # Overall statistics (all human attempts, not random baseline)
    all_crossings = df['number_of_crossings']
    overall_zero_pct = (all_crossings == 0).mean() * 100
    overall_mean = all_crossings.mean()
    overall_median = all_crossings.median()

    results = {
        'n_problems': len(problems),
        'n_iterations': n_iterations,
        'n_participants': df['participant_number'].nunique(),
        'total_attempts': len(df),
        'seed': seed,
        # Random baseline (bootstrap)
        'baseline_mean': mean_rate,
        'baseline_median': median_rate,
        'baseline_std': std_rate,
        'baseline_ci_lower': ci_95_lower,
        'baseline_ci_upper': ci_95_upper,
        # Overall (all attempts)
        'overall_zero_pct': overall_zero_pct,
        'overall_mean_crossings': overall_mean,
        'overall_median_crossings': overall_median,
    }

    return results


def print_results(results):
    """Print formatted results."""
    print('='*70)
    print('RANDOM HUMAN BASELINE: ZERO CROSSINGS RATE')
    print('='*70)
    print(f'Data summary:')
    print(f'  Total human attempts: {results["total_attempts"]}')
    print(f'  Unique problems: {results["n_problems"]}')
    print(f'  Unique participants: {results["n_participants"]}')
    print(f'  Bootstrap iterations: {results["n_iterations"]}')
    print(f'  Random seed: {results["seed"]}')
    print()
    print(f'Random baseline (one human per problem):')
    print(f'  Mean zero-crossings rate: {results["baseline_mean"]:.2f}%')
    print(f'  Median zero-crossings rate: {results["baseline_median"]:.2f}%')
    print(f'  Std deviation: {results["baseline_std"]:.2f}%')
    print(f'  95% CI: [{results["baseline_ci_lower"]:.2f}%, {results["baseline_ci_upper"]:.2f}%]')
    print()
    print(f'Overall statistics (all {results["total_attempts"]} attempts):')
    print(f'  Zero crossings rate: {results["overall_zero_pct"]:.2f}%')
    print(f'  Mean crossings: {results["overall_mean_crossings"]:.2f}')
    print(f'  Median crossings: {results["overall_median_crossings"]:.2f}')
    print('='*70)
    print()
    print(f'For plotting: use {results["baseline_mean"]:.1f}% and {results["overall_mean_crossings"]:.2f}')
    print(f'For LaTeX: ${results["baseline_mean"]:.1f}\\%$ (95\\% CI: [{results["baseline_ci_lower"]:.1f}\\%, {results["baseline_ci_upper"]:.1f}\\%])')
    print('='*70)


def main():
    """Main function."""
    csv_path = Path('marupudi_data/tsp-singles.csv')

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return 1

    # Calculate baseline with fixed seed for reproducibility
    results = calculate_human_baseline(csv_path=csv_path, n_iterations=1000, seed=42)

    # Print results
    print_results(results)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

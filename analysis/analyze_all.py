#!/usr/bin/env python3
"""
Main analysis pipeline for TSP experiments.

Runs all heuristic measurements and generates visualizations comparing
VLM and human performance.
"""

import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import subprocess


def run_command(cmd, description):
    """Run a shell command and print status."""
    print(f"\n{'='*60}")
    print(f"{description}...")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠ Warning: {description} failed with code {result.returncode}")
        return False
    print(f"✓ {description} completed")
    return True


def add_metadata_to_results(results_df, metadata_csv="problem_metadata.csv"):
    """Add problem metadata (group, optimal_length) to results."""
    if not Path(metadata_csv).exists():
        print(f"⚠ Metadata file {metadata_csv} not found, skipping metadata")
        return results_df

    metadata_df = pd.read_csv(metadata_csv)
    results_df = results_df.merge(
        metadata_df[['problem_id', 'group', 'optimal_length']],
        on='problem_id',
        how='left'
    )
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Run complete TSP analysis pipeline"
    )
    parser.add_argument("vlm_solutions", help="VLM solutions JSON file")
    parser.add_argument("--vlm-clusters", help="VLM clusters JSON file (optional)")
    parser.add_argument("--human-trials", default="marupudi_data/tsp-trials-anonymized.json",
                       help="Human trials JSON (default: marupudi_data/tsp-trials-anonymized.json)")
    parser.add_argument("--human-singles", default="marupudi_data/tsp-singles.csv",
                       help="Human summary CSV (default: marupudi_data/tsp-singles.csv)")
    parser.add_argument("--optimal-dir", default="optimal-tsps",
                       help="Directory with optimal TSP files (default: optimal-tsps)")
    parser.add_argument("-o", "--output-dir", default=None,
                       help="Output directory (default: results/[model_name])")
    parser.add_argument("--skip-human", action="store_true",
                       help="Skip human analysis")

    args = parser.parse_args()

    # Extract model name from solutions file
    vlm_name = Path(args.vlm_solutions).stem

    # Create model-specific output directory
    if args.output_dir is None:
        output_dir = Path("results") / vlm_name
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# TSP Analysis Pipeline")
    print(f"# VLM: {vlm_name}")
    print(f"# Output: {output_dir}/")
    print(f"{'#'*60}\n")

    # ========================================
    # VLM Analysis
    # ========================================

    # 1. Convex hull adherence (VLM)
    hull_vlm_csv = output_dir / f"{vlm_name}_hull.csv"
    run_command(
        f"uv run python analysis/heuristics/convex_hull_vlm.py {args.vlm_solutions} "
        f"-d {args.optimal_dir} -m vlm-inputs -o {hull_vlm_csv}",
        "Analyzing VLM convex hull adherence"
    )

    # 2. Crossings (VLM)
    crossings_csv = output_dir / f"{vlm_name}_crossings.csv"
    run_command(
        f"uv run python analysis/heuristics/crossings.py {args.vlm_solutions} "
        f"-d {args.optimal_dir} -m vlm-inputs -o {crossings_csv}",
        "Counting VLM crossings"
    )

    # 3. Clustering deviance (VLM) - if clusters provided
    if args.vlm_clusters:
        cluster_dev_csv = output_dir / f"{vlm_name}_cluster_deviance.csv"
        run_command(
            f"uv run python analysis/heuristics/clustering.py {args.vlm_clusters} "
            f"{args.vlm_solutions} -o {args.optimal_dir} --output {cluster_dev_csv}",
            "Analyzing VLM cluster deviance"
        )

    # ========================================
    # Human Analysis
    # ========================================

    if not args.skip_human:
        # 4. Convex hull adherence (Human)
        hull_human_csv = output_dir / "human_hull.csv"
        if Path(args.human_trials).exists():
            run_command(
                f"uv run python analysis/heuristics/convex_hull_human.py {args.human_trials} "
                f"-o {hull_human_csv}",
                "Analyzing human convex hull adherence"
            )
        else:
            print(f"\n⚠ Skipping human analysis: {args.human_trials} not found")

    # ========================================
    # Combine Results
    # ========================================

    print(f"\n{'='*60}")
    print("Combining results into master CSV...")
    print(f"{'='*60}")

    # Load and merge VLM results
    vlm_data = {}

    if hull_vlm_csv.exists():
        hull_df = pd.read_csv(hull_vlm_csv)
        vlm_data = hull_df.set_index('problem_id').to_dict('index')

    if crossings_csv.exists():
        cross_df = pd.read_csv(crossings_csv)
        for _, row in cross_df.iterrows():
            pid = row['problem_id']
            if pid in vlm_data:
                vlm_data[pid]['crossings'] = row['crossings']
            else:
                vlm_data[pid] = {'problem_id': pid, 'crossings': row['crossings']}

    if args.vlm_clusters and (output_dir / f"{vlm_name}_cluster_deviance.csv").exists():
        clust_df = pd.read_csv(output_dir / f"{vlm_name}_cluster_deviance.csv")
        for _, row in clust_df.iterrows():
            pid = row['problem_id']
            if pid in vlm_data:
                vlm_data[pid]['cluster_deviance'] = row['cluster_deviance']
                vlm_data[pid]['num_clusters'] = row['num_clusters']
                vlm_data[pid]['cluster_transitions'] = row['transitions']

    # Save combined VLM results
    if vlm_data:
        combined_vlm = pd.DataFrame.from_dict(vlm_data, orient='index')
        combined_vlm = combined_vlm.reset_index().rename(columns={'index': 'problem_id'})

        # Add metadata (group: clustered/disperse)
        if Path("problem_metadata.csv").exists():
            combined_vlm = add_metadata_to_results(combined_vlm, "problem_metadata.csv")

        combined_vlm_csv = output_dir / f"{vlm_name}.csv"
        combined_vlm.to_csv(combined_vlm_csv, index=False)
        print(f"✓ Saved combined VLM results: {combined_vlm_csv}")
        print(f"  Columns: {list(combined_vlm.columns)}")
        print(f"  Problems analyzed: {len(combined_vlm)}")

    # ========================================
    # Generate Visualizations
    # ========================================

    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")

    viz_cmd = f"uv run python analysis/create_plots.py {output_dir}/{vlm_name}.csv"

    if not args.skip_human and Path(args.human_singles).exists():
        viz_cmd += f" --human-singles {args.human_singles}"

    if not args.skip_human and hull_human_csv.exists():
        viz_cmd += f" --human-hull {hull_human_csv}"

    viz_cmd += f" -o {viz_dir}"

    run_command(viz_cmd, "Creating visualization plots")

    # ========================================
    # Summary
    # ========================================

    print(f"\n{'#'*60}")
    print(f"# Analysis Complete!")
    print(f"# Results: {output_dir}/")
    print(f"# Visualizations: {viz_dir}/")
    print(f"{'#'*60}\n")

    # Print quick stats
    if vlm_data:
        print("VLM Performance Summary:")
        if 'hull_contiguous' in combined_vlm.columns:
            contiguous_pct = combined_vlm['hull_contiguous'].mean() * 100
            print(f"  Contiguous hull: {contiguous_pct:.1f}%")
        if 'adherence_score' in combined_vlm.columns:
            avg_adherence = combined_vlm['adherence_score'].mean()
            print(f"  Avg hull adherence: {avg_adherence:.3f}")
        if 'crossings' in combined_vlm.columns:
            zero_cross_pct = (combined_vlm['crossings'] == 0).mean() * 100
            avg_crossings = combined_vlm['crossings'].mean()
            print(f"  Zero crossings: {zero_cross_pct:.1f}%")
            print(f"  Avg crossings: {avg_crossings:.2f}")
        if 'cluster_deviance' in combined_vlm.columns:
            avg_dev = combined_vlm['cluster_deviance'].mean()
            perfect_pct = (combined_vlm['cluster_deviance'] == 0).mean() * 100
            print(f"  Perfect cluster congruence: {perfect_pct:.1f}%")
            print(f"  Avg cluster deviance: {avg_dev:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Create visualizations for TSP analysis.

Recreates plots from human data and adds VLM comparisons.
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import pandas as pd
import numpy as np

# Set nice default styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def create_hull_adherence_plot(vlm_df, human_df, output_dir):
    """
    Convex hull adherence comparison.
    Compare adherence scores and contiguous hull percentages.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Convex Hull Heuristic Adherence", fontsize=16, fontweight="bold")

    # 1. Adherence score distribution
    ax = axes[0]
    if 'adherence_score' in vlm_df.columns:
        vlm_scores = vlm_df['adherence_score'].dropna()
        ax.hist(vlm_scores, bins=20, alpha=0.6, label='VLM',
               color='#3498db', edgecolor='black')

    if human_df is not None and 'adherence_score' in human_df.columns:
        human_scores = human_df['adherence_score'].dropna()
        ax.hist(human_scores, bins=20, alpha=0.6, label='Human',
               color='#e74c3c', edgecolor='black')

    ax.set_xlabel("Hull Adherence Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Adherence Score Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Contiguous hull percentage
    ax = axes[1]
    data = []
    labels = []

    if 'hull_contiguous' in vlm_df.columns:
        vlm_cont_pct = (vlm_df['hull_contiguous'].sum() / len(vlm_df)) * 100
        data.append(vlm_cont_pct)
        labels.append('VLM')

    if human_df is not None and 'hull_contiguous' in human_df.columns:
        human_cont_pct = (human_df['hull_contiguous'].sum() / len(human_df)) * 100
        data.append(human_cont_pct)
        labels.append('Human')

    if data:
        bars = ax.bar(labels, data, color=['#3498db', '#e74c3c'][:len(data)],
                     alpha=0.7, edgecolor='black')
        ax.set_ylabel("Contiguous Hull (%)", fontsize=12)
        ax.set_title("Percentage with Contiguous Hull", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 100)

        for bar, val in zip(bars, data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                   f"{val:.1f}%", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    output_file = Path(output_dir) / "hull_adherence_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_crossings_plot(vlm_df, human_df, output_dir):
    """
    Path crossings comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Path Crossings Analysis", fontsize=16, fontweight="bold")

    # 1. Crossings distribution
    ax = axes[0]
    data_to_plot = []
    labels_to_plot = []

    if 'crossings' in vlm_df.columns:
        data_to_plot.append(vlm_df['crossings'].dropna())
        labels_to_plot.append('VLM')

    if human_df is not None and 'number_of_crossings' in human_df.columns:
        data_to_plot.append(human_df['number_of_crossings'].dropna())
        labels_to_plot.append('Human')

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels_to_plot, patch_artist=True, widths=0.6)
        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)

        ax.set_ylabel("Number of Crossings", fontsize=12)
        ax.set_title("Crossings Distribution", fontsize=13, fontweight="bold")
        ax.grid(axis='y', alpha=0.3)

    # 2. Zero crossings percentage
    ax = axes[1]
    data = []
    labels = []

    if 'crossings' in vlm_df.columns:
        vlm_zero_pct = (vlm_df['crossings'] == 0).mean() * 100
        data.append(vlm_zero_pct)
        labels.append('VLM')

    if human_df is not None and 'number_of_crossings' in human_df.columns:
        human_zero_pct = (human_df['number_of_crossings'] == 0).mean() * 100
        data.append(human_zero_pct)
        labels.append('Human')

    if data:
        bars = ax.bar(labels, data, color=['#3498db', '#e74c3c'][:len(data)],
                     alpha=0.7, edgecolor='black')
        ax.set_ylabel("Solutions with Zero Crossings (%)", fontsize=12)
        ax.set_title("Zero Crossings Rate", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 100)

        for bar, val in zip(bars, data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                   f"{val:.1f}%", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    output_file = Path(output_dir) / "crossings_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_cluster_deviance_plot(vlm_df, output_dir):
    """
    Cluster deviance analysis (Figure 2 from paper).
    Shows prevalence of perfect congruence.
    """
    if 'cluster_deviance' not in vlm_df.columns:
        print("  Skipping cluster deviance plot: no cluster data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cluster Deviance Analysis", fontsize=16, fontweight="bold")

    # 1. Perfect congruence rate
    ax = axes[0]
    perfect_count = (vlm_df['cluster_deviance'] == 0).sum()
    imperfect_count = (vlm_df['cluster_deviance'] > 0).sum()

    bars = ax.bar(['No\n(Deviance > 0)', 'Yes\n(Deviance = 0)'],
                   [imperfect_count, perfect_count],
                   color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Perfect Cluster Congruence", fontsize=13, fontweight="bold")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
               f"{int(height)}", ha="center", va="bottom", fontsize=11)

    # 2. Deviance distribution
    ax = axes[1]
    deviance = vlm_df['cluster_deviance'].dropna()
    ax.hist(deviance, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect (0)')
    ax.set_xlabel("Cluster Deviance", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Cluster Deviance Distribution", fontsize=13, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    output_file = Path(output_dir) / "cluster_deviance.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_by_num_points_plot(vlm_df, human_df, output_dir):
    """
    Metrics by number of points (Figure 3 from paper).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Performance by Problem Size", fontsize=16, fontweight="bold")

    # 1. Hull adherence by size
    ax = axes[0, 0]
    if 'num_points' in vlm_df.columns and 'adherence_score' in vlm_df.columns:
        vlm_by_size = vlm_df.groupby('num_points')['adherence_score'].mean()
        ax.plot(vlm_by_size.index, vlm_by_size.values, marker='o',
               label='VLM', linewidth=2, markersize=8, color='#3498db')

    if human_df is not None and 'num_points' in human_df.columns and 'adherence_score' in human_df.columns:
        human_by_size = human_df.groupby('num_points')['adherence_score'].mean()
        ax.plot(human_by_size.index, human_by_size.values, marker='s',
               label='Human', linewidth=2, markersize=8, color='#e74c3c')

    ax.set_xlabel("Number of Points", fontsize=12)
    ax.set_ylabel("Avg Hull Adherence Score", fontsize=12)
    ax.set_title("Hull Adherence by Problem Size", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Crossings by size
    ax = axes[0, 1]
    if 'num_points' in vlm_df.columns and 'crossings' in vlm_df.columns:
        vlm_by_size = vlm_df.groupby('num_points')['crossings'].mean()
        ax.plot(vlm_by_size.index, vlm_by_size.values, marker='o',
               label='VLM', linewidth=2, markersize=8, color='#3498db')

    if human_df is not None and 'number_of_points' in human_df.columns and 'number_of_crossings' in human_df.columns:
        human_by_size = human_df.groupby('number_of_points')['number_of_crossings'].mean()
        ax.plot(human_by_size.index, human_by_size.values, marker='s',
               label='Human', linewidth=2, markersize=8, color='#e74c3c')

    ax.set_xlabel("Number of Points", fontsize=12)
    ax.set_ylabel("Avg Number of Crossings", fontsize=12)
    ax.set_title("Crossings by Problem Size", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Perfect congruence by size (if cluster data available)
    ax = axes[1, 0]
    if 'num_points' in vlm_df.columns and 'cluster_deviance' in vlm_df.columns:
        perfect_by_size = vlm_df.groupby('num_points')['cluster_deviance'].apply(
            lambda x: (x == 0).mean() * 100
        )
        ax.plot(perfect_by_size.index, perfect_by_size.values, marker='o',
               linewidth=2, markersize=8, color='#2ecc71')
        ax.set_xlabel("Number of Points", fontsize=12)
        ax.set_ylabel("Perfect Cluster Congruence (%)", fontsize=12)
        ax.set_title("Cluster Congruence by Problem Size", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    else:
        ax.text(0.5, 0.5, "Cluster data not available", ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # 4. Contiguous hull by size
    ax = axes[1, 1]
    if 'num_points' in vlm_df.columns and 'hull_contiguous' in vlm_df.columns:
        vlm_by_size = vlm_df.groupby('num_points')['hull_contiguous'].mean() * 100
        ax.plot(vlm_by_size.index, vlm_by_size.values, marker='o',
               label='VLM', linewidth=2, markersize=8, color='#3498db')

    if human_df is not None and 'num_points' in human_df.columns and 'hull_contiguous' in human_df.columns:
        human_by_size = human_df.groupby('num_points')['hull_contiguous'].mean() * 100
        ax.plot(human_by_size.index, human_by_size.values, marker='s',
               label='Human', linewidth=2, markersize=8, color='#e74c3c')

    ax.set_xlabel("Number of Points", fontsize=12)
    ax.set_ylabel("Contiguous Hull (%)", fontsize=12)
    ax.set_title("Contiguous Hull by Problem Size", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    output_file = Path(output_dir) / "by_problem_size.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_summary_table(vlm_df, human_df, output_dir):
    """
    Create a summary statistics table.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    summary_data = [["Metric", "VLM", "Human"]]

    # Hull adherence
    if 'adherence_score' in vlm_df.columns:
        vlm_val = f"{vlm_df['adherence_score'].mean():.3f}"
    else:
        vlm_val = "N/A"

    if human_df is not None and 'adherence_score' in human_df.columns:
        human_val = f"{human_df['adherence_score'].mean():.3f}"
    else:
        human_val = "N/A"

    summary_data.append(["Avg Hull Adherence", vlm_val, human_val])

    # Contiguous hull
    if 'hull_contiguous' in vlm_df.columns:
        vlm_val = f"{(vlm_df['hull_contiguous'].mean() * 100):.1f}%"
    else:
        vlm_val = "N/A"

    if human_df is not None and 'hull_contiguous' in human_df.columns:
        human_val = f"{(human_df['hull_contiguous'].mean() * 100):.1f}%"
    else:
        human_val = "N/A"

    summary_data.append(["Contiguous Hull", vlm_val, human_val])

    # Crossings
    if 'crossings' in vlm_df.columns:
        vlm_val = f"{vlm_df['crossings'].mean():.2f}"
        vlm_zero = f"{((vlm_df['crossings'] == 0).mean() * 100):.1f}%"
    else:
        vlm_val = "N/A"
        vlm_zero = "N/A"

    if human_df is not None and 'number_of_crossings' in human_df.columns:
        human_val = f"{human_df['number_of_crossings'].mean():.2f}"
        human_zero = f"{((human_df['number_of_crossings'] == 0).mean() * 100):.1f}%"
    else:
        human_val = "N/A"
        human_zero = "N/A"

    summary_data.append(["Avg Crossings", vlm_val, human_val])
    summary_data.append(["Zero Crossings", vlm_zero, human_zero])

    # Cluster deviance
    if 'cluster_deviance' in vlm_df.columns:
        vlm_val = f"{vlm_df['cluster_deviance'].mean():.3f}"
        vlm_perfect = f"{((vlm_df['cluster_deviance'] == 0).mean() * 100):.1f}%"
        summary_data.append(["Avg Cluster Deviance", vlm_val, "N/A"])
        summary_data.append(["Perfect Congruence", vlm_perfect, "N/A"])

    # Sample sizes
    summary_data.append(["", "", ""])
    summary_data.append(["Sample Size", str(len(vlm_df)),
                         str(len(human_df)) if human_df is not None else "N/A"])

    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.5, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title("Performance Summary Statistics", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    output_file = Path(output_dir) / "summary_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create TSP analysis visualizations"
    )
    parser.add_argument("vlm_results", help="Combined VLM results CSV")
    parser.add_argument("--human-singles", help="Human summary CSV (optional)")
    parser.add_argument("--human-hull", help="Human hull adherence CSV (optional)")
    parser.add_argument("-o", "--output-dir", default="visualizations",
                       help="Output directory (default: visualizations)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nLoading data...")
    vlm_df = pd.read_csv(args.vlm_results)
    print(f"  VLM results: {len(vlm_df)} problems")

    human_df = None
    if args.human_singles and Path(args.human_singles).exists():
        human_df = pd.read_csv(args.human_singles)
        print(f"  Human summary: {len(human_df)} trials")

    human_hull_df = None
    if args.human_hull and Path(args.human_hull).exists():
        human_hull_df = pd.read_csv(args.human_hull)
        print(f"  Human hull data: {len(human_hull_df)} trials")

    print(f"\nGenerating plots in {output_dir}/...")

    # Create all plots
    create_hull_adherence_plot(vlm_df, human_hull_df, output_dir)
    create_crossings_plot(vlm_df, human_df, output_dir)
    create_cluster_deviance_plot(vlm_df, output_dir)
    create_by_num_points_plot(vlm_df, human_hull_df, output_dir)
    create_summary_table(vlm_df, human_df if human_df is not None else human_hull_df, output_dir)

    print(f"\n✓ All visualizations created successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Compare multiple VLM models side-by-side.

Creates cross-model comparison visualizations.
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from scipy import stats
import json


def calculate_tour_length(points, tour_indices):
    """Calculate Euclidean tour length."""
    total = 0
    n = len(tour_indices)
    for i in range(n):
        p1 = points[tour_indices[i]]
        p2 = points[tour_indices[(i + 1) % n]]
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        total += (dx**2 + dy**2)**0.5
    return total


def load_model_results(results_dir):
    """
    Load all model results from results directory.

    Returns dict: display_name -> (DataFrame, original_name)
    """
    results_dir = Path(results_dir)
    models = {}

    # Mapping from file names to display names
    name_mapping = {
        'claude_opus_4.5_solutions': 'Opus 4.5',
        'openai_gpt_5.2_solutions': 'GPT-5.2',
        'gemini_3_pro_solutions': 'Gemini 3'
    }

    for model_dir in results_dir.glob("*"):
        if not model_dir.is_dir():
            continue

        # Look for the model CSV file
        csv_files = list(model_dir.glob("*.csv"))
        # Filter out the individual metric CSVs
        main_csv = [f for f in csv_files if not any(x in f.stem for x in ['_hull', '_crossings', '_cluster_deviance'])]

        if main_csv:
            original_name = main_csv[0].stem
            try:
                df = pd.read_csv(main_csv[0])

                # Add optimality and tour length calculations
                df = calculate_optimality_metrics(df, original_name)

                # Use display name
                display_name = name_mapping.get(original_name, original_name)
                models[display_name] = df
                print(f"✓ Loaded {display_name}: {len(models[display_name])} problems")
            except Exception as e:
                print(f"⚠ Error loading {original_name}: {e}")

    return models


def calculate_optimality_metrics(df, model_name):
    """Calculate tour lengths and optimality for VLM solutions."""
    tour_lengths = []
    is_optimal = []
    pct_increase = []

    # Load solutions
    solutions_file = f"vlm-outputs/solutions/{model_name}.json"
    if not Path(solutions_file).exists():
        print(f"  ⚠ Solutions file not found: {solutions_file}")
        df['tour_length'] = np.nan
        df['is_optimal'] = False
        df['pct_increase'] = np.nan
        return df

    with open(solutions_file) as f:
        solutions = json.load(f)

    for _, row in df.iterrows():
        problem_id = row['problem_id']
        optimal_length = row.get('optimal_length', np.nan)

        if problem_id not in solutions or pd.isna(optimal_length):
            tour_lengths.append(np.nan)
            is_optimal.append(False)
            pct_increase.append(np.nan)
            continue

        # Load mapping to get points and convert labels
        mapping_file = f"vlm-inputs/{problem_id}_mapping.json"
        if not Path(mapping_file).exists():
            tour_lengths.append(np.nan)
            is_optimal.append(False)
            pct_increase.append(np.nan)
            continue

        try:
            with open(mapping_file) as f:
                mapping_data = json.load(f)
                label_to_index = {int(k): v for k, v in mapping_data['label_to_index'].items()}
                points = mapping_data['points']

            tour_labels = solutions[problem_id]
            tour_indices = [label_to_index[label] for label in tour_labels]

            # Calculate tour length
            length = calculate_tour_length(points, tour_indices)
            tour_lengths.append(length)

            # Check if optimal (within 0.01% tolerance)
            optimal = abs(length - optimal_length) / optimal_length < 0.0001
            is_optimal.append(optimal)

            # Calculate percentage increase
            if optimal:
                pct_increase.append(0.0)
            else:
                pct_increase.append((length - optimal_length) / optimal_length * 100)

        except Exception as e:
            print(f"  ⚠ Error processing {problem_id}: {e}")
            tour_lengths.append(np.nan)
            is_optimal.append(False)
            pct_increase.append(np.nan)

    df['tour_length'] = tour_lengths
    df['is_optimal'] = is_optimal
    df['pct_increase'] = pct_increase

    return df


def create_hull_comparison(models, output_dir):
    """Compare hull adherence across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Convex Hull Adherence - Model Comparison", fontsize=16, fontweight="bold")

    colors = plt.cm.Set2(range(len(models)))

    # 1. Adherence score by model
    ax = axes[0]
    model_names = []
    adherence_scores = []

    for i, (model_name, df) in enumerate(models.items()):
        if 'adherence_score' in df.columns:
            model_names.append(model_name)
            adherence_scores.append(df['adherence_score'].mean())

    bars = ax.bar(range(len(model_names)), adherence_scores, color=colors[:len(model_names)],
                   alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Avg Hull Adherence Score", fontsize=12)
    ax.set_title("Average Hull Adherence", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, adherence_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
               f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    # 2. Contiguous hull percentage
    ax = axes[1]
    model_names = []
    contiguous_pcts = []

    for i, (model_name, df) in enumerate(models.items()):
        if 'hull_contiguous' in df.columns:
            model_names.append(model_name)
            contiguous_pcts.append((df['hull_contiguous'].sum() / len(df)) * 100)

    bars = ax.bar(range(len(model_names)), contiguous_pcts, color=colors[:len(model_names)],
                   alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Contiguous Hull (%)", fontsize=12)
    ax.set_title("Contiguous Hull Rate", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)

    for bar, val in zip(bars, contiguous_pcts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
               f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    output_file = Path(output_dir) / "model_comparison_hull.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_crossings_comparison(models, output_dir):
    """Compare path crossings across models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Path Crossings - Model Comparison", fontsize=16, fontweight="bold")

    colors = plt.cm.Set2(range(len(models)))

    # 1. Average crossings
    ax = axes[0]
    model_names = []
    avg_crossings = []

    for i, (model_name, df) in enumerate(models.items()):
        if 'crossings' in df.columns:
            model_names.append(model_name)
            avg_crossings.append(df['crossings'].mean())

    # Add human baseline
    model_names.append('Human')
    avg_crossings.append(0.41)  # Hardcoded human baseline

    # Create bars with human in red
    bar_colors = list(colors[:len(models)]) + ['#e74c3c']
    bars = ax.bar(range(len(model_names)), avg_crossings, color=bar_colors,
                   alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Avg Crossings", fontsize=12)
    ax.set_title("Average Path Crossings", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, avg_crossings):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
               f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    # 2. Zero crossings percentage
    ax = axes[1]
    model_names = []
    zero_pcts = []

    for i, (model_name, df) in enumerate(models.items()):
        if 'crossings' in df.columns:
            model_names.append(model_name)
            zero_pcts.append((df['crossings'] == 0).mean() * 100)

    # Add human baseline (32 problems, seed=42)
    model_names.append('Human')
    zero_pcts.append(79.9)  # Hardcoded human baseline from calculate_human_baseline.py

    # Create bars with human in red
    bar_colors = list(colors[:len(models)]) + ['#e74c3c']
    bars = ax.bar(range(len(model_names)), zero_pcts, color=bar_colors,
                   alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Zero Crossings (%)", fontsize=12)
    ax.set_title("Zero Crossings Rate", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)

    for bar, val in zip(bars, zero_pcts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
               f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    output_file = Path(output_dir) / "model_comparison_crossings.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_cluster_comparison(models, output_dir):
    """Compare cluster deviance across models."""
    # Filter models that have cluster data
    models_with_clusters = {name: df for name, df in models.items()
                           if 'cluster_deviance' in df.columns}

    if not models_with_clusters:
        print("  Skipping cluster comparison: no models with cluster data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cluster Deviance - Model Comparison", fontsize=16, fontweight="bold")

    colors = plt.cm.Set2(range(len(models_with_clusters)))

    # 1. Average deviance
    ax = axes[0]
    model_names = list(models_with_clusters.keys())
    avg_deviance = [df['cluster_deviance'].mean() for df in models_with_clusters.values()]

    bars = ax.bar(range(len(model_names)), avg_deviance, color=colors[:len(model_names)],
                   alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Avg Cluster Deviance", fontsize=12)
    ax.set_title("Average Cluster Deviance", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, avg_deviance):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
               f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    # 2. Perfect congruence percentage
    ax = axes[1]
    perfect_pcts = [(df['cluster_deviance'] == 0).mean() * 100
                    for df in models_with_clusters.values()]

    bars = ax.bar(range(len(model_names)), perfect_pcts, color=colors[:len(model_names)],
                   alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Perfect Congruence (%)", fontsize=12)
    ax.set_title("Perfect Cluster Congruence Rate", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)

    for bar, val in zip(bars, perfect_pcts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
               f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    output_file = Path(output_dir) / "model_comparison_clusters.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_by_group_comparison(models, output_dir):
    """Compare performance on clustered vs disperse problems."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Performance by Problem Type (Clustered vs Disperse)", fontsize=16, fontweight="bold")

    colors = plt.cm.Set2(range(len(models)))

    # Check if models have group data
    models_with_groups = {name: df for name, df in models.items()
                         if 'group' in df.columns}

    if not models_with_groups:
        print("  Skipping by-group comparison: no models with group metadata")
        return

    # 1. Hull adherence by group
    ax = axes[0, 0]
    x = np.arange(2)
    width = 0.8 / len(models_with_groups)

    for i, (model_name, df) in enumerate(models_with_groups.items()):
        if 'adherence_score' in df.columns:
            clustered_score = df[df['group'] == 'clustered']['adherence_score'].mean()
            disperse_score = df[df['group'] == 'disperse']['adherence_score'].mean()
            offset = (i - len(models_with_groups)/2 + 0.5) * width
            ax.bar(x + offset, [clustered_score, disperse_score], width,
                  label=model_name, color=colors[i], alpha=0.7, edgecolor='black')

    ax.set_ylabel("Avg Hull Adherence", fontsize=12)
    ax.set_title("Hull Adherence by Group", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(['Clustered', 'Disperse'])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)

    # 2. Crossings by group
    ax = axes[0, 1]
    for i, (model_name, df) in enumerate(models_with_groups.items()):
        if 'crossings' in df.columns:
            clustered_cross = df[df['group'] == 'clustered']['crossings'].mean()
            disperse_cross = df[df['group'] == 'disperse']['crossings'].mean()
            offset = (i - len(models_with_groups)/2 + 0.5) * width
            ax.bar(x + offset, [clustered_cross, disperse_cross], width,
                  label=model_name, color=colors[i], alpha=0.7, edgecolor='black')

    ax.set_ylabel("Avg Crossings", fontsize=12)
    ax.set_title("Crossings by Group", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(['Clustered', 'Disperse'])
    ax.legend(loc='upper right')

    # 3. Cluster deviance by group (if available)
    ax = axes[1, 0]
    models_with_both = {name: df for name, df in models_with_groups.items()
                        if 'cluster_deviance' in df.columns}

    if models_with_both:
        for i, (model_name, df) in enumerate(models_with_both.items()):
            clustered_dev = df[df['group'] == 'clustered']['cluster_deviance'].mean()
            disperse_dev = df[df['group'] == 'disperse']['cluster_deviance'].mean()
            offset = (i - len(models_with_both)/2 + 0.5) * width
            ax.bar(x + offset, [clustered_dev, disperse_dev], width,
                  label=model_name, color=colors[i], alpha=0.7, edgecolor='black')

        ax.set_ylabel("Avg Cluster Deviance", fontsize=12)
        ax.set_title("Cluster Deviance by Group", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(['Clustered', 'Disperse'])
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "No cluster data available", ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')

    summary_data = [["Metric", "Model", "Clustered", "Disperse"]]

    for model_name, df in models_with_groups.items():
        if 'adherence_score' in df.columns:
            c_score = df[df['group'] == 'clustered']['adherence_score'].mean()
            d_score = df[df['group'] == 'disperse']['adherence_score'].mean()
            summary_data.append(["Hull Adh.", model_name, f"{c_score:.3f}", f"{d_score:.3f}"])

        if 'crossings' in df.columns:
            c_cross = df[df['group'] == 'clustered']['crossings'].mean()
            d_cross = df[df['group'] == 'disperse']['crossings'].mean()
            summary_data.append(["Crossings", model_name, f"{c_cross:.1f}", f"{d_cross:.1f}"])

    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.35, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.tight_layout()
    output_file = Path(output_dir) / "model_comparison_by_group.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_cluster_congruence_by_size(models, output_dir):
    """Compare cluster congruence by problem size (VLMs + hardcoded human data)."""
    # Hardcoded estimates
    human_sizes = [10, 15, 20, 25, 30]
    human_clustered = [0.75, 0.625, 0.575, 0.43, 0.5]
    human_dispersed = [0.7, 0.39, 0.43, 0.343, 0.25]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    color_idx = 0

    # Plot human data first (two lines)
    ax.plot(human_sizes, human_clustered, color=colors[color_idx], linestyle='-',
            marker='o', linewidth=2, markersize=6, label='Human (Clustered)')
    ax.plot(human_sizes, human_dispersed, color=colors[color_idx], linestyle='--',
            marker='s', linewidth=2, markersize=6, label='Human (Dispersed)')
    color_idx += 1

    # Plot VLM data
    for model_name, df in models.items():
        if 'cluster_deviance' not in df.columns or 'group' not in df.columns:
            continue

        # Calculate perfect congruence rate (deviance == 0) by size and group
        for group_name, linestyle in [('clustered', '-'), ('disperse', '--')]:
            group_df = df[df['group'] == group_name]

            if len(group_df) == 0:
                continue

            # Group by problem size
            size_groups = group_df.groupby('num_points')
            sizes = []
            congruence_rates = []

            for size, size_df in size_groups:
                sizes.append(size)
                # Perfect congruence = cluster_deviance == 0
                congruence_rate = (size_df['cluster_deviance'] == 0).mean()
                congruence_rates.append(congruence_rate)

            if sizes:
                marker = 'o' if linestyle == '-' else 's'
                label_suffix = 'Clustered' if group_name == 'clustered' else 'Dispersed'
                ax.plot(sizes, congruence_rates, color=colors[color_idx],
                       linestyle=linestyle, marker=marker, linewidth=2, markersize=6,
                       label=f'{model_name} ({label_suffix})')

        color_idx += 1

    ax.set_xlabel("Problem Size (Number of Points)", fontsize=12)
    ax.set_ylabel("Perfect Cluster Congruence Rate", fontsize=12)
    ax.set_title("Cluster Congruence by Problem Size", fontsize=14, fontweight="bold")

    # Set x-axis ticks every 5
    ax.set_xticks([10, 15, 20, 25, 30])
    ax.set_ylim(0, 1)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    output_file = Path(output_dir) / "cluster_congruence_by_size.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_cluster_congruence_by_size_overall(models, output_dir):
    """Compare cluster congruence by problem size (overall, combining clustered and dispersed)."""
    # Hardcoded estimates for human overall
    human_sizes = [10, 15, 20, 25, 30]
    # Average the clustered and dispersed values
    human_clustered = [0.75, 0.65, 0.57, 0.43, 0.52]
    human_dispersed = [0.7, 0.39, 0.43, 0.31, 0.28]
    human_overall = [(c + d) / 2 for c, d in zip(human_clustered, human_dispersed)]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette (same as cluster_congruence_by_size)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    color_idx = 0

    # Plot human data (single line)
    ax.plot(human_sizes, human_overall, color=colors[color_idx], linestyle='-',
            marker='o', linewidth=2, markersize=6, label='Human')
    color_idx += 1

    # Plot VLM data (single line per model)
    for model_name, df in models.items():
        if 'cluster_deviance' not in df.columns or 'num_points' not in df.columns:
            continue

        # Group by problem size (combining clustered and dispersed)
        size_groups = df.groupby('num_points')
        sizes = []
        congruence_rates = []

        for size, size_df in size_groups:
            sizes.append(size)
            # Perfect congruence = cluster_deviance == 0
            congruence_rate = (size_df['cluster_deviance'] == 0).mean()
            congruence_rates.append(congruence_rate)

        if sizes:
            ax.plot(sizes, congruence_rates, color=colors[color_idx],
                   linestyle='-', marker='o', linewidth=2, markersize=6,
                   label=model_name)

        color_idx += 1

    ax.set_xlabel("Problem Size (Number of Points)", fontsize=12)
    ax.set_ylabel("Perfect Cluster Congruence Rate", fontsize=12)
    ax.set_title("Overall Cluster Congruence by Problem Size", fontsize=14, fontweight="bold")

    # Set x-axis ticks every 5
    ax.set_xticks([10, 15, 20, 25, 30])
    ax.set_ylim(0, 1)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    output_file = Path(output_dir) / "cluster_congruence_by_size_overall.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_crossings_by_size(models, output_dir):
    """Compare crossings by problem size across models (overall, not split by group)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette (matching cluster_congruence_by_size)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    # Start from index 1 to match the VLM colors from cluster_congruence_by_size
    color_idx = 1

    # Plot VLM data
    for model_name, df in models.items():
        if 'crossings' not in df.columns or 'num_points' not in df.columns:
            continue

        # Group by problem size (combining clustered and dispersed)
        size_groups = df.groupby('num_points')
        sizes = []
        avg_crossings = []

        for size, size_df in size_groups:
            sizes.append(size)
            avg_crossings.append(size_df['crossings'].mean())

        if sizes:
            ax.plot(sizes, avg_crossings, color=colors[color_idx],
                   linestyle='-', marker='o', linewidth=2, markersize=6,
                   label=model_name)

        color_idx += 1

    ax.set_xlabel("Problem Size (Number of Points)", fontsize=12)
    ax.set_ylabel("Average Crossings", fontsize=12)
    ax.set_title("Path Crossings by Problem Size", fontsize=14, fontweight="bold")

    # Set x-axis ticks every 5
    ax.set_xticks([10, 15, 20, 25, 30])

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    output_file = Path(output_dir) / "crossings_by_size.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_crossings_combined(models, output_dir):
    """Combined view: crossings by size (left) and zero crossings count (right)."""
    # Load human data from tsp-singles.csv
    human_data = None
    try:
        csv_path = Path("marupudi_data/tsp-singles.csv")
        if csv_path.exists():
            human_df = pd.read_csv(csv_path)
            # Group by problem size and calculate average crossings
            human_by_size = human_df.groupby('number_of_points').agg({
                'number_of_crossings': ['mean', lambda x: (x == 0).sum()]
            }).reset_index()
            human_by_size.columns = ['size', 'avg_crossings', 'zero_count']
            human_data = human_by_size
    except Exception as e:
        print(f"  ⚠ Could not load human data: {e}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Path Crossings Analysis", fontsize=16, fontweight="bold")

    # Color palette (matching cluster_congruence_by_size)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    # LEFT: Crossings by size
    ax = axes[0]

    # Plot human data first (in red)
    if human_data is not None:
        ax.plot(human_data['size'], human_data['avg_crossings'],
               color=colors[0], linestyle='-', marker='o', linewidth=2,
               markersize=6, label='Human')

    # Plot VLM data (start from index 1)
    color_idx = 1
    for model_name, df in models.items():
        if 'crossings' not in df.columns or 'num_points' not in df.columns:
            continue

        # Group by problem size
        size_groups = df.groupby('num_points')
        sizes = []
        avg_crossings = []

        for size, size_df in size_groups:
            sizes.append(size)
            avg_crossings.append(size_df['crossings'].mean())

        if sizes:
            ax.plot(sizes, avg_crossings, color=colors[color_idx],
                   linestyle='-', marker='o', linewidth=2, markersize=6,
                   label=model_name)

        color_idx += 1

    ax.set_xlabel("Problem Size (Number of Points)", fontsize=12)
    ax.set_ylabel("Average Crossings", fontsize=12)
    ax.set_title("Average Path Crossings by Problem Size", fontsize=13, fontweight="bold")
    ax.set_xticks([10, 15, 20, 25, 30])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # RIGHT: Zero crossings rate (percentage)
    ax = axes[1]

    # Prepare data
    all_names = []
    all_pcts = []
    all_colors = []

    # Add human data first (in red) - hardcoded baseline
    all_names.append('Human')
    all_pcts.append(79.9)  # Hardcoded human baseline from calculate_human_baseline.py (seed=42)
    all_colors.append(colors[0])

    # Add VLM data
    color_idx = 1
    for model_name, df in models.items():
        if 'crossings' in df.columns:
            all_names.append(model_name)
            zero_pct = (df['crossings'] == 0).mean() * 100
            all_pcts.append(zero_pct)
            all_colors.append(colors[color_idx])
            color_idx += 1

    bars = ax.bar(range(len(all_names)), all_pcts,
                   color=all_colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names)
    ax.set_ylabel("Zero Crossings (%)", fontsize=12)
    ax.set_title("Zero Crossings Rate", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)

    for bar, val in zip(bars, all_pcts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
               f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    output_file = Path(output_dir) / "crossings_analysis_combined.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_cluster_congruence_combined(models, output_dir):
    """Combined view: cluster congruence by size overall (left) and perfect rate (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Cluster Congruence Analysis", fontsize=16, fontweight="bold")

    # Color palette (same as cluster_congruence_by_size)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    # LEFT: Cluster congruence by size (overall)
    ax = axes[0]
    color_idx = 0

    # Hardcoded estimates for human overall
    human_sizes = [10, 15, 20, 25, 30]
    human_clustered = [0.75, 0.625, 0.575, 0.43, 0.5]
    human_dispersed = [0.7, 0.39, 0.43, 0.343, 0.25]
    human_overall = [(c + d) / 2 for c, d in zip(human_clustered, human_dispersed)]

    # Plot human data
    ax.plot(human_sizes, human_overall, color=colors[color_idx], linestyle='-',
            marker='o', linewidth=2, markersize=6, label='Human')
    color_idx += 1

    # Plot VLM data
    for model_name, df in models.items():
        if 'cluster_deviance' not in df.columns or 'num_points' not in df.columns:
            continue

        size_groups = df.groupby('num_points')
        sizes = []
        congruence_rates = []

        for size, size_df in size_groups:
            sizes.append(size)
            congruence_rate = (size_df['cluster_deviance'] == 0).mean()
            congruence_rates.append(congruence_rate)

        if sizes:
            ax.plot(sizes, congruence_rates, color=colors[color_idx],
                   linestyle='-', marker='o', linewidth=2, markersize=6,
                   label=model_name)

        color_idx += 1

    ax.set_xlabel("Problem Size (Number of Points)", fontsize=12)
    ax.set_ylabel("Perfect Cluster Congruence Rate", fontsize=12)
    ax.set_title("Perfect Cluster Congruence by Problem Size", fontsize=13, fontweight="bold")
    ax.set_xticks([10, 15, 20, 25, 30])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # RIGHT: Perfect cluster congruence rate
    ax = axes[1]

    models_with_clusters = {name: df for name, df in models.items()
                           if 'cluster_deviance' in df.columns}

    if models_with_clusters:
        # Prepare data with human baseline first
        all_names = ['Human']
        all_pcts = [52.0]  # Hardcoded human baseline
        all_colors = [colors[0]]  # Red for human

        # Add VLM data
        for i, (model_name, df) in enumerate(models_with_clusters.items()):
            all_names.append(model_name)
            all_pcts.append((df['cluster_deviance'] == 0).mean() * 100)
            all_colors.append(colors[i+1])  # Use colors starting from index 1

        bars = ax.bar(range(len(all_names)), all_pcts,
                       color=all_colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(all_names)))
        ax.set_xticklabels(all_names)
        ax.set_ylabel("Perfect Congruence (%)", fontsize=12)
        ax.set_title("Perfect Cluster Congruence Rate", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 100)

        for bar, val in zip(bars, all_pcts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                   f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    else:
        ax.text(0.5, 0.5, "No cluster data available", ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    output_file = Path(output_dir) / "cluster_congruence_analysis_combined.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_accuracy_by_size(models, output_dir):
    """Scatter plot of optimality rate by problem size with linear fit."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Consistent color mapping across all plots
    color_map = {
        'Opus 4.5': '#3498db',  # blue
        'GPT-5.2': '#2ecc71',   # green
        'Gemini 3': '#9b59b6'   # purple
    }

    for model_name, df in models.items():
        if 'is_optimal' not in df.columns or 'num_points' not in df.columns:
            continue

        # Get all problem sizes and optimality
        sizes = df['num_points'].values
        is_optimal = df['is_optimal'].astype(int).values

        # Remove NaN values
        valid_mask = ~np.isnan(sizes) & ~np.isnan(is_optimal)
        sizes_clean = sizes[valid_mask]
        is_optimal_clean = is_optimal[valid_mask]

        if len(sizes_clean) == 0:
            continue

        # Get color for this model
        color = color_map.get(model_name, '#34495e')  # default gray if model not in map

        # Scatter plot
        ax.scatter(sizes_clean, is_optimal_clean, alpha=0.5, s=50,
                  color=color, label=f'{model_name} (data)')

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(sizes_clean, is_optimal_clean)

        # Plot regression line
        x_line = np.array([10, 30])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=color, linewidth=2,
               linestyle='--', label=f'{model_name} (fit, r²={r_value**2:.3f})')

    ax.set_xlabel("Problem Size (Number of Points)", fontsize=12)
    ax.set_ylabel("Optimal Solution (0=No, 1=Yes)", fontsize=12)
    ax.set_title("Solution Optimality by Problem Size", fontsize=14, fontweight="bold")

    # Set x-axis ticks every 5
    ax.set_xticks([10, 15, 20, 25, 30])
    ax.set_ylim(-0.1, 1.1)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    output_file = Path(output_dir) / "optimality_by_size.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def create_deviation_by_size(models, output_dir):
    """Scatter plot of deviation from optimal by problem size with linear fit."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Consistent color mapping across all plots
    color_map = {
        'Opus 4.5': '#3498db',  # blue
        'GPT-5.2': '#2ecc71',   # green
        'Gemini 3': '#9b59b6'   # purple
    }

    for model_name, df in models.items():
        if 'pct_increase' not in df.columns or 'num_points' not in df.columns:
            continue

        # Filter to only suboptimal solutions (pct_increase > 0)
        suboptimal_df = df[df['pct_increase'] > 0]

        if len(suboptimal_df) == 0:
            continue

        sizes = suboptimal_df['num_points'].values
        pct_increase = suboptimal_df['pct_increase'].values

        # Remove NaN values
        valid_mask = ~np.isnan(sizes) & ~np.isnan(pct_increase)
        sizes_clean = sizes[valid_mask]
        pct_increase_clean = pct_increase[valid_mask]

        if len(sizes_clean) == 0:
            continue

        # Get color for this model
        color = color_map.get(model_name, '#34495e')  # default gray if model not in map

        # Scatter plot
        ax.scatter(sizes_clean, pct_increase_clean, alpha=0.5, s=50,
                  color=color, label=f'{model_name} (data)')

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(sizes_clean, pct_increase_clean)

        # Plot regression line
        x_line = np.array([10, 30])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=color, linewidth=2,
               linestyle='--', label=f'{model_name} (fit, r²={r_value**2:.3f})')

    ax.set_xlabel("Problem Size (Number of Points)", fontsize=12)
    ax.set_ylabel("% Increase from Optimal", fontsize=12)
    ax.set_title("Deviation from Optimal by Problem Size", fontsize=14, fontweight="bold")

    # Set x-axis ticks every 5
    ax.set_xticks([10, 15, 20, 25, 30])

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    output_file = Path(output_dir) / "deviation_by_size.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple VLM models"
    )
    parser.add_argument("--results-dir", default="results",
                       help="Results directory containing model subdirectories (default: results)")
    parser.add_argument("-o", "--output-dir", default="results/comparisons",
                       help="Output directory for comparison plots (default: results/comparisons)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nLoading model results from {args.results_dir}...")
    models = load_model_results(args.results_dir)

    if len(models) < 2:
        print(f"\n⚠ Found only {len(models)} model(s). Need at least 2 for comparison.")
        return 1

    print(f"\nGenerating comparison plots in {output_dir}/...")

    create_hull_comparison(models, output_dir)
    create_crossings_comparison(models, output_dir)
    create_cluster_comparison(models, output_dir)
    create_by_group_comparison(models, output_dir)
    create_cluster_congruence_by_size(models, output_dir)
    create_cluster_congruence_by_size_overall(models, output_dir)
    create_crossings_by_size(models, output_dir)
    create_crossings_combined(models, output_dir)
    create_cluster_congruence_combined(models, output_dir)
    create_accuracy_by_size(models, output_dir)
    create_deviation_by_size(models, output_dir)

    print(f"\n✓ Model comparison complete!")
    print(f"  Models compared: {', '.join(models.keys())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Add missing problems to problem_metadata.csv.

Classifies problems as clustered vs disperse based on z-scores
(clustering metric from spatial statistics) and calculates optimal lengths.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform


def calculate_z_score(points):
    """
    Calculate z-score metric for clustering (higher = more clustered).

    Based on vacuumed z-score from the human experiment data.
    Measures how much the nearest-neighbor distances deviate from random.
    """
    coords = np.array([(p['x'], p['y']) for p in points])
    n = len(coords)

    if n < 2:
        return 0.0

    # Calculate pairwise distances
    distances = squareform(pdist(coords, 'euclidean'))

    # For each point, find nearest neighbor distance
    nn_distances = []
    for i in range(n):
        # Get distances to all other points
        dists = distances[i]
        dists = dists[dists > 0]  # Exclude self (distance 0)
        if len(dists) > 0:
            nn_distances.append(dists.min())

    if len(nn_distances) == 0:
        return 0.0

    # Mean nearest neighbor distance
    mean_nn = np.mean(nn_distances)

    # Expected mean NN distance for random distribution
    # Assuming points in 800x600 area (typical canvas size)
    area = 800 * 600
    expected_nn = 0.5 / np.sqrt(n / area)

    # Z-score: how many std devs away from random
    # Higher = more clustered (shorter NN distances than expected)
    std_nn = np.std(nn_distances)
    if std_nn > 0:
        z_score = (expected_nn - mean_nn) / std_nn
    else:
        z_score = 0.0

    return z_score


def calculate_tour_length(points):
    """Calculate Euclidean tour length from ordered points."""
    total = 0
    n = len(points)
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        total += np.sqrt(dx**2 + dy**2)
    return total


def main():
    # Load existing metadata
    existing_df = pd.read_csv('problem_metadata.csv')
    existing_ids = set(existing_df['problem_id'])

    print(f"Existing metadata: {len(existing_ids)} problems")
    print(f"  Clustered: {(existing_df['group'] == 'clustered').sum()}")
    print(f"  Disperse: {(existing_df['group'] == 'disperse').sum()}")

    # Load all VLM problems
    with open('vlm-outputs/solutions/claude_opus_4.5_solutions.json') as f:
        all_problems = set(json.load(f).keys())

    missing_ids = all_problems - existing_ids
    print(f"\nMissing problems: {len(missing_ids)}")

    if len(missing_ids) == 0:
        print("\n✓ No missing problems - metadata is already complete!")
        return 0

    # Calculate z-scores and metadata for missing problems
    missing_data = []

    for problem_id in sorted(missing_ids):
        # Load points from mapping file
        mapping_file = Path('vlm-inputs') / f'{problem_id}_mapping.json'
        with open(mapping_file) as f:
            mapping_data = json.load(f)
            points = mapping_data['points']

        # Calculate z-score
        z_score = calculate_z_score(points)

        # Load optimal tour and calculate length
        optimal_file = Path('optimal-tsps') / problem_id
        if not optimal_file.exists():
            optimal_file = Path('optimal-tsps') / f'{problem_id}.json'

        with open(optimal_file) as f:
            optimal_points = json.load(f)

        optimal_length = calculate_tour_length(optimal_points)

        # Count hull points (for completeness)
        from scipy.spatial import ConvexHull
        coords = np.array([(p['x'], p['y']) for p in points])
        try:
            hull = ConvexHull(coords)
            hull_points = len(hull.vertices)
        except:
            hull_points = len(points)

        missing_data.append({
            'problem_id': problem_id,
            'num_points': len(points),
            'z_score': z_score,
            'hull_points': hull_points,
            'optimal_length': round(optimal_length, 1)
        })

    missing_df = pd.DataFrame(missing_data)

    print(f"\nZ-score statistics for missing problems:")
    print(f"  Mean: {missing_df['z_score'].mean():.1f}")
    print(f"  Median: {missing_df['z_score'].median():.1f}")
    print(f"  Min: {missing_df['z_score'].min():.1f}")
    print(f"  Max: {missing_df['z_score'].max():.1f}")

    # Classify as clustered vs disperse to achieve 36/36 split overall
    current_clustered = (existing_df['group'] == 'clustered').sum()
    current_disperse = (existing_df['group'] == 'disperse').sum()

    target_clustered = 36
    target_disperse = 36

    need_clustered = target_clustered - current_clustered
    need_disperse = target_disperse - current_disperse

    print(f"\nTarget distribution (36/36 split):")
    print(f"  Currently: {current_clustered} clustered, {current_disperse} disperse")
    print(f"  Need to add: {need_clustered} clustered, {need_disperse} disperse")

    # Sort by z-score (descending) and assign top N as clustered
    missing_df = missing_df.sort_values('z_score', ascending=False)
    missing_df['group'] = ''
    missing_df.iloc[:need_clustered, missing_df.columns.get_loc('group')] = 'clustered'
    missing_df.iloc[need_clustered:need_clustered + need_disperse,
                    missing_df.columns.get_loc('group')] = 'disperse'

    print(f"\nZ-score thresholds:")
    clustered_z = missing_df[missing_df['group'] == 'clustered']['z_score']
    disperse_z = missing_df[missing_df['group'] == 'disperse']['z_score']
    print(f"  Clustered: {clustered_z.min():.1f} to {clustered_z.max():.1f}")
    print(f"  Disperse: {disperse_z.min():.1f} to {disperse_z.max():.1f}")

    # Combine with existing metadata
    missing_df = missing_df[['problem_id', 'num_points', 'group', 'hull_points', 'optimal_length']]
    combined_df = pd.concat([existing_df, missing_df], ignore_index=True)

    # Verify final distribution
    print(f"\nFinal distribution:")
    print(f"  Total problems: {len(combined_df)}")
    print(f"  Clustered: {(combined_df['group'] == 'clustered').sum()}")
    print(f"  Disperse: {(combined_df['group'] == 'disperse').sum()}")
    print("\nBy size:")
    print(combined_df.groupby(['num_points', 'group']).size().unstack(fill_value=0))

    # Save updated metadata
    combined_df.to_csv('problem_metadata.csv', index=False)
    print(f"\n✓ Updated problem_metadata.csv with {len(missing_df)} new problems")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

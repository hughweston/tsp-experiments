#!/usr/bin/env python3
"""
Measure VLM cluster deviance.

Analyzes how well VLM TSP solutions respect their own clustering boundaries.
Implements the cluster deviance metric from Marapudi et al.
"""

import argparse
import json
import sys
import csv
from pathlib import Path


def load_tsp_points(tsp_file):
    """Load points from TSP file."""
    with open(tsp_file) as f:
        data = json.load(f)
    if isinstance(data, dict) and "points" in data:
        return data["points"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected TSP file format")


def load_solutions(solutions_file):
    """Load VLM TSP solutions from JSON file."""
    with open(solutions_file) as f:
        return json.load(f)


def load_clusters(clusters_file):
    """Load VLM cluster assignments from JSON file."""
    with open(clusters_file) as f:
        data = json.load(f)
    # Convert string keys to integers
    return {
        problem_id: {int(k): v for k, v in assignments.items()} if assignments else {}
        for problem_id, assignments in data.items()
    }


def parse_tour(tour_input):
    """Parse tour from string or list."""
    if isinstance(tour_input, list):
        return tour_input
    return [int(x.strip()) for x in tour_input.split(",")]




def calculate_cluster_deviance(tour, cluster_assignments):
    """
    Calculate cluster deviance metric from Marapudi et al.

    Cluster deviance = (t - c) / (n - c)

    Where:
        t = number of cross-cluster transitions in TSP tour
        c = number of clusters
        n = total number of points

    Returns:
        - transitions: Number of cross-cluster transitions (t)
        - num_clusters: Number of clusters (c)
        - num_points: Total number of points (n)
        - cluster_deviance: Normalized deviance score [0, 1]
            - 0 = perfect congruence (tour respects clusters)
            - 1 = maximum deviance
    """
    n = len(tour)
    if n < 2:
        return {
            "transitions": 0,
            "num_clusters": 0,
            "num_points": n,
            "cluster_deviance": 0.0
        }

    # Count cross-cluster transitions
    transitions = 0
    for i in range(n):
        current = tour[i]
        next_point = tour[(i + 1) % n]  # Wrap around for closed tour

        if cluster_assignments[current] != cluster_assignments[next_point]:
            transitions += 1

    # Count number of unique clusters
    num_clusters = len(set(cluster_assignments.values()))

    # Calculate normalized cluster deviance
    # (t - c) / (n - c)
    if n == num_clusters:
        # Edge case: if every point is its own cluster
        cluster_deviance = 0.0
    else:
        cluster_deviance = (transitions - num_clusters) / (n - num_clusters)

    return {
        "transitions": transitions,
        "num_clusters": num_clusters,
        "num_points": n,
        "cluster_deviance": cluster_deviance,
    }




def batch_analyze(clusters_file, solutions_file, optimal_dir, output_file):
    """
    Batch analyze cluster deviance for all problems.

    Args:
        clusters_file: JSON file with cluster assignments
        solutions_file: JSON file with TSP solutions
        optimal_dir: Directory with optimal TSP JSON files
        output_file: CSV file to write results
    """
    optimal_dir = Path(optimal_dir)

    # Load clusters and solutions
    clusters = load_clusters(clusters_file)
    solutions = load_solutions(solutions_file)

    results = []

    # Process each problem
    for problem_id in sorted(clusters.keys()):
        # Skip if no cluster assignments
        if not clusters[problem_id]:
            print(f"⚠ Skipping {problem_id}: No cluster assignments")
            continue

        # Skip if solution not available
        if problem_id not in solutions or not solutions[problem_id]:
            print(f"⚠ Skipping {problem_id}: No TSP solution found")
            continue

        # Load optimal TSP points
        # Try with and without .json extension
        optimal_file = optimal_dir / problem_id
        if not optimal_file.exists():
            optimal_file = optimal_dir / f"{problem_id}.json"

        if not optimal_file.exists():
            print(f"⚠ Skipping {problem_id}: Optimal TSP file not found")
            continue

        try:
            points = load_tsp_points(optimal_file)
            tour = parse_tour(solutions[problem_id])
            cluster_assignments = clusters[problem_id]

            # Validate
            if len(tour) != len(points):
                print(f"⚠ Skipping {problem_id}: Tour length mismatch ({len(tour)} vs {len(points)})")
                continue

            if len(cluster_assignments) != len(points):
                print(f"⚠ Skipping {problem_id}: Cluster coverage mismatch ({len(cluster_assignments)} vs {len(points)})")
                continue

            # Calculate metrics
            metrics = calculate_cluster_deviance(tour, cluster_assignments)

            results.append({
                'problem_id': problem_id,
                'num_points': metrics['num_points'],
                'num_clusters': metrics['num_clusters'],
                'transitions': metrics['transitions'],
                'cluster_deviance': metrics['cluster_deviance'],
            })

            print(f"✓ {problem_id}: {metrics['num_clusters']} clusters, deviance={metrics['cluster_deviance']:.3f}")

        except Exception as e:
            print(f"✗ Error processing {problem_id}: {e}")
            continue

    # Write results to CSV
    if results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['problem_id', 'num_points', 'num_clusters', 'transitions', 'cluster_deviance'])
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✓ Wrote {len(results)} results to {output_file}")

        # Print summary statistics
        avg_deviance = sum(r['cluster_deviance'] for r in results) / len(results)
        avg_clusters = sum(r['num_clusters'] for r in results) / len(results)
        print(f"\nSummary Statistics:")
        print(f"  Average cluster deviance: {avg_deviance:.3f}")
        print(f"  Average number of clusters: {avg_clusters:.1f}")
    else:
        print("\n⚠ No results to write")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Measure VLM cluster deviance (Marapudi et al. metric)"
    )
    parser.add_argument("clusters_file", help="JSON file with cluster assignments (e.g., claude_clusters.json)")
    parser.add_argument("solutions_file", help="JSON file with TSP solutions (e.g., claude_opus_4.5_solutions.json)")
    parser.add_argument("-o", "--optimal-dir", default="optimal-tsps", help="Directory with optimal TSP files (default: optimal-tsps)")
    parser.add_argument("--output", default="cluster_deviance_results.csv", help="Output CSV file (default: cluster_deviance_results.csv)")

    args = parser.parse_args()

    return batch_analyze(args.clusters_file, args.solutions_file, args.optimal_dir, args.output)


if __name__ == "__main__":
    sys.exit(main())

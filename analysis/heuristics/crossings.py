#!/usr/bin/env python3
"""
Detect and count crossings in TSP solutions.

A crossing occurs when two edges in the tour intersect.
"""

import argparse
import json
import sys
import csv
from pathlib import Path


def load_tsp_points(tsp_file):
    """Load points from TSP file (points in optimal order)."""
    with open(tsp_file) as f:
        data = json.load(f)
    if isinstance(data, dict) and "points" in data:
        return data["points"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected TSP file format: {tsp_file}")


def load_solutions(solutions_file):
    """Load VLM TSP solutions from JSON file."""
    with open(solutions_file) as f:
        return json.load(f)


def parse_tour(tour_input):
    """Parse tour from string or list."""
    if isinstance(tour_input, list):
        return tour_input
    return [int(x.strip()) for x in tour_input.split(",")]


def ccw(A, B, C):
    """
    Check if three points are in counter-clockwise order.

    Used for line segment intersection test.
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(A, B, C, D):
    """
    Check if line segment AB intersects with line segment CD.

    Returns True if they intersect (crossing), False otherwise.
    Endpoints touching don't count as crossings.
    """
    # Check if they share an endpoint
    if A == C or A == D or B == C or B == D:
        return False

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def count_crossings(points, tour_indices):
    """
    Count the number of edge crossings in a TSP tour.

    Args:
        points: List of {x, y} dicts (point coordinates in order)
        tour_indices: List of indices representing visit order (0-indexed)

    Returns:
        Number of crossings (edge intersections)
    """
    n = len(tour_indices)
    if n < 4:
        return 0  # Need at least 4 points to have a crossing

    crossings = 0

    # Check all pairs of edges
    for i in range(n):
        for j in range(i + 2, n):
            # Skip adjacent edges (they share a vertex)
            if j == (i + n - 1) % n or i == (j + 1) % n:
                continue

            # Get the two edges
            edge1_start = points[tour_indices[i]]
            edge1_end = points[tour_indices[(i + 1) % n]]
            edge2_start = points[tour_indices[j]]
            edge2_end = points[tour_indices[(j + 1) % n]]

            # Convert to tuples for intersection test
            A = (edge1_start["x"], edge1_start["y"])
            B = (edge1_end["x"], edge1_end["y"])
            C = (edge2_start["x"], edge2_start["y"])
            D = (edge2_end["x"], edge2_end["y"])

            if segments_intersect(A, B, C, D):
                crossings += 1

    return crossings


def batch_analyze(solutions_file, optimal_dir, output_file, mapping_dir="vlm-inputs"):
    """
    Batch analyze crossings for all VLM solutions.

    Args:
        solutions_file: JSON file with TSP solutions
        optimal_dir: Directory with optimal TSP files
        output_file: CSV file to write results
        mapping_dir: Directory with label mapping files
    """
    optimal_dir = Path(optimal_dir)
    mapping_dir = Path(mapping_dir)

    # Load solutions
    solutions = load_solutions(solutions_file)

    results = []

    # Process each problem
    for problem_id in sorted(solutions.keys()):
        if not solutions[problem_id]:
            print(f"⚠ Skipping {problem_id}: No solution found")
            continue

        # Load mapping file to convert labels to indices
        mapping_file = mapping_dir / f"{problem_id}_mapping.json"
        if not mapping_file.exists():
            print(f"⚠ Skipping {problem_id}: Mapping file not found")
            continue

        # Load optimal TSP points
        # Try with and without .json extension
        optimal_file = optimal_dir / problem_id
        if not optimal_file.exists():
            optimal_file = optimal_dir / f"{problem_id}.json"

        if not optimal_file.exists():
            print(f"⚠ Skipping {problem_id}: TSP file not found")
            continue

        try:
            # Load mapping
            with open(mapping_file) as f:
                mapping_data = json.load(f)
                label_to_index = {int(k): v for k, v in mapping_data['label_to_index'].items()}
                points = mapping_data['points']  # Use points from mapping file

            tour_labels = parse_tour(solutions[problem_id])

            # Validate
            if len(tour_labels) != len(points):
                print(f"⚠ Skipping {problem_id}: Tour length mismatch ({len(tour_labels)} vs {len(points)})")
                continue

            # Convert labels to indices using the mapping
            tour_indices = [label_to_index[label] for label in tour_labels]

            # Count crossings using the correct point order
            crossings = count_crossings(points, tour_indices)

            results.append({
                'problem_id': problem_id,
                'num_points': len(points),
                'crossings': crossings,
            })

            status = "✓" if crossings == 0 else "✗"
            print(f"{status} {problem_id}: {crossings} crossing(s)")

        except Exception as e:
            print(f"✗ Error processing {problem_id}: {e}")
            continue

    # Write results to CSV
    if results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['problem_id', 'num_points', 'crossings'])
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✓ Wrote {len(results)} results to {output_file}")

        # Print summary statistics
        total_crossings = sum(r['crossings'] for r in results)
        zero_crossing_count = sum(1 for r in results if r['crossings'] == 0)
        avg_crossings = total_crossings / len(results) if results else 0

        print(f"\nSummary Statistics:")
        print(f"  Total solutions: {len(results)}")
        print(f"  Zero crossings: {zero_crossing_count} ({zero_crossing_count/len(results)*100:.1f}%)")
        print(f"  Average crossings: {avg_crossings:.2f}")
        print(f"  Max crossings: {max(r['crossings'] for r in results)}")
    else:
        print("\n⚠ No results to write")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Count edge crossings in TSP solutions"
    )
    parser.add_argument("solutions_file", help="JSON file with TSP solutions")
    parser.add_argument("-d", "--optimal-dir", default="optimal-tsps",
                       help="Directory with TSP point files (default: optimal-tsps)")
    parser.add_argument("-m", "--mapping-dir", default="vlm-inputs",
                       help="Directory with label mapping files (default: vlm-inputs)")
    parser.add_argument("-o", "--output", default="crossings_results.csv",
                       help="Output CSV file (default: crossings_results.csv)")

    args = parser.parse_args()

    return batch_analyze(args.solutions_file, args.optimal_dir, args.output, args.mapping_dir)


if __name__ == "__main__":
    sys.exit(main())
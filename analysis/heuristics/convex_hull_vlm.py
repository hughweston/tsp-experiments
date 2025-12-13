#!/usr/bin/env python3
"""
Measure VLM adherence to convex hull heuristic.

The convex hull heuristic suggests visiting points on the convex hull
(outer boundary) consecutively, often at the beginning of the tour,
before visiting interior points.
"""

import argparse
import json
import sys
import csv
from pathlib import Path
from scipy.spatial import ConvexHull


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


def parse_tour(tour_input):
    """Parse tour from string or list."""
    if isinstance(tour_input, list):
        return tour_input
    return [int(x.strip()) for x in tour_input.split(",")]


def compute_hull_indices(points):
    """
    Compute convex hull and return set of point indices on the hull.

    Returns:
        set of indices (0-indexed) of points on convex hull
    """
    if len(points) < 3:
        return set(range(len(points)))

    coords = [(p["x"], p["y"]) for p in points]
    try:
        hull = ConvexHull(coords)
        return set(hull.vertices)
    except Exception as e:
        print(f"Warning: Could not compute hull: {e}")
        return set()


def analyze_hull_adherence(tour_indices, hull_indices):
    """
    Analyze how well tour adheres to convex hull heuristic.

    Convex hull heuristic: Visit hull points consecutively (typically first),
    forming the outer boundary, before visiting interior points.

    Args:
        tour_indices: List of point indices in visit order (0-indexed)
        hull_indices: Set of indices on convex hull

    Returns dict with metrics:
        - hull_contiguous: bool, whether hull points form a contiguous segment
        - hull_switches: number of times tour switches between hull/interior
        - hull_first_position: position of first hull point in tour (0-indexed)
        - hull_segment_length: length of longest contiguous hull segment
        - adherence_score: 0-1 score (1.0 = perfect hull adherence)
    """
    n = len(tour_indices)
    total_hull_points = len(hull_indices)

    if total_hull_points == 0:
        return {
            "hull_contiguous": True,
            "hull_switches": 0,
            "hull_first_position": 0,
            "hull_segment_length": 0,
            "adherence_score": 1.0,
            "hull_count": 0,
        }

    # Check if each point is on hull
    is_hull = [idx in hull_indices for idx in tour_indices]

    # Find first hull point
    hull_first_position = next((i for i, h in enumerate(is_hull) if h), 0)

    # Count switches between hull and interior
    switches = 0
    for i in range(n):
        if is_hull[i] != is_hull[(i + 1) % n]:
            switches += 1

    # Find longest contiguous hull segment
    max_segment = 0
    current_segment = 0
    for i in range(n * 2):  # Go around twice to handle wrap-around
        if is_hull[i % n]:
            current_segment += 1
            max_segment = max(max_segment, current_segment)
        else:
            current_segment = 0

    # Hull is contiguous if all hull points form one segment
    hull_contiguous = max_segment >= total_hull_points

    # Adherence score: perfect if hull points are contiguous, penalized by switches
    # Max switches for non-contiguous hull is 2 * total_hull_points
    max_possible_switches = 2 * total_hull_points if total_hull_points > 0 else 1
    adherence_score = 1.0 - (switches / max_possible_switches)

    return {
        "hull_contiguous": hull_contiguous,
        "hull_switches": switches,
        "hull_first_position": hull_first_position,
        "hull_segment_length": max_segment,
        "adherence_score": adherence_score,
        "hull_count": total_hull_points,
    }


def batch_analyze(solutions_file, optimal_dir, output_file, mapping_dir="vlm-inputs"):
    """
    Batch analyze convex hull adherence for all VLM solutions.

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

            # Compute hull and analyze adherence
            hull_indices = compute_hull_indices(points)
            metrics = analyze_hull_adherence(tour_indices, hull_indices)

            results.append({
                'problem_id': problem_id,
                'num_points': len(points),
                'hull_count': metrics['hull_count'],
                'hull_contiguous': metrics['hull_contiguous'],
                'hull_switches': metrics['hull_switches'],
                'hull_segment_length': metrics['hull_segment_length'],
                'hull_first_position': metrics['hull_first_position'],
                'adherence_score': metrics['adherence_score'],
            })

            status = "✓" if metrics['hull_contiguous'] else "○"
            print(f"{status} {problem_id}: adherence={metrics['adherence_score']:.3f}, contiguous={metrics['hull_contiguous']}")

        except Exception as e:
            print(f"✗ Error processing {problem_id}: {e}")
            continue

    # Write results to CSV
    if results:
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['problem_id', 'num_points', 'hull_count', 'hull_contiguous',
                         'hull_switches', 'hull_segment_length', 'hull_first_position',
                         'adherence_score']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✓ Wrote {len(results)} results to {output_file}")

        # Print summary statistics
        avg_adherence = sum(r['adherence_score'] for r in results) / len(results)
        contiguous_count = sum(1 for r in results if r['hull_contiguous'])

        print(f"\nSummary Statistics:")
        print(f"  Average adherence score: {avg_adherence:.3f}")
        print(f"  Contiguous hull: {contiguous_count}/{len(results)} ({contiguous_count/len(results)*100:.1f}%)")
    else:
        print("\n⚠ No results to write")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Measure convex hull heuristic adherence for VLM solutions"
    )
    parser.add_argument("solutions_file", help="JSON file with TSP solutions")
    parser.add_argument("-d", "--optimal-dir", default="optimal-tsps",
                       help="Directory with TSP point files (default: optimal-tsps)")
    parser.add_argument("-m", "--mapping-dir", default="vlm-inputs",
                       help="Directory with label mapping files (default: vlm-inputs)")
    parser.add_argument("-o", "--output", default="hull_vlm_results.csv",
                       help="Output CSV file (default: hull_vlm_results.csv)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed per-problem analysis")

    args = parser.parse_args()

    return batch_analyze(args.solutions_file, args.optimal_dir, args.output, args.mapping_dir)


if __name__ == "__main__":
    sys.exit(main())

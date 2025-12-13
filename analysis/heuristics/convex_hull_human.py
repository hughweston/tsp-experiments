#!/usr/bin/env python3
"""
Measure human adherence to convex hull heuristic from tsp-trials-anonymized.json.

Analyzes human TSP solutions to see if they follow the convex hull heuristic
(visiting boundary points consecutively before interior points).
"""

import argparse
import json
import sys
from scipy.spatial import ConvexHull


def compute_hull_indices(points):
    """
    Compute convex hull and return set of point indices on the hull.

    Args:
        points: List of {x, y} dicts

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


def find_point_index(clicked_point, all_points, tolerance=5):
    """
    Find which point in all_points matches the clicked coordinates.

    Args:
        clicked_point: {x, y} from human click
        all_points: List of {x, y} from stimulus
        tolerance: Max distance to consider a match

    Returns:
        Index of matching point, or None if not found
    """
    for i, p in enumerate(all_points):
        dx = abs(p["x"] - clicked_point["x"])
        dy = abs(p["y"] - clicked_point["y"])
        if dx <= tolerance and dy <= tolerance:
            return i
    return None


def analyze_hull_adherence(tour_indices, hull_indices):
    """
    Analyze how well tour adheres to convex hull heuristic.

    Returns dict with metrics (same as measure_hull_heuristic.py).
    """
    n = len(tour_indices)
    total_hull_points = len(hull_indices)

    if total_hull_points == 0 or n == 0:
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

    # Adherence score
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


def process_trial(trial):
    """Process a single trial from the JSON and return hull adherence metrics."""
    stimulus = trial["stimulus"]
    points = stimulus["points"]
    tsp_sequence = trial["tsp"]

    # Compute hull
    hull_indices = compute_hull_indices(points)

    # Convert human clicks to point indices
    tour_indices = []
    for click in tsp_sequence:
        idx = find_point_index(click, points)
        if idx is not None:
            tour_indices.append(idx)
        else:
            # Skip if we can't match a click (shouldn't happen)
            print(f"Warning: Could not match click at ({click['x']}, {click['y']})")

    # Analyze adherence
    metrics = analyze_hull_adherence(tour_indices, hull_indices)

    # Add trial metadata
    metrics["participant"] = trial["participant_number"]
    metrics["base_uuid"] = stimulus["base_uuid"]
    metrics["unique_uuid"] = stimulus["unique_uuid"]
    metrics["group"] = stimulus["group"]
    metrics["num_points"] = stimulus["number_of_points"]

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Measure human convex hull heuristic adherence"
    )
    parser.add_argument(
        "json_file",
        help="Path to tsp-trials-anonymized.json",
    )
    parser.add_argument(
        "--participant",
        help="Filter by specific participant number",
    )
    parser.add_argument(
        "--group",
        choices=["clustered", "disperse"],
        help="Filter by point distribution group",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        choices=[10, 15, 20, 25],
        help="Filter by number of points",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output CSV file with results (default: print summary)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-trial details",
    )

    args = parser.parse_args()

    # Load trials
    with open(args.json_file) as f:
        trials = json.load(f)

    # Filter trials
    if args.participant:
        trials = [t for t in trials if t["participant_number"] == args.participant]
    if args.group:
        trials = [t for t in trials if t["stimulus"]["group"] == args.group]
    if args.num_points:
        trials = [t for t in trials if t["stimulus"]["number_of_points"] == args.num_points]

    print(f"Analyzing {len(trials)} trials...")

    # Process all trials
    results = []
    for trial in trials:
        metrics = process_trial(trial)
        results.append(metrics)

        if args.verbose:
            print(f"\nTrial: {metrics['unique_uuid'][:8]}... ({metrics['group']}, {metrics['num_points']} points)")
            print(f"  Hull contiguous: {metrics['hull_contiguous']}")
            print(f"  Hull switches: {metrics['hull_switches']}")
            print(f"  Adherence score: {metrics['adherence_score']:.3f}")

    # Calculate summary statistics
    if results:
        avg_adherence = sum(r["adherence_score"] for r in results) / len(results)
        avg_switches = sum(r["hull_switches"] for r in results) / len(results)
        contiguous_count = sum(1 for r in results if r["hull_contiguous"])
        contiguous_pct = (contiguous_count / len(results)) * 100

        print(f"\n{'='*60}")
        print(f"Summary Statistics ({len(results)} trials)")
        print(f"{'='*60}")
        print(f"Average adherence score: {avg_adherence:.3f}")
        print(f"Average hull switches: {avg_switches:.2f}")
        print(f"Hull contiguous: {contiguous_count}/{len(results)} ({contiguous_pct:.1f}%)")
        print(f"{'='*60}\n")

    # Output to CSV if requested
    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            fieldnames = [
                "participant",
                "base_uuid",
                "unique_uuid",
                "group",
                "num_points",
                "hull_count",
                "hull_contiguous",
                "hull_switches",
                "hull_segment_length",
                "hull_first_position",
                "adherence_score",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
